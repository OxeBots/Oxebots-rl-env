import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
import wandb
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from walk_env import WalkEnv


class WalkActor(nn.Module):
    def __init__(self, obs_dim=78, action_dim=23, init_noise_std=0.3):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_noise_std))

    def forward(self, x):
        x = torch.nn.functional.elu(self.layer_norm(self.fc1(x)))
        x = torch.nn.functional.elu(self.fc2(x))
        x = torch.nn.functional.elu(self.fc3(x))
        return self.fc4(x)

    def distribution(self, obs):
        return torch.distributions.Normal(self.forward(obs), torch.exp(self.log_std))

    def act(self, obs):
        dist = self.distribution(obs)
        actions = dist.sample()
        return actions, dist.log_prob(actions).sum(dim=-1)

    def evaluate(self, obs, actions):
        dist = self.distribution(obs)
        return dist.log_prob(actions).sum(dim=-1), dist.entropy().sum(dim=-1)


class WalkCritic(nn.Module):
    def __init__(self, obs_dim=78):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.nn.functional.elu(self.layer_norm(self.fc1(x)))
        x = torch.nn.functional.elu(self.fc2(x))
        x = torch.nn.functional.elu(self.fc3(x))
        return self.fc4(x).squeeze(-1)


class ActorForExport(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.fc1 = actor.fc1
        self.layer_norm = actor.layer_norm
        self.fc2 = actor.fc2
        self.fc3 = actor.fc3
        self.fc4 = actor.fc4

    def forward(self, x):
        x = torch.nn.functional.elu(self.layer_norm(self.fc1(x)))
        x = torch.nn.functional.elu(self.fc2(x))
        x = torch.nn.functional.elu(self.fc3(x))
        return torch.clamp(self.fc4(x), -10.0, 10.0)


def load_onnx_weights(actor, onnx_path):
    onnx_model = onnx.load(onnx_path)
    weights = {}
    for init in onnx_model.graph.initializer:
        weights[init.name] = torch.from_numpy(numpy_helper.to_array(init).copy())

    state = actor.state_dict()
    for name in [
        "fc1.weight", "fc1.bias",
        "layer_norm.weight", "layer_norm.bias",
        "fc2.weight", "fc2.bias",
        "fc3.weight", "fc3.bias",
        "fc4.weight", "fc4.bias",
    ]:
        if name in weights:
            state[name] = weights[name]
    actor.load_state_dict(state, strict=False)


def export_actor_onnx(actor, obs_dim, path):
    export_model = ActorForExport(actor)
    export_model.eval()
    torch.onnx.export(
        export_model,
        torch.randn(1, obs_dim),
        path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, action_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        self.step = 0

    def add(self, obs, actions, rewards, dones, log_probs, values):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.log_probs[self.step] = log_probs
        self.values[self.step] = values
        self.step += 1

    def compute_gae(self, last_values, gamma, lam):
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
        self.returns = self.advantages + self.values

    def mini_batches(self, num_batches):
        total = self.num_steps * self.num_envs
        size = total // num_batches
        indices = torch.randperm(total, device=self.device)
        obs = self.obs.reshape(-1, self.obs.shape[-1])
        act = self.actions.reshape(-1, self.actions.shape[-1])
        lp = self.log_probs.reshape(-1)
        ret = self.returns.reshape(-1)
        adv = self.advantages.reshape(-1)
        for start in range(0, total, size):
            idx = indices[start:start + size]
            yield obs[idx], act[idx], lp[idx], ret[idx], adv[idx]

    def reset(self):
        self.step = 0


def train(onnx_path):
    device = torch.device("cpu")
    log_dir = "./training/logs/walk_finetune/"
    model_dir = "./training/models/walk/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    num_envs = max(1, os.cpu_count() - 1)
    num_steps_per_env = 4096
    num_learning_epochs = 5
    num_mini_batches = 4
    clip_param = 0.2
    gamma = 0.99
    lam = 0.95
    value_loss_coef = 1.0
    entropy_coef = 0.01
    learning_rate = 1e-3
    max_grad_norm = 1.0
    desired_kl = 0.01
    init_noise_std = 0.3
    total_timesteps = 100_000_000
    save_interval = 50

    steps_per_iter = num_steps_per_env * num_envs
    total_iters = total_timesteps // steps_per_iter
    obs_dim = 78
    action_dim = 23

    print(f"Configurando {num_envs} ambientes em paralelo...")
    vec_env = make_vec_env(WalkEnv, n_envs=num_envs, vec_env_cls=SubprocVecEnv)

    actor = WalkActor(obs_dim, action_dim, init_noise_std).to(device)
    critic = WalkCritic(obs_dim).to(device)

    print(f"Carregando pesos do ator: {onnx_path}")
    load_onnx_weights(actor, onnx_path)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=learning_rate
    )

    buffer = RolloutBuffer(num_steps_per_env, num_envs, obs_dim, action_dim, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = wandb.init(
        project="oxebots_walk_train",
        name=f"ppo-walk-finetune-{timestamp}",
        config={
            "num_envs": num_envs,
            "num_steps_per_env": num_steps_per_env,
            "num_learning_epochs": num_learning_epochs,
            "num_mini_batches": num_mini_batches,
            "clip_param": clip_param,
            "gamma": gamma,
            "lam": lam,
            "entropy_coef": entropy_coef,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "desired_kl": desired_kl,
            "init_noise_std": init_noise_std,
            "total_timesteps": total_timesteps,
            "resumed_from": onnx_path,
        },
        save_code=True,
    )

    obs = torch.from_numpy(vec_env.reset()).float().to(device)
    ep_rewards = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs, dtype=int)
    completed_rewards = []
    completed_lengths = []
    total_steps = 0
    best_mean_reward = float("-inf")

    print(f"=== Fine-tuning Walk ===")
    print(f"  Modelo base: {onnx_path}")
    print(f"  Ambientes: {num_envs}")
    print(f"  Steps/iteração: {steps_per_iter:,}")
    print(f"  Total iterações: {total_iters:,}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print()

    try:
        for it in range(total_iters):
            t_start = time.time()

            actor.eval()
            critic.eval()
            with torch.no_grad():
                for _ in range(num_steps_per_env):
                    actions, log_probs = actor.act(obs)
                    values = critic(obs)

                    actions_env = actions.clamp(-1.0, 1.0).cpu().numpy()
                    obs_np, rewards_np, dones_np, infos = vec_env.step(actions_env)

                    buffer.add(
                        obs, actions,
                        torch.from_numpy(rewards_np.astype(np.float32)).to(device),
                        torch.from_numpy(dones_np.astype(np.float32)).to(device),
                        log_probs, values,
                    )

                    obs = torch.from_numpy(obs_np.astype(np.float32)).to(device)
                    ep_rewards += rewards_np
                    ep_lengths += 1
                    total_steps += num_envs

                    for i in range(num_envs):
                        if dones_np[i]:
                            completed_rewards.append(ep_rewards[i])
                            completed_lengths.append(ep_lengths[i])
                            ep_rewards[i] = 0
                            ep_lengths[i] = 0

                last_values = critic(obs)
                buffer.compute_gae(last_values, gamma, lam)

            actor.train()
            critic.train()
            p_loss_acc, v_loss_acc, ent_acc, n_updates = 0.0, 0.0, 0.0, 0

            for _ in range(num_learning_epochs):
                for b_obs, b_act, b_old_lp, b_ret, b_adv in buffer.mini_batches(num_mini_batches):
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                    new_lp, entropy = actor.evaluate(b_obs, b_act)
                    ratio = torch.exp(new_lp - b_old_lp)

                    p_loss = -torch.min(
                        ratio * b_adv,
                        torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * b_adv,
                    ).mean()

                    v_loss = (b_ret - critic(b_obs)).pow(2).mean()
                    loss = p_loss + value_loss_coef * v_loss - entropy_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(actor.parameters()) + list(critic.parameters()), max_grad_norm
                    )
                    optimizer.step()

                    p_loss_acc += p_loss.item()
                    v_loss_acc += v_loss.item()
                    ent_acc += entropy.mean().item()
                    n_updates += 1

            with torch.no_grad():
                new_lp, _ = actor.evaluate(
                    buffer.obs.reshape(-1, obs_dim),
                    buffer.actions.reshape(-1, action_dim),
                )
                kl = (buffer.log_probs.reshape(-1) - new_lp).mean().item()

            if kl > desired_kl * 2.0:
                learning_rate = max(1e-5, learning_rate / 1.5)
            elif kl < desired_kl / 2.0:
                learning_rate = min(1e-2, learning_rate * 1.5)
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rate

            buffer.reset()

            elapsed = time.time() - t_start
            fps = steps_per_iter / elapsed

            log_data = {
                "loss/policy": p_loss_acc / n_updates,
                "loss/value": v_loss_acc / n_updates,
                "loss/entropy": ent_acc / n_updates,
                "ppo/kl": kl,
                "ppo/lr": learning_rate,
                "perf/fps": fps,
                "perf/total_steps": total_steps,
            }

            if completed_rewards:
                mean_rew = np.mean(completed_rewards[-100:])
                mean_len = np.mean(completed_lengths[-100:])
                log_data["episode/mean_reward"] = mean_rew
                log_data["episode/mean_length"] = mean_len

                if mean_rew > best_mean_reward and len(completed_rewards) >= 10:
                    best_mean_reward = mean_rew
                    best_path = os.path.join(model_dir, "best_finetune")
                    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, best_path + ".pt")
                    export_actor_onnx(actor, obs_dim, best_path + ".onnx")

            wandb.log(log_data, step=total_steps)

            if (it + 1) % save_interval == 0:
                ckpt_path = os.path.join(model_dir, f"finetune_iter_{it+1}")
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, ckpt_path + ".pt")

            if (it + 1) % 10 == 0 or it == 0:
                rew_str = f"{np.mean(completed_rewards[-100:]):.1f}" if completed_rewards else "---"
                print(
                    f"[{it+1}/{total_iters}] "
                    f"steps={total_steps:,} "
                    f"rew={rew_str} "
                    f"p={p_loss_acc/n_updates:.4f} "
                    f"v={v_loss_acc/n_updates:.4f} "
                    f"kl={kl:.4f} "
                    f"lr={learning_rate:.1e} "
                    f"fps={fps:.0f}"
                )

    except KeyboardInterrupt:
        print("\nTreinamento interrompido.")

    final_path = os.path.join(model_dir, f"walk_finetune_{timestamp}")
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, final_path + ".pt")
    export_actor_onnx(actor, obs_dim, final_path + ".onnx")
    print(f"Modelo salvo: {final_path}.pt")
    print(f"ONNX exportado: {final_path}.onnx")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    train(args.model)

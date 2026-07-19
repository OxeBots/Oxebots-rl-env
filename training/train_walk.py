from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from walk_env import WalkEnv
import os
from datetime import datetime


def train():
    N_ENVS = 8
    TOTAL_TIMESTEPS = 1_000_000

    log_dir = "./training/logs/walk/"
    checkpoint_dir = "./training/checkpoints/walk/"
    model_dir = "./training/models/walk/"
    for d in [log_dir, checkpoint_dir, model_dir]:
        os.makedirs(d, exist_ok=True)

    vec_env = make_vec_env(WalkEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MlpPolicy", vec_env, verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(model_dir, f"ppo_walk_{timestamp}")

    # EvalCallback precisa de um ambiente separado (não pode usar o vec_env)
    eval_env = make_vec_env(WalkEnv, n_envs=1)

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(250_000 // N_ENVS, 1),
            save_path=checkpoint_dir,
            name_prefix="walk_model",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best/"),
            log_path=log_dir,
            eval_freq=max(50_000 // N_ENVS, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),
    ])

    print(f"=== Treinamento Walk ===")
    print(f"  Ambientes paralelos: {N_ENVS}")
    print(f"  Buffer por update: {model.n_steps * N_ENVS} steps")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Device: {model.device}")
    print(f"  Modelo final: {final_model_path}")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=True,
            tb_log_name=f"PPO_Walk_{timestamp}",
            callback=callbacks,
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")

    model.save(final_model_path)
    print(f"Modelo salvo: {final_model_path}")


if __name__ == "__main__":
    train()

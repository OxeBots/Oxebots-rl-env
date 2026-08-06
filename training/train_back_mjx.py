import os
import time
from datetime import datetime
import jax

from getup_env_mjx import GetUpBackMjxEnv

try:
    from brax.training.agents.ppo import train as ppo_train
    import wandb
    HAS_BRAX_TRAIN = True
except ImportError:
    HAS_BRAX_TRAIN = False


if not hasattr(jax, 'device_put_replicated'):
    import jax.numpy as jnp
    def _device_put_replicated(x, devices):
        return jax.tree.map(lambda leaf: jax.device_put(jnp.stack([leaf] * len(devices))), x)
    jax.device_put_replicated = _device_put_replicated

def render_and_log_video(env, make_policy_fn, params, num_steps, max_steps=200):
    try:
        import numpy as np
        import mujoco
        import wandb

        if not wandb.run:
            return

        policy = make_policy_fn(params)
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        state = jit_reset(jax.random.PRNGKey(0))

        mj_model = env.mj_model
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=360, width=480)

        frames = []
        for _ in range(max_steps):
            qpos = np.array(state.pipeline_state.qpos)
            qvel = np.array(state.pipeline_state.qvel)
            mj_data.qpos[:len(qpos)] = qpos
            mj_data.qvel[:len(qvel)] = qvel
            mujoco.mj_forward(mj_model, mj_data)

            renderer.update_scene(mj_data)
            frames.append(renderer.render())

            action, _ = policy(state.obs, jax.random.PRNGKey(0))
            state = jit_step(state, action)
            if bool(state.done):
                break

        if frames:
            video_array = np.array(frames).transpose(0, 3, 1, 2)
            wandb.log({
                "video": wandb.Video(video_array, fps=30, format="mp4"),
                "num_steps": num_steps
            })
            print(f"\n📹 [WandB] Vídeo gravado e enviado para o passo {num_steps}!")
    except Exception as e:
        print(f"\n⚠️ Aviso ao gerar vídeo para WandB: {e}")

def train():
    if not HAS_BRAX_TRAIN:
        print("Erro: Brax e WandB são necessários para o treino em GPU via MJX.")
        print("Instale com: pip install brax wandb mujoco-mjx")
        return

   
    log_dir = "./training/logs/back_mjx/"
    model_dir = "./training/models/back_mjx/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # hip
    total_timesteps = 100_000_000
    num_envs = 4096  
    learning_rate = 3e-4
    batch_size = 2048
    unroll_length = 20
    num_minibatches = 32
    num_updates_per_batch = 8

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo-getup-back-mjx-{timestamp}"

    # wandB
    run = wandb.init(
        project="bahiart-mujoco-getup-mjx",
        name=run_name,
        config={
            "env": "GetUpBackMjxEnv",
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "backend": "mjx_gpu"
        }
    )

    print(f"🚀 Iniciando treino BACK 100% na GPU (MJX)...")
    print(f"Ambientes paralelos na VRAM: {num_envs}")
    print(f"Dispositivos JAX detectados: {jax.devices()}")

    env = GetUpBackMjxEnv()

    from tqdm import tqdm

    start_time = time.time()
    pbar = tqdm(total=total_timesteps, unit="step", desc="🏋️ Treino BACK (MJX)", dynamic_ncols=True)
    last_steps = 0

    def progress_callback(num_steps, metrics):
        nonlocal last_steps
        delta_steps = num_steps - last_steps
        last_steps = num_steps

        elapsed = time.time() - start_time
        sps = num_steps / elapsed if elapsed > 0 else 0

        pbar.update(delta_steps)

        postfix = {"SPS": f"{sps:.0f}"}
        if "eval/episode_reward" in metrics:
            postfix["Reward"] = f"{float(metrics['eval/episode_reward']):.2f}"
        elif "training/reward" in metrics:
            postfix["Reward"] = f"{float(metrics['training/reward']):.2f}"

        pbar.set_postfix(postfix)

        if wandb.run:
            wandb.log({"num_steps": num_steps, "sps": sps, **metrics})

    def policy_params_callback(num_steps, make_policy_fn, params):
        render_and_log_video(env, make_policy_fn, params, num_steps)

    try:
        
        make_inference_fn, params, _ = ppo_train.train(
            environment=env,
            num_timesteps=total_timesteps,
            num_envs=num_envs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            unroll_length=unroll_length,
            num_minibatches=num_minibatches,
            num_updates_per_batch=num_updates_per_batch,
            progress_fn=progress_callback,
            policy_params_fn=policy_params_callback,
            num_evals=10,
            seed=42
        )
    finally:
        pbar.close()

    print(f"✅ Treino BACK concluído em {(time.time() - start_time)/60:.2f} minutos!")

    # Salvar parâmetros do modelo
    model_path = os.path.join(model_dir, f"ppo_getup_back_mjx_{timestamp}.pkl")
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Modelo salvo em: {model_path}")

    run.finish()

if __name__ == "__main__":
    train()

import os
import time
from datetime import datetime
import jax

from getup_env_mjx import GetUpFrontMjxEnv

try:
    from brax.training.agents.ppo import train as ppo_train
    import wandb
    HAS_BRAX_TRAIN = True
except ImportError:
    HAS_BRAX_TRAIN = False

def train():
    if not HAS_BRAX_TRAIN:
        print("Erro: Brax e WandB são necessários para o treino em GPU via MJX.")
        print("Instale com: pip install brax wandb mujoco-mjx")
        return

    # Pastas de log e modelo
    log_dir = "./training/logs/front_mjx/"
    model_dir = "./training/models/front_mjx/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Hiperparâmetros acelerados em GPU
    total_timesteps = 100_000_000
    num_envs = 4096  # 4096 ambientes paralelos rodando 100% dentro da VRAM da GPU!
    learning_rate = 3e-4
    batch_size = 2048
    unroll_length = 20
    num_minibatches = 32
    num_updates_per_batch = 8

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo-getup-front-mjx-{timestamp}"

    # Inicializar WandB
    run = wandb.init(
        project="bahiart-mujoco-getup-mjx",
        name=run_name,
        config={
            "env": "GetUpFrontMjxEnv",
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "backend": "mjx_gpu"
        }
    )

    print(f"🚀 Iniciando treino FRONT 100% na GPU (MJX)...")
    print(f"Ambientes paralelos na VRAM: {num_envs}")
    print(f"Dispositivos JAX detectados: {jax.devices()}")

    env = GetUpFrontMjxEnv()

    start_time = time.time()

    def progress_callback(num_steps, metrics):
        elapsed = time.time() - start_time
        sps = num_steps / elapsed if elapsed > 0 else 0
        print(f"Passos: {num_steps}/{total_timesteps} | FPS (SPS): {sps:.0f} | Tempo Decorrido: {elapsed/60:.2f} min")
        if wandb.run:
            wandb.log({"num_steps": num_steps, "sps": sps, **metrics})

    # Rodar o PPO compilado em GPU via JAX / Brax
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
        seed=42
    )

    print(f"✅ Treino concluído em {(time.time() - start_time)/60:.2f} minutos!")

    # Salvar parâmetros do modelo
    model_path = os.path.join(model_dir, f"ppo_getup_front_mjx_{timestamp}.pkl")
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Modelo salvo em: {model_path}")

    run.finish()

if __name__ == "__main__":
    train()

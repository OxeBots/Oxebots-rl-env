import os
import time
import multiprocessing as mp
from datetime import datetime
import jax

# Garantir método 'spawn' no multiprocessing para evitar deadlocks com JAX XLA
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from getup_env_mjx import GetUpBackMjxEnv

try:
    from brax.training.agents.ppo import train as ppo_train
    HAS_BRAX_TRAIN = True
except ImportError:
    HAS_BRAX_TRAIN = False

try:
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# WandB e Renderização de Vídeo desativados para evitar gargalos na GPU/CPU
# import wandb

# Monkeypatch para compatibilidade com versões recentes do JAX no Brax
if not hasattr(jax, 'device_put_replicated'):
    import jax.numpy as jnp
    def _device_put_replicated(x, devices):
        return jax.tree.map(lambda leaf: jax.device_put(jnp.stack([leaf] * len(devices))), x)
    jax.device_put_replicated = _device_put_replicated

def render_and_log_video(env, make_policy_fn, params, num_steps, max_steps=200):
    # Desativado para evitar overhead de CPU e fork deadlocks durante o treino GPU JAX
    pass

def train():
    if not HAS_BRAX_TRAIN:
        print("Erro: Brax é necessário para o treino em GPU via MJX.")
        print("Instale com: pip install brax mujoco-mjx")
        return

    # Pastas de log e modelo
    log_dir = "./training/logs/back_mjx/"
    model_dir = "./training/models/back_mjx/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir) if HAS_TENSORBOARD else None

    # Hiperparâmetros otimizados para VRAM (2048 ambientes paralelos 100% em GPU)
    total_timesteps = 100_000_000
    num_envs = 2048  # ambientes paralelos rodando 
    episode_length = 1000  # Tamanho do episódio (passos de simulação por episódio)
    learning_rate = 3e-4
    unroll_length = 32
    batch_size = 2048
    num_minibatches = 8
    num_updates_per_batch = 4
    num_evals = 20  # Log por epoch/update (sem interrupções frequentes de transferência GPU-CPU)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"🚀 Iniciando treino BACK 100% na GPU (MJX)...")
    print(f"Ambientes paralelos na VRAM: {num_envs}")
    print(f"Dispositivos JAX detectados: {jax.devices()}")
    if writer:
        print(f"📊 Logs do TensorBoard sendo salvos em: {log_dir}")

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

        # Logar métricas no TensorBoard sem pausar a GPU
        if writer:
            for k, v in metrics.items():
                try:
                    writer.add_scalar(k, float(v), num_steps)
                except (ValueError, TypeError):
                    pass
            writer.add_scalar("eval/sps", sps, num_steps)
            writer.flush()

    def policy_params_callback(num_steps, make_policy_fn, params):
        pass

    try:
        # Rodar o PPO compilado em GPU via JAX / Brax por epochs
        make_inference_fn, params, _ = ppo_train.train(
            environment=env,
            num_timesteps=total_timesteps,
            num_envs=num_envs,
            episode_length=episode_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            unroll_length=unroll_length,
            num_minibatches=num_minibatches,
            num_updates_per_batch=num_updates_per_batch,
            progress_fn=progress_callback,
            policy_params_fn=policy_params_callback,
            num_evals=num_evals,
            seed=42
        )
    finally:
        pbar.close()
        if writer:
            writer.close()

    print(f"✅ Treino BACK concluído em {(time.time() - start_time)/60:.2f} minutos!")

    # Salvar parâmetros do modelo
    model_path = os.path.join(model_dir, f"ppo_getup_back_mjx_{timestamp}.pkl")
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Modelo salvo em: {model_path}")

if __name__ == "__main__":
    train()


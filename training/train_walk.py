import gymnasium as gym
import wandb
import os
import glob
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from walk_env import WalkEnv
from wandb.integration.sb3 import WandbCallback
from datetime import datetime

class UploadVideoCallback(BaseCallback):
    def __init__(self, video_folder: str, verbose=0):
        super().__init__(verbose)
        self.video_folder = video_folder
        self.videos_uploaded = set()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        arquivos_videos = glob.glob(os.path.join(self.video_folder, "*.mp4"))

        for video_file in arquivos_videos:
            if (video_file not in self.videos_uploaded and os.path.getsize(video_file) > 10_000):
                wandb.log({"video": wandb.Video(video_file, format="mp4")})
                self.videos_uploaded.add(video_file)

def train():
    log_dir = "./training/logs/walk/"
    checkpoint_dir = "./training/checkpoints/walk/"
    model_dir = "./training/models/walk/"

    for d in [log_dir, checkpoint_dir, model_dir]:
        os.makedirs(d, exist_ok=True)

    num_cpu = max(1, os.cpu_count() - 1)
    print(f"Configurando {num_cpu} ambientes em paralelo...")

    vec_env = make_vec_env(WalkEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    # Hiperparâmetros ajustados para estabilidade PPO em locomoção
    learning_rate = 3e-4
    n_steps = 4096
    batch_size = 4096
    n_epochs = 5
    gamma = 0.99
    total_timesteps = 50_000_000
    gae_lambda=0.95
    ent_coef=0.005
    max_grad_norm=1.0

    model = PPO(
        "MlpPolicy", vec_env, verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(model_dir, f"ppo_walk_{timestamp}")

    # Configuração de Hiperparâmetros para o WandB
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "n_envs": num_cpu,
    }

    # Inicializar o WandB
    run = wandb.init(
        project="oxebots_walk_train",
        name=f"ppo-walk-{timestamp}",
        config=config,       
        sync_tensorboard=True,  # Sincroniza com TensorBoard
        save_code=True,         # Salva o estado do código no wandb
    )

    gym.register(
        id="walkEnv-v0",
        entry_point="walk_env:WalkEnv",
        )
    
    def make_env():
        env_eval = gym.make("walkEnv-v0", render_mode="rgb_array")
        env_eval = gym.wrappers.RecordVideo(
            env_eval,
            video_folder=os.path.join(log_dir, "videos"),
            episode_trigger=lambda x:True
        )
        env_eval = Monitor(env_eval) 
        return env_eval

    vec_env_eval = DummyVecEnv([make_env])

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(1_000_000 // num_cpu, 1),
            save_path=checkpoint_dir,
            name_prefix="walk_model",
        ),

        EvalCallback(
            vec_env_eval,
            best_model_save_path=os.path.join(model_dir, "best/"),
            log_path=log_dir,
            eval_freq=max(250_000 // num_cpu, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),

        WandbCallback(
        gradient_save_freq=100,
        model_save_path=os.path.join(model_dir, run.id),
        verbose=2,
    ),

        UploadVideoCallback(
        video_folder=os.path.join(log_dir, "videos"))
    ])

    print(f"=== Treinamento Walk ===")
    print(f"  Ambientes paralelos: {num_cpu}")
    print(f"  Buffer por update: {model.n_steps * num_cpu} steps")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Device: {model.device}")
    print(f"  Modelo final: {final_model_path}")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            tb_log_name=f"PPO_Walk_{timestamp}",
            callback=callbacks,
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")

    model.save(final_model_path)
    print(f"Modelo salvo: {final_model_path}")

    print("Limpando checkpoints intermediários...")
    for file in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Erro ao apagar {file_path}: {e}")
    
    run.finish()


if __name__ == "__main__":
    train()

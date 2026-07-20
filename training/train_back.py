import gymnasium as gym
import wandb
import os
import glob
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from getup_env import GetUpBackEnv
from datetime import datetime
from wandb.integration.sb3 import WandbCallback

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
    # Pastas organizadas
    log_dir = "./training/logs/back/"
    checkpoint_dir = "./training/checkpoints/back/"
    model_dir = "./training/models/back/"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Detecta automaticamente a quantidade de núcleos da CPU (deixando 1 livre para o SO)
    num_cpu = max(1, os.cpu_count() - 1)
    print(f"Configurando {num_cpu} ambientes em paralelo...")
    
    env = make_vec_env(GetUpBackEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    checkpoint_callback = CheckpointCallback(
        save_freq=140_000,
        save_path=checkpoint_dir,
        name_prefix="getup_back_model"
    )

    # Hiperparâmetros
    learning_rate = 3e-4
    n_steps = 4096
    batch_size = 128
    n_epochs = 10
    gamma = 0.99
    total_timesteps = 100_000_000


    # Configuração de Hiperparâmetros para o WandB
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "n_envs": num_cpu,
    }

    # Nome único com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(model_dir, f"ppo_getup_back_{timestamp}")

    # Inicializar o WandB
    run = wandb.init(
        project="bahiart-mujoco-getup",
        name=f"ppo-getup-back-{timestamp}",
        config=config,       
        sync_tensorboard=True,  # Sincroniza com TensorBoard
        save_code=True,         # Salva o estado do código no wandb
    )

    # Configuração de Ambiente para Gravação de Vídeo no WandB
    gym.register(
        id="GetUpBackEnv-v0",
        entry_point="getup_env:GetUpBackEnv",
        )
    
    def make_env():
        env_eval = gym.make("GetUpBackEnv-v0", render_mode="rgb_array")
        env_eval = gym.wrappers.RecordVideo(
            env_eval,
            video_folder=os.path.join(log_dir, "videos"),
            episode_trigger=lambda x:True
        )
        env_eval = Monitor(env_eval) 
        return env_eval

    vec_env_eval = DummyVecEnv([make_env])

    eval_callback = EvalCallback(
        vec_env_eval,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=140_000, 
        n_eval_episodes=1,
        deterministic=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=os.path.join(model_dir, run.id),
        verbose=2,
    )

    video_upload_callback = UploadVideoCallback(video_folder=os.path.join(log_dir, "videos"))

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        tensorboard_log=log_dir,
        device="auto"
    )

    print(f"Iniciando treino BACK... Logs em {log_dir}")
    print(f"Modelo final será salvo em: {final_model_path}")

    try:
        model.learn(
            total_timesteps=total_timesteps, 
            progress_bar=True,
            tb_log_name=f"PPO_GetUp_Back_{timestamp}",
            callback=[checkpoint_callback, eval_callback, wandb_callback, video_upload_callback]
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido.")

    model.save(final_model_path)
    print(f"Modelo salvo com sucesso: {final_model_path}")

    print("Limpando checkpoints intermediários...")
    for file in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Erro ao apagar {file_path}: {e}")

    # Finalizar a execução do wandb
    run.finish()

if __name__ == "__main__":
    train()


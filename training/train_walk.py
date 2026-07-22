import gymnasium as gym
import torch
import wandb
import os
import glob
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from walk_env import WalkEnv
from wandb.integration.sb3 import WandbCallback
from datetime import datetime


class CurriculumCallback(BaseCallback):
    """Propaga o progresso de treinamento para os ambientes (curriculum learning)."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 10_000 == 0:
            self.training_env.env_method("set_training_progress", self.num_timesteps)
        return True


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
    # Normalização de observações e recompensas — estabiliza significativamente o PPO
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Hiperparâmetros otimizados para locomoção bípede
    learning_rate = 3e-4
    n_steps = 4096            # Mais dados por update (era 2048)
    batch_size = 512           # Mini-batches menores → mais diversidade de gradiente (era 2048)
    n_epochs = 5               # Reduzido de 10 → evitar overfitting no buffer
    gamma = 0.99
    total_timesteps = 100_000_000
    gae_lambda = 0.95
    ent_coef = 0.005           # Reduzido de 0.01 → permitir convergência mais rápida
    max_grad_norm = 0.5        # Reduzido de 1.0 → mais estabilidade
    clip_range = 0.2
    vf_coef = 0.5

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
        clip_range=clip_range,
        vf_coef=vf_coef,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            activation_fn=torch.nn.ELU,
        ),
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
        "clip_range": clip_range,
        "vf_coef": vf_coef,
        "n_envs": num_cpu,
        "net_arch": "pi=[256,256,128] vf=[256,256,128]",
        "activation": "ELU",
        "vec_normalize": True,
        "curriculum_learning": True,
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
    # Eval env PRECISA de VecNormalize para sync com o training env
    vec_env_eval = VecNormalize(vec_env_eval, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec_env_eval.training = False  # Não atualizar stats durante avaliação

    callbacks = CallbackList([
        CurriculumCallback(),

        CheckpointCallback(
            save_freq=max(1_000_000 // num_cpu, 1),
            save_path=checkpoint_dir,
            name_prefix="walk_model",
        ),

        EvalCallback(
            vec_env_eval,
            best_model_save_path=os.path.join(model_dir, "best/"),
            log_path=log_dir,
            eval_freq=max(1_000_000 // num_cpu, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),

        WandbCallback(
        gradient_save_freq=1000,
        model_save_path=os.path.join(model_dir, run.id),
        verbose=2,
    ),

        UploadVideoCallback(
        video_folder=os.path.join(log_dir, "videos"))
    ])

    print(f"=== Treinamento Walk ===")
    print(f"  Ambientes paralelos: {num_cpu}")
    print(f"  Buffer por update: {model.n_steps * num_cpu} steps")
    print(f"  Batch size: {batch_size}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Device: {model.device}")
    print(f"  VecNormalize: ON")
    print(f"  Curriculum: ON")
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
    # Salvar VecNormalize stats para uso em inferência
    vec_env.save(os.path.join(model_dir, f"vecnormalize_{timestamp}.pkl"))
    print(f"Modelo salvo: {final_model_path}")
    print(f"VecNormalize salvo: vecnormalize_{timestamp}.pkl")

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

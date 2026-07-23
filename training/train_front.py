from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, DummyVecEnv
from getup_env import GetUpFrontEnv
import os
from datetime import datetime
import wandb
from wandb.integration.sb3 import WandbCallback

def train():
    # Pastas organizadas
    log_dir = "./training/logs/front/"
    checkpoint_dir = "./training/checkpoints/front/"
    model_dir = "./training/models/front/"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Detecta automaticamente a quantidade de núcleos da CPU (deixando 1 livre para o SO)
    num_cpu = max(1, os.cpu_count() - 1)
    print(f"Configurando {num_cpu} ambientes em paralelo...")
    
    env = make_vec_env(GetUpFrontEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)


    # Checkpoints continuam salvando por passos (já evita sobrescrita)
    checkpoint_callback = CheckpointCallback(
        save_freq=2500000,
        save_path=checkpoint_dir,
        name_prefix="getup_front_model"
    )

    # Hiperparâmetros
    learning_rate = 3e-4
    n_steps = 4096
    batch_size = 128
    n_epochs = 10
    gamma = 0.99
    total_timesteps = 100000000


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
    final_model_path = os.path.join(model_dir, f"ppo_getup_front_{timestamp}")

    # Inicializar o WandB (com fallback para modo offline se não houver login)
    wandb_mode = os.getenv("WANDB_MODE")
    try:
        run = wandb.init(
            project="bahiart-mujoco-getup",
            name=f"ppo-getup-front-{timestamp}",
            config=config,
            sync_tensorboard=True,  # Sincroniza logs do TensorBoard automaticamente
            monitor_gym=True,       # Monitora o Gym/Gymnasium
            save_code=True,         # Salva o estado do código no wandb
            mode=wandb_mode,
        )
    except Exception as e:
        print(f"Aviso: WandB login/conexão falhou ({e}). Executando em modo offline...")
        run = wandb.init(
            project="bahiart-mujoco-getup",
            name=f"ppo-getup-front-{timestamp}",
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            mode="offline",
        )

    # Configuração de Ambiente para Gravação de Vídeo no WandB
    eval_env = DummyVecEnv([lambda: GetUpFrontEnv(render_mode="rgb_array")])
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=os.path.join(log_dir, "videos"),
        record_video_trigger=lambda step: step % 1000 == 0,
        video_length=1000,
        name_prefix="ppo-getup-front-eval"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=2500000,  # Avalia e grava vídeo a cada 2.5M passos
        n_eval_episodes=1,
        deterministic=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=os.path.join(model_dir, run.id),
        verbose=2,
    )

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

    print(f"Iniciando treino FRONT... Logs em {log_dir}")
    print(f"Modelo final será salvo em: {final_model_path}")

    try:
        model.learn(
            total_timesteps=total_timesteps, 
            progress_bar=True,
            tb_log_name=f"PPO_GetUp_Front_{timestamp}",
            callback=[checkpoint_callback, eval_callback, wandb_callback]
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido.")

    model.save(final_model_path)
    print(f"Modelo salvo com sucesso: {final_model_path}")
    
    # Finalizar a execução do wandb
    run.finish()

if __name__ == "__main__":
    train()


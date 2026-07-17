from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
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

    # Inicializar o WandB
    run = wandb.init(
        project="bahiart-mujoco-getup",
        name=f"ppo-getup-front-{timestamp}",
        config=config,
        sync_tensorboard=True,  # Sincroniza logs do TensorBoard automaticamente
        monitor_gym=True,       # Monitora o Gym/Gymnasium
        save_code=True,         # Salva o estado do código no wandb
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
            callback=[checkpoint_callback, wandb_callback]
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido.")

    model.save(final_model_path)
    print(f"Modelo salvo com sucesso: {final_model_path}")
    
    # Finalizar a execução do wandb
    run.finish()

if __name__ == "__main__":
    train()


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from getup_env import GetUpEnv
import os

def train():
    # Pastas de logs e checkpoints
    log_dir = "./training/logs/"
    checkpoint_dir = "./training/checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Cria o ambiente
    env = GetUpEnv()

    # Configura o salvamento automático a cada 500.000 passos
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path=checkpoint_dir,
        name_prefix="getup_t1_model"
    )

    # Define o modelo PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=log_dir,
        device="auto"
    )

    model_name = "getup_t1_ppo"

    print(f"Iniciando treinamento... Logs em {log_dir}, Checkpoints em {checkpoint_dir}")

    try:
        # Treina com o callback de salvamento
        model.learn(
            total_timesteps=10000000, 
            progress_bar=True,
            tb_log_name="PPO_GetUp",
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")

    # Salva a versão final
    model.save(model_name)
    print(f"Modelo final salvo como {model_name}")

if __name__ == "__main__":
    train()


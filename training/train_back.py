from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from getup_env import GetUpBackEnv
import os
from datetime import datetime

def train():
    # Pastas organizadas
    log_dir = "./training/logs/back/"
    checkpoint_dir = "./training/checkpoints/back/"
    model_dir = "./training/models/back/"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = GetUpBackEnv()

    checkpoint_callback = CheckpointCallback(
        save_freq=250000,
        save_path=checkpoint_dir,
        name_prefix="getup_back_model"
    )

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

    # Nome único com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(model_dir, f"ppo_getup_back_{timestamp}")

    print(f"Iniciando treino BACK... Logs em {log_dir}")
    print(f"Modelo final será salvo em: {final_model_path}")

    try:
        model.learn(
            total_timesteps=10000000, 
            progress_bar=True,
            tb_log_name=f"PPO_GetUp_Back_{timestamp}",
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido.")

    model.save(final_model_path)
    print(f"Modelo salvo com sucesso: {final_model_path}")

 # Limpeza de checkpoints após sucesso
    print("Limpando checkpoints intermediários...")
    import shutil
    for file in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Erro ao apagar {file_path}: {e}")

if __name__ == "__main__":
    train()

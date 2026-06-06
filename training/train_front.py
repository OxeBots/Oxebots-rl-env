from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from getup_env import GetUpFrontEnv
import os
from datetime import datetime

def train():
    # Pastas organizadas
    log_dir = "./training/logs/front/"
    checkpoint_dir = "./training/checkpoints/front/"
    model_dir = "./training/models/front/"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = GetUpFrontEnv()

    # Checkpoints continuam salvando por passos (já evita sobrescrita)
    checkpoint_callback = CheckpointCallback(
        save_freq=250000,
        save_path=checkpoint_dir,
        name_prefix="getup_front_model"
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

    # Nome único com timestamp para o modelo final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(model_dir, f"ppo_getup_front_{timestamp}")

    print(f"Iniciando treino FRONT... Logs em {log_dir}")
    print(f"Modelo final será salvo em: {final_model_path}")

    try:
        model.learn(
            total_timesteps=10000000, 

            progress_bar=True,
            tb_log_name=f"PPO_GetUp_Front_{timestamp}",
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido.")

    model.save(final_model_path)
    print(f"Modelo salvo com sucesso: {final_model_path}")

if __name__ == "__main__":
    train()

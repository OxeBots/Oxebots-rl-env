import gymnasium as gym
from stable_baselines3 import PPO
from getup_env import GetUpEnv
import mujoco.viewer
import time
import numpy as np

def enjoy():
    env = GetUpEnv()
    # Tenta carregar o modelo mais recente
    try:
        model = PPO.load("getup_t1_ppo", env=env)
        print("Modelo carregado!")
    except:
        print("Modelo não encontrado. Rodando com ações aleatórias...")
        model = None

    obs, _ = env.reset()
    
    # Abre o visualizador do MuJoCo
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                
            obs, reward, done, trunc, _ = env.step(action)
            
            viewer.sync()
            # Tenta manter ~50 FPS na visualização
            time.sleep(0.02)
            
            if done or trunc:
                obs, _ = env.reset()

if __name__ == "__main__":
    enjoy()

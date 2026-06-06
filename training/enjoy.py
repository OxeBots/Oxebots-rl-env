import gymnasium as gym
from stable_baselines3 import PPO
from getup_env import GetUpFrontEnv, GetUpBackEnv
import mujoco.viewer
import time
import numpy as np
import sys
import os
import glob

def get_latest_model(model_dir, checkpoint_dir):
    """
    Busca o modelo mais recente. 
    Primeiro tenta na pasta de modelos finais, depois na de checkpoints.
    """
    # 1. Tenta modelos finais
    final_files = glob.glob(os.path.join(model_dir, "*.zip"))
    if final_files:
        latest_file = max(final_files, key=os.path.getctime)
        return latest_file.replace(".zip", ""), "MODELO FINAL"

    # 2. Se não houver, tenta checkpoints
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if checkpoint_files:
        latest_file = max(checkpoint_files, key=os.path.getctime)
        return latest_file.replace(".zip", ""), "CHECKPOINT"

    return None, None

def enjoy():
    mode = sys.argv[1] if len(sys.argv) > 1 else "front"
    
    if mode == "back":
        env = GetUpBackEnv()
        model_dir = "./training/models/back/"
        checkpoint_dir = "./training/checkpoints/back/"
    else:
        env = GetUpFrontEnv()
        model_dir = "./training/models/front/"
        checkpoint_dir = "./training/checkpoints/front/"

    print(f"Modo: {mode.upper()}")
    
    model_path, model_type = get_latest_model(model_dir, checkpoint_dir)

    if model_path:
        try:
            model = PPO.load(model_path, env=env)
            print(f"[{model_type}] Carregado: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            model = None
    else:
        print(f"Nenhum modelo ou checkpoint encontrado em {model_dir} ou {checkpoint_dir}. Usando ações aleatórias.")
        model = None

    obs, _ = env.reset()
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                                
            obs, reward, done, trunc, _ = env.step(action)
            
            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)
            
            if done or trunc:
                obs, _ = env.reset()

if __name__ == "__main__":
    enjoy()

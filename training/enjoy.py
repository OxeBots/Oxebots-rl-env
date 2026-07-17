import gymnasium as gym
from stable_baselines3 import PPO
from getup_env import GetUpFrontEnv, GetUpBackEnv
import mujoco.viewer
import time
import numpy as np
import sys
import os
import glob

def get_latest_model(checkpoint_dir):
    """
    Busca o checkpoint mais recente na pasta de checkpoints.
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if checkpoint_files:
        latest_file = max(checkpoint_files, key=os.path.getctime)
        return latest_file.replace(".zip", ""), "CHECKPOINT"

    return None, None

def enjoy():
    mode = "front"
    specific_model = None

    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1 in ("front", "back"):
            mode = arg1
            if len(sys.argv) > 2:
                specific_model = sys.argv[2]
        else:
            # O usuário passou o arquivo diretamente como primeiro argumento
            specific_model = arg1
            # Tenta inferir o modo pelo caminho do arquivo
            if "back" in arg1.lower():
                mode = "back"
            else:
                mode = "front"
    
    if mode == "back":
        env = GetUpBackEnv()
        model_dir = "./training/models/back/"
        checkpoint_dir = "./training/checkpoints/back/"
    else:
        env = GetUpFrontEnv()
        model_dir = "./training/models/front/"
        checkpoint_dir = "./training/checkpoints/front/"

    print(f"Modo: {mode.upper()}")
    
    if specific_model:
        model_path = specific_model.replace(".zip", "")
        model_type = "ESPECÍFICO"
    else:
        model_path, model_type = get_latest_model(checkpoint_dir)


    if model_path:
        try:
            model = PPO.load(model_path, env=env)
            print(f"[{model_type}] Carregado: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            model = None
    else:
        print(f"Nenhum modelo ou checkpoint encontrado. Usando ações aleatórias.")
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

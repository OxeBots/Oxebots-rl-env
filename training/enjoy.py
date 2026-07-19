import gymnasium as gym
from stable_baselines3 import PPO
from getup_env import GetUpFrontEnv, GetUpBackEnv
from walk_env import WalkEnv
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
    final_files = glob.glob(os.path.join(model_dir, "*.zip"))
    if final_files:
        latest_file = max(final_files, key=os.path.getctime)
        return latest_file.replace(".zip", ""), "MODELO FINAL"

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if checkpoint_files:
        latest_file = max(checkpoint_files, key=os.path.getctime)
        return latest_file.replace(".zip", ""), "CHECKPOINT"

    return None, None

def enjoy():
    mode = sys.argv[1] if len(sys.argv) > 1 else "front"
    
    if mode == "walk":
        env = WalkEnv()
        model_dir = "./training/models/walk/"
        checkpoint_dir = "./training/checkpoints/walk/"
    elif mode == "back":
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
        print(f"Nenhum modelo encontrado em {model_dir} ou {checkpoint_dir}. Ações aleatórias.")
        model = None

    obs, _ = env.reset()
    
    # Walk: exibir e opcionalmente sobrescrever o comando de velocidade
    if mode == "walk":
        if len(sys.argv) > 2:
            cmd = [float(x) for x in sys.argv[2].split()]
            env.velocity_command = np.array(cmd)
            print(f"Comando manual: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, yaw={cmd[2]:.2f}")
        else:
            print(f"Comando: vx={env.velocity_command[0]:.2f}, "
                  f"vy={env.velocity_command[1]:.2f}, yaw={env.velocity_command[2]:.2f}")
    
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
                if mode == "walk":
                    print(f"Novo episódio — Comando: vx={env.velocity_command[0]:.2f}, "
                          f"vy={env.velocity_command[1]:.2f}, yaw={env.velocity_command[2]:.2f}")

if __name__ == "__main__":
    enjoy()

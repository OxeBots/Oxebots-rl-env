import os
import sys
import glob
import time
import pickle
import numpy as np
import mujoco
import mujoco.viewer

from getup_env_mjx import GetUpFrontMjxEnv, GetUpBackMjxEnv

def get_latest_model(model_dir):
    model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if model_files:
        return max(model_files, key=os.path.getctime)
    return None

def enjoy():
    mode = "front"
    if len(sys.argv) > 1:
        if sys.argv[1] in ("front", "back"):
            mode = sys.argv[1]

    if mode == "back":
        env = GetUpBackMjxEnv()
        model_dir = "./training/models/back_mjx/"
    else:
        env = GetUpFrontMjxEnv()
        model_dir = "./training/models/front_mjx/"

    print(f"🎮 Modo de Visualização: {mode.upper()}")

    model_path = get_latest_model(model_dir)
    params = None

    if model_path and os.path.exists(model_path):
        print(f"📦 Modelo Carregado: {os.path.basename(model_path)}")
        try:
            with open(model_path, "rb") as f:
                params = pickle.load(f)
        except Exception as e:
            print(f"Erro ao ler parâmetros do modelo: {e}")
    else:
        print("⚠️ Nenhum modelo .pkl encontrado. Executando com ações aleatórias.")

    # Inicializa simulação física local para visualização gráfica
    with mujoco.viewer.launch_passive(env.mj_model, env.sys.init_q) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Ação aleatória ou inferida
            action = np.random.uniform(-1.0, 1.0, size=(env.mj_model.nu,))

            time.sleep(0.02)
            viewer.sync()

if __name__ == "__main__":
    enjoy()

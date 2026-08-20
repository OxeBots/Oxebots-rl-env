import os
import sys

# Permitir execução a partir de qualquer pasta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from getup_env import GetUpFrontEnv, GetUpBackEnv
from walk_env import WalkEnv
import mujoco
import mujoco.viewer
import time
import numpy as np
import sys
import os
import glob


class ONNXWrapper:
    def __init__(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, obs, deterministic=True):
        obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
        action = self.session.run([self.output_name], {self.input_name: obs_array})[0]
        return action.flatten(), None


def get_latest_model(model_dir, checkpoint_dir):
    """
    Busca o modelo mais recente nas pastas de modelo e checkpoint.
    Prioriza o best_model.zip se existir, senão busca o arquivo .zip ou .onnx mais recente.
    """
    best_model_path = os.path.join(model_dir, "best_model", "best_model.zip")
    if os.path.exists(best_model_path):
        return best_model_path.replace(".zip", ""), "BEST_MODEL"

    candidates = []
    for d in [model_dir, checkpoint_dir]:
        if os.path.exists(d):
            candidates.extend(glob.glob(os.path.join(d, "*.zip")))
            candidates.extend(glob.glob(os.path.join(d, "*.onnx")))
            candidates.extend(glob.glob(os.path.join(d, "**", "*.zip"), recursive=True))

    if candidates:
        latest = max(candidates, key=os.path.getctime)
        model_type = "CHECKPOINT" if "checkpoint" in latest.lower() else "MODELO RECENTE"
        return latest.replace(".zip", ""), model_type

    return None, None


def find_vecnormalize_stats(model_dir):
    """Busca o arquivo VecNormalize mais recente na pasta de modelos."""
    pkl_files = glob.glob(os.path.join(model_dir, "vecnormalize_*.pkl"))
    if pkl_files:
        return max(pkl_files, key=os.path.getctime)
    return None


def enjoy():
    mode = "front"
    specific_model = None

    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1 in ("front", "back", "walk"):
            mode = arg1
            if len(sys.argv) > 2:
                specific_model = sys.argv[2]
        else:
            # O usuário passou o caminho do arquivo diretamente como primeiro argumento
            specific_model = arg1
            if "back" in arg1.lower():
                mode = "back"
            elif "walk" in arg1.lower():
                mode = "walk"
            else:
                mode = "front"

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

    print(f"🎮 Modo de Visualização CPU: {mode.upper()}")

    if specific_model:
        model_path = specific_model
        model_type = "ESPECÍFICO"
    else:
        model_path, model_type = get_latest_model(model_dir, checkpoint_dir)

    # Para walk com modelos SB3: carregar VecNormalize stats se disponível
    vec_normalize = None
    if mode == "walk" and model_path and not model_path.endswith(".onnx"):
        stats_path = find_vecnormalize_stats(model_dir)
        if stats_path:
            vec_env = DummyVecEnv([lambda: env])
            vec_normalize = VecNormalize.load(stats_path, vec_env)
            vec_normalize.training = False
            vec_normalize.norm_reward = False
            print(f"📦 VecNormalize carregado: {os.path.basename(stats_path)}")

    if model_path:
        try:
            if model_path.endswith(".onnx"):
                model = ONNXWrapper(model_path)
            else:
                model = PPO.load(model_path.replace(".zip", ""), env=env)
            print(f"📦 [{model_type}] Modelo Carregado: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo ({e}). Executando com política aleatória.")
            model = None
    else:
        print("⚠️ Nenhum modelo ou checkpoint encontrado. Executando com ações aleatórias.")
        model = None

    if vec_normalize is not None:
        obs = vec_normalize.reset()
    else:
        obs, _ = env.reset()

    if mode == "walk":
        if len(sys.argv) > 2 and not specific_model:
            try:
                cmd_str = " ".join(sys.argv[2:])
                cmd = [float(x) for x in cmd_str.replace(',', ' ').split()]
                if len(cmd) >= 3:
                    env.velocity_command = np.array(cmd[:3])
                    print(f"Comando manual: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, yaw={cmd[2]:.2f}")
            except (ValueError, IndexError):
                print(f"Comando padrão: vx={env.velocity_command[0]:.2f}, "
                      f"vy={env.velocity_command[1]:.2f}, yaw={env.velocity_command[2]:.2f}")
        else:
            print(f"Comando: vx={env.velocity_command[0]:.2f}, "
                  f"vy={env.velocity_command[1]:.2f}, yaw={env.velocity_command[2]:.2f}")

    dt = float(env.n_frames * env.model.opt.timestep) if hasattr(env, 'n_frames') else 0.02

    print("🚀 Iniciando simulação e visualização MuJoCo 3D no modo CPU...")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            if vec_normalize is not None:
                obs, reward, done, info = vec_normalize.step(np.array([action]))
                obs = obs[0] if len(obs.shape) > 1 else obs
                done = done[0] if hasattr(done, '__len__') else done
            else:
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            if done:
                if vec_normalize is not None:
                    obs = vec_normalize.reset()
                else:
                    obs, _ = env.reset()
                if mode == "walk":
                    print(f"🔄 Novo episódio — Comando: vx={env.velocity_command[0]:.2f}, "
                          f"vy={env.velocity_command[1]:.2f}, yaw={env.velocity_command[2]:.2f}")


if __name__ == "__main__":
    enjoy()

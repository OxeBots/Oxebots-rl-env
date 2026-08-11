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

    policy = None
    if params is not None:
        try:
            import jax
            import jax.numpy as jnp
            from brax.training.agents.ppo import networks as ppo_networks

            ppo_network = ppo_networks.make_ppo_networks(
                env.observation_size,
                env.action_size
            )
            make_policy = ppo_networks.make_inference_fn(ppo_network)
            jit_inference_fn = jax.jit(make_policy(params))
            policy = lambda obs: jit_inference_fn(obs, jax.random.PRNGKey(0))[0]
            print("🧠 Rede neural da política carregada para inferência!")
        except Exception as e:
            print(f"⚠️ Não foi possível compilar a política para inferência: {e}")

    # Inicializa estrutura de dados da física MuJoCo
    mj_data = mujoco.MjData(env.mj_model)
    mj_data.qpos[:] = np.array(env.sys.init_q)

    if mode == "back":
        mj_data.qpos[2] = 0.25
        mj_data.qpos[3] = 0.707
        mj_data.qpos[4] = 0.0
        mj_data.qpos[5] = -0.707
        mj_data.qpos[6] = 0.0
    else:
        mj_data.qpos[2] = 0.25
        mj_data.qpos[3] = 0.707
        mj_data.qpos[4] = 0.0
        mj_data.qpos[5] = 0.707
        mj_data.qpos[6] = 0.0

    # Inicializa simulação física local para visualização gráfica
    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if policy is not None:
                try:
                    import jax.numpy as jnp
                    torso_height = mj_data.qpos[2]
                    height_progress = float(np.clip((torso_height - 0.15) / (0.65 - 0.15), 0.0, 1.0))
                    target_pose = env.get_target_pose(jnp.array(height_progress))

                    actuated_qpos = mj_data.qpos[7:30]
                    joint_err = np.array(target_pose) - actuated_qpos
                    obs = np.concatenate([mj_data.qpos, mj_data.qvel, joint_err])
                    obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
                    obs = np.clip(obs, -100.0, 100.0)

                    action = np.array(policy(obs))
                except Exception as e:
                    action = np.random.uniform(-1.0, 1.0, size=(env.mj_model.nu,))
            else:
                action = np.random.uniform(-1.0, 1.0, size=(env.mj_model.nu,))

            mj_data.ctrl[:] = action
            mujoco.mj_step(env.mj_model, mj_data)

            time_until_next_step = env.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            viewer.sync()

if __name__ == "__main__":
    enjoy()

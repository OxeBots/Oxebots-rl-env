import os
# Evitar pré-alocação agressiva de 90% da VRAM da GPU pelo JAX/XLA no modo de visualização
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"

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
            from brax.training.agents.ppo import networks as ppo_networks

            ppo_network = ppo_networks.make_ppo_networks(
                env.observation_size,
                env.action_size
            )
            make_policy = ppo_networks.make_inference_fn(ppo_network)
            jit_inference_fn = jax.jit(make_policy(params))
            policy = lambda obs, key: jit_inference_fn(obs, key)[0]
            print("🧠 Rede neural da política carregada para inferência!")
        except Exception as e:
            print(f"⚠️ Não foi possível compilar a política para inferência: {e}")

    import jax
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    # Inicializa estrutura de dados da física MuJoCo para renderização
    mj_data = mujoco.MjData(env.mj_model)

    max_steps_per_episode = 500  # Reinicia o episódio a cada 500 passos (aprox. 10s)
    step_count = 0

    print("🚀 Iniciando simulação física...")

    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            rng, policy_rng = jax.random.split(rng)

            if policy is not None:
                try:
                    action = policy(state.obs, policy_rng)
                except Exception as e:
                    action = jax.random.uniform(policy_rng, shape=(env.action_size,), minval=-1.0, maxval=1.0)
            else:
                action = jax.random.uniform(policy_rng, shape=(env.action_size,), minval=-1.0, maxval=1.0)

            # Executa passo de simulação via Brax / MJX
            state = jit_step(state, action)
            step_count += 1

            # Atualiza pose e velocidade no visualizador 3D do MuJoCo
            mj_data.qpos[:] = np.array(state.pipeline_state.qpos)
            mj_data.qvel[:] = np.array(state.pipeline_state.qvel)
            mujoco.mj_forward(env.mj_model, mj_data)

            # Reinicia se o episódio terminar ou atingir o limite de tempo
            is_done = bool(np.array(state.done) > 0.5) or step_count >= max_steps_per_episode
            if is_done:
                print("🔄 Reiniciando episódio...")
                rng, reset_rng = jax.random.split(rng)
                state = jit_reset(reset_rng)
                step_count = 0

            # Manter taxa de quadros realista (~50 Hz / 20ms)
            dt = float(env.dt) if hasattr(env, 'dt') else 0.02
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            viewer.sync()

if __name__ == "__main__":
    enjoy()

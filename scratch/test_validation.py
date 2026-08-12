import sys
import os

# Adiciona o diretório training ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from getup_env_mjx import GetUpFrontMjxEnv, GetUpBackMjxEnv

def validate():
    print("🔍 Executando validação dos ambientes MJX...\n")

    for name, env_cls in [("FRONT", GetUpFrontMjxEnv), ("BACK", GetUpBackMjxEnv)]:
        env = env_cls()
        
        print(f"--- AMBIENTE {name} ---")
        print(f"1. ID do Torso (torso_id): {env.torso_id}")
        print(f"   Nome do corpo torso_id no MjModel: {env.mj_model.body(env.torso_id).name}")
        print(f"2. Mimic YAML Carregado (_has_mimic): {env._has_mimic}")
        print(f"   Número de Fases (n_phases): {env.n_phases}\n")

    print("✅ Validação concluída com sucesso!")

if __name__ == "__main__":
    validate()

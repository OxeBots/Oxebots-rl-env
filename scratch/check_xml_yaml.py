import os
import yaml
import mujoco

def check():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(repo_root, "resources", "robots", "T1", "robot.xml")
    
    if os.path.exists(model_path):
        model = mujoco.MjModel.from_xml_path(model_path)
        torso_id = model.body('torso').id
        print(f"1. [XML] ID do torso no MjModel: {torso_id}")
        print(f"   Nome do corpo torso_id: {model.body(torso_id).name}")
    else:
        print(f"1. [XML] MjModel robot.xml não encontrado em {model_path}.")

    yaml_front = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "skills", "keyframe", "get_up", "get_up_front.yaml"))
    yaml_back = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "skills", "keyframe", "get_up", "get_up_back.yaml"))

    print(f"\n2. [YAML FRONT] Existe: {os.path.exists(yaml_front)}")
    if os.path.exists(yaml_front):
        with open(yaml_front, 'r') as f:
            data = yaml.safe_load(f)
        print(f"   Fases de keyframe carregadas (n_phases): {len(data.get('keyframes', []))}")

    print(f"\n3. [YAML BACK] Existe: {os.path.exists(yaml_back)}")
    if os.path.exists(yaml_back):
        with open(yaml_back, 'r') as f:
            data = yaml.safe_load(f)
        print(f"   Fases de keyframe carregadas (n_phases): {len(data.get('keyframes', []))}")

if __name__ == "__main__":
    check()

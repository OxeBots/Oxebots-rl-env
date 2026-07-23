import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
import os
import re
from abc import abstractmethod


class BaseRobotEnv(gym.Env):
    """Classe Base com infraestrutura MuJoCo compartilhada para todos os ambientes de treinamento."""

    # Posição nominal das juntas (em radianos) — de walk.py L12-37
    JOINT_NOMINAL_POSITION = np.array([
        0.0, 0.0, 0.0, 1.4, 0.0, -0.4,       # he1,he2,lae1,lae2,lae3,lae4
        0.0, -1.4, 0.0, 0.4,                    # rae1,rae2,rae3,rae4
        0.0, -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,   # te1,lle1..lle6
        0.4, 0.0, 0.0, -0.8, 0.4, 0.0,          # rle1..rle6
    ])

    # Fator de espelhamento de eixos simulação/treinamento — de walk.py L39-64
    TRAIN_SIM_FLIP = np.array([
        1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
        -1.0, -1.0, 1.0, 1.0,
        1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    ])
    SCALING_FACTOR = 0.5

    def __init__(self, model_path=None):
        if model_path is None or not os.path.exists(model_path):
            home = os.path.expanduser("~")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.abspath(os.path.join(current_dir, "..", "..", "mojucco_simulator3D", "src", "rcsssmj", "resources", "robots", "T1", "robot.xml")),
                os.path.abspath(os.path.join(current_dir, "..", "mojucco_simulator3D", "src", "rcsssmj", "resources", "robots", "T1", "robot.xml")),
                os.path.join(home, "rcssservermj", "src", "rcsssmj", "resources", "robots", "T1", "robot.xml"),
                os.path.join(home, "mojucco_simulator3D", "src", "rcsssmj", "resources", "robots", "T1", "robot.xml"),
            ]
            if "ROBOT_XML_PATH" in os.environ:
                candidates.insert(0, os.environ["ROBOT_XML_PATH"])

            model_path = None
            for cand in candidates:
                if cand and os.path.exists(cand):
                    model_path = cand
                    break

            if model_path is None:
                raise FileNotFoundError(f"Não foi possível encontrar robot.xml. Caminhos buscados: {candidates}")

        super().__init__()

        # Carregar e configurar o XML do robô
        with open(model_path, 'r') as f:
            xml_string = f.read()

        mesh_dir = os.path.join(os.path.dirname(model_path), "meshes").replace('\\', '/')
        xml_string = re.sub(r'meshdir="[^"]*"', f'meshdir="{mesh_dir}"', xml_string)

        if "plane" not in xml_string:
            ground_xml = """
    <geom name="floor" type="plane" size="10 10 0.05" rgba="0.8 0.9 0.8 1" pos="0 0 0"/>
    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
"""
            xml_string = re.sub(r'<worldbody>', f'<worldbody>{ground_xml}', xml_string)

        try:
            self.model = mujoco.MjModel.from_xml_string(xml_string)
        except Exception:
            self.model = mujoco.MjModel.from_xml_path(model_path)

        self.data = mujoco.MjData(self.model)

        # Configuração de Atuadores e Torques
        self.motor_indices = []
        self.torque_limits = []
        class_torques = {
            "he": 7, "lae": 30, "rae": 30, "te": 30,
            "lle1": 45, "lle2": 30, "lle3": 30, "lle4": 60, "lle5": 20, "lle6": 15,
            "rle1": 45, "rle2": 30, "rle3": 30, "rle4": 60, "rle5": 20, "rle6": 15
        }

        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            if "tau" in name:
                self.motor_indices.append(i)
                limit = 18
                for prefix, torque in class_torques.items():
                    if name.startswith(prefix):
                        limit = torque
                        break
                self.torque_limits.append(limit)

        self.torque_limits = np.array(self.torque_limits)
        self.n_actions = len(self.motor_indices)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)

        # IDs de corpos para contato
        self.hand_ids = [self.model.body('left_hand_link').id, self.model.body('right_hand_link').id]
        self.foot_ids = [self.model.body('left_foot_link').id, self.model.body('right_foot_link').id]
        self.forearm_ids = [self.model.body('AL3').id, self.model.body('AR3').id]
        self.shank_ids = [self.model.body('Shank_Left').id, self.model.body('Shank_Right').id]

        # Mapeamento de juntas importantes para Reward Shaping
        self.l_knee_idx = self.model.joint('Left_Knee_Pitch').qposadr[0]
        self.r_knee_idx = self.model.joint('Right_Knee_Pitch').qposadr[0]
        self.l_hip_pitch_idx = self.model.joint('Left_Hip_Pitch').qposadr[0]
        self.r_hip_pitch_idx = self.model.joint('Right_Hip_Pitch').qposadr[0]

        self.l_shoulder_roll_idx = self.model.joint('Left_Shoulder_Roll').qposadr[0]
        self.r_shoulder_roll_idx = self.model.joint('Right_Shoulder_Roll').qposadr[0]

        self.max_steps = 1000
        self.current_step = 0
        self.prev_action = None
        self.total_training_steps = 0

    def set_training_progress(self, total_steps):
        """Atualizado pelo callback de curriculum durante o treinamento."""
        self.total_training_steps = total_steps

    def _get_contact_info(self):
        """Detecta contatos com o chão. Retorna 8 flags binárias."""
        contacts = np.zeros(8)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 == 0 or con.geom2 == 0:
                body1 = self.model.geom_bodyid[con.geom1]
                body2 = self.model.geom_bodyid[con.geom2]
                if body1 == self.hand_ids[0] or body2 == self.hand_ids[0]: contacts[0] = 1.0
                if body1 == self.hand_ids[1] or body2 == self.hand_ids[1]: contacts[1] = 1.0
                if body1 == self.foot_ids[0] or body2 == self.foot_ids[0]: contacts[2] = 1.0
                if body1 == self.foot_ids[1] or body2 == self.foot_ids[1]: contacts[3] = 1.0
                if body1 == self.forearm_ids[0] or body2 == self.forearm_ids[0]: contacts[4] = 1.0
                if body1 == self.forearm_ids[1] or body2 == self.forearm_ids[1]: contacts[5] = 1.0
                if body1 == self.shank_ids[0] or body2 == self.shank_ids[0]: contacts[6] = 1.0
                if body1 == self.shank_ids[1] or body2 == self.shank_ids[1]: contacts[7] = 1.0
        return contacts

    @abstractmethod
    def _get_obs(self):
        """Cada ambiente define sua própria observação."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Cada ambiente define seu próprio step."""
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self, action, *args, **kwargs):
        """Cada ambiente define sua própria recompensa."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        """Inicializa o RNG do Gymnasium. Subclasses devem chamar super().reset(seed=seed)."""
        super().reset(seed=seed)

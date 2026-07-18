import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
import os
import re
import yaml


def _load_keyframes_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    keyframes = []
    deltas = []
    for kf in data['keyframes']:
        motors = kf['motor_positions']
        keyframes.append({name: np.radians(val) for name, val in motors.items()})
        deltas.append(kf.get('delta', 1.0))
    return keyframes, deltas


class GetUpEnv(gym.Env):
    """Classe Base para o ambiente de levantar do robô T1."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, model_path=None, render_mode=None):
        self.render_mode = render_mode
        self.renderer = None

        if model_path is None:
            home = os.path.expanduser("~")
            model_path = os.path.join(home, "rcssservermj/src/rcsssmj/resources/robots/T1/robot.xml")

        super().__init__()

        # 1. Carregar e configurar o XML (idêntico ao original)
        with open(model_path, 'r') as f:
            xml_string = f.read()

        mesh_dir = os.path.join(os.path.dirname(model_path), "meshes")
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

        # Índice da cintura (Waist)
        self.waist_idx = self.model.joint('Waist').qposadr[0]

        # Índices para Simetria (mapeados dinamicamente)
        self.left_joints_idx = [
            self.model.joint('Left_Shoulder_Pitch').qposadr[0],
            self.model.joint('Left_Shoulder_Roll').qposadr[0],
            self.model.joint('Left_Elbow_Pitch').qposadr[0],
            self.model.joint('Left_Hip_Pitch').qposadr[0],
            self.model.joint('Left_Hip_Roll').qposadr[0],
            self.model.joint('Left_Knee_Pitch').qposadr[0],
            self.model.joint('Left_Ankle_Pitch').qposadr[0],
        ]
        self.right_joints_idx = [
            self.model.joint('Right_Shoulder_Pitch').qposadr[0],
            self.model.joint('Right_Shoulder_Roll').qposadr[0],
            self.model.joint('Right_Elbow_Pitch').qposadr[0],
            self.model.joint('Right_Hip_Pitch').qposadr[0],
            self.model.joint('Right_Hip_Roll').qposadr[0],
            self.model.joint('Right_Knee_Pitch').qposadr[0],
            self.model.joint('Right_Ankle_Pitch').qposadr[0],
        ]

        obs_shape = self.model.nsensordata + self.model.nq + self.model.nv + 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        self.max_steps = 1000
        self.current_step = 0

    def _get_contact_info(self):
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

    def _get_obs(self):
        return np.concatenate([
            self.data.sensordata.flatten(),
            self.data.qpos.flatten(),
            self.data.qvel.flatten(),
            self._get_contact_info()
        ]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        ctrl = np.zeros(self.model.nu)
        ctrl[self.motor_indices] = action * self.torque_limits
        self.data.ctrl[:] = ctrl

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self.compute_reward(action, obs)

        terminated = False
        truncated = self.current_step >= self.max_steps
        return obs, float(reward), terminated, truncated, {}

    def compute_reward(self, action, obs):
        return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self.prev_action = np.zeros(self.n_actions)
        self.data.qpos[2] = 0.25
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera=-1)
            return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class _MimicGetUpMixin:
    """Carrega um YAML de keyframes e fornece reward de imitação (DeepMimic-style)."""

    DEFAULT_YAML_NAME = None

    def _init_mimic(self, keyframe_yaml=None):
        if keyframe_yaml is None:
            keyframe_yaml = os.path.join(
                os.path.dirname(__file__), "..", "mujococodebase", "skills", "keyframe", "get_up", self.DEFAULT_YAML_NAME
            )
            if not os.path.exists(keyframe_yaml):
                keyframe_yaml = os.path.join(
                    os.path.dirname(__file__), "keyframe", "get_up", self.DEFAULT_YAML_NAME
                )
        self.KEYFRAMES, self.KEYFRAME_DELTAS = _load_keyframes_from_yaml(keyframe_yaml)

        try:
            self.head_id = self.model.body('head').id
        except KeyError:
            self.head_id = None

        self.left_joints_map = {
            "Hip_Pitch":      self._qadr("Left_Hip_Pitch"),
            "Hip_Roll":       self._qadr("Left_Hip_Roll"),
            "Hip_Yaw":        self._qadr("Left_Hip_Yaw"),
            "Knee_Pitch":     self._qadr("Left_Knee_Pitch"),
            "Ankle_Pitch":    self._qadr("Left_Ankle_Pitch"),
            "Ankle_Roll":     self._qadr("Left_Ankle_Roll"),
            "Shoulder_Pitch": self._qadr("Left_Shoulder_Pitch"),
            "Shoulder_Roll":  self._qadr("Left_Shoulder_Roll"),
            "Elbow_Pitch":    self._qadr("Left_Elbow_Pitch"),
            "Elbow_Yaw":      self._qadr("Left_Elbow_Yaw"),
        }
        self.right_joints_map = {
            "Hip_Pitch":      self._qadr("Right_Hip_Pitch"),
            "Hip_Roll":       self._qadr("Right_Hip_Roll"),
            "Hip_Yaw":        self._qadr("Right_Hip_Yaw"),
            "Knee_Pitch":     self._qadr("Right_Knee_Pitch"),
            "Ankle_Pitch":    self._qadr("Right_Ankle_Pitch"),
            "Ankle_Roll":     self._qadr("Right_Ankle_Roll"),
            "Shoulder_Pitch": self._qadr("Right_Shoulder_Pitch"),
            "Shoulder_Roll":  self._qadr("Right_Shoulder_Roll"),
            "Elbow_Pitch":    self._qadr("Right_Elbow_Pitch"),
            "Elbow_Yaw":      self._qadr("Right_Elbow_Yaw"),
        }
        self.head_yaw_qadr = self._qadr("Head_Yaw")
        self.head_pitch_qadr = self._qadr("Head_Pitch")

        self.phase = 0
        self._stand_hold_steps = 0

    def _qadr(self, joint_name):
        try:
            return self.model.joint(joint_name).qposadr[0]
        except KeyError:
            return None

    def _target_qpos(self, keyframe):
        target = self.data.qpos[7:].copy()

        if self.head_yaw_qadr is not None and "Head_yaw" in keyframe:
            target[self.head_yaw_qadr - 7] = keyframe["Head_yaw"]
        if self.head_pitch_qadr is not None and "Head_pitch" in keyframe:
            target[self.head_pitch_qadr - 7] = keyframe["Head_pitch"]
        if self.waist_idx is not None and "Waist" in keyframe:
            target[self.waist_idx - 7] = keyframe["Waist"]

        for name, val in keyframe.items():
            if name in ("Head_yaw", "Head_pitch", "Waist"):
                continue
            joint_type = name.split("_")[-1]
            sign = -1.0 if joint_type in ("Roll", "Yaw") else 1.0

            l_idx = self.left_joints_map.get(name)
            r_idx = self.right_joints_map.get(name)
            if l_idx is not None:
                target[l_idx - 7] = val
            if r_idx is not None:
                target[r_idx - 7] = val * sign
        return target

    def _mimic_reward_term(self):
        cur_joints = self.data.qpos[7:]
        target = self._target_qpos(self.KEYFRAMES[self.phase])
        joint_err = np.mean(np.square(cur_joints - target))

        tol = 0.20
        if joint_err < tol and self.phase < len(self.KEYFRAMES) - 1:
            self.phase += 1

        return 40.0 * np.exp(-3.0 * joint_err)

    # ORIGINAL: Mantido intacto para não quebrar o GetUpFrontEnv
    def _common_shaping_terms(self, action, obs):
        contact_info = obs[-8:]
        torso_height = self.data.body('torso').xpos[2]

        reward = 0.0

        if contact_info[2] > 0:
            reward += (max(0.0, self.data.body('left_foot_link').xmat[8]) ** 3) * 5.0
        if contact_info[3] > 0:
            reward += (max(0.0, self.data.body('right_foot_link').xmat[8]) ** 3) * 5.0

        if self.head_id is not None:
            head_height = self.data.xpos[self.head_id][2]
            if head_height < 0.20:
                reward -= 10.0
            if head_height < torso_height:
                reward -= 5.0

        waist_qpos = self.data.qpos[self.waist_idx]
        reward -= 10.0 * (waist_qpos ** 2)

        left_j = self.data.qpos[self.left_joints_idx]
        right_j = self.data.qpos[self.right_joints_idx]
        symmetry_error = np.mean(np.square(left_j - right_j))
        reward -= 5.0 * symmetry_error

        reward -= 0.05 * np.sum(np.square(action))
        reward -= 0.05 * np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()

        return reward

    # ORIGINAL: Mantido intacto para não quebrar o GetUpFrontEnv
    def _standing_bonus(self, torso_height, torso_up):
        reward = 0.0
        is_standing = torso_height > 0.65 and torso_up > 0.9
        if is_standing:
            self._stand_hold_steps += 1
            reward += 5.0
            if self._stand_hold_steps >= 30:
                reward += 2000.0
        else:
            self._stand_hold_steps = 0
        return reward


class GetUpBackEnv(_MimicGetUpMixin, GetUpEnv):
    """Levantamento de Costas - Robô inicia com as COSTAS no chão (Barriga para CIMA)."""

    DEFAULT_YAML_NAME = "get_up_back.yaml"

    def __init__(self, model_path=None, keyframe_yaml=None, render_mode=None):
        super().__init__(model_path=model_path, render_mode=render_mode)
        self._init_mimic(keyframe_yaml=keyframe_yaml)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.data.qpos[3:7] = [0.707, 0, -0.707, 0]
        mujoco.mj_forward(self.model, self.data)

        self.phase = 0
        self._stand_hold_steps = 0

        return self._get_obs(), info

    # MÉTODOS EXCLUSIVOS DO BACK ENV COM AS MUDANÇAS PROPOSTAS
    def _back_shaping_terms(self, action, obs, torso_height, torso_up):
        contact_info = obs[-8:]
        reward = 0.0

        # 1. PÉ CHATO (Chapado no chão)
        l_foot_up = self.data.body('left_foot_link').xmat[8]
        r_foot_up = self.data.body('right_foot_link').xmat[8]
        if contact_info[2] > 0:
            reward += (max(0.0, l_foot_up) ** 3) * 10.0 # Peso dobrado para forçar a chapar o pé
        if contact_info[3] > 0:
            reward += (max(0.0, r_foot_up) ** 3) * 10.0

        # ANTI-BREAKDANCE
        if self.head_id is not None:
            head_height = self.data.xpos[self.head_id][2]
            if head_height < 0.20:
                reward -= 10.0
            if head_height < torso_height:
                reward -= 5.0

        # ANTI-TORÇÃO
        waist_qpos = self.data.qpos[self.waist_idx]
        reward -= 10.0 * (waist_qpos ** 2)

        # 2. BÔNUS DE APOIO E CURVATURA DA PERNA
        if torso_height < 0.45: 
            # Apoio do cotovelo/antebraço
            if contact_info[4] > 0: reward += 15.0 
            if contact_info[5] > 0: reward += 15.0 

            # TUCK: Incentivo ativo para dobrar os joelhos ("curvadas")
            l_knee = self.data.qpos[self.l_knee_idx]
            r_knee = self.data.qpos[self.r_knee_idx]
            # O valor absoluto encoraja a junta a dobrar, saindo da posição reta (0)
            reward += (abs(l_knee) + abs(r_knee)) * 5.0 

        # 3. SIMETRIA DIVIDIDA
        # Índices 3 ao 6 na sua lista são o quadril, joelho e tornozelo
        left_legs = self.data.qpos[self.left_joints_idx[3:]]
        right_legs = self.data.qpos[self.right_joints_idx[3:]]
        # Punição forte e constante para garantir pernas sempre simétricas
        reward -= 5.0 * np.mean(np.square(left_legs - right_legs)) 

        # Índices 0 ao 2 são ombro e cotovelo
        left_arms = self.data.qpos[self.left_joints_idx[:3]]
        right_arms = self.data.qpos[self.right_joints_idx[:3]]
        # Braços com liberdade no início, mas exigindo alinhamento na hora de ficar em pé
        if torso_height > 0.50 and torso_up > 0.8:
            reward -= 1.0 * np.mean(np.square(left_arms - right_arms))
        else:
            reward -= 0.1 * np.mean(np.square(left_arms - right_arms))

        # AÇÃO / SUAVIDADE
        reward -= 0.05 * np.sum(np.square(action))
        reward -= 0.05 * np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()

        return reward

    def _back_standing_bonus(self, torso_height, torso_up):
        """Condição de vitória suavizada específica para o BackEnv."""
        reward = 0.0
        is_standing = torso_height > 0.65 and torso_up > 0.9
        if is_standing:
            self._stand_hold_steps += 1
            reward += 10.0 
            victory_capped_bonus = min(500.0, 2.0 * self._stand_hold_steps)
            reward += victory_capped_bonus
        else:
            self._stand_hold_steps = 0
        return reward

    def compute_reward(self, action, obs):
        torso_height = self.data.body('torso').xpos[2]
        torso_up = self.data.body('torso').xmat[8]

        reward = 0.0

        # 1. MIMIC REWARD
        reward += self._mimic_reward_term()

        # 2. ALTURA
        if torso_height > 0.30:
            height_progress = np.clip((torso_height - 0.30) / (0.65 - 0.30), 0.0, 1.0)
            reward += (height_progress ** 3) * 50.0 * max(0.0, torso_up)

        # 3-8. TERMOS DE SHAPING ESPECÍFICOS DO BACK
        reward += self._back_shaping_terms(action, obs, torso_height, torso_up)

        # POSTURA DAS PERNAS: forçar os pés sob o corpo e evitar espacate
        torso_xy = self.data.body('torso').xpos[:2]
        l_foot_xy = self.data.body('left_foot_link').xpos[:2]
        r_foot_xy = self.data.body('right_foot_link').xpos[:2]
        
        # 1. Puxa os pés para perto do centro de massa (peso 5.0)
        dist_l_foot = np.linalg.norm(torso_xy - l_foot_xy)
        dist_r_foot = np.linalg.norm(torso_xy - r_foot_xy)
        reward -= (dist_l_foot + dist_r_foot) * 5.0

        # 2. NOVO: Punição direta pela distância ENTRE os pés (mata o espacate)
        dist_feet = np.linalg.norm(l_foot_xy - r_foot_xy)
        reward -= dist_feet * 10.0 

        # pressão leve pra resolver rápido
        reward -= 1.0

        # 9. CONDIÇÃO DE VITÓRIA ESPECÍFICA DO BACK
        reward += self._back_standing_bonus(torso_height, torso_up)

        return reward


class GetUpFrontEnv(_MimicGetUpMixin, GetUpEnv):
    """Levantamento de Frente - Robô inicia de FRENTE (Barriga para BAIXO)."""

    DEFAULT_YAML_NAME = "get_up_front.yaml"

    def __init__(self, model_path=None, keyframe_yaml=None, render_mode=None):
        super().__init__(model_path=model_path, render_mode=render_mode)
        self._init_mimic(keyframe_yaml=keyframe_yaml)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.data.qpos[3:7] = [0.707, 0, 0.707, 0]
        mujoco.mj_forward(self.model, self.data)

        self.phase = 0
        self._stand_hold_steps = 0

        return self._get_obs(), info

    def compute_reward(self, action, obs):
        contact_info = obs[-8:]
        torso_height = self.data.body('torso').xpos[2]
        torso_up = self.data.body('torso').xmat[8]

        reward = 0.0

        # 1. MIMIC REWARD
        reward += self._mimic_reward_term()

        # 2. ALTURA
        if torso_height > 0.30:
            height_progress = np.clip((torso_height - 0.30) / (0.65 - 0.30), 0.0, 1.0)
            reward += (height_progress ** 3) * 50.0 * max(0.0, torso_up)

        # 3-8. TERMOS COMPARTILHADOS ORIGINAIS
        reward += self._common_shaping_terms(action, obs)

        # "TUCK"
        l_knee = self.data.qpos[self.l_knee_idx]
        r_knee = self.data.qpos[self.r_knee_idx]
        l_hip = self.data.qpos[self.l_hip_pitch_idx]
        r_hip = self.data.qpos[self.r_hip_pitch_idx]
        if torso_height < 0.50:
            reward += (-l_knee - r_knee) * 5.0
            reward += (l_hip + r_hip) * 5.0

        # Joelhos como apoio
        if torso_height < 0.45 and (contact_info[6] > 0 or contact_info[7] > 0):
            reward += 25.0

        # pressão leve pra resolver rápido
        reward -= 0.5

        # 9. CONDIÇÃO DE VITÓRIA ORIGINAL
        reward += self._standing_bonus(torso_height, torso_up)

        return reward
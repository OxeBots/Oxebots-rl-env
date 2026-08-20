import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
import os
import re
import yaml

JOINT_NAMES = [
    'AAHead_yaw', 'Head_pitch',
    'Left_Shoulder_Pitch', 'Left_Shoulder_Roll', 'Left_Elbow_Pitch', 'Left_Elbow_Yaw',
    'Right_Shoulder_Pitch', 'Right_Shoulder_Roll', 'Right_Elbow_Pitch', 'Right_Elbow_Yaw',
    'Waist',
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw', 'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw', 'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll'
]

YAML_MAPPING = {
    'Head_yaw': [('AAHead_yaw', 1.0)],
    'Head_pitch': [('Head_pitch', 1.0)],
    'Waist': [('Waist', 1.0)],
    'Shoulder_Pitch': [('Left_Shoulder_Pitch', 1.0), ('Right_Shoulder_Pitch', 1.0)],
    'Shoulder_Roll': [('Left_Shoulder_Roll', 1.0), ('Right_Shoulder_Roll', -1.0)],
    'Elbow_Pitch': [('Left_Elbow_Pitch', 1.0), ('Right_Elbow_Pitch', 1.0)],
    'Elbow_Yaw': [('Left_Elbow_Yaw', 1.0), ('Right_Elbow_Yaw', -1.0)],
    'Hip_Pitch': [('Left_Hip_Pitch', 1.0), ('Right_Hip_Pitch', 1.0)],
    'Hip_Roll': [('Left_Hip_Roll', 1.0), ('Right_Hip_Roll', -1.0)],
    'Hip_Yaw': [('Left_Hip_Yaw', 1.0), ('Right_Hip_Yaw', -1.0)],
    'Knee_Pitch': [('Left_Knee_Pitch', 1.0), ('Right_Knee_Pitch', 1.0)],
    'Ankle_Pitch': [('Left_Ankle_Pitch', 1.0), ('Right_Ankle_Pitch', 1.0)],
    'Ankle_Roll': [('Left_Ankle_Roll', 1.0), ('Right_Ankle_Roll', -1.0)],
}


def _load_keyframes_from_yaml(yaml_path):
    if not yaml_path or not os.path.exists(yaml_path):
        return np.zeros((1, 23), dtype=np.float32)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    kf_targets = []
    for kf in data.get('keyframes', []):
        motors = kf['motor_positions']
        radians_dict = {name: float(np.radians(val)) for name, val in motors.items()}
        target_vec = {}
        for y_name, val in radians_dict.items():
            if y_name in YAML_MAPPING:
                for j_name, mult in YAML_MAPPING[y_name]:
                    target_vec[j_name] = float(val * mult)
        vec = [target_vec.get(j, 0.0) for j in JOINT_NAMES]
        kf_targets.append(vec)
    return np.array(kf_targets, dtype=np.float32) if len(kf_targets) > 0 else np.zeros((1, 23), dtype=np.float32)


class GetUpEnv(gym.Env):
    """
    Classe Base de Levantamento para CPU via MuJoCo + Gymnasium.
    Modelo HÍBRIDO: Guia de Poses dos Keyframes YAML (Soft-DeepMimic) + Recompensas Biomecânicas Físicas.
    """
    metadata = {"render_modes": ["rgb_array"]}
    DEFAULT_YAML_NAME = None
    DEFAULT_N_FRAMES = 5

    def __init__(self, model_path=None, keyframe_yaml=None, render_mode=None, n_frames=None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.renderer = None

        if model_path is None or not os.path.exists(model_path):
            home = os.path.expanduser("~")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.abspath(os.path.join(current_dir, "..", "robots", "T1", "robot.xml")),
                os.path.abspath(os.path.join(current_dir, "..", "resources", "robots", "T1", "robot.xml")),
                os.path.abspath(os.path.join(current_dir, "..", "mojucco_simulator3D", "src", "rcsssmj", "resources", "robots", "T1", "robot.xml")),
                os.path.abspath(os.path.join(current_dir, "..", "..", "mojucco_simulator3D", "src", "rcsssmj", "resources", "robots", "T1", "robot.xml")),
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
        self.torque_limits = np.array(self.torque_limits, dtype=np.float32)
        self.n_actions = len(self.motor_indices)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)

        # Mapeamento de corpos
        try:
            self.torso_id = self.model.body('torso').id
        except KeyError:
            self.torso_id = 1

        try:
            self.left_foot_id = self.model.body('left_foot_link').id
        except KeyError:
            self.left_foot_id = self.torso_id
        try:
            self.right_foot_id = self.model.body('right_foot_link').id
        except KeyError:
            self.right_foot_id = self.torso_id

        # Mapeamento de juntas
        self._l_knee_idx = self._get_qadr('Left_Knee_Pitch')
        self._r_knee_idx = self._get_qadr('Right_Knee_Pitch')
        self._l_hip_pitch_idx = self._get_qadr('Left_Hip_Pitch')
        self._r_hip_pitch_idx = self._get_qadr('Right_Hip_Pitch')
        self._l_hip_roll_idx = self._get_qadr('Left_Hip_Roll')
        self._r_hip_roll_idx = self._get_qadr('Right_Hip_Roll')
        self._waist_idx = self._get_qadr('Waist')
        self._l_shoulder_pitch_idx = self._get_qadr('Left_Shoulder_Pitch')
        self._r_shoulder_pitch_idx = self._get_qadr('Right_Shoulder_Pitch')
        self._l_shoulder_roll_idx = self._get_qadr('Left_Shoulder_Roll')
        self._r_shoulder_roll_idx = self._get_qadr('Right_Shoulder_Roll')
        self._l_elbow_pitch_idx = self._get_qadr('Left_Elbow_Pitch')
        self._r_elbow_pitch_idx = self._get_qadr('Right_Elbow_Pitch')

        # Carregar Keyframes do YAML para guiamento híbrido
        if keyframe_yaml is None and self.DEFAULT_YAML_NAME:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.abspath(os.path.join(current_dir, "..", "keyframes", self.DEFAULT_YAML_NAME)),
                os.path.abspath(os.path.join(current_dir, "..", "keyframes", "get_up", self.DEFAULT_YAML_NAME)),
                os.path.abspath(os.path.join(current_dir, "..", "mujococodebase", "skills", "keyframe", "get_up", self.DEFAULT_YAML_NAME)),
                os.path.abspath(os.path.join(current_dir, "..", "resources", "skills", "keyframe", "get_up", self.DEFAULT_YAML_NAME)),
                os.path.abspath(os.path.join(current_dir, "keyframe", "get_up", self.DEFAULT_YAML_NAME)),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    keyframe_yaml = cand
                    break

        self.keyframe_targets = _load_keyframes_from_yaml(keyframe_yaml)
        self.n_phases = len(self.keyframe_targets)
        self._has_mimic = self.n_phases > 0

        # Espaço de Observação Híbrido (119 dimensões)
        # qpos(30) + qvel(29) + torso_height(1) + torso_up(1) + torso_angvel(3) + torso_linvel(3) + left_foot(3) + right_foot(3) + joint_err(23) + last_action(23)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(119,), dtype=np.float32)

        self.n_frames = n_frames if n_frames is not None else self.DEFAULT_N_FRAMES
        self.max_steps = 1000
        self.current_step = 0
        self.last_action = np.zeros(self.n_actions, dtype=np.float32)
        self.standing_count = 0.0

    def _get_qadr(self, joint_name):
        try:
            return self.model.joint(joint_name).qposadr[0]
        except KeyError:
            return 0

    def get_target_pose(self, progress: float) -> np.ndarray:
        """Interpolador suave de pose alvo dos Keyframes em função da fase de altura [0, 1]."""
        if not self._has_mimic or self.n_phases <= 1:
            return self.keyframe_targets[0]
        idx_float = float(np.clip(progress, 0.0, 1.0) * (self.n_phases - 1))
        idx_low = int(np.floor(idx_float))
        idx_high = min(idx_low + 1, self.n_phases - 1)
        alpha = idx_float - idx_low
        return (1.0 - alpha) * self.keyframe_targets[idx_low] + alpha * self.keyframe_targets[idx_high]

    def _get_obs(self, last_action=None, target_pose=None) -> np.ndarray:
        """
        Observações enriquecidas HÍBRIDAS:
        qpos + qvel + torso height + torso up + torso angvel/linvel + foot pos + joint_err + last_action.
        """
        torso_pos = self.data.xpos[self.torso_id]
        torso_height = np.array([torso_pos[2]], dtype=np.float32)

        q = self.data.qpos[3:7]
        torso_up = np.array([1.0 - 2.0 * (q[1]**2 + q[2]**2)], dtype=np.float32)

        torso_cvel = self.data.cvel[self.torso_id]
        torso_angvel = torso_cvel[:3].astype(np.float32)
        torso_linvel = torso_cvel[3:].astype(np.float32)

        left_foot_pos = self.data.xpos[self.left_foot_id].astype(np.float32)
        right_foot_pos = self.data.xpos[self.right_foot_id].astype(np.float32)

        actuated_qpos = self.data.qpos[7:30].astype(np.float32)
        if target_pose is None:
            target_pose = actuated_qpos
        joint_err_vec = (target_pose - actuated_qpos).astype(np.float32)  # (23,)

        if last_action is None:
            last_action = np.zeros(self.n_actions, dtype=np.float32)

        obs = np.concatenate([
            self.data.qpos.astype(np.float32),        # 30
            self.data.qvel.astype(np.float32),        # 29
            torso_height,                             # 1
            torso_up,                                 # 1
            torso_angvel,                             # 3
            torso_linvel,                             # 3
            left_foot_pos,                            # 3
            right_foot_pos,                           # 3
            joint_err_vec,                            # 23
            last_action.astype(np.float32),           # 23
        ])
        return np.nan_to_num(obs, nan=0.0, posinf=1e2, neginf=-1e2).astype(np.float32)

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


class GetUpFrontEnv(GetUpEnv):
    """
    Levantamento HÍBRIDO de Frente (Front / Prone) para CPU.
    Combina Poses Alvo do Keyframe YAML (Soft DeepMimic) + Recompensas Biomecânicas de Física.
    """
    DEFAULT_YAML_NAME = "get_up_front.yaml"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        init_qpos = self.model.key_qpos[0].copy() if self.model.nkey > 0 else np.zeros(self.model.nq)
        if np.all(init_qpos == 0):
            init_qpos[:7] = [0, 0, 0.6735, 1, 0, 0, 0]

        # 1. Pose deitado de bruços (Front)
        qpos_front = init_qpos.copy()
        qpos_front[2] = 0.22
        qpos_front[3] = 0.707
        qpos_front[4] = 0.707
        qpos_front[5] = 0.0
        qpos_front[6] = 0.0

        # 2. Pose intermediária baseada no Keyframe 2 do YAML (Ajoelhado/Crouch)
        qpos_crouch = init_qpos.copy()
        qpos_crouch[2] = 0.42
        qpos_crouch[3] = 0.924
        qpos_crouch[4] = 0.383
        qpos_crouch[5] = 0.0
        qpos_crouch[6] = 0.0
        if self._has_mimic and self.n_phases >= 3:
            kf2 = self.keyframe_targets[2]
            qpos_crouch[7:30] = kf2

        # 3. Pose quase de pé baseada no Keyframe final do YAML
        qpos_stand = init_qpos.copy()
        qpos_stand[2] = 0.60
        qpos_stand[3] = 1.0
        qpos_stand[4] = 0.0
        qpos_stand[5] = 0.0
        qpos_stand[6] = 0.0
        if self._has_mimic and self.n_phases >= 4:
            kf_last = self.keyframe_targets[-1]
            qpos_stand[7:30] = kf_last

        # Curriculum de Reset (60% deitado, 25% ajoelhado/keyframe, 15% de pé/keyframe)
        u = self.np_random.uniform(0.0, 1.0)
        if u < 0.60:
            qpos = qpos_front
        elif u < 0.85:
            qpos = qpos_crouch
        else:
            qpos = qpos_stand

        # Ruído nas juntas
        joint_noise = self.np_random.uniform(-0.10, 0.10, size=qpos.shape[0] - 7)
        qpos[7:] += joint_noise

        # Ruído em qvel
        qvel = self.np_random.uniform(-0.2, 0.2, size=self.model.nv)
        qvel[:6] = 0.0

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.last_action = np.zeros(self.n_actions, dtype=np.float32)
        self.standing_count = 0.0

        target_pose = self.get_target_pose(0.0)
        obs = self._get_obs(self.last_action, target_pose)
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.current_step += 1

        ctrl = np.zeros(self.model.nu)
        ctrl[self.motor_indices] = action * self.torque_limits
        self.data.ctrl[:] = ctrl

        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)

        torso_pos = self.data.xpos[self.torso_id]
        torso_height = float(torso_pos[2])

        q = self.data.qpos[3:7]
        torso_up = float(1.0 - 2.0 * (q[1]**2 + q[2]**2))

        # 1. Progresso Vertical Relativo
        h_ground = 0.24
        h_target = 0.65
        h_rel = float(np.clip((torso_height - h_ground) / (h_target - h_ground), 0.0, 1.0))
        height_reward = (h_rel ** 2.0) * 120.0

        # 2. Keyframe Target Pose Suave (Soft DeepMimic RL)
        target_pose = self.get_target_pose(h_rel)
        actuated_qpos = self.data.qpos[7:30]
        joint_err = float(np.mean(np.square(actuated_qpos - target_pose)))

        # Recompensa Gaussiana Suave de Mimic (Nunca Zera! Fornece gradiente contínuo)
        mimic_reward = 80.0 * np.exp(-1.5 * joint_err)

        # 3. Orientação Upright
        upright_pos = float(np.clip((torso_up + 0.2) / 1.2, 0.0, 1.0))
        upright_reward = (upright_pos ** 2.0) * 60.0

        # 4. Velocidade Vertical Positiva
        torso_vz = float(self.data.cvel[self.torso_id][5])
        velocity_reward = max(0.0, min(torso_vz, 2.0)) * 40.0

        # 5. Sinal de Standing Contínuo
        standing_signal = h_rel * upright_pos
        standing_reward = (standing_signal ** 3.0) * 350.0

        # 6. Biomecânica de Bruços
        l_shoulder = float(self.data.qpos[self._l_shoulder_pitch_idx])
        r_shoulder = float(self.data.qpos[self._r_shoulder_pitch_idx])
        push_reward = max(0.0, min(l_shoulder + r_shoulder, 4.0)) * 10.0 if torso_height < 0.40 else 0.0

        l_knee = float(self.data.qpos[self._l_knee_idx])
        r_knee = float(self.data.qpos[self._r_knee_idx])
        tuck_reward = max(0.0, min(l_knee + r_knee, 4.0)) * 10.0 if torso_height < 0.50 else 0.0

        symmetry_penalty = -2.0 * ((l_knee - r_knee)**2 + (l_shoulder - r_shoulder)**2)

        # 7. Penalidades
        action_penalty = -0.001 * float(np.sum(np.square(action)))
        action_smooth_penalty = -0.005 * float(np.sum(np.square(action - self.last_action)))
        torso_angvel = self.data.cvel[self.torso_id][:3]
        angvel_penalty = -0.01 * float(np.sum(np.square(torso_angvel)))
        step_penalty = -0.2

        # 8. Bônus de Sucesso e Estabilidade
        is_standing = (torso_height > 0.58) and (torso_up > 0.80)
        standing_bonus = 150.0 if is_standing else 0.0

        if is_standing:
            self.standing_count += 1.0
        else:
            self.standing_count = 0.0
        stability_bonus = 250.0 if self.standing_count >= 10 else 0.0

        total_reward = float(
            height_reward + upright_reward + velocity_reward + standing_reward + mimic_reward
            + push_reward + tuck_reward + symmetry_penalty
            + action_penalty + action_smooth_penalty + angvel_penalty + step_penalty
            + standing_bonus + stability_bonus
        )

        fell = bool((torso_height < 0.05) or np.isnan(torso_height))
        terminated = fell
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs(action, target_pose)
        self.last_action = action.copy()

        info = {
            'reward_height': height_reward,
            'reward_upright': upright_reward,
            'reward_velocity': velocity_reward,
            'reward_standing': standing_reward + standing_bonus,
            'reward_mimic': mimic_reward,
            'joint_error': joint_err,
            'torso_height': torso_height,
            'torso_up': torso_up,
            'standing_count': self.standing_count,
        }

        return obs, total_reward, terminated, truncated, info


class GetUpBackEnv(GetUpEnv):
    """
    Levantamento HÍBRIDO de Costas (Back / Supine) para CPU.
    Combina Poses Alvo do Keyframe YAML (Soft DeepMimic) + Recompensas Biomecânicas de Física.
    """
    DEFAULT_YAML_NAME = "get_up_back.yaml"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        init_qpos = self.model.key_qpos[0].copy() if self.model.nkey > 0 else np.zeros(self.model.nq)
        if np.all(init_qpos == 0):
            init_qpos[:7] = [0, 0, 0.6735, 1, 0, 0, 0]

        # 1. Pose deitado de costas (Back)
        qpos_back = init_qpos.copy()
        qpos_back[2] = 0.22
        qpos_back[3] = 0.707
        qpos_back[4] = 0.0
        qpos_back[5] = -0.707
        qpos_back[6] = 0.0

        # 2. Pose intermediária baseada no Keyframe 2 do YAML
        qpos_crouch = init_qpos.copy()
        qpos_crouch[2] = 0.42
        qpos_crouch[3] = 0.924
        qpos_crouch[4] = -0.383
        qpos_crouch[5] = 0.0
        qpos_crouch[6] = 0.0
        if self._has_mimic and self.n_phases >= 2:
            kf1 = self.keyframe_targets[1]
            qpos_crouch[7:30] = kf1

        # 3. Pose quase de pé baseada no Keyframe final do YAML
        qpos_stand = init_qpos.copy()
        qpos_stand[2] = 0.60
        qpos_stand[3] = 1.0
        qpos_stand[4] = 0.0
        qpos_stand[5] = 0.0
        qpos_stand[6] = 0.0
        if self._has_mimic and self.n_phases >= 3:
            kf_last = self.keyframe_targets[-1]
            qpos_stand[7:30] = kf_last

        # Curriculum de Reset (60% deitado, 25% ajoelhado/keyframe, 15% de pé/keyframe)
        u = self.np_random.uniform(0.0, 1.0)
        if u < 0.60:
            qpos = qpos_back
        elif u < 0.85:
            qpos = qpos_crouch
        else:
            qpos = qpos_stand

        # Ruído nas juntas
        joint_noise = self.np_random.uniform(-0.10, 0.10, size=qpos.shape[0] - 7)
        qpos[7:] += joint_noise

        # Ruído em qvel
        qvel = self.np_random.uniform(-0.2, 0.2, size=self.model.nv)
        qvel[:6] = 0.0

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.last_action = np.zeros(self.n_actions, dtype=np.float32)
        self.standing_count = 0.0

        target_pose = self.get_target_pose(0.0)
        obs = self._get_obs(self.last_action, target_pose)
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.current_step += 1

        ctrl = np.zeros(self.model.nu)
        ctrl[self.motor_indices] = action * self.torque_limits
        self.data.ctrl[:] = ctrl

        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)

        torso_pos = self.data.xpos[self.torso_id]
        torso_height = float(torso_pos[2])

        q = self.data.qpos[3:7]
        torso_up = float(1.0 - 2.0 * (q[1]**2 + q[2]**2))

        # 1. Progresso Vertical Relativo
        h_ground = 0.24
        h_target = 0.65
        h_rel = float(np.clip((torso_height - h_ground) / (h_target - h_ground), 0.0, 1.0))
        height_reward = (h_rel ** 2.0) * 120.0

        # 2. Keyframe Target Pose Suave (Soft DeepMimic RL)
        target_pose = self.get_target_pose(h_rel)
        actuated_qpos = self.data.qpos[7:30]
        joint_err = float(np.mean(np.square(actuated_qpos - target_pose)))

        # Recompensa Gaussiana Suave de Mimic (Nunca Zera! Fornece gradiente contínuo)
        mimic_reward = 80.0 * np.exp(-1.5 * joint_err)

        # 3. Orientação Upright
        upright_pos = float(np.clip((torso_up + 0.2) / 1.2, 0.0, 1.0))
        upright_reward = (upright_pos ** 2.0) * 60.0

        # 4. Velocidade Vertical Positiva
        torso_vz = float(self.data.cvel[self.torso_id][5])
        velocity_reward = max(0.0, min(torso_vz, 2.0)) * 40.0

        # 5. Sinal de Standing Contínuo
        standing_signal = h_rel * upright_pos
        standing_reward = (standing_signal ** 3.0) * 350.0

        # 6. Biomecânica de Costas
        l_shoulder = float(self.data.qpos[self._l_shoulder_pitch_idx])
        r_shoulder = float(self.data.qpos[self._r_shoulder_pitch_idx])
        push_reward = max(0.0, min(-l_shoulder - r_shoulder, 4.0)) * 10.0 if torso_height < 0.40 else 0.0

        l_knee = float(self.data.qpos[self._l_knee_idx])
        r_knee = float(self.data.qpos[self._r_knee_idx])
        tuck_reward = max(0.0, min(l_knee + r_knee, 4.0)) * 10.0 if torso_height < 0.40 else 0.0

        l_hip = float(self.data.qpos[self._l_hip_pitch_idx])
        r_hip = float(self.data.qpos[self._r_hip_pitch_idx])
        hip_drive_reward = max(0.0, min(-l_hip - r_hip, 3.0)) * 10.0 if (0.30 < torso_height < 0.55) else 0.0

        symmetry_penalty = -2.0 * ((l_knee - r_knee)**2 + (l_shoulder - r_shoulder)**2)

        # 7. Penalidades
        action_penalty = -0.001 * float(np.sum(np.square(action)))
        action_smooth_penalty = -0.005 * float(np.sum(np.square(action - self.last_action)))
        torso_angvel = self.data.cvel[self.torso_id][:3]
        angvel_penalty = -0.01 * float(np.sum(np.square(torso_angvel)))
        step_penalty = -0.2

        # 8. Bônus de Sucesso e Estabilidade
        is_standing = (torso_height > 0.58) and (torso_up > 0.80)
        standing_bonus = 150.0 if is_standing else 0.0

        if is_standing:
            self.standing_count += 1.0
        else:
            self.standing_count = 0.0
        stability_bonus = 250.0 if self.standing_count >= 10 else 0.0

        total_reward = float(
            height_reward + upright_reward + velocity_reward + standing_reward + mimic_reward
            + push_reward + tuck_reward + hip_drive_reward + symmetry_penalty
            + action_penalty + action_smooth_penalty + angvel_penalty + step_penalty
            + standing_bonus + stability_bonus
        )

        fell = bool((torso_height < 0.05) or np.isnan(torso_height))
        terminated = fell
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs(action, target_pose)
        self.last_action = action.copy()

        info = {
            'reward_height': height_reward,
            'reward_upright': upright_reward,
            'reward_velocity': velocity_reward,
            'reward_standing': standing_reward + standing_bonus,
            'reward_mimic': mimic_reward,
            'joint_error': joint_err,
            'torso_height': torso_height,
            'torso_up': torso_up,
            'standing_count': self.standing_count,
        }

        return obs, total_reward, terminated, truncated, info

import os
import re
import yaml
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

try:
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf
    HAS_BRAX = True
except ImportError:
    HAS_BRAX = False
    PipelineEnv = object
    State = None


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
        return jnp.zeros((1, 23))
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    kf_targets = []
    for kf in data.get('keyframes', []):
        motors = kf['motor_positions']
        radians_dict = {name: float(jnp.radians(val)) for name, val in motors.items()}
        target_vec = {}
        for y_name, val in radians_dict.items():
            if y_name in YAML_MAPPING:
                for j_name, mult in YAML_MAPPING[y_name]:
                    target_vec[j_name] = float(val * mult)
        vec = [target_vec.get(j, 0.0) for j in JOINT_NAMES]
        kf_targets.append(vec)
    return jnp.array(kf_targets) if len(kf_targets) > 0 else jnp.zeros((1, 23))


class BaseGetUpMjxEnv(PipelineEnv):
    """
    Classe Base de Levantamento acelerada 100% em GPU via MuJoCo MJX + Brax.
    Modelo HÍBRIDO: Guia de Poses dos Keyframes YAML (Soft-DeepMimic) + Recompensas Biomecânicas Físicas.
    """
    DEFAULT_YAML_NAME = None
    DEFAULT_N_FRAMES = 5

    def __init__(self, model_path=None, keyframe_yaml=None, **kwargs):
        if not HAS_BRAX:
            raise ImportError("Brax e MJX são necessários. Instale com: pip install brax mujoco-mjx")

        if model_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_path = os.path.join(repo_root, "resources", "robots", "T1", "robot.xml")

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

        self.mj_model = mujoco.MjModel.from_xml_string(xml_string)
        sys = mjcf.load_model(self.mj_model)

        # Mapeamento de corpos
        try:
            self.torso_id = self.mj_model.body('torso').id
        except KeyError:
            self.torso_id = 1

        try:
            self.left_foot_id = self.mj_model.body('left_foot_link').id
        except KeyError:
            self.left_foot_id = self.torso_id
        try:
            self.right_foot_id = self.mj_model.body('right_foot_link').id
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
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            keyframe_yaml = os.path.join(repo_root, "resources", "skills", "keyframe", "get_up", self.DEFAULT_YAML_NAME)

        self.keyframe_targets = _load_keyframes_from_yaml(keyframe_yaml)
        self.n_phases = len(self.keyframe_targets)
        self._has_mimic = self.n_phases > 0

        n_frames = kwargs.pop('n_frames', self.DEFAULT_N_FRAMES)
        super().__init__(sys=sys, backend='mjx', n_frames=n_frames, **kwargs)

    def _get_qadr(self, joint_name):
        try:
            return self.mj_model.joint(joint_name).qposadr[0]
        except KeyError:
            return 0

    def get_target_pose(self, progress: jax.Array) -> jax.Array:
        """Interpolador suave de pose alvo dos Keyframes em função da fase de altura [0, 1]."""
        if not self._has_mimic or self.n_phases <= 1:
            return self.keyframe_targets[0]
        idx_float = jnp.clip(progress, 0.0, 1.0) * (self.n_phases - 1)
        idx_low = jnp.floor(idx_float).astype(jnp.int32)
        idx_high = jnp.minimum(idx_low + 1, self.n_phases - 1)
        alpha = idx_float - idx_low
        return (1.0 - alpha) * self.keyframe_targets[idx_low] + alpha * self.keyframe_targets[idx_high]

    def _get_obs(self, pipeline_state, last_action=None, target_pose=None) -> jax.Array:
        """
        Observações enriquecidas HÍBRIDAS:
        qpos + qvel + torso height + torso up + torso angvel/linvel + foot pos + joint_err + last_action.
        """
        torso_height = pipeline_state.x.pos[self.torso_id][2:3]
        torso_rot = pipeline_state.x.rot[self.torso_id]
        torso_up = (1.0 - 2.0 * (torso_rot[1]**2 + torso_rot[2]**2)).reshape(1)
        torso_angvel = pipeline_state.xd.ang[self.torso_id]
        torso_linvel = pipeline_state.xd.vel[self.torso_id]

        left_foot_pos = pipeline_state.x.pos[self.left_foot_id]
        right_foot_pos = pipeline_state.x.pos[self.right_foot_id]

        actuated_qpos = pipeline_state.qpos[7:30]
        if target_pose is None:
            target_pose = actuated_qpos
        joint_err_vec = target_pose - actuated_qpos  # Delta para a pose alvo do keyframe (23,)

        if last_action is None:
            last_action = jnp.zeros(self.sys.nu)

        obs = jnp.concatenate([
            pipeline_state.qpos,        # 30
            pipeline_state.qvel,        # 29
            torso_height,               # 1
            torso_up,                   # 1
            torso_angvel,               # 3
            torso_linvel,               # 3
            left_foot_pos,              # 3
            right_foot_pos,             # 3
            joint_err_vec,              # 23 (Guiamento humano direto!)
            last_action,                # nu (69)
        ])
        return jnp.nan_to_num(obs, nan=0.0, posinf=1e2, neginf=-1e2)


class GetUpFrontMjxEnv(BaseGetUpMjxEnv):
    """
    Levantamento HÍBRIDO de Frente (Front / Prone).
    Combina Poses Alvo do Keyframe YAML (Soft DeepMimic) + Recompensas Biomecânicas de Física.
    """
    DEFAULT_YAML_NAME = "get_up_front.yaml"

    def reset(self, rng: jax.Array) -> State:
        rng, pose_rng, noise_rng, vel_rng = jax.random.split(rng, 4)
        init_qpos = self.sys.init_q

        # 1. Pose deitado de bruços (Front)
        qpos_front = init_qpos.at[2].set(0.22)
        qpos_front = qpos_front.at[3].set(0.707)
        qpos_front = qpos_front.at[4].set(0.707)
        qpos_front = qpos_front.at[5].set(0.0)
        qpos_front = qpos_front.at[6].set(0.0)

        # 2. Pose intermediária baseada no Keyframe 2 do YAML (Ajoelhado/Crouch)
        qpos_crouch = init_qpos.at[2].set(0.42)
        qpos_crouch = qpos_crouch.at[3].set(0.924)
        qpos_crouch = qpos_crouch.at[4].set(0.383)
        qpos_crouch = qpos_crouch.at[5].set(0.0)
        qpos_crouch = qpos_crouch.at[6].set(0.0)
        if self._has_mimic and self.n_phases >= 3:
            kf2 = self.keyframe_targets[2]
            qpos_crouch = qpos_crouch.at[7:30].set(kf2)

        # 3. Pose quase de pé baseada no Keyframe final do YAML
        qpos_stand = init_qpos.at[2].set(0.60)
        qpos_stand = qpos_stand.at[3].set(1.0)
        qpos_stand = qpos_stand.at[4].set(0.0)
        qpos_stand = qpos_stand.at[5].set(0.0)
        qpos_stand = qpos_stand.at[6].set(0.0)
        if self._has_mimic and self.n_phases >= 4:
            kf_last = self.keyframe_targets[-1]
            qpos_stand = qpos_stand.at[7:30].set(kf_last)

        # Curriculum de Reset (60% deitado, 25% ajoelhado/keyframe, 15% de pé/keyframe)
        u = jax.random.uniform(pose_rng)
        qpos = jnp.where(u < 0.60, qpos_front, jnp.where(u < 0.85, qpos_crouch, qpos_stand))

        # Ruído nas juntas
        joint_noise = jax.random.uniform(noise_rng, shape=qpos.shape, minval=-0.10, maxval=0.10)
        root_mask = jnp.concatenate([jnp.zeros(7), jnp.ones(qpos.shape[0] - 7)])
        qpos = qpos + joint_noise * root_mask

        # Ruído em qvel
        qvel = jax.random.uniform(vel_rng, shape=(self.sys.nv,), minval=-0.2, maxval=0.2)
        root_vel_mask = jnp.concatenate([jnp.zeros(6), jnp.ones(self.sys.nv - 6)])
        qvel = qvel * root_vel_mask

        pipeline_state = self.pipeline_init(qpos, qvel)
        last_action = jnp.zeros(self.sys.nu)

        # Target pose inicial para obs
        target_pose = self.get_target_pose(jnp.array(0.0))
        obs = self._get_obs(pipeline_state, last_action, target_pose)

        metrics = {
            'reward': jnp.zeros(()),
            'reward_height': jnp.zeros(()),
            'reward_upright': jnp.zeros(()),
            'reward_velocity': jnp.zeros(()),
            'reward_standing': jnp.zeros(()),
            'reward_mimic': jnp.zeros(()),
            'joint_error': jnp.zeros(()),
            'torso_height': jnp.zeros(()),
            'torso_up': jnp.zeros(()),
        }

        info = {
            'last_action': last_action,
            'standing_count': jnp.zeros(()),
        }

        return State(pipeline_state, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        last_action = state.info.get('last_action', jnp.zeros(self.sys.nu))
        standing_count = state.info.get('standing_count', jnp.zeros(()))

        torso_pos = pipeline_state.x.pos[self.torso_id]
        torso_height = torso_pos[2]

        torso_rot = pipeline_state.x.rot[self.torso_id]
        torso_up = 1.0 - 2.0 * (torso_rot[1]**2 + torso_rot[2]**2)

        # 1. Progresso Vertical Relativo
        h_ground = 0.24
        h_target = 0.65
        h_rel = jnp.clip((torso_height - h_ground) / (h_target - h_ground), 0.0, 1.0)
        height_reward = (h_rel ** 2.0) * 120.0

        # 2. Keyframe Target Pose Suave (Soft DeepMimic RL)
        target_pose = self.get_target_pose(h_rel)
        actuated_qpos = pipeline_state.qpos[7:30]
        joint_err = jnp.mean(jnp.square(actuated_qpos - target_pose))

        # Recompensa Gaussiana Suave de Mimic (Nunca Zera! Fornece gradiente contínuo)
        mimic_reward = 80.0 * jnp.exp(-1.5 * joint_err)

        # Atualizar Observações Híbridas
        obs = self._get_obs(pipeline_state, action, target_pose)

        # 3. Orientação Upright
        upright_pos = jnp.clip((torso_up + 0.2) / 1.2, 0.0, 1.0)
        upright_reward = (upright_pos ** 2.0) * 60.0

        # 4. Velocidade Vertical Positiva
        torso_vz = pipeline_state.xd.vel[self.torso_id][2]
        velocity_reward = jnp.clip(torso_vz, 0.0, 2.0) * 40.0

        # 5. Sinal de Standing Contínuo
        standing_signal = h_rel * upright_pos
        standing_reward = (standing_signal ** 3.0) * 350.0

        # 6. Biomecânica de Bruços
        l_shoulder = pipeline_state.qpos[self._l_shoulder_pitch_idx]
        r_shoulder = pipeline_state.qpos[self._r_shoulder_pitch_idx]
        push_reward = jnp.where(torso_height < 0.40, jnp.clip(l_shoulder + r_shoulder, 0.0, 4.0) * 10.0, 0.0)

        l_knee = pipeline_state.qpos[self._l_knee_idx]
        r_knee = pipeline_state.qpos[self._r_knee_idx]
        tuck_reward = jnp.where(torso_height < 0.50, jnp.clip(l_knee + r_knee, 0.0, 4.0) * 10.0, 0.0)

        symmetry_penalty = -2.0 * (jnp.square(l_knee - r_knee) + jnp.square(l_shoulder - r_shoulder))

        # 7. Penalidades
        action_penalty = -0.001 * jnp.sum(jnp.square(action))
        action_smooth_penalty = -0.005 * jnp.sum(jnp.square(action - last_action))
        angvel_penalty = -0.01 * jnp.sum(jnp.square(pipeline_state.xd.ang[self.torso_id]))
        step_penalty = -0.2

        # 8. Bônus de Sucesso e Estabilidade
        is_standing = (torso_height > 0.58) & (torso_up > 0.80)
        standing_bonus = jnp.where(is_standing, 150.0, 0.0)

        standing_count = jnp.where(is_standing, standing_count + 1.0, 0.0)
        stability_bonus = jnp.where(standing_count >= 10, 250.0, 0.0)

        total_reward = (
            height_reward + upright_reward + velocity_reward + standing_reward + mimic_reward
            + push_reward + tuck_reward + symmetry_penalty
            + action_penalty + action_smooth_penalty + angvel_penalty + step_penalty
            + standing_bonus + stability_bonus
        )

        fell = (torso_height < 0.05) | jnp.isnan(torso_height)
        done = jnp.where(fell, 1.0, 0.0)

        metrics = {
            'reward': total_reward,
            'reward_height': height_reward,
            'reward_upright': upright_reward,
            'reward_velocity': velocity_reward,
            'reward_standing': standing_reward + standing_bonus,
            'reward_mimic': mimic_reward,
            'joint_error': joint_err,
            'torso_height': torso_height,
            'torso_up': torso_up,
        }

        info = state.info.copy()
        info['last_action'] = action
        info['standing_count'] = standing_count

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=total_reward, done=done, metrics=metrics, info=info
        )


class GetUpBackMjxEnv(BaseGetUpMjxEnv):
    """
    Levantamento HÍBRIDO de Costas (Back / Supine).
    Combina Poses Alvo do Keyframe YAML (Soft DeepMimic) + Recompensas Biomecânicas de Física.
    """
    DEFAULT_YAML_NAME = "get_up_back.yaml"

    def reset(self, rng: jax.Array) -> State:
        rng, pose_rng, noise_rng, vel_rng = jax.random.split(rng, 4)
        init_qpos = self.sys.init_q

        # 1. Pose deitado de costas (Back)
        qpos_back = init_qpos.at[2].set(0.22)
        qpos_back = qpos_back.at[3].set(0.707)
        qpos_back = qpos_back.at[4].set(0.0)
        qpos_back = qpos_back.at[5].set(-0.707)
        qpos_back = qpos_back.at[6].set(0.0)

        # 2. Pose intermediária baseada no Keyframe 2 do YAML
        qpos_crouch = init_qpos.at[2].set(0.42)
        qpos_crouch = qpos_crouch.at[3].set(0.924)
        qpos_crouch = qpos_crouch.at[4].set(-0.383)
        qpos_crouch = qpos_crouch.at[5].set(0.0)
        qpos_crouch = qpos_crouch.at[6].set(0.0)
        if self._has_mimic and self.n_phases >= 2:
            kf1 = self.keyframe_targets[1]
            qpos_crouch = qpos_crouch.at[7:30].set(kf1)

        # 3. Pose quase de pé baseada no Keyframe final do YAML
        qpos_stand = init_qpos.at[2].set(0.60)
        qpos_stand = qpos_stand.at[3].set(1.0)
        qpos_stand = qpos_stand.at[4].set(0.0)
        qpos_stand = qpos_stand.at[5].set(0.0)
        qpos_stand = qpos_stand.at[6].set(0.0)
        if self._has_mimic and self.n_phases >= 3:
            kf_last = self.keyframe_targets[-1]
            qpos_stand = qpos_stand.at[7:30].set(kf_last)

        # Curriculum de Reset (60% deitado, 25% ajoelhado/keyframe, 15% de pé/keyframe)
        u = jax.random.uniform(pose_rng)
        qpos = jnp.where(u < 0.60, qpos_back, jnp.where(u < 0.85, qpos_crouch, qpos_stand))

        # Ruído nas juntas
        joint_noise = jax.random.uniform(noise_rng, shape=qpos.shape, minval=-0.10, maxval=0.10)
        root_mask = jnp.concatenate([jnp.zeros(7), jnp.ones(qpos.shape[0] - 7)])
        qpos = qpos + joint_noise * root_mask

        # Ruído em qvel
        qvel = jax.random.uniform(vel_rng, shape=(self.sys.nv,), minval=-0.2, maxval=0.2)
        root_vel_mask = jnp.concatenate([jnp.zeros(6), jnp.ones(self.sys.nv - 6)])
        qvel = qvel * root_vel_mask

        pipeline_state = self.pipeline_init(qpos, qvel)
        last_action = jnp.zeros(self.sys.nu)

        # Target pose inicial para obs
        target_pose = self.get_target_pose(jnp.array(0.0))
        obs = self._get_obs(pipeline_state, last_action, target_pose)

        metrics = {
            'reward': jnp.zeros(()),
            'reward_height': jnp.zeros(()),
            'reward_upright': jnp.zeros(()),
            'reward_velocity': jnp.zeros(()),
            'reward_standing': jnp.zeros(()),
            'reward_mimic': jnp.zeros(()),
            'joint_error': jnp.zeros(()),
            'torso_height': jnp.zeros(()),
            'torso_up': jnp.zeros(()),
        }

        info = {
            'last_action': last_action,
            'standing_count': jnp.zeros(()),
        }

        return State(pipeline_state, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        last_action = state.info.get('last_action', jnp.zeros(self.sys.nu))
        standing_count = state.info.get('standing_count', jnp.zeros(()))

        torso_pos = pipeline_state.x.pos[self.torso_id]
        torso_height = torso_pos[2]

        torso_rot = pipeline_state.x.rot[self.torso_id]
        torso_up = 1.0 - 2.0 * (torso_rot[1]**2 + torso_rot[2]**2)

        # 1. Progresso Vertical Relativo
        h_ground = 0.24
        h_target = 0.65
        h_rel = jnp.clip((torso_height - h_ground) / (h_target - h_ground), 0.0, 1.0)
        height_reward = (h_rel ** 2.0) * 120.0

        # 2. Keyframe Target Pose Suave (Soft DeepMimic RL)
        target_pose = self.get_target_pose(h_rel)
        actuated_qpos = pipeline_state.qpos[7:30]
        joint_err = jnp.mean(jnp.square(actuated_qpos - target_pose))

        # Recompensa Gaussiana Suave de Mimic (Nunca Zera! Fornece gradiente contínuo)
        mimic_reward = 80.0 * jnp.exp(-1.5 * joint_err)

        # Atualizar Observações Híbridas
        obs = self._get_obs(pipeline_state, action, target_pose)

        # 3. Orientação Upright
        upright_pos = jnp.clip((torso_up + 0.2) / 1.2, 0.0, 1.0)
        upright_reward = (upright_pos ** 2.0) * 60.0

        # 4. Velocidade Vertical Positiva
        torso_vz = pipeline_state.xd.vel[self.torso_id][2]
        velocity_reward = jnp.clip(torso_vz, 0.0, 2.0) * 40.0

        # 5. Sinal de Standing Contínuo
        standing_signal = h_rel * upright_pos
        standing_reward = (standing_signal ** 3.0) * 350.0

        # 6. Biomecânica de Costas
        l_shoulder = pipeline_state.qpos[self._l_shoulder_pitch_idx]
        r_shoulder = pipeline_state.qpos[self._r_shoulder_pitch_idx]
        push_reward = jnp.where(torso_height < 0.40, jnp.clip(-l_shoulder - r_shoulder, 0.0, 4.0) * 10.0, 0.0)

        l_knee = pipeline_state.qpos[self._l_knee_idx]
        r_knee = pipeline_state.qpos[self._r_knee_idx]
        tuck_reward = jnp.where(torso_height < 0.40, jnp.clip(l_knee + r_knee, 0.0, 4.0) * 10.0, 0.0)

        l_hip = pipeline_state.qpos[self._l_hip_pitch_idx]
        r_hip = pipeline_state.qpos[self._r_hip_pitch_idx]
        hip_drive_reward = jnp.where(
            (torso_height > 0.30) & (torso_height < 0.55),
            jnp.clip(-l_hip - r_hip, 0.0, 3.0) * 10.0,
            0.0
        )

        symmetry_penalty = -2.0 * (jnp.square(l_knee - r_knee) + jnp.square(l_shoulder - r_shoulder))

        # 7. Penalidades
        action_penalty = -0.001 * jnp.sum(jnp.square(action))
        action_smooth_penalty = -0.005 * jnp.sum(jnp.square(action - last_action))
        angvel_penalty = -0.01 * jnp.sum(jnp.square(pipeline_state.xd.ang[self.torso_id]))
        step_penalty = -0.2

        # 8. Bônus de Sucesso e Estabilidade
        is_standing = (torso_height > 0.58) & (torso_up > 0.80)
        standing_bonus = jnp.where(is_standing, 150.0, 0.0)

        standing_count = jnp.where(is_standing, standing_count + 1.0, 0.0)
        stability_bonus = jnp.where(standing_count >= 10, 250.0, 0.0)

        total_reward = (
            height_reward + upright_reward + velocity_reward + standing_reward + mimic_reward
            + push_reward + tuck_reward + hip_drive_reward + symmetry_penalty
            + action_penalty + action_smooth_penalty + angvel_penalty + step_penalty
            + standing_bonus + stability_bonus
        )

        fell = (torso_height < 0.05) | jnp.isnan(torso_height)
        done = jnp.where(fell, 1.0, 0.0)

        metrics = {
            'reward': total_reward,
            'reward_height': height_reward,
            'reward_upright': upright_reward,
            'reward_velocity': velocity_reward,
            'reward_standing': standing_reward + standing_bonus,
            'reward_mimic': mimic_reward,
            'joint_error': joint_err,
            'torso_height': torso_height,
            'torso_up': torso_up,
        }

        info = state.info.copy()
        info['last_action'] = action
        info['standing_count'] = standing_count

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=total_reward, done=done, metrics=metrics, info=info
        )
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
        return [], []
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    keyframes = []
    deltas = []
    for kf in data.get('keyframes', []):
        motors = kf['motor_positions']
        keyframes.append({name: float(jnp.radians(val)) for name, val in motors.items()})
        deltas.append(float(kf.get('delta', 1.0)))
    return keyframes, deltas


class BaseGetUpMjxEnv(PipelineEnv):
    """
    Classe Base de Levantamento acelerada 100% em GPU via MuJoCo MJX + Brax.
    Suporta rastreamento de keyframes do YAML (DeepMimic RL) e vetorização em JAX.
    """
    DEFAULT_YAML_NAME = None

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

        # Mapeamento de corpos e juntas
        try:
            self.torso_id = self.mj_model.body('torso').id
        except KeyError:
            self.torso_id = 1

        self._l_knee_idx = self._get_qadr('Left_Knee_Pitch')
        self._r_knee_idx = self._get_qadr('Right_Knee_Pitch')
        self._l_hip_pitch_idx = self._get_qadr('Left_Hip_Pitch')
        self._r_hip_pitch_idx = self._get_qadr('Right_Hip_Pitch')
        self._waist_idx = self._get_qadr('Waist')

        self.left_joints_idx = jnp.array([
            self._get_qadr('Left_Shoulder_Pitch'),
            self._get_qadr('Left_Shoulder_Roll'),
            self._get_qadr('Left_Elbow_Pitch'),
            self._get_qadr('Left_Hip_Pitch'),
            self._get_qadr('Left_Hip_Roll'),
            self._get_qadr('Left_Knee_Pitch'),
            self._get_qadr('Left_Ankle_Pitch'),
        ])
        self.right_joints_idx = jnp.array([
            self._get_qadr('Right_Shoulder_Pitch'),
            self._get_qadr('Right_Shoulder_Roll'),
            self._get_qadr('Right_Elbow_Pitch'),
            self._get_qadr('Right_Hip_Pitch'),
            self._get_qadr('Right_Hip_Roll'),
            self._get_qadr('Right_Knee_Pitch'),
            self._get_qadr('Right_Ankle_Pitch'),
        ])

        self._pitch_indices = jnp.array([0, 2, 3, 5, 6])
        self._roll_indices = jnp.array([1, 4])
        self._leg_pitch_indices = jnp.array([0, 2, 3])

        # Carregar YAML 
        if keyframe_yaml is None and self.DEFAULT_YAML_NAME:
            base_dir = os.path.dirname(__file__)
            repo_root = os.path.abspath(os.path.join(base_dir, ".."))
            keyframe_yaml = os.path.join(repo_root, "resources", "skills", "keyframe", "get_up", self.DEFAULT_YAML_NAME)

        self.keyframes, self.keyframe_deltas = _load_keyframes_from_yaml(keyframe_yaml)
        self._has_mimic = len(self.keyframes) > 0
        self.n_phases = len(self.keyframes)

        # Construir matriz de alvos de keyframe em JAX (N_phases, 23)
        kf_targets = []
        for kf in self.keyframes:
            target_vec = {}
            for y_name, val in kf.items():
                if y_name in YAML_MAPPING:
                    for j_name, mult in YAML_MAPPING[y_name]:
                        target_vec[j_name] = float(val * mult)
            vec = [target_vec.get(j, 0.0) for j in JOINT_NAMES]
            kf_targets.append(vec)

        if len(kf_targets) > 0:
            self.keyframe_targets = jnp.array(kf_targets)
        else:
            self.keyframe_targets = jnp.zeros((1, 23))

        super().__init__(sys=sys, backend='mjx', n_frames=5, **kwargs)

    def _get_qadr(self, joint_name):
        try:
            return self.mj_model.joint(joint_name).qposadr[0]
        except KeyError:
            return 0

    def get_target_pose(self, progress: jax.Array) -> jax.Array:
        """Interpolador de pose alvo dos Keyframes do YAML em função do progresso [0, 1]."""
        if not self._has_mimic or self.n_phases <= 1:
            return self.keyframe_targets[0]
        idx_float = jnp.clip(progress, 0.0, 1.0) * (self.n_phases - 1)
        idx_low = jnp.floor(idx_float).astype(jnp.int32)
        idx_high = jnp.minimum(idx_low + 1, self.n_phases - 1)
        alpha = idx_float - idx_low
        return (1.0 - alpha) * self.keyframe_targets[idx_low] + alpha * self.keyframe_targets[idx_high]

    def _get_obs(self, pipeline_state, target_pose: jax.Array = None) -> jax.Array:
        if target_pose is None:
            target_pose = jnp.zeros(23)
        actuated_qpos = pipeline_state.qpos[7:30]
        joint_err = target_pose - actuated_qpos
        obs = jnp.concatenate([pipeline_state.qpos, pipeline_state.qvel, joint_err])
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        return jnp.clip(obs, -100.0, 100.0)


class GetUpFrontMjxEnv(BaseGetUpMjxEnv):
    
    DEFAULT_YAML_NAME = "get_up_front.yaml"

    def reset(self, rng: jax.Array) -> State:
        qpos = self.sys.init_q
        qpos = qpos.at[2].set(0.25)
        qpos = qpos.at[3].set(0.707)
        qpos = qpos.at[4].set(0.0)
        qpos = qpos.at[5].set(0.707)
        qpos = qpos.at[6].set(0.0)

        qvel = jnp.zeros(self.sys.nv)
        pipeline_state = self.pipeline_init(qpos, qvel)

        target_pose = self.get_target_pose(jnp.array(0.0))
        obs = self._get_obs(pipeline_state, target_pose)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = {
            'reward': jnp.zeros(()),
            'reward_height': jnp.zeros(()),
            'reward_standing': jnp.zeros(()),
            'reward_mimic': jnp.zeros(()),
            'joint_error': jnp.zeros(())
        }

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        torso_pos = pipeline_state.x.pos[self.torso_id]
        torso_height = torso_pos[2]
        
        torso_rot = pipeline_state.x.rot[self.torso_id]
        torso_up = 1.0 - 2.0 * (torso_rot[1]**2 + torso_rot[2]**2)

        # Progresso de altura [0.0, 1.0]
        height_progress = jnp.clip((torso_height - 0.15) / (0.65 - 0.15), 0.0, 1.0)
        height_reward = (height_progress ** 2) * 40.0 * jnp.maximum(0.0, torso_up)

        # Obter pose alvo interpolada dos keyframes do YAML (Mimic RL)
        target_pose = self.get_target_pose(height_progress)
        actuated_qpos = pipeline_state.qpos[7:30]
        joint_error = jnp.mean(jnp.square(actuated_qpos - target_pose))

        # Recompensa de acompanhamento de pose dos Keyframes
        mimic_reward = jnp.where(self._has_mimic, 40.0 * jnp.exp(-4.0 * joint_error), 0.0)

        obs = self._get_obs(pipeline_state, target_pose)

        waist_qpos = pipeline_state.qpos[self._waist_idx]
        waist_penalty = -5.0 * (waist_qpos ** 2)

        left_j = pipeline_state.qpos[self.left_joints_idx]
        right_j = pipeline_state.qpos[self.right_joints_idx]
        pitch_diff = left_j[self._pitch_indices] - right_j[self._pitch_indices]
        roll_sum = left_j[self._roll_indices] + right_j[self._roll_indices]
        symmetry_penalty = -2.5 * (jnp.mean(jnp.square(pitch_diff)) + jnp.mean(jnp.square(roll_sum)))

        action_penalty = -0.01 * jnp.sum(jnp.square(action))

        l_knee = pipeline_state.qpos[self._l_knee_idx]
        r_knee = pipeline_state.qpos[self._r_knee_idx]
        l_hip = pipeline_state.qpos[self._l_hip_pitch_idx]
        r_hip = pipeline_state.qpos[self._r_hip_pitch_idx]

        tuck_reward = jnp.where(
            torso_height < 0.45,
            (-l_knee - r_knee) * 2.0 + (l_hip + r_hip) * 2.0,
            0.0
        )

        step_penalty = -0.05

        standing_bonus = jnp.where(
            (torso_height > 0.60) & (torso_up > 0.85),
            30.0,
            0.0
        )

        total_reward = height_reward + mimic_reward + waist_penalty + symmetry_penalty + action_penalty + tuck_reward + step_penalty + standing_bonus
        total_reward = jnp.nan_to_num(total_reward, nan=0.0, posinf=100.0, neginf=-100.0)
        total_reward = jnp.clip(total_reward, -100.0, 100.0)

        done = jnp.where(jnp.isnan(torso_height) | (torso_height < 0.05), 1.0, 0.0)

        metrics = {
            'reward': total_reward,
            'reward_height': height_reward,
            'reward_standing': standing_bonus,
            'reward_mimic': mimic_reward,
            'joint_error': joint_error,
        }

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=total_reward, done=done, metrics=metrics
        )


class GetUpBackMjxEnv(BaseGetUpMjxEnv):
    
    DEFAULT_YAML_NAME = "get_up_back.yaml"

    def reset(self, rng: jax.Array) -> State:
        qpos = self.sys.init_q
        qpos = qpos.at[2].set(0.25)
        qpos = qpos.at[3].set(0.707)
        qpos = qpos.at[4].set(0.0)
        qpos = qpos.at[5].set(-0.707)
        qpos = qpos.at[6].set(0.0)

        qvel = jnp.zeros(self.sys.nv)
        pipeline_state = self.pipeline_init(qpos, qvel)

        target_pose = self.get_target_pose(jnp.array(0.0))
        obs = self._get_obs(pipeline_state, target_pose)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = {
            'reward': jnp.zeros(()),
            'reward_height': jnp.zeros(()),
            'reward_standing': jnp.zeros(()),
            'reward_mimic': jnp.zeros(()),
            'joint_error': jnp.zeros(())
        }

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        torso_pos = pipeline_state.x.pos[self.torso_id]
        torso_height = torso_pos[2]

        torso_rot = pipeline_state.x.rot[self.torso_id]
        torso_up = 1.0 - 2.0 * (torso_rot[1]**2 + torso_rot[2]**2)

        # Progresso de altura [0.0, 1.0]
        height_progress = jnp.clip((torso_height - 0.15) / (0.65 - 0.15), 0.0, 1.0)
        height_reward = (height_progress ** 2) * 40.0 * jnp.maximum(0.0, torso_up)

        # Obter pose alvo interpolada dos keyframes do YAML (Mimic RL)
        target_pose = self.get_target_pose(height_progress)
        actuated_qpos = pipeline_state.qpos[7:30]
        joint_error = jnp.mean(jnp.square(actuated_qpos - target_pose))

        # Recompensa de acompanhamento de pose dos Keyframes
        mimic_reward = jnp.where(self._has_mimic, 40.0 * jnp.exp(-4.0 * joint_error), 0.0)

        obs = self._get_obs(pipeline_state, target_pose)

        waist_qpos = pipeline_state.qpos[self._waist_idx]
        waist_penalty = -5.0 * (waist_qpos ** 2)

        l_knee = pipeline_state.qpos[self._l_knee_idx]
        r_knee = pipeline_state.qpos[self._r_knee_idx]
        tuck_reward = jnp.where(torso_height < 0.45, (jnp.abs(l_knee) + jnp.abs(r_knee)) * 2.0, 0.0)

        left_legs = pipeline_state.qpos[self.left_joints_idx[3:]]
        right_legs = pipeline_state.qpos[self.right_joints_idx[3:]]
        leg_pitch_diff = left_legs[self._leg_pitch_indices] - right_legs[self._leg_pitch_indices]
        leg_roll_sum = left_legs[1] + right_legs[1]
        leg_symmetry_penalty = -2.5 * (jnp.mean(jnp.square(leg_pitch_diff)) + jnp.square(leg_roll_sum))

        action_penalty = -0.01 * jnp.sum(jnp.square(action))
        step_penalty = -0.05

        standing_bonus = jnp.where(
            (torso_height > 0.60) & (torso_up > 0.85),
            30.0,
            0.0
        )

        total_reward = height_reward + mimic_reward + waist_penalty + tuck_reward + leg_symmetry_penalty + action_penalty + step_penalty + standing_bonus
        total_reward = jnp.nan_to_num(total_reward, nan=0.0, posinf=100.0, neginf=-100.0)
        total_reward = jnp.clip(total_reward, -100.0, 100.0)

        done = jnp.where(jnp.isnan(torso_height) | (torso_height < 0.05), 1.0, 0.0)

        metrics = {
            'reward': total_reward,
            'reward_height': height_reward,
            'reward_standing': standing_bonus,
            'reward_mimic': mimic_reward,
            'joint_error': joint_error,
        }

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=total_reward, done=done, metrics=metrics
        )
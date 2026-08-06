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
    Suporta mapeamento de juntas, YAML de keyframes (Mimic), e vetorização em JAX.
    """
    DEFAULT_YAML_NAME = None

    def __init__(self, model_path=None, keyframe_yaml=None, **kwargs):
        if not HAS_BRAX:
            raise ImportError("Brax e MJX são necessários. Instale com: pip install brax mujoco-mjx")

        if model_path is None:
            home = os.path.expanduser("~")
            model_path = os.path.join(home, "rcssservermj/src/rcsssmj/resources/robots/T1/robot.xml")

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

        # mapeamento de corpos e juntas
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

        # carregar YAML 
        if keyframe_yaml is None and self.DEFAULT_YAML_NAME:
            base_dir = os.path.dirname(__file__)
            possible_paths = [
                os.path.join(base_dir, "..", "mujococodebase", "skills", "keyframe", "get_up", self.DEFAULT_YAML_NAME),
                os.path.join(base_dir, "keyframe", "get_up", self.DEFAULT_YAML_NAME),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    keyframe_yaml = p
                    break

        self.keyframes, self.keyframe_deltas = _load_keyframes_from_yaml(keyframe_yaml)
        self._has_mimic = len(self.keyframes) > 0
        self.n_phases = len(self.keyframes)

        super().__init__(sys=sys, backend='mjx', n_frames=5, **kwargs)

    def _get_qadr(self, joint_name):
        try:
            return self.mj_model.joint(joint_name).qposadr[0]
        except KeyError:
            return 0

    def _get_obs(self, pipeline_state) -> jax.Array:
        return jnp.concatenate([pipeline_state.qpos, pipeline_state.qvel])


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

        obs = self._get_obs(pipeline_state)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = {'reward_height': jnp.zeros(()), 'reward_standing': jnp.zeros(())}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)

        # utiliza torso_id dinâmico para pegar a posição z do torso 
        torso_pos = pipeline_state.x.pos[self.torso_id]
        torso_height = torso_pos[2]
        
        torso_rot = pipeline_state.x.rot[self.torso_id]
        torso_up = 1.0 - 2.0 * (torso_rot[1]**2 + torso_rot[2]**2)

        # recompensa de altura e postura
        height_progress = jnp.clip((torso_height - 0.30) / (0.65 - 0.30), 0.0, 1.0)
        height_reward = (height_progress ** 3) * 50.0 * jnp.maximum(0.0, torso_up)

        # recompensa de Mimic (se YAML foi carregado)
        mimic_reward = jnp.where(self._has_mimic, 40.0 * jnp.exp(-3.0 * (1.0 - jnp.maximum(0.0, torso_up))), 0.0)

       
        waist_qpos = pipeline_state.qpos[self._waist_idx]
        waist_penalty = -10.0 * (waist_qpos ** 2)

        left_j = pipeline_state.qpos[self.left_joints_idx]
        right_j = pipeline_state.qpos[self.right_joints_idx]
        symmetry_penalty = -5.0 * jnp.mean(jnp.square(left_j - right_j))

        action_penalty = -0.05 * jnp.sum(jnp.square(action))

        l_knee = pipeline_state.qpos[self._l_knee_idx]
        r_knee = pipeline_state.qpos[self._r_knee_idx]
        l_hip = pipeline_state.qpos[self._l_hip_pitch_idx]
        r_hip = pipeline_state.qpos[self._r_hip_pitch_idx]

        tuck_reward = jnp.where(
            torso_height < 0.50,
            (-l_knee - r_knee) * 5.0 + (l_hip + r_hip) * 5.0,
            0.0
        )

        step_penalty = -0.5

        standing_bonus = jnp.where(
            (torso_height > 0.65) & (torso_up > 0.9),
            2005.0,
            0.0
        )

        total_reward = height_reward + mimic_reward + waist_penalty + symmetry_penalty + action_penalty + tuck_reward + step_penalty + standing_bonus
        done = jnp.where(torso_height < 0.05, 1.0, 0.0)

        metrics = {
            'reward_height': height_reward,
            'reward_standing': standing_bonus
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

        obs = self._get_obs(pipeline_state)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = {'reward_height': jnp.zeros(()), 'reward_standing': jnp.zeros(())}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)

        torso_pos = pipeline_state.x.pos[self.torso_id]
        torso_height = torso_pos[2]

        torso_rot = pipeline_state.x.rot[self.torso_id]
        torso_up = 1.0 - 2.0 * (torso_rot[1]**2 + torso_rot[2]**2)

        height_progress = jnp.clip((torso_height - 0.30) / (0.65 - 0.30), 0.0, 1.0)
        height_reward = (height_progress ** 3) * 50.0 * jnp.maximum(0.0, torso_up)

        mimic_reward = jnp.where(self._has_mimic, 40.0 * jnp.exp(-3.0 * (1.0 - jnp.maximum(0.0, torso_up))), 0.0)

        waist_qpos = pipeline_state.qpos[self._waist_idx]
        waist_penalty = -10.0 * (waist_qpos ** 2)

        l_knee = pipeline_state.qpos[self._l_knee_idx]
        r_knee = pipeline_state.qpos[self._r_knee_idx]
        tuck_reward = jnp.where(torso_height < 0.45, (jnp.abs(l_knee) + jnp.abs(r_knee)) * 5.0, 0.0)

        left_legs = pipeline_state.qpos[self.left_joints_idx[3:]]
        right_legs = pipeline_state.qpos[self.right_joints_idx[3:]]
        leg_symmetry_penalty = -5.0 * jnp.mean(jnp.square(left_legs - right_legs))

        action_penalty = -0.05 * jnp.sum(jnp.square(action))
        step_penalty = -1.0

        standing_bonus = jnp.where(
            (torso_height > 0.65) & (torso_up > 0.9),
            510.0,
            0.0
        )

        total_reward = height_reward + mimic_reward + waist_penalty + tuck_reward + leg_symmetry_penalty + action_penalty + step_penalty + standing_bonus
        done = jnp.where(torso_height < 0.05, 1.0, 0.0)

        metrics = {
            'reward_height': height_reward,
            'reward_standing': standing_bonus
        }

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=total_reward, done=done, metrics=metrics
        )
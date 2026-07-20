from base_env import BaseRobotEnv
import numpy as np
import mujoco
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R


class WalkEnv(BaseRobotEnv):
    """Ambiente de treinamento para locomoção bípede do robô T1."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # Controle PD (mesmos valores do walk.py runtime)
    PD_KP = 25.0
    PD_KD = 0.6
    N_SUBSTEPS = 10

    # Referências de altura
    STANDING_HEIGHT = 0.65
    FALL_THRESHOLD = 0.30

    def __init__(self, model_path=None, render_mode=None):
        super().__init__(model_path)
        self.render_mode = render_mode
        self.renderer = None

        # Observation space: 69 (joints interleaved) + 3 (gyro) + 3 (vel cmd) + 3 (gravity) = 78
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(78,), dtype=np.float32
        )

    def _get_obs(self):
        joint_qpos = self.data.qpos[7:]   # Posições das juntas (exclui free joint)
        joint_qvel = self.data.qvel[6:]   # Velocidades das juntas (exclui free joint)

        qpos_norm = (joint_qpos * self.TRAIN_SIM_FLIP - self.JOINT_NOMINAL_POSITION) / 4.6
        qvel_norm = joint_qvel / 110.0 * self.TRAIN_SIM_FLIP
        prev_action_norm = self.prev_action / 10.0

        # Interleave: [q0, v0, a0, q1, v1, a1, ..., q22, v22, a22] → 69 valores
        qpos_qvel_prev = np.vstack([qpos_norm, qvel_norm, prev_action_norm]).T.flatten()

        # Velocidade angular do giroscópio (rad/s do MuJoCo)
        gyro = self.data.qvel[3:6]
        ang_vel = np.clip(gyro / 50.0, -1.0, 1.0)

        # Gravidade projetada no frame do robô
        quat_mj = self.data.qpos[3:7]  # MuJoCo: [qw, qx, qy, qz]
        quat_scipy = np.array([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]])  # scipy: [x, y, z, w]
        orientation_inv = R.from_quat(quat_scipy).inv()
        projected_gravity = orientation_inv.apply(np.array([0.0, 0.0, -1.0]))

        obs = np.concatenate([
            qpos_qvel_prev,        # 69
            ang_vel,                # 3
            self.velocity_command,  # 3
            projected_gravity,      # 3
        ]).astype(np.float32)

        return np.clip(obs, -10.0, 10.0)

    def step(self, action):
        self.current_step += 1

        # Converter ação [-1,1] → posição alvo das juntas
        target_positions = self.JOINT_NOMINAL_POSITION + self.SCALING_FACTOR * action
        target_positions *= self.TRAIN_SIM_FLIP

        # Aplicar controle PD
        joint_qpos = self.data.qpos[7:]
        joint_qvel = self.data.qvel[6:]

        ctrl = np.zeros(self.model.nu)
        for i, motor_idx in enumerate(self.motor_indices):
            position_error = target_positions[i] - joint_qpos[i]
            velocity_error = -joint_qvel[i]
            torque = self.PD_KP * position_error + self.PD_KD * velocity_error
            ctrl[motor_idx] = np.clip(torque, -self.torque_limits[i], self.torque_limits[i])

        self.data.ctrl[:] = ctrl

        # Frame skip (decimation)
        for _ in range(self.N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self.compute_reward(action)

        # Terminação: robô caiu
        torso_height = self.data.body('torso').xpos[2]
        terminated = torso_height < self.FALL_THRESHOLD
        truncated = self.current_step >= self.max_steps

        self.prev_action = action.copy()

        return obs, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self.prev_action = np.zeros(self.n_actions)

        # Robô começa em pé
        self.data.qpos[2] = self.STANDING_HEIGHT
        self.data.qpos[3:7] = [1, 0, 0, 0]  # Quaternion identidade

        # Setar juntas na posição nominal
        self.data.qpos[7:] = self.JOINT_NOMINAL_POSITION

        mujoco.mj_forward(self.model, self.data)

        # Randomizar o comando de velocidade a cada episódio
        self.velocity_command = np.array([
            self.np_random.uniform(-0.5, 0.5),     # vx
            self.np_random.uniform(-0.25, 0.25),   # vy
            self.np_random.uniform(-0.25, 0.25),   # yaw_rate
        ])

        return self._get_obs(), {}

    def compute_reward(self, action):
        reward = 1.0  # Bônus de sobrevivência (Survival Bonus)

        # === VELOCIDADE (Componente dominante) ===
        # Converter velocidade global para frame local do robô
        global_vel = self.data.qvel[:2]  # vx, vy globais

        # Extrair yaw do quaternion
        quat_mj = self.data.qpos[3:7]
        quat_scipy = np.array([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]])
        yaw = R.from_quat(quat_scipy).as_euler('xyz')[2]

        # Rotacionar para frame local
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        local_vx = cos_yaw * global_vel[0] + sin_yaw * global_vel[1]
        local_vy = -sin_yaw * global_vel[0] + cos_yaw * global_vel[1]

        # Velocidade angular real (yaw rate)
        actual_yaw_rate = self.data.qvel[5]

        # 1. Tracking de velocidade linear (exponencial — nunca fica negativa)
        lin_vel_error = np.square(local_vx - self.velocity_command[0]) \
                      + np.square(local_vy - self.velocity_command[1])
        reward += np.exp(-lin_vel_error / 0.25) * 2.0

        # 2. Tracking de velocidade angular
        ang_vel_error = np.square(actual_yaw_rate - self.velocity_command[2])
        reward += np.exp(-ang_vel_error / 0.25) * 1.0

        # === POSTURA ===

        # 3. Orientação do torso (ficar em pé)
        torso_up = self.data.body('torso').xmat[8]
        reward += max(0, torso_up) * 0.5

        # 4. Altura do torso (bônus por manter altura estável)
        torso_height = self.data.body('torso').xpos[2]
        height_error = np.square(torso_height - self.STANDING_HEIGHT)
        reward += np.exp(-height_error / 0.01) * 0.3

        # === PENALIDADES ===

        # 5. Suavidade da ação
        reward -= 0.005 * np.sum(np.square(action))

        # 6. Suavidade temporal
        if self.prev_action is not None:
            reward -= 0.01 * np.sum(np.square(action - self.prev_action))

        # 7. Consumo de energia
        torques = self.data.ctrl[self.motor_indices]
        reward -= 0.0001 * np.sum(np.square(torques))

        # 8. Velocidade das juntas
        joint_vel = self.data.qvel[6:]
        reward -= 0.0001 * np.sum(np.square(joint_vel))

        # === CONTATOS ===
        contacts = self._get_contact_info()

        # 9. Penalidade por contato indevido (mãos, antebraços, canelas)
        undesired_contact = contacts[0] + contacts[1] + contacts[4] + contacts[5] + contacts[6] + contacts[7]
        reward -= 0.5 * undesired_contact

        # 10. Bônus por contato dos pés
        feet_contact = contacts[2] + contacts[3]
        reward += 0.2 * feet_contact

        # === TERMINAÇÃO PRECOCE ===
        # O fato do episódio terminar (truncando o survival bonus futuro) já é penalidade suficiente
        # Removido a penalidade explícita de -50.0 para não criar abismos de gradiente.

        return reward
    
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
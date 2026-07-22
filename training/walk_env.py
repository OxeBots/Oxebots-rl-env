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

    # Parâmetros de marcha
    GAIT_FREQUENCY = 2.5       # Hz — frequência para corrida bípede (futebol)
    DESIRED_AIR_TIME = 0.15    # Segundos — compatível com corrida rápida

    def __init__(self, model_path=None, render_mode=None):
        super().__init__(model_path)
        self.render_mode = render_mode
        self.renderer = None

        # Observation space: 69 (joints interleaved) + 3 (gyro) + 3 (vel cmd) + 3 (gravity) = 78
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(78,), dtype=np.float32
        )

        # Timestep da simulação
        self.sim_dt = self.model.opt.timestep * self.N_SUBSTEPS

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

        # Atualizar fase da marcha
        self.gait_phase = (self.gait_phase + 2 * np.pi * self.GAIT_FREQUENCY * self.sim_dt) % (2 * np.pi)

        # Atualizar estado de contato dos pés e air-time
        contacts = self._get_contact_info()
        current_foot_contacts = np.array([contacts[2], contacts[3]])  # [pé_esq, pé_dir]

        for foot_idx in range(2):
            if current_foot_contacts[foot_idx] > 0:
                # Pé acabou de tocar o chão — registrar air-time
                if self.last_foot_contacts[foot_idx] == 0:
                    self.feet_air_time[foot_idx] = self.feet_air_time_counter[foot_idx]
                    self.feet_air_time_counter[foot_idx] = 0.0
            else:
                # Pé no ar — incrementar contador
                self.feet_air_time_counter[foot_idx] += self.sim_dt

        self.last_foot_contacts = current_foot_contacts.copy()

        obs = self._get_obs()
        reward = self.compute_reward(action, contacts)

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

        # Estado de marcha
        self.gait_phase = 0.0
        self.last_foot_contacts = np.zeros(2)
        self.feet_air_time = np.zeros(2)
        self.feet_air_time_counter = np.zeros(2)

        # Robô começa em pé
        self.data.qpos[2] = self.STANDING_HEIGHT
        self.data.qpos[3:7] = [1, 0, 0, 0]  # Quaternion identidade

        # Setar juntas na posição nominal
        self.data.qpos[7:] = self.JOINT_NOMINAL_POSITION

        mujoco.mj_forward(self.model, self.data)

        # Curriculum Learning: range de velocidade cresce com o progresso do treinamento
        progress = min(1.0, self.total_training_steps / 15_000_000)
        max_vx = 0.15 + 0.85 * progress     # 0.15 → 1.0  (velocidade de jogo)
        max_vy = 0.1 + 0.4 * progress       # 0.1  → 0.5  (lateral rápida)
        max_yaw = 0.1 + 0.9 * progress      # 0.1  → 1.0  (giros rápidos)

        self.velocity_command = np.array([
            self.np_random.uniform(-max_vx, max_vx),
            self.np_random.uniform(-max_vy, max_vy),
            self.np_random.uniform(-max_yaw, max_yaw),
        ])

        return self._get_obs(), {}

    def compute_reward(self, action, contacts):
        reward = 0.0

        # === 1. SURVIVAL BONUS ===
        # Incentivo constante para ficar vivo — cair corta esse ganho futuro
        reward += 0.5

        # === 2. VELOCITY TRACKING (Componente dominante) ===
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

        # 2a. Tracking de velocidade linear (σ=0.25 — tracking preciso para futebol)
        lin_vel_error = np.square(local_vx - self.velocity_command[0]) \
                      + np.square(local_vy - self.velocity_command[1])
        reward += np.exp(-lin_vel_error / 0.25) * 2.0

        # 2b. Tracking de velocidade angular
        ang_vel_error = np.square(actual_yaw_rate - self.velocity_command[2])
        reward += np.exp(-ang_vel_error / 0.25) * 1.0

        # === 3. POSTURA ===

        # 3a. Orientação do torso (ficar em pé)
        torso_up = self.data.body('torso').xmat[8]
        reward += max(0, torso_up) * 0.3

        # 3b. Altura do torso (bônus por manter altura estável)
        torso_height = self.data.body('torso').xpos[2]
        height_error = np.square(torso_height - self.STANDING_HEIGHT)
        reward += np.exp(-height_error / 0.02) * 0.5

        # === 4. MARCHA (Gait incentives) ===

        # 4a. Foot air-time: recompensar pés que ficam no ar pelo tempo desejado
        for foot_idx in range(2):
            if self.last_foot_contacts[foot_idx] > 0 and self.feet_air_time[foot_idx] > 0:
                # Pé acabou de tocar — recompensar se ficou no ar ~DESIRED_AIR_TIME
                air_time_error = np.square(self.feet_air_time[foot_idx] - self.DESIRED_AIR_TIME)
                reward += np.exp(-air_time_error / 0.04) * 0.5

        # 4b. Gait phase: incentivar alternância de pernas
        # Fase 0-π: pé esquerdo deveria estar no ar; π-2π: pé direito no ar
        left_phase_target = 1.0 if self.gait_phase < np.pi else 0.0   # 1=ar, 0=chão
        right_phase_target = 1.0 if self.gait_phase >= np.pi else 0.0

        left_contact = self.last_foot_contacts[0]
        right_contact = self.last_foot_contacts[1]

        # Recompensar quando o estado de contato alinha com a fase desejada
        left_match = 1.0 if (left_phase_target == 1.0 and left_contact == 0) or \
                           (left_phase_target == 0.0 and left_contact > 0) else 0.0
        right_match = 1.0 if (right_phase_target == 1.0 and right_contact == 0) or \
                            (right_phase_target == 0.0 and right_contact > 0) else 0.0
        reward += (left_match + right_match) * 0.25

        # 4c. Bônus base por contato dos pés (pelo menos um pé no chão)
        feet_contact = contacts[2] + contacts[3]
        reward += 0.1 * feet_contact

        # === 5. PENALIDADES (suaves — não sufocar exploração) ===

        # 5a. Suavidade da ação (L2)
        reward -= 0.001 * np.sum(np.square(action))

        # 5b. Suavidade temporal
        if self.prev_action is not None:
            reward -= 0.002 * np.sum(np.square(action - self.prev_action))

        # 5c. Consumo de energia (torques)
        torques = self.data.ctrl[self.motor_indices]
        reward -= 0.00005 * np.sum(np.square(torques))

        # 5d. Velocidade das juntas
        joint_vel = self.data.qvel[6:]
        reward -= 0.00005 * np.sum(np.square(joint_vel))

        # 5e. Penalidade por contato indevido (mãos, antebraços, canelas)
        undesired_contact = contacts[0] + contacts[1] + contacts[4] + contacts[5] + contacts[6] + contacts[7]
        reward -= 0.3 * undesired_contact

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
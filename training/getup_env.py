import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
import os
import re

class GetUpEnv(gym.Env):
    """Classe Base para o ambiente de levantar do robô T1."""
    def __init__(self, model_path=None):
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

class GetUpBackEnv(GetUpEnv):
    """Levantamento de Costas - Robô inicia com as COSTAS no chão (Barriga para CIMA)."""
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.data.qpos[3:7] = [0.707, 0, -0.707, 0] 
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), info

    def compute_reward(self, action, obs):
        contact_info = obs[-8:]
        torso_height = self.data.body('torso').xpos[2]
        torso_up = self.data.body('torso').xmat[8]
        # Pegar a velocidade Z do tronco ajuda a recompensar o *movimento* de subida
        torso_z_vel = self.data.qvel[2] 
        
        reward = 0.0
        
        # 1. Altura e Postura: Recompensa exponencial
        # Em vez de dar pontos lineares, fazemos a recompensa explodir só quando ele realmente chega perto de ficar em pé.
        if torso_height > 0.30:
            # Usando uma curva exponencial para desencorajar ficar no meio do caminho
            height_progress = (torso_height - 0.30) / (0.65 - 0.30) # Normalizado de 0 a 1
            reward += (height_progress ** 3) * 100.0 * max(0, torso_up)

        # 2. Shaping: Pé Plano
        l_foot_up = self.data.body('left_foot_link').xmat[8]
        r_foot_up = self.data.body('right_foot_link').xmat[8]
        if contact_info[2] > 0: reward += l_foot_up * 5.0 # Reduzi um pouco para não ser exploitável
        if contact_info[3] > 0: reward += r_foot_up * 5.0

        # 3. Impulso de braços APENAS se estiver subindo
        # Agora ele só ganha os 30 pontos se o braço estiver no chão E ele estiver ganhando altura (velocidade positiva)
        if torso_height < 0.50 and (contact_info[0] > 0 or contact_info[1] > 0):
            if torso_z_vel > 0.1: # Exige velocidade de subida
                reward += 10.0

        # 4. Puxar os pés para perto do centro de massa (Evitar espacate da foto)
        # Opcional, mas muito útil: penalizar se os pés estiverem muito longe do tronco no eixo X/Y
        torso_xy = self.data.body('torso').xpos[:2]
        l_foot_xy = self.data.body('left_foot_link').xpos[:2]
        r_foot_xy = self.data.body('right_foot_link').xpos[:2]
        dist_l_foot = np.linalg.norm(torso_xy - l_foot_xy)
        dist_r_foot = np.linalg.norm(torso_xy - r_foot_xy)
        # Penaliza levemente se os pés estiverem muito espalhados
        reward -= (dist_l_foot + dist_r_foot) * 2.0 

        # Penalidades de ação
        reward -= 0.05 * np.sum(np.square(action)) # Aumentei um pouco para evitar movimentos bruscos
        if hasattr(self, 'prev_action'):
            reward -= 0.05 * np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()

        # Condição de Vitória MANTIDA
        if torso_height > 0.65 and torso_up > 0.9:
            reward += 2000.0 
            
        # Penalidade de Sobrevivência AUMENTADA
        # A penalidade tem que ser maior que qualquer recompensa "fácil" que ele consiga farmar parado.
        reward -= 2.0 
        
        return reward

class GetUpFrontEnv(GetUpEnv):
    """Levantamento de Frente - Robô inicia de FRENTE (Barriga para BAIXO)."""
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.data.qpos[3:7] = [0.707, 0, 0.707, 0] 
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), info

    def compute_reward(self, action, obs):
        contact_info = obs[-8:]
        torso_height = self.data.body('torso').xpos[2]
        torso_up = self.data.body('torso').xmat[8]
        
        reward = 0.0
        if torso_height > 0.30:
            reward += (torso_height - 0.30) * 500.0 * max(0, torso_up)

        # 1. SHAPING: Pé Plano
        l_foot_up = self.data.body('left_foot_link').xmat[8]
        r_foot_up = self.data.body('right_foot_link').xmat[8]
        if contact_info[2] > 0: reward += l_foot_up * 15.0
        if contact_info[3] > 0: reward += r_foot_up * 15.0

        # O "Tuck" (Quadril e Joelho)
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

        # Estabilização Pose Neutra
        if torso_height > 0.60 and torso_up > 0.8:
            current_joints = self.data.qpos[7:]
            target_joints = np.zeros_like(current_joints)
            target_joints[self.l_shoulder_roll_idx - 7] = -1.57
            target_joints[self.r_shoulder_roll_idx - 7] = 1.57
            error = np.sum(np.square(current_joints - target_joints))
            reward += 150.0 * np.exp(-error)

        reward -= 0.01 * np.sum(np.square(action))
        if hasattr(self, 'prev_action'):
            reward -= 0.05 * np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()

        if torso_height > 0.65 and torso_up > 0.9:
            reward += 2000.0 # Aumentado
        reward -= 0.5
        return reward

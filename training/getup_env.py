import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
import os
import re

class GetUpEnv(gym.Env):
    def __init__(self, model_path="/home/matheus/rcssservermj/src/rcsssmj/resources/robots/T1/robot.xml"):
        super().__init__()
        
        # 1. Carregar o XML original
        with open(model_path, 'r') as f:
            xml_string = f.read()
        
        # 2. Corrigir o caminho das meshes para ser absoluto
        mesh_dir = os.path.join(os.path.dirname(model_path), "meshes")
        xml_string = re.sub(r'meshdir="[^"]*"', f'meshdir="{mesh_dir}"', xml_string)
        
        # 3. Adicionar o chão (plane) se não existir
        if "plane" not in xml_string:
            ground_xml = """
    <geom name="floor" type="plane" size="10 10 0.05" rgba="0.8 0.9 0.8 1" pos="0 0 0"/>
    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
"""
            # Inserir logo após a tag worldbody
            xml_string = re.sub(r'<worldbody>', f'<worldbody>{ground_xml}', xml_string)

        # 4. Carregar o modelo no MuJoCo
        try:
            self.model = mujoco.MjModel.from_xml_string(xml_string)
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            self.model = mujoco.MjModel.from_xml_path(model_path)

        self.data = mujoco.MjData(self.model)
        
        # 5. Mapeamento de torques reais
        self.torque_limits = []
        self.motor_indices = []
        
        class_torques = {
            "he": 7,
            "lae": 30,
            "rae": 30,
            "te": 30,
            "lle1": 45,
            "lle2": 30,
            "lle3": 30,
            "lle4": 60,
            "lle5": 20,
            "lle6": 15,
            "rle1": 45,
            "rle2": 30,
            "rle3": 30,
            "rle4": 60,
            "rle5": 20,
            "rle6": 15
        }

        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            if "tau" in name:
                self.motor_indices.append(i)
                limit = 18 # Default
                for prefix, torque in class_torques.items():
                    if name.startswith(prefix):
                        limit = torque
                        break
                self.torque_limits.append(limit)
        
        self.torque_limits = np.array(self.torque_limits)
        self.n_actions = len(self.motor_indices)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)
        
        # IDs dos corpos para sensores de contato
        self.hand_ids = [self.model.body('left_hand_link').id, self.model.body('right_hand_link').id]
        self.foot_ids = [self.model.body('left_foot_link').id, self.model.body('right_foot_link').id]
        self.forearm_ids = [self.model.body('AL3').id, self.model.body('AR3').id]
        self.shank_ids = [self.model.body('Shank_Left').id, self.model.body('Shank_Right').id]
        
        # Índices dos qpos para Ankle Roll
        self.l_ankle_roll_idx = self.model.joint('Left_Ankle_Roll').qposadr[0]
        self.r_ankle_roll_idx = self.model.joint('Right_Ankle_Roll').qposadr[0]
        
        # Obs: Sensores + Qpos + Qvel + Contatos (8 booleanos/floats)
        # Contatos: [L_Hand, R_Hand, L_Foot, R_Foot, L_Forearm, R_Forearm, L_Shank, R_Shank]
        obs_shape = self.model.nsensordata + self.model.nq + self.model.nv + 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        
        self.max_steps = 500 
        self.current_step = 0

    def _get_contact_info(self):
        # [L_Hand, R_Hand, L_Foot, R_Foot, L_Forearm, R_Forearm, L_Shank, R_Shank]
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
        
        # Aplicar força escalada pelo limite individual de cada junta
        ctrl = np.zeros(self.model.nu)
        ctrl[self.motor_indices] = action * self.torque_limits
        self.data.ctrl[:] = ctrl
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        contact_info = obs[-8:]
        
        # --- RECOMPENSA ---
        torso_height = self.data.body('torso').xpos[2]
        torso_up = self.data.body('torso').xmat[8] # 1.0 é em pé
        
        reward = 0.0
        
        # 1. Recompensa de Altura MULTIPLICADA pela Verticalidade
        if torso_height > 0.40:
            h_reward = (torso_height - 0.40) * 300.0
            v_mult = max(0, torso_up) 
            reward += h_reward * v_mult
        
        # 2. Bônus de Verticalidade pura
        if torso_height > 0.45:
            reward += max(0, torso_up) * 50.0
            
        # 3. Penalidades de Controle (Energia, Instabilidade e Suavidade)
        reward -= 0.01 * np.sum(np.square(action))
        torso_angvel = self.data.sensor('torso_gyro').data
        reward -= 0.05 * np.linalg.norm(torso_angvel)
        
        # NOVO: Penalidade de Suavidade (Evita tremedeira)
        if hasattr(self, 'prev_action'):
            reward -= 0.05 * np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()
        
        # NOVO: Penalidade de Limite de Junta (Evita travar nos limites)
        # qpos[7:] são as juntas articuladas
        qpos_joints = self.data.qpos[7:]
        for i in range(len(qpos_joints)):
            jnt_min, jnt_max = self.model.jnt_range[i]
            # Se estiver a menos de 5% do limite, penaliza
            margin = (jnt_max - jnt_min) * 0.05
            if qpos_joints[i] < jnt_min + margin or qpos_joints[i] > jnt_max - margin:
                reward -= 1.0

        # --- BÔNUS CONDICIONAIS ---
        if torso_height > 0.45:
            # Bônus por usar as MÃOS para push-off
            torso_vel_z = self.data.qvel[2]
            if (contact_info[0] > 0 or contact_info[1] > 0) and torso_vel_z > 0.1:
                reward += 20.0 # Aumentado
                
            # PENALIDADE POR USAR O COTOVELO OU CANELA
            if contact_info[4] > 0 or contact_info[5] > 0: reward -= 25.0
            if contact_info[6] > 0 or contact_info[7] > 0: reward -= 25.0
                
            # Bônus de Sola do Pé Reta
            l_foot_up = self.data.body('left_foot_link').xmat[8]
            r_foot_up = self.data.body('right_foot_link').xmat[8]
            if contact_info[2] > 0 and l_foot_up > 0.9: reward += 15.0
            if contact_info[3] > 0 and r_foot_up > 0.9: reward += 15.0
                
            # Bônus de Pés para Dentro (Inversão)
            l_roll = self.data.qpos[self.l_ankle_roll_idx]
            r_roll = self.data.qpos[self.r_ankle_roll_idx]
            if l_roll > 0.1: reward += 5.0
            if r_roll < -0.1: reward += 5.0
        
        # 4. Bônus massivo por ficar de pé
        standing = torso_height > 0.65 and torso_up > 0.9
        if standing:
            reward += 200.0 # Aumentado para valorizar a meta final
            
        # 5. Penalidade de tempo
        reward -= 0.5 
            
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return obs, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self.prev_action = np.zeros(self.n_actions) # Reset da ação anterior
        self.data.qpos[2] = 0.25 
        if np.random.rand() > 0.5:
            self.data.qpos[3:7] = [0.707, 0, 0.707, 0]
        else:
            self.data.qpos[3:7] = [0.707, 0, -0.707, 0]
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

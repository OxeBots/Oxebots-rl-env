import torch
import torch.nn as nn
from stable_baselines3 import PPO

CAMINHO_MODELO = "ppo_getup_back_20260621_191115.zip"
CAMINHO_ONNX = "ppo_getup_back.onnx"

print("Carregando o modelo Stable Baselines 3 na CPU...")
model = PPO.load(CAMINHO_MODELO, device="cpu")
print("Modelo carregado com sucesso!")

class SB3ActorWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        
    def forward(self, observation):
        return self.policy._predict(observation, deterministic=True)

ator_para_exportar = SB3ActorWrapper(model.policy)
ator_para_exportar.eval()

formato_observacao = model.observation_space.shape
print(f"Formato da observação detectado: {formato_observacao}")

dummy_input = torch.randn(1, *formato_observacao, device="cpu")

print("Iniciando a exportação para ONNX...")
torch.onnx.export(
    ator_para_exportar,            
    dummy_input,                   
    CAMINHO_ONNX,                  
    export_params=True,            
    opset_version=12,              
    input_names=['observation'],   
    output_names=['action'],       
    dynamic_axes={
        'observation': {0: 'batch_size'}, 
        'action': {0: 'batch_size'}
    },
    dynamo=False  # <--- Desativa o motor novo e usa o exportador clássico
)

print(f"\n✅ SUCESSO ABSOLUTO! Modelo convertido e salvo como: {CAMINHO_ONNX}")
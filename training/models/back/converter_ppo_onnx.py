import torch
import torch.nn as nn
from stable_baselines3 import PPO

# 1. Nomes dos arquivos
CAMINHO_MODELO = "ppo_getup_back_mjx_20260810_230406.pkl"
CAMINHO_ONNX = "ppo_getup_back_mjx.onnx"

# =====================================================================
# PASSO 1: Carregar o modelo do Stable Baselines 3 corretamente
# =====================================================================
print(f"Carregando o modelo {CAMINHO_MODELO}...")
# Usamos a função de load do próprio PPO (ele lida com o arquivo zipado/pickled)
model = PPO.load(CAMINHO_MODELO)
print("Modelo carregado com sucesso!")

# =====================================================================
# PASSO 2: Criar um Wrapper para extrair apenas o "Ator" (Actor)
# ONNX precisa de uma função simples de Entrada -> Saída.
# O SB3 tem muitas coisas internas, então essa classe simplifica.
# =====================================================================
class SB3ActorWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        
    def forward(self, observation):
        # '_predict' retorna diretamente o tensor da Ação baseado na Observação.
        # 'deterministic=True' faz com que ele escolha a melhor ação (sem explorar).
        return self.policy._predict(observation, deterministic=True)

# Instanciamos o Wrapper passando a política do modelo e colocamos em modo de avaliação
ator_para_exportar = SB3ActorWrapper(model.policy)
ator_para_exportar.eval()

# =====================================================================
# PASSO 3: Criar um dado de exemplo (Dummy Input)
# =====================================================================
# O ONNX precisa rodar a rede uma vez com dados falsos para "mapear" o caminho.
# Pegamos o tamanho exato da observação do seu ambiente MuJoCo
formato_observacao = model.observation_space.shape

print(f"Formato da observação do ambiente: {formato_observacao}")

# Criamos um tensor aleatório (batch_size=1, + formato da observação)
# E enviamos para o mesmo dispositivo (CPU ou GPU) onde o modelo está
dummy_input = torch.randn(1, *formato_observacao).to(model.device)

# =====================================================================
# PASSO 4: Exportar para ONNX
# =====================================================================
print("Iniciando a exportação para ONNX...")

torch.onnx.export(
    ator_para_exportar,            # O modelo simplificado
    dummy_input,                   # O dado de exemplo
    CAMINHO_ONNX,                  # Nome do arquivo de saída
    export_params=True,            # Salvar os pesos dentro do arquivo
    opset_version=12,              # Versão compatível do ONNX
    input_names=['observation'],   # Nome da camada de entrada
    output_names=['action'],       # Nome da camada de saída
    dynamic_axes={
        'observation': {0: 'batch_size'}, 
        'action': {0: 'batch_size'}
    } # Permite enviar lotes (batches) de qualquer tamanho no futuro
)

print(f"\n✅ SUCESSO! Modelo convertido e salvo como: {CAMINHO_ONNX}")
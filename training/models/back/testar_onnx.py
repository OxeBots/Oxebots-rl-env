import numpy as np
import onnxruntime as ort

# Caminho do modelo ONNX gerado
CAMINHO_ONNX = "ppo_getup_back.onnx"

print("Carregando o modelo ONNX na CPU...")
# Inicializa a sessão de inferência forçando o provedor de CPU
session = ort.InferenceSession(CAMINHO_ONNX, providers=["CPUExecutionProvider"])

# Descobre o nome e o formato da entrada esperada pelo modelo
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

print(f"Nome da entrada: {input_name}")
print(f"Formato esperado pela entrada: {input_shape}")

# Corrige dimensões dinâmicas (ex: substitui 'None' ou strings por 1 para o teste)
batch_size = 1
fixed_shape = [batch_size if (dim is None or isinstance(dim, str)) else dim for dim in input_shape]

# Cria uma observação fictícia aleatória com o mesmo formato que o robô espera 
# (Lembre-se que vimos antes que o seu modelo espera 126 features)
print(f"Criando dados de teste fictícios com o formato: {fixed_shape}")
dummy_observation = np.random.randn(*fixed_shape).astype(np.float32)

# Executa a inferência na CPU
print("Executando a predição...")
outputs = session.run(None, {input_name: dummy_observation})

# Mostra o resultado (a ação que o robô deve tomar)
print("\n✅ TESTE CONCLUÍDO COM SUCESSO!")
print(f"Formato da ação de saída: {outputs[0].shape}")
print(f"Valores das ações geradas:\n{outputs[0]}")
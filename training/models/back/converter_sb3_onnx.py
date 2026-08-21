from pathlib import Path
import torch
import torch.nn as nn

# Defina o caminho do seu modelo (.zip, .pkl ou .pkr) e o nome de saída
CAMINHO_MODELO = "ppo_getup_back_20260621_191115.zip"
CAMINHO_ONNX = "ppo_getup_back.onnx"

# Identifica a extensão do arquivo automaticamente
caminho_path = Path(CAMINHO_MODELO)
extensao = caminho_path.suffix.lower()

if not caminho_path.exists():
    raise FileNotFoundError(f"O arquivo {CAMINHO_MODELO} não foi encontrado!")

print(f"Arquivo detectado com extensão: {extensao}")

# =====================================================================
# CASO 1: Modelo do Stable Baselines 3 (.zip)
# =====================================================================
if extensao == ".zip":
    from stable_baselines3 import PPO
    
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

    print("Iniciando a exportação para ONNX (SB3)...")
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
        dynamo=False  # Usa o exportador clássico
    )

# =====================================================================
# CASO 2: Modelo do Brax / JAX (.pkl ou .pkr)
# =====================================================================
elif extensao in [".pkl", ".pkr"]:
    import pickle
    import jax
    from jax.experimental import jax2tf
    import tensorflow as tf
    import tf2onnx
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics

    print("Carregando o modelo Brax/JAX...")
    with open(CAMINHO_MODELO, "rb") as f:
        normalizer_params, actor_params, critic_params = pickle.load(f)

    ENV_OBS_SIZE = normalizer_params.mean.shape[0]
    print(f"Tamanho da observação detectado: {ENV_OBS_SIZE}")
    
    # ATENÇÃO: Ajuste este valor para o número de ações do seu robô se for usar .pkl/.pkr
    ENV_ACT_SIZE = 10  

    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=ENV_OBS_SIZE,
        action_size=ENV_ACT_SIZE,
        preprocess_observations_fn=running_statistics.normalize,
    )

    make_policy = ppo_networks.make_inference_fn(ppo_network)
    policy_fn = make_policy((normalizer_params, actor_params), deterministic=True)

    def forward_pass(obs):
        rng = jax.random.PRNGKey(0)
        action, _ = policy_fn(obs, rng)
        return action

    jit_forward = jax.jit(forward_pass)
    tf_fn = jax2tf.convert(jit_forward, with_gradient=False)

    class TFModule(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, ENV_OBS_SIZE], dtype=tf.float32)])
        def __call__(self, x):
            return tf_fn(x)

    tf_model = TFModule()

    print("Iniciando a exportação para ONNX (Brax/JAX via TF)...")
    model_proto, _ = tf2onnx.convert.from_function(
        tf_model.__call__,
        input_signature=[tf.TensorSpec(shape=[None, ENV_OBS_SIZE], dtype=tf.float32, name='observation')],
        output_path=CAMINHO_ONNX,
        opset=13
    )

else:
    raise ValueError(f"Extensão não suportada: {extensao}. Use .zip, .pkl ou .pkr")

print(f"\n✅ SUCESSO ABSOLUTO! Modelo convertido e salvo como: {CAMINHO_ONNX}")
import argparse
import sys
import os
import glob
import shutil
import numpy as np
import torch
import onnxruntime as ort
from stable_baselines3 import PPO


def extract_actor(sb3_model_path: str) -> tuple[torch.nn.Module, int]:
    """
    Extrai a rede do Ator de um modelo SB3 PPO.

    Returns:
        actor_model: Rede PyTorch sequencial (policy_net + action_net)
        input_size: Dimensão da observação de entrada
    """
    model = PPO.load(sb3_model_path, device="cpu")

    # Extrair as duas partes do ator
    policy_net = model.policy.mlp_extractor.policy_net  # Camadas ocultas
    action_net = model.policy.action_net                 # Camada de saída

    # Montar modelo sequencial completo
    actor = torch.nn.Sequential(policy_net, action_net)
    actor.eval()

    # Tamanho da entrada
    input_size = model.observation_space.shape[0]

    return actor, input_size


def export_to_onnx(actor: torch.nn.Module, input_size: int, output_path: str):
    """
    Exporta o modelo ator para ONNX.

    Os nomes de input/output DEVEM ser 'obs' e 'action' para compatibilidade
    com load_network() e run_network() do neural_network.py.
    """
    dummy_input = torch.randn(1, input_size)

    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Modelo exportado com sucesso: {output_path}")
    print(f"  Input shape:  (batch, {input_size})")
    print(f"  Output shape: (batch, N_ACTIONS)")


def validate(sb3_model_path: str, onnx_path: str, n_tests: int = 100):
    """
    Compara as saídas do modelo SB3 original com o ONNX exportado.
    """
    model = PPO.load(sb3_model_path, device="cpu")

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_size = model.observation_space.shape[0]
    max_error = 0.0

    for _ in range(n_tests):
        # Gerar observação aleatória
        obs = np.random.randn(input_size).astype(np.float32)
        obs = np.clip(obs, -10.0, 10.0)

        # Predição SB3
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            features = model.policy.extract_features(obs_tensor, model.policy.pi_features_extractor)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            sb3_action = model.policy.action_net(latent_pi).numpy().flatten()

        # Predição ONNX
        onnx_action = session.run(
            [output_name],
            {input_name: obs.reshape(1, -1)}
        )[0].flatten()

        error = np.max(np.abs(sb3_action - onnx_action))
        max_error = max(max_error, error)

    print(f"\nValidação com {n_tests} amostras:")
    print(f"  Erro máximo absoluto: {max_error:.10f}")

    if max_error < 1e-5:
        print("  ✅ Conversão validada com sucesso!")
    else:
        print("  ⚠️  Erro acima do esperado — investigar!")

    return max_error


def get_latest_model(model_dir: str) -> str | None:
    """Encontra o modelo .zip mais recente em um diretório."""
    files = glob.glob(os.path.join(model_dir, "*.zip"))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def main():
    parser = argparse.ArgumentParser(description="Exportar modelo SB3 para ONNX")
    parser.add_argument(
        "mode",
        choices=["walk", "front", "back"],
        help="Tipo do modelo a exportar"
    )
    parser.add_argument(
        "--input", "-i",
        help="Caminho do .zip (se omitido, usa o mais recente em models/<mode>/)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Caminho do .onnx de saída (default: mesmo diretório do input)"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Executar validação cruzada SB3 vs ONNX"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Copiar o .onnx para a pasta de runtime do skill"
    )

    args = parser.parse_args()

    # Encontrar modelo de entrada
    if args.input:
        model_path = args.input
    else:
        model_dir = f"./training/models/{args.mode}/"
        model_path = get_latest_model(model_dir)
        if not model_path:
            print(f"Nenhum modelo encontrado em {model_dir}")
            sys.exit(1)

    print(f"Modelo de entrada: {model_path}")

    # Definir saída
    if args.output:
        onnx_path = args.output
    else:
        onnx_path = model_path.replace(".zip", ".onnx")

    # Exportar
    actor, input_size = extract_actor(model_path)
    export_to_onnx(actor, input_size, onnx_path)

    # Validar
    if args.validate:
        validate(model_path, onnx_path)

    # Deploy (copiar para runtime)
    if args.deploy:
        deploy_paths = {
            "walk": "mujococodebase/skills/walk/walk.onnx",
            "front": "mujococodebase/skills/keyframe/get_up/get_up_front.onnx",
            "back": "mujococodebase/skills/keyframe/get_up/get_up_back.onnx",
        }
        dest = deploy_paths[args.mode]
        shutil.copy2(onnx_path, dest)
        print(f"\nModelo deployado em: {dest}")


if __name__ == "__main__":
    main()

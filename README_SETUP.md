# Guia de Configuração Rápida - BahiaRT Mujoco Base

> [!IMPORTANT]
> **O SIMULADOR DEVE ESTAR RODANDO PRIMEIRO.**
> Você deve abrir o servidor (RCSSServerMJ) e visualizar o campo antes de tentar rodar qualquer script Python. Caso contrário, a conexão será recusada.

Este guia detalha os passos realizados para configurar o ambiente e rodar o simulador.

## 1. Requisitos de Sistema
- Python 3.12 ou superior (O projeto recomenda 3.13, mas funciona no 3.12 com as dependências instaladas via pip).
- Servidor RCSSServerMJ instalado e rodando.

## 2. Configuração do Ambiente Virtual (venv)

Para manter o sistema limpo e evitar conflitos de versão, utilizamos um ambiente virtual:

```bash
# Criar o ambiente virtual
python3 -m venv venv

# Ativar o ambiente virtual
source venv/bin/activate
```

## 3. Instalação das Dependências

Com o ambiente ativado, instale as bibliotecas necessárias:

```bash
pip install --upgrade pip
pip install numpy scipy pyyaml onnxruntime onnx
```

## 4. Como Executar

### Iniciar um Único Jogador
Certifique-se de que o simulador MuJoCo está aberto e em modo de espera. No terminal com a `venv` ativa:

```bash
python3 run_player.py -n 1 -t MeuTime
```

### Iniciar o Time Completo (3v3)
O script `start3v3.sh` automatiza a inicialização de 3 jogadores:

```bash
./start3v3.sh
```

## 5. Dicas Importantes
- **Sempre ative a venv** antes de rodar os scripts: `source venv/bin/activate`.
- Se o robô não aparecer, verifique se o servidor MuJoCo está rodando na porta padrão (60000).
- Os logs de erro aparecerão diretamente no terminal caso alguma dependência ainda esteja faltando.

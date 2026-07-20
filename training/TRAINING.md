# Guia de Treinamento Especializado - Skills de RL (GetUp e Walk)

Este guia descreve o sistema de treinamento por Aprendizado por Reforço (RL) para o robô T1. O projeto suporta tanto as manobras de levantar (frente/costas) quanto o controle de locomoção bípede.

## 1. Dependências

Para rodar o treinamento, instale as bibliotecas base:

```bash
pip install gymnasium stable-baselines3[extra] mujoco numpy tensorboard scipy pyyaml onnx onnxruntime wandb
```

Ou instale pelo arquivo de requirements:

```bash
pip install -r training/requirements.txt
```

## 2. Estrutura de Treinamento

O treinamento é dividido por skills para evitar interferência:

### A. Treinar Levantamento de FRENTE e COSTAS (GetUp)

```bash
python3 training/train_front.py
python3 training/train_back.py
```

### B. Treinar Locomoção do Zero (Walk - SB3)

Treina a caminhada bípede omnidirecional a partir de pesos aleatórios usando Stable Baselines 3.

```bash
python3 training/train_walk.py
```

### C. Fine-tuning de Locomoção (Walk - Nativo PyTorch/ONNX)

Script especializado para carregar os pesos exatos de um modelo ONNX em produção (com arquitetura própria, ex: ELU + LayerNorm) e aperfeiçoá-lo usando PPO customizado.

```bash
python3 training/train_walk_finetune.py mujococodebase/skills/walk/walk.onnx
```

## 3. Visualização do Aprendizado (Enjoy)

O script `enjoy.py` renderiza o comportamento das redes treinadas. Ele suporta modelos `.zip` (gerados pelo SB3) e `.onnx` (gerados pelo fine-tuning).

### Ver o Modelo mais recente

O script busca automaticamente na pasta `training/models/` ou `training/checkpoints/`:

```bash
python3 training/enjoy.py front
python3 training/enjoy.py back
python3 training/enjoy.py walk
```

### Ver um Modelo Específico (.zip ou .onnx)

Você pode testar diretamente o modelo salvo (ideal para testar um fine-tuning interrompido na metade ou o checkpoint de melhor recompensa):

```bash
# SB3 (Treino do zero)
python3 training/enjoy.py walk training/checkpoints/walk/walk_model_1000000_steps.zip

# Fine-tuning ONNX (Melhor modelo)
python3 training/enjoy.py walk training/models/walk/best_finetune.onnx

# Fine-tuning ONNX (Modelo ao ser interrompido com Ctrl+C)
python3 training/enjoy.py walk training/models/walk/walk_finetune_20260720_195501.onnx
```

### Testar Comandos de Velocidade (Apenas modo Walk)

Você pode injetar comandos manuais de velocidade (vx, vy, yaw_rate) para ver se o robô obedece a direção correta:

```bash
python3 training/enjoy.py walk training/models/walk/best_finetune.onnx "0.5 0.0 0.0"
```

---

## 4. Monitoramento (WandB & TensorBoard)

### A. Weights & Biases (WandB) - Recomendado

Os logs e gráficos são enviados à nuvem em tempo real.

1. Faça login: `wandb login`
2. Os scripts sincronizam hiperparâmetros, vídeos de validação e métricas de PPO (Recompensa, KL, Loss).

### B. TensorBoard (Local)

```bash
tensorboard --logdir ./training/logs/
```

---

## 5. Arquitetura de Recompensas e Hiperparâmetros (Boas Práticas)

Para que o robô aprenda a andar e evite comportamentos como "congelamento" (ficar paralisado esperando cair), os scripts de *Walk* seguem as seguintes diretrizes essenciais de modelagem:

1. **Survival Bonus vs Penalidades de Morte**:
   Não aplicamos punições explícitas gigantes (ex: `-50.0`) quando o robô cai. Isso aterroriza a rede, pois qualquer exploração arriscada gera um impacto devastador no gradiente, induzindo o "congelamento". Em vez disso, aplicamos um **Survival Bonus** (`+1.0` constante a cada passo). Cair encerra o episódio e corta esse ganho. O PPO aprende sozinho que deve estender a vida.

2. **Tamanho do Batch e Overfitting (SB3)**:
   Ao simular múltiplos ambientes (`num_cpu = 15`), o buffer de coleta cresce rápido. Garantimos que `batch_size = 4096` seja suficientemente grande e que o número de épocas (`n_epochs = 5`) seja pequeno. Isso evita que a rede sofra "catastrophic forgetting" fazendo milhares de mini-atualizações de gradiente na exata mesma amostragem de dados.

3. **Learning Rate**:
   O padrão ouro para PPO em robótica de controle contínuo contendo MLPs é na casa de `3e-4` no SB3, ou `1e-3` com *learning rate adaptativo baseado em Kullback-Leibler (KL divergence)* no script de fine-tuning.

4. **Curvas de Aprendizado**:
   * **Até 2M steps**: O robô agitará as pernas de forma descontrolada e cairá muito.
   * **5M a 10M steps**: Começa a arrastar os pés e dar passos grosseiros.
   * **Acima de 30M steps**: Convergência para marchas simétricas e obediência total ao controle linear e angular.
   O teto de treinamento oficial é de 50.000.000 timesteps, sendo seguro interromper antes ao observar um longo platô na métrica `ep_rew_mean`.

## 6. Otimizações de Desempenho e Dicas Técnicas

### Paralelização de Ambientes (Multi-CPU)

Os scripts [train_front.py](file:///home/matheus/bahiart-mujoco-base/training/train_front.py) e [train_back.py](file:///home/matheus/bahiart-mujoco-base/training/train_back.py) usam `SubprocVecEnv` para paralelizar a simulação física do MuJoCo.

* **Auto-Detecção**: O código detecta automaticamente os núcleos de CPU disponíveis e cria `N - 1` ambientes simultâneos.
* **Como alterar o número de núcleos manualmente**: Caso queira definir um número fixo de ambientes em paralelo (ex: 8), abra os scripts [train_front.py](file:///home/matheus/bahiart-mujoco-base/training/train_front.py) ou [train_back.py](file:///home/matheus/bahiart-mujoco-base/training/train_back.py) e altere a linha abaixo:

  ```python
  # De:
  num_cpu = max(1, os.cpu_count() - 1)

  # Para:
  num_cpu = 8  # Substitua pelo número desejado
  ```

* **Impacto**: Reduz o tempo de treinamento necessário de dias para poucas horas (escala quase linear com o número de CPUs físicas).

### Aceleração por GPU (CUDA)

A execução das atualizações de rede neural (PPO) é feita na GPU se o CUDA estiver disponível.

* O código utiliza `device="auto"`.
* Para treinar usando a placa de vídeo (ex: GTX 1080), garanta que o driver oficial da NVIDIA está instalado e que o PyTorch foi instalado com suporte a CUDA no ambiente virtual:

  ```bash
  pip install torch --extra-index-url https://download.pytorch.org/whl/cu121 --force-reinstall
  ```

* **Como validar se a GPU está sendo detectada**:
  1. No terminal, verifique se a placa de vídeo é exibida pelo driver:

     ```bash
     nvidia-smi
     ```

  2. Teste se o PyTorch no ambiente virtual está detectando a GPU:

     ```bash
     python3 -c "import torch; print('CUDA Disponível:', torch.cuda.is_available()); print('GPU Detectada:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma')"
     ```

     Se retornar `CUDA Disponível: True`, o treinamento usará a placa automaticamente.

### Outras Informações

* **Duração do Episódio**: O robô tem 1000 passos (~20s) para levantar e estabilizar.
* **Pose Neutra**: O bônus de estabilização final incentiva o robô a terminar na postura padrão de jogo.
* **Reward Shaping**: O aprendizado é guiado pelos ângulos de junta definidos nos Keyframes YAML (encolher pernas e usar braços/joelhos como pivô).

## 7. Estrutura de Diretórios do Projeto

A estrutura de pastas foi organizada para separar o código-fonte da execução e dos modelos treinados.

```text
3d_strategy/
├── mujococodebase/             # Código-fonte do robô e simulação
├── training/                   # Scripts de treinamento e execução
│   ├── requirements.txt        # Dependências do ambiente
│   ├── train_front.py          # Treinamento da skill GetUp (Frente)
│   ├── train_back.py           # Treinamento da skill GetUp (Costas)
│   ├── train_walk.py           # Treinamento da skill Walk (SB3)
│   ├── train_walk_finetune.py  # Fine-tuning de Walk com ONNX
│   ├── enjoy.py                # Visualização dos modelos treinados
│   ├── logs/                   # Logs de execução e TensorBoard
│   │   ├── front/
│   │   ├── back/
│   │   └── walk/
│   ├── checkpoints/            # Checkpoints intermediários do SB3
│   │   └── walk/
│   └── models/                 # Modelos finais (.zip e .onnx)
│       ├── front/
│       ├── back/
│       └── walk/
```

## 8. Melhores Práticas para Treinamento (Resumo)

* **Paralelização**: Use múltiplos CPUs (`num_cpu = N-1`) para acelerar a coleta de experiências.
* **Aceleração GPU**: Garanta que o PyTorch com CUDA esteja instalado para processamento rápido das redes neurais.
* **Monitoramento**: Sempre use `wandb.init()` para acompanhar métricas e vídeos de validação em tempo real.
* **Interrupção Segura**: O treinamento pode ser interrompido a qualquer momento com `Ctrl+C`. O modelo mais recente é salvo automaticamente no diretório `training/models/walk/`.

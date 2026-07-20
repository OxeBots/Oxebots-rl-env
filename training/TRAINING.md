# Guia de Treinamento Especializado - Skill GetUp (RL)

Este guia descreve o novo sistema de treinamento especializado para o robô T1, focado em manobras separadas de levantar de frente e de costas.

## 1. Dependências

Para rodar o treinamento, você precisará instalar as seguintes bibliotecas:

```bash
pip install gymnasium stable-baselines3[extra] mujoco numpy tensorboard scipy pyyaml
```

Ou, se preferir usar o arquivo de requirements:

```bash
pip install -r training/requirements.txt
```

## 2. Estrutura de Treinamento

O treinamento agora é dividido em dois modos independentes para evitar interferência de aprendizado:

### A. Treinar Levantamento de FRENTE (Front)
```bash
python3 training/train_front.py
```

### B. Treinar Levantamento de COSTAS (Back)
```bash
python3 training/train_back.py
```

## 3. Monitoramento

O treinamento pode ser monitorado de duas formas simultâneas:

### A. Weights & Biases (WandB) - Recomendado (Nuvem)
Os logs e o uso de recursos (GPU/CPU) são sincronizados em tempo real na nuvem do [wandb.ai](https://wandb.ai).
1. Faça login na sua conta WandB pelo terminal antes de iniciar:
   ```bash
   wandb login
   ```
2. Insira a sua chave de API quando solicitado. Os scripts iniciarão o rastreamento automaticamente.

### B. TensorBoard (Local)
Visualize os gráficos localmente em tempo real:
```bash
tensorboard --logdir ./training/logs/
```
No navegador: **http://localhost:6006**

---

## 4. Visualização do Aprendizado (Enjoy)

### Ver o Modelo mais recente
O script busca automaticamente o arquivo mais recente em `training/models/` (modelos finais) ou `training/checkpoints/` (se não houver modelos finais).
```bash
python3 training/enjoy.py front
# OU
python3 training/enjoy.py back
```

### Ver um Modelo Específico (Checkpoints ou Antigos)
Você pode passar o caminho de um arquivo `.zip` específico como segundo argumento:
```bash
# Exemplo para visualizar um checkpoint específico de 1 milhão de passos:
python3 training/enjoy.py front training/checkpoints/front/getup_front_model_1000000_steps.zip
```

---

## 5. Onde ficam os modelos?

1. **Modelos Finais**: `training/models/[front|back]/`. Salvos ao fim do treino com timestamp.
2. **Checkpoints**: `training/checkpoints/[front|back]/`. Salvos automaticamente a cada 250.000 passos.
3. **Logs**: `training/logs/[front|back]/`. Dados para o TensorBoard.

---

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
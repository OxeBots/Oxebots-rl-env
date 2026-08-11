# OxeBots Training - Guia de Treinamento Especializado (100% GPU / MJX)

Este guia descreve o sistema de treinamento ultra-acelerado do robô T1 para a equipe **OxeBots**, focado em manobras de levantar (frente/costas) 100% executadas na GPU via **MuJoCo MJX + Brax (JAX)**.

---

## 1. Dependências e Instalação

Para rodar o treinamento acelerado em GPU, instale as bibliotecas necessárias no ambiente virtual:

```bash
pip install -r training/requirements.txt
```

*(Pacotes principais: `jax[cuda12]`, `mujoco-mjx`, `brax`, `tensorboardX`, `mujoco`, `pyyaml`).*

---

## 2. Estrutura de Treinamento (100% GPU na VRAM)

O treinamento simula **2.048 ambientes paralelos simultaneamente** dentro da memória VRAM da GPU, sem gargalos de CPU/RAM.

### A. Treinar Levantamento de FRENTE (Front)
```bash
python3 training/train_front_mjx.py
```

### B. Treinar Levantamento de COSTAS (Back)
```bash
python3 training/train_back_mjx.py
```

---

## 3. Monitoramento em Tempo Real (TensorBoard)

Para garantir **zero gargalos na GPU**:
* **WandB e Renderização de Vídeo por Frame** foram desativados para evitar chamadas de rede e overhead da CPU durante o loop de treino.
* **TensorBoard** foi configurado para registrar logs assíncronos diretamente na pasta `./training/logs/`.

### Como rodar o TensorBoard:

No terminal, execute:

```bash
tensorboard --logdir=training/logs
```

Em seguida, abra o navegador em:
👉 **[http://localhost:6006](http://localhost:6006)**

No dashboard do TensorBoard você poderá acompanhar em tempo real:
* `eval/episode_reward`: Recompensa total por episódio.
* `reward_height`: Evolução do ganho de altura do robô.
* `reward_standing`: Bônus por posição em pé estável.
* `reward_mimic`: Recompensa de imitação dos Keyframes do YAML (DeepMimic RL).
* `joint_error`: Erro quadrático médio em relação às poses do YAML.
* `eval/sps`: Passos de simulação por segundo (SPS) na GPU.

---

## 4. Visualização do Aprendizado (Enjoy)

Para visualizar a simulação gráfica 3D do robô executando o modelo treinado:

```bash
# Visualizar o modelo de FRENTE mais recente
python3 training/enjoy.py front

# Visualizar o modelo de COSTAS mais recente
python3 training/enjoy.py back
```

---

## 5. Onde ficam os modelos e logs?

1. **Modelos Finais (JAX/Brax)**: `training/models/[front_mjx|back_mjx]/`. Salvos no formato `.pkl`.
2. **Logs do TensorBoard**: `training/logs/[front_mjx|back_mjx]/`.
3. **Ambiente MJX**: `training/getup_env_mjx.py` (contém `GetUpFrontMjxEnv` e `GetUpBackMjxEnv`).

---

## 6. Arquitetura Técnica e Desempenho

* **Física & PPO 100% em GPU**: O motor físico (MJX) e o algoritmo PPO rodam nativamente compilados na GPU (`@jax.jit`).
* **DeepMimic RL (Keyframes YAML)**: Rastreamento dinâmico de pose em tempo real com interpolação contínua das fases salvas em `get_up_front.yaml` e `get_up_back.yaml`.
* **Zero Host-Device Sync Overhead**: A comunicação GPU $\to$ CPU ocorre apenas 20 vezes durante todo o treino (epochs de avaliação).
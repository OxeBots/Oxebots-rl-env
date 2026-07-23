# Guia de Treinamento Especializado - Skill GetUp (100% GPU / MJX)

Este guia descreve o sistema de treinamento ultra-acelerado para o robô T1, focado em manobras separadas de levantar de frente e de costas 100% executadas na GPU via **MuJoCo MJX + Brax (JAX)**.

---

## 1. Dependências e Instalação

Para rodar o treinamento acelerado em GPU, instale as bibliotecas necessárias no ambiente virtual:

```bash
pip install -r training/requirements.txt
```

*(Pacotes principais: `jax[cuda12]`, `mujoco-mjx`, `brax`, `wandb`, `mujoco`, `pyyaml`).*

---

## 2. Estrutura de Treinamento (100% GPU na VRAM)

O treinamento simula **4.096 ambientes paralelos simultaneamente** dentro da memória da GPU (RTX 3080), reduzindo o tempo de treino de 9 horas para ~10 a 20 minutos.

### A. Treinar Levantamento de FRENTE (Front)
```bash
python3 training/train_front_mjx.py
```

### B. Treinar Levantamento de COSTAS (Back)
```bash
python3 training/train_back_mjx.py
```

---

## 3. Monitoramento

O treinamento monitora logs em tempo real:

### Weights & Biases (WandB) - Recomendado
Os logs e FPS do treino em GPU são sincronizados na nuvem do [wandb.ai](https://wandb.ai).
1. Faça login no terminal:
   ```bash
   wandb login
   ```
2. Insira a sua chave de API quando solicitado.

---

## 4. Visualização do Aprendizado (Enjoy)

Para visualizar a simulação gráfica do robô aprendendo:

```bash
# Visualizar o modelo de FRENTE mais recente
python3 training/enjoy.py front

# Visualizar o modelo de COSTAS mais recente
python3 training/enjoy.py back
```

---

## 5. Onde ficam os modelos e logs?

1. **Modelos Finais (JAX/Brax)**: `training/models/[front_mjx|back_mjx]/`. Salvos no formato `.pkl`.
2. **Logs de Treinamento**: `training/logs/[front_mjx|back_mjx]/`.
3. **Ambiente MJX**: `training/getup_env_mjx.py` (contém `GetUpFrontMjxEnv` e `GetUpBackMjxEnv`).

---

## 6. Validação e Arquitetura Técnica

* **Física & Rede em GPU**: Tanto o motor de física do MuJoCo canto a execução do PPO rodam nativamente na GPU usando JAX (`@jax.jit`).
* **Mimic Reward**: Recompensa de imitação baseada nas fases dos arquivos Keyframe YAML (`get_up_front.yaml` e `get_up_back.yaml`).
* **Resolução Dinâmica do Torso**: O ID da torso é mapeado dinamicamente para garantir medição precisa de altura e estabilidade.
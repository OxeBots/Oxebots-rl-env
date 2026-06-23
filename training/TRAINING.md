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

## 2. Monitoramento (TensorBoard)

Visualize os gráficos de aprendizado (recompensa, tempo de episódio) em tempo real:
```bash
tensorboard --logdir ./training/logs/
```
No navegador: **http://localhost:6006**

## 3. Visualização do Aprendizado (Enjoy)

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

## 4. Onde ficam os modelos?

1.  **Modelos Finais**: `training/models/[front|back]/`. Salvos ao fim do treino com timestamp.
2.  **Checkpoints**: `training/checkpoints/[front|back]/`. Salvos automaticamente a cada 250.000 passos.
3.  **Logs**: `training/logs/[front|back]/`. Dados para o TensorBoard.

## 5. Dicas Técnicas

- **Duração do Episódio**: O robô tem 1000 passos (~20s) para levantar e estabilizar.
- **Pose Neutra**: O bônus de estabilização final incentiva o robô a terminar na postura padrão de jogo.
- **Reward Shaping**: O aprendizado é guiado pelos ângulos de junta definidos nos Keyframes YAML (encolher pernas e usar braços/joelhos como pivô).

---
o
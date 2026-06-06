# Guia de Treinamento Especializado - Skill GetUp (RL)

Este guia descreve o novo sistema de treinamento especializado para o robô T1, focado em manobras separadas de levantar de frente e de costas.

## 1. Estrutura de Treinamento

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

### Ver o Modelo Final (após o treino acabar)
O script busca automaticamente o arquivo mais recente em `training/models/`.
```bash
python3 training/enjoy.py front
# OU
python3 training/enjoy.py back
```

### Ver o Progresso "Ao Vivo" (Checkpoints)
Se o treino ainda estiver rodando, você pode ver o que o robô já aprendeu carregando um checkpoint da pasta `training/checkpoints/`.

Para fazer isso, use o script de teste rápido (substitua o caminho pelo checkpoint desejado):
```bash
# Exemplo para visualizar um checkpoint de 1 milhão de passos do modo Front:
python3 -c "import gymnasium as gym; from stable_baselines3 import PPO; from training.getup_env import GetUpFrontEnv; env = GetUpFrontEnv(); model = PPO.load('training/checkpoints/front/getup_front_model_1000000_steps', env=env); obs, _ = env.reset(); import mujoco.viewer; import time; viewer = mujoco.viewer.launch_passive(env.model, env.data); [ (model.predict(obs, deterministic=True), (action := model.predict(obs, deterministic=True)[0]), (step_res := env.step(action)), (obs := step_res[0]), (done := step_res[2]), (trunc := step_res[3]), viewer.sync(), time.sleep(0.02), (obs := env.reset()[0] if (done or trunc) else obs)) for _ in range(10000) ]"
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
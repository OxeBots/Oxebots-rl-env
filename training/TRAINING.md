# Guia Completo de Treinamento - Skill GetUp (RL)

Este guia contém todos os comandos necessários para treinar, monitorar e visualizar o aprendizado do robô T1.

## 1. Instalação do Ambiente

Certifique-se de estar com o ambiente virtual ativado e instale as dependências:

```bash
pip install stable-baselines3 gymnasium mujoco shimmy tensorboard tqdm rich
```

## 2. Comandos de Execução

Abra terminais separados para cada uma das funções abaixo:

### A. Iniciar o Treinamento
Este comando inicia o processo de aprendizado.
```bash
python3 training/train.py
```

### B. Monitorar o Progresso (Gráficos)
Este comando abre uma interface web para acompanhar o aprendizado em tempo real.
```bash
tensorboard --logdir ./training/logs/
```
> Após rodar, acesse no navegador: **http://localhost:6006**

### C. Ver o Robô Aprendendo (Visualização)
Este comando abre uma janela 3D para você ver fisicamente o que o robô já aprendeu.
```bash
python3 training/enjoy.py
```

## 3. O que observar no TensorBoard

Dentro do link do navegador, procure pela aba **SCALARS**:

- **rollout/ep_rew_mean**: É o gráfico principal. Mostra a pontuação média do robô. No começo estará muito negativo (ex: -7000) e deve começar a subir (ex: -2000, 0, +500) conforme ele aprende a levantar.
- **rollout/ep_len_mean**: Mostra quanto tempo as tentativas estão durando. Se o robô aprender a levantar rápido, este tempo pode cair.
- **train/learning_rate**: A velocidade com que a rede neural está se ajustando.

## 4. Dicas de Treinamento

- **Exploração Inicial**: Nos primeiros 100.000 passos, o robô apenas se debate. É normal a recompensa ficar estagnada no negativo.
- **Interrupção e Salvamento**: Se você estiver satisfeito com o comportamento visto no `enjoy.py`, vá no terminal do treino e pressione `Ctrl + C`. O modelo será salvo automaticamente como `getup_t1_ppo.zip`.
- **Histerese de Recompensa**: Se a curva de aprendizado (ep_rew_mean) ficar plana por muito tempo sem subir, pode ser necessário ajustar a função de recompensa no arquivo `training/getup_env.py`.

---

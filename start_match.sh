

#!/bin/bash
export OMP_NUM_THREADS=1

# Configurações padrão
host=${3:-localhost}
port=${4:-60000}
team1=${1:-TimeEsquerda}
team2=${2:-TimeDireita}
players=${5:-7} # Quantidade de jogadores por time

echo "Iniciando partida: $team1 vs $team2"

# Iniciar Time 1
echo "Lançando $team1..."
for i in $(seq 1 $players); do
  python3 run_player.py --host $host --port $port -n $i -t "$team1" &
done

# Pequena pausa para o servidor processar as conexões
sleep 1

# Iniciar Time 2
echo "Lançando $team2..."
for i in $(seq 1 $players); do
  python3 run_player.py --host $host --port $port -n $i -t "$team2" &
done

echo "Todos os jogadores foram lançados. Use ./kill.sh para encerrar."
wait

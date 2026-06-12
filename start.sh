#!/bin/bash
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-60000}
team_name=${3:-MujocoCodebase}

for i in {1..7}; do
  python3 run_player.py --host $host --port $port -n $i -t $team_name &
done

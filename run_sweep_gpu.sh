#!/bin/bash

session="sweep_$1"

tmux new-session -d -s $session

seed=0
gpus="0 1 2 3 4 5 6 7"
first_iteration="true"

for gpu in $gpus; do
    if [ "$first_iteration" != "true" ]; then
        tmux split-window -h -t $session
        tmux select-layout -t $session tiled
    fi
    cmd="echo ./run.sh CUDA_VISIBLE_DEVICES=$gpu wandb agent georgw7777/mil/$1"
    tmux send-keys -t $session "$cmd" ENTER
    first_iteration="false"
done

tmux select-layout -t $session even-vertical

tmux a -t $session

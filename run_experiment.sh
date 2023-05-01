#!/bin/bash

tmux new-session -d

seed=0
gpus="3 4 5 6 7"

for gpu in $gpus; do
    cmd="./run.sh ${@} device=$gpu seed=$seed"
    seed=$((seed+1))
    tmux split-window -h
    tmux send-keys "$cmd" ENTER
done


tmux select-pane -t 0
tmux kill-pane

tmux select-layout even-vertical

tmux a
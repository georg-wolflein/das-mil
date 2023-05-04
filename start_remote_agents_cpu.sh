#!/bin/bash

USR="gw66"
INSTANCES_PER_MACHINE=1
DO_INSTALL="false"

# Check if WANDB_API_KEY environment variable is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY environment variable is not set"
    exit 1
fi

# Check if machines.txt file exists
if [ ! -f machines.txt ]; then
    echo "machines.txt file does not exist"
    exit 1
fi

session="sweeps_cpu"
tmux new-session -d -s $session

first_iteration="true"

for machine in $(cat machines.txt); do
    for instance in $(seq 1 $INSTANCES_PER_MACHINE); do
        if [ "$first_iteration" != "true" ]; then
            tmux split-window -h -t $session
            tmux select-layout -t $session tiled
        fi
        ssh_cmd="\"cd /cs/home/$USR/das-mil"
        if [ "$DO_INSTALL" == "true" ]; then
            ssh_cmd="$ssh_cmd && ./install_cpu.sh"
        fi
        ssh_cmd="$ssh_cmd && WANDB_API_KEY=$WANDB_API_KEY ./run_agent_for_sweeps_cpu.sh\""
        cmd="ssh -o StrictHostKeyChecking=no $USR@$machine.cs.st-andrews.ac.uk $ssh_cmd"
        tmux send-keys -t $session "$cmd" ENTER
        first_iteration="false"
    done
done

# tmux select-layout -t $session even-vertical

tmux a -t $session
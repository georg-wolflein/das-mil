#!/bin/bash

sweeps_file="sweeps_cpu.txt"
stop_sweeps_file="stop_sweeps_cpu"

# Current line in sweeps.txt
curr_line=1

while true; do

    # Check if stop_sweeps file exists
    if [ -f $stop_sweeps_file ]; then
        echo "Stopping sweeps"
        break
    fi

    # Check if there are any sweeps left in sweeps.txt
    if [ ! -s $sweeps_file ]; then
        echo "No more sweeps left"
        break
    fi

    # Read current line from sweeps.txt
    sweep=$(sed -n ${curr_line}p $sweeps_file)

    # If sweep is empty, reset curr_line
    if [ -z "$sweep" ]; then
        curr_line=1
        continue
    fi

    # Remove everything after "#"
    sweep=$(echo "$sweep" | cut -d'#' -f1)

    echo "Running one task from sweep: $sweep"
    mil_env/bin/wandb agent --count 1 $sweep

    curr_line=$((curr_line+1))
    sleep 1
done
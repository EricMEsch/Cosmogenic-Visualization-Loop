#!/bin/bash

cleanup() {
    echo "Stopping workers..."

    tmux ls 2>/dev/null | grep worker | cut -d: -f1 | while read -r session; do
        tmux kill-session -t "$session"
    done

    echo "Clearing job queue..."
    : > out/queue.yaml

    exit 0
}

# Trap Ctrl+C and termination signals
trap cleanup SIGINT SIGTERM EXIT

python3 scheduling/launch_workers.py
python3 scheduling/scheduler.py

#!/bin/bash

cleanup() {
    echo "Stopping workers..."

    tmux ls 2>/dev/null | grep worker | cut -d: -f1 | while read -r session; do
        tmux kill-session -t "$session"
    done

    exit 0
}

# Trap Ctrl+C and termination signals
trap cleanup SIGINT SIGTERM EXIT

python3 scheduling/launch_workers.py
python3 scheduling/scheduler.py

# Things to do:
# Speed up the rendering even more.
# Add some sort of crash safety to the workers. (it just errors into "There is no crash recovery, because we dont crash")
# rename workers to jeff, jeb and jenny
# Fix any issues like the maint bug print
# Add more random idle prints to the workers, like "Jeff is taking a coffee break" or "Jenny is doing some yoga"

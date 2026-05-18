import subprocess

MAX_WORKERS = 3

# Launch worker processes in tmux sessions
for i in range(MAX_WORKERS):
    log_file = f"worker_{i}.log"

    # cmd = f"python3 worker.py {i} > {log_file} 2>&1"
    cmd = f"python3 worker.py {i}"

    subprocess.Popen(["tmux", "new-session", "-d", "-s", f"worker{i}", cmd])

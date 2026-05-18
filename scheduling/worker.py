import time
import yaml
import os
import sys

sys.path.append(".")  # Because we run this from the parent folder.
from run import run_gif
import fcntl

QUEUE_FILE = "out/queue.yaml"
PLAYLIST_FILE = "out/playlist.yaml"
LOCK_FILE = "out/worker_pause.flag"


def append_playlist(file):
    playlist = []
    if os.path.exists(PLAYLIST_FILE):
        with open(PLAYLIST_FILE, "r") as f:
            playlist = yaml.safe_load(f) or []

    playlist.append(file)

    with open(PLAYLIST_FILE, "w") as f:
        yaml.dump(playlist, f)


def is_paused():
    return os.path.exists(LOCK_FILE)


def pop_job_atomic():
    """
    Atomically selects a pending job and marks it as running.
    Returns job or None.
    """
    with open(QUEUE_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            queue = yaml.safe_load(f) or []

            # find first pending job
            for job in queue:
                if job.get("status") == "pending":
                    job["status"] = "running"

                    # rewrite file
                    f.seek(0)
                    f.truncate()
                    yaml.safe_dump(queue, f)

                    return job

            return None

        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def remove_job(job_id):
    with open(QUEUE_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            queue = yaml.safe_load(f) or []

            queue = [job for job in queue if job.get("job_id") != job_id]

            f.seek(0)
            f.truncate()
            yaml.safe_dump(queue, f)

        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_worker(worker_id):
    BLUE = "\033[34m"
    RESET = "\033[0m"
    print(f"{BLUE}[Worker {worker_id}] started{RESET}")

    while True:
        if is_paused():
            print(f"{BLUE}[Worker {worker_id}] paused...{RESET}")
            time.sleep(20)
            continue

        job = pop_job_atomic()

        if job is None:
            print(f"{BLUE}[Worker {worker_id}] no pending jobs, sleeping...{RESET}")
            time.sleep(30)
            continue

        job_id = job["job_id"]
        print(f"{BLUE}[Worker {worker_id}] running job {job_id}{RESET}")

        # run simulation
        files = run_gif("musun/part_*.dat", 1, worker_id)
        result_file = files[0]

        print(f"{BLUE}[Worker {worker_id}] finished job {job_id}: {result_file}{RESET}")

        append_playlist(result_file)
        remove_job(job_id)


if __name__ == "__main__":
    import sys

    run_worker(sys.argv[1])

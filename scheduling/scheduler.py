import os
import yaml
import time

QUEUE_FILE = "out/queue.yaml"
PLAYLIST_FILE = "out/playlist.yaml"
TARGET_PLAYLIST_SIZE = 10

CURR_JOB_ID = 0


def load(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return yaml.safe_load(f) or []
    return []


def save(file, data):
    with open(file, "w") as f:
        yaml.dump(data, f)


def add_jobs(n):
    global CURR_JOB_ID
    queue = load(QUEUE_FILE)

    for i in range(n):
        queue.append(
            {
                "job_id": CURR_JOB_ID,
                "status": "pending",
            }
        )
        CURR_JOB_ID = (CURR_JOB_ID + 1) % 1000

    save(QUEUE_FILE, queue)


def main():
    BLUE = "\033[34m"
    RESET = "\033[0m"
    print(f"{BLUE}[Scheduler] started{RESET}")
    while True:
        playlist = load(PLAYLIST_FILE)

        # refill playlist target indirectly via queue size
        queue = load(QUEUE_FILE)

        missing = TARGET_PLAYLIST_SIZE - (len(queue) + len(playlist))

        if missing > 0:
            print(f"{BLUE}[Scheduler] adding {missing} jobs{RESET}")
            add_jobs(missing)

        print(f"{BLUE}[Scheduler] sleeping...{RESET}")
        time.sleep(30)


if __name__ == "__main__":
    main()

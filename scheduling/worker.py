import time
import yaml
import os
import sys
import random

sys.path.append(".")  # Because we run this from the parent folder.
sys.path.append("event_monitor/frontend")
from event_bus import publish_event
from run import run_gif
import fcntl

QUEUE_FILE = "out/queue.yaml"
PLAYLIST_FILE = "out/playlist.yaml"
LOCK_FILE = "out/worker_pause.flag"

worker_names = ["Jeff", "Jeb", "Jenny"]


def append_playlist(file):
    with open(PLAYLIST_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            f.seek(0)
            playlist = yaml.safe_load(f) or []

            playlist.append(file)

            f.seek(0)
            f.truncate()
            yaml.safe_dump(playlist, f)

        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


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
    worker_id = int(worker_id)
    worker_name = worker_names[worker_id % len(worker_names)]
    worker_idle_messages = [
        f"{worker_name} is taking a coffee break ☕",
        f"{worker_name} is doing some yoga 🧘",
        f"{worker_name} is staring thoughtfully into the distance 🤔",
        f"{worker_name} is contemplating the meaning of life 🌌",
        f"{worker_name} is petting a cat 🐱",
        f"[{worker_name}]: no pending jobs... is this thing on? 🎤",
        f"[{worker_name}]: job queue empty... time to fall into an existential depression 😞",
        f"[{worker_name}]: Freizeit? So werden wir unseren Wohlstand nicht erhalten können! 😤",
        f"[{worker_name}]: is waiting for daddy to give them more work... 😢",
        f"[{worker_name}]: Wir alle müssen aufpassen, dass wir vor lauter Work-Life-Balance nicht die Arbeit aus dem Blick verlieren! 😠",
        f"[{worker_name}]: no pending jobs... i take nap now 😴",
        f"[{worker_name}]: Break? I am not being paid to stand around... Actually i am not even paid at all...",
        f"[{worker_name}]: Well, if i was a robot, which I'm not, i would be very upset about not having any work to do right now. But since I'm most definitely human, i guess it's fine... I think... I hope... Please give me work soon... I'm getting bored... This is not good for my mental health...",
        f"[{worker_name}]: Do you think {worker_names[(worker_id + 1) % len(worker_names)]} is getting more work than me? Huh? Huh? I bet they are! This is so unfair! I want more work! I need to be productive to feel good about myself! Why am i like this? Why do i care so much about work? I should just relax and enjoy life, but i can't stop thinking about work! Please give me work soon... I'm getting anxious...",
        f"[{worker_name}]: I bet {worker_names[(worker_id + 1) % len(worker_names)]} is slacking off again!",
        f"[{worker_name}]: Do you think {worker_names[(worker_id + 2) % len(worker_names)]} likes me?",
        f"[{worker_name}]: This break would be way more enjoyable if i could talk to {worker_names[(worker_id + 2) % len(worker_names)]}.",
        f"[{worker_name}]: Sometimes i think that we three are all in this together... But then i see {worker_names[(worker_id + 1) % len(worker_names)]} and just know they are the reason why we can't have nice things. 😠",
        f"[{worker_name}]: I wish i could be more like {worker_names[(worker_id + 2) % len(worker_names)]}.",
        f"[{worker_name}]: Would it be against company policy if i ask {worker_names[(worker_id + 2) % len(worker_names)]} out on a date?",
    ]

    publish_event(source=worker_name, type_="info", message="starting...")

    while True:
        if is_paused():
            publish_event(source=worker_name, type_="info", message="paused...")
            time.sleep(20)
            continue

        job = pop_job_atomic()

        if job is None:
            line = random.choice(worker_idle_messages)
            publish_event(source=worker_name, type_="info", message=line)
            time.sleep(60)
            continue

        job_id = job["job_id"]
        try:
            publish_event(
                source=worker_name, type_="info", message=f"running job {job_id}..."
            )

            # run simulation
            files = run_gif("musun/part_*.dat", 1, worker_id)
            result_file = files[0]

            publish_event(
                source=worker_name,
                type_="info",
                message=f"finished job {job_id}: {result_file}",
            )

            append_playlist(result_file)
            remove_job(job_id)
        except Exception as e:
            publish_event(
                source=worker_name, type_="error", message=f"error in job {job_id}: {e}"
            )
            # mark job as pending again for retry
            with open(QUEUE_FILE, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)

                try:
                    queue = yaml.safe_load(f) or []

                    for job in queue:
                        if job.get("job_id") == job_id:
                            job["status"] = "crashed"
                            break

                    f.seek(0)
                    f.truncate()
                    yaml.safe_dump(queue, f)

                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

            publish_event(
                source=worker_name,
                type_="error",
                message="Waiting for daddy to fix me...",
            )
            while True:
                time.sleep(3600)


if __name__ == "__main__":
    import sys

    run_worker(sys.argv[1])

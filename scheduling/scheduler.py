import os
import yaml
import time
import subprocess
from launch_workers import MAX_WORKERS
import sys
import numpy as np

sys.path.append("event_monitor/frontend")
sys.path.append("scheduling")
from event_bus import publish_event
from worker import worker_names

OUT_DIR = "out"
QUEUE_FILE = "out/queue.yaml"
PLAYLIST_FILE = "out/playlist.yaml"
TARGET_PLAYLIST_SIZE = 10

CURR_JOB_ID = 0

CLEANUP_INTERVAL = 60 * 60 * 2  # every 2 hours

LAST_TEN_CPU_TEMPS = []
ACCEPTABLE_CPU_FLUCTUATION = 4.0  # Celsius


def get_cpu_temp():
    # 1. Try Raspberry Pi thermal zone
    try:
        path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(path):
            with open(path, "r") as f:
                return float(f.read().strip()) / 1000.0
    except Exception:
        pass
    try:
        import psutil

        temps = psutil.sensors_temperatures()
        for entries in temps.values():
            if entries:
                return entries[0].current
    except Exception:
        pass

    return None


def get_disk_usage(path="/"):
    try:
        stat = os.statvfs(path)
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bfree * stat.f_frsize
        used = total - free
        usage_percent = used / total * 100
        return usage_percent
    except Exception:
        return None


def check_system_health():
    global LAST_TEN_CPU_TEMPS
    global ACCEPTABLE_CPU_FLUCTUATION
    publish_event(
        source="CurOS",
        type_="info",
        message="Monitoring system health...",
    )
    temp = get_cpu_temp()
    disk = get_disk_usage("/")

    if disk is not None:
        if disk > 90:
            publish_event(
                source="CurOS",
                type_="warning",
                message=f"Low disk space! ({disk:.1f}%)",
            )

    if temp is not None:
        LAST_TEN_CPU_TEMPS.append(temp)
        if len(LAST_TEN_CPU_TEMPS) > 10:
            np.mean(LAST_TEN_CPU_TEMPS[-10:])
            LAST_TEN_CPU_TEMPS.pop(0)
            if (temp - np.mean(LAST_TEN_CPU_TEMPS)) > ACCEPTABLE_CPU_FLUCTUATION:
                publish_event(
                    source="CurOS",
                    type_="info",
                    message="Warning: Core temperature rising...",
                )
            elif (temp - np.mean(LAST_TEN_CPU_TEMPS)) < -ACCEPTABLE_CPU_FLUCTUATION:
                publish_event(
                    source="CurOS",
                    type_="info",
                    message="Core temperature dropping...",
                )
        if temp > 100:
            publish_event(
                source="CurOS",
                type_="warning",
                message=f"Core overheating! Temperature at {temp:.1f}°C! Starting emergency shutdown...",
            )
            with open("overheat_shutdown.log", "a") as f:
                f.write(
                    f"{time.ctime()}: CPU temp {temp:.1f}°C exceeded threshold. Initiating shutdown.\n"
                )
            time.sleep(5)
            subprocess.run(["sudo", "shutdown", "-h", "now"])
        elif (
            temp > 75
        ):  # not like the elif is necessary. We shouldn't be here if temp > 100
            publish_event(
                source="CurOS",
                type_="warning",
                message=f"High Core temperature detected: {temp:.1f}°C! Danger imminent...",
            )
            with open("overheat_shutdown.log", "a") as f:
                f.write(f"{time.ctime()}: CPU temp {temp:.1f}°C Warning.\n")
    publish_event(
        source="Monitor",
        type_="info",
        message=f"core_temp: {temp:.1f}°C\ndisk_usage: {disk:.1f}%",
    )


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


def worker_alive(i):
    session = f"worker{i}"
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def check_workers():

    for i in range(MAX_WORKERS):
        worker_name = worker_names[i % len(worker_names)]
        if not worker_alive(i):
            publish_event(
                source="Scheduler",
                type_="info",
                message=f"Worker {worker_name} is DEAD",
            )
        else:
            publish_event(
                source="Scheduler",
                type_="info",
                message=f"Worker {worker_name} is still alive ✅",
            )


def cleanup():
    # you will implement this later
    publish_event(source="Scheduler", type_="info", message="Running cleanup...")
    # iterate only over subfolders in out/
    for root, dirs, files in os.walk(OUT_DIR, topdown=False):
        # skip the top-level OUT_DIR itself (important safety rule)
        if root == OUT_DIR:
            continue

        # process yaml files only
        for file in files:
            if not file.endswith(".yaml"):
                continue

            yaml_path = os.path.join(root, file)

            # corresponding video file
            base_name = file[:-5]  # remove ".yaml"
            video_path = os.path.join(root, base_name + ".mp4")

            try:
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                publish_event(
                    source="Scheduler",
                    type_="warning",
                    message=f"Failed reading {yaml_path}: {e}",
                )
                continue

            # extract fields safely
            gif_shown = data.get("gif_shown", False)
            dangerous_muon = data.get("dangerous_muon", True)
            primary_muon_vetoed = data.get("primary_muon_vetoed", False)
            ge_77_captures = data.get("ge_77_captures", 1)
            detected_neutrons = data.get("detected_neutrons", 999)

            # delete if gif has been shown,
            # was not a dangerous muon (or dangerous but was vetoed),
            # had no Ge-77
            # and was not vetoed by us
            # (So we keep all ge-77 but also all wrong ge-77 vetoes)
            should_delete = (
                gif_shown is True
                and (primary_muon_vetoed is True or dangerous_muon is False)
                and ge_77_captures == 0
                and detected_neutrons < 5
            )

            if should_delete:
                # delete yaml
                if os.path.exists(yaml_path):
                    os.remove(yaml_path)

                # delete video
                if os.path.exists(video_path):
                    os.remove(video_path)

                publish_event(
                    source="Scheduler", type_="info", message=f"Deleted: {base_name} 🔫"
                )

        # remove folder if empty
        # try:
        #    if root != OUT_DIR and not os.listdir(root):
        #        os.rmdir(root)
        #        publish_event(
        #            source="Scheduler",
        #            type_="info",
        #            message=f"Removed empty folder: {root}",
        #        )
        # except Exception as e:
        #    publish_event(
        #        source="Scheduler",
        #        type_="warning",
        #        message=f"Could not remove folder {root}: {e}",
        #    )

    publish_event(source="Scheduler", type_="info", message="Cleanup finished")


def main():

    publish_event(source="Scheduler", type_="info", message="Scheduler started")
    last_cleanup_time = time.time()
    while True:
        playlist = load(PLAYLIST_FILE)

        # refill playlist target indirectly via queue size
        queue = load(QUEUE_FILE)

        missing = TARGET_PLAYLIST_SIZE - (len(queue) + len(playlist))

        if missing > 0:
            publish_event(
                source="Scheduler",
                type_="info",
                message=f"Playlist low ({len(playlist)} items). Adding {missing} jobs to queue.",
            )
            add_jobs(missing)

        publish_event(
            source="Scheduler", type_="info", message="Checking worker status..."
        )
        check_workers()

        now = time.time()
        if now - last_cleanup_time >= CLEANUP_INTERVAL:
            cleanup()
            last_cleanup_time = now

        publish_event(source="Scheduler", type_="info", message="Sleeping...")
        time.sleep(15)
        check_system_health()
        time.sleep(15)


if __name__ == "__main__":
    main()

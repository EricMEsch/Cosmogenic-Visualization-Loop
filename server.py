from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO
import yaml
import fcntl
import os
import redis
import json

# -----------------------------
# APP SETUP
# -----------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# ABSOLUTE BASE DIR
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(BASE_DIR, "event_monitor", "frontend")
OUT_DIR = os.path.join(BASE_DIR, "out")

PLAYLIST_FILE = os.path.join(OUT_DIR, "playlist.yaml")
GLOBAL_METADATA_FILE = os.path.join(OUT_DIR, "global_metadata.yaml")

# Keep track of the last played video, to update metadata only after the next has been requested
last_served_item = None
# -----------------------------
# REDIS SETUP
# -----------------------------
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_CHANNEL = "events"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def update_metadata_played_field(metadata_path):
    with open(metadata_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            data = yaml.safe_load(f) or []

            if not data:
                return

            data["gif_shown"] = True

            f.seek(0)
            f.truncate()
            yaml.safe_dump(data, f, sort_keys=False)

            return

        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# -----------------------------
# FRONTEND ROUTES
# -----------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/app.js")
def js():
    return send_from_directory(FRONTEND_DIR, "app.js")


@app.route("/style.css")
def css():
    return send_from_directory(FRONTEND_DIR, "style.css")


# -----------------------------
# API
# -----------------------------
@app.route("/playlist")
def playlist():
    with open(PLAYLIST_FILE, "r") as f:
        data = yaml.safe_load(f)
    return jsonify(data)


@app.route("/global_metadata")
def global_metadata():
    with open(GLOBAL_METADATA_FILE, "r") as f:
        data = yaml.safe_load(f)
    return jsonify(data)


@app.route("/playlist/next")
def playlist_next():
    global last_served_item
    with open(PLAYLIST_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            data = yaml.safe_load(f) or []

            if not data:
                return jsonify({"item": None})

            item = data.pop(0)

            f.seek(0)
            f.truncate()
            yaml.safe_dump(data, f)

            # Update the metadata of the last served item to mark it as played
            if last_served_item is not None:
                update_metadata_played_field(last_served_item.replace(".mp4", ".yaml"))
            # Set the current item as the last served item for the next request
            last_served_item = item
            return jsonify({"item": item})

        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# -----------------------------
# VIDEO FILES
# -----------------------------
@app.route("/out/<path:filename>")
def out_files(filename):
    return send_from_directory(OUT_DIR, filename)


# -----------------------------
# WEBSOCKET CONNECTION
# -----------------------------
@socketio.on("connect")
def on_connect():
    print("Client connected")


# -----------------------------
# REDIS → SOCKETIO BRIDGE
# -----------------------------
def redis_listener():
    pubsub = r.pubsub()
    pubsub.subscribe(REDIS_CHANNEL)

    print(f"[Redis] Listening on channel: {REDIS_CHANNEL}")

    for msg in pubsub.listen():
        if msg["type"] != "message":
            continue

        try:
            event = json.loads(msg["data"])
        except Exception as e:
            print("Failed to decode event:", e)
            continue

        # Broadcast to all web clients
        socketio.emit("event", event)


# -----------------------------
# START SERVER
# -----------------------------
if __name__ == "__main__":
    socketio.start_background_task(redis_listener)

    socketio.run(app, host="0.0.0.0", port=8000, debug=True)

import json
import redis
from datetime import datetime, timezone

# Connect to Redis (default localhost)
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True, protocol=2)

CHANNEL = "events"


def publish_event(source: str, type_: str, message: str, data=None):
    """
    Publish a structured event to Redis.
    """
    event = {
        "source": source,
        "type": type_,
        "message": message,
        "data": data or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    r.publish(CHANNEL, json.dumps(event))

// -----------------------------
// Socket.IO connection
// -----------------------------
const socket = io();

socket.on("connect", () => {
  log("Connected to server log stream");
});

socket.on("event", (msg) => {
  handleEvent(msg);
});

// -----------------------------
// UI helpers
// -----------------------------
function log(msg) {
  const logBox = document.getElementById("log");

  const line = document.createElement("div");
  line.textContent =
    "[" + new Date().toLocaleTimeString() + "] " + msg;

  logBox.appendChild(line);

  // Scroll to bottom
  logBox.scrollTop = logBox.scrollHeight;
}

function updateSystem(msg) {
  document.getElementById("system").innerText = msg;
}

function updateStatus(msg) {
  document.getElementById("status").innerText = msg;
}

function updateScheduler(msg) {
  const box = document.getElementById("scheduler");

  const line = document.createElement("div");
  line.textContent = msg;

  box.appendChild(line);

  // optional: auto-scroll like log
  box.scrollTop = box.scrollHeight;
}

function updateJeff(msg) {
  const box = document.getElementById("jeff");

  const line = document.createElement("div");
  line.textContent = msg;

  box.appendChild(line);

  // optional: auto-scroll like log
  box.scrollTop = box.scrollHeight;
}

function updateJeb(msg) {
  const box = document.getElementById("jeb");

  const line = document.createElement("div");
  line.textContent = msg;

  box.appendChild(line);

  // optional: auto-scroll like log
  box.scrollTop = box.scrollHeight;
}

function updateJenny(msg) {
  const box = document.getElementById("jenny");

  const line = document.createElement("div");
  line.textContent = msg;

  box.appendChild(line);

  // optional: auto-scroll like log
  box.scrollTop = box.scrollHeight;
}

function updateEventPopup(msg_green, msg_inner) {
  document.getElementById("event-title").innerText = msg_green;
  document.getElementById("event-meta").innerText = msg_inner;
}

function updateGlobalMetadata(globalMetadata) {
  const startTime = new Date(globalMetadata.start_time);
  const pad = (n) => String(n).padStart(2, "0");

  const startTimeText =
    `${startTime.getFullYear()}-${pad(startTime.getMonth() + 1)}-${pad(startTime.getDate())} ` +
    `${pad(startTime.getHours())}:${pad(startTime.getMinutes())}:${pad(startTime.getSeconds())}`;
  // simulated time in hours
  const simulatedHours = globalMetadata.muons_sampled
    ? globalMetadata.muons_sampled / 504
    : null; // 504 muons per hour is
  // convert hours → ms
  const simulatedMs = simulatedHours !== null
    ? simulatedHours * 60 * 60 * 1000
    : null;
  let endTimeText = "—";

  if (simulatedMs !== null && !isNaN(startTime)) {
    const endTime = new Date(startTime.getTime() + simulatedMs);
    endTimeText =
    `${endTime.getFullYear()}-${pad(endTime.getMonth() + 1)}-${pad(endTime.getDate())} ` +
    `${pad(endTime.getHours())}:${pad(endTime.getMinutes())}:${pad(endTime.getSeconds())}`;
  }


  const summaryTitle = document.getElementById("summary-title");
  summaryTitle.textContent = `Start time of simulation:\n${startTimeText ?? "-"}`;
  const summaryContent = document.getElementById("summary-content");
  summaryContent.textContent =
  `  events_simulated: ${globalMetadata.events_simulated ?? "—"}
  muons_sampled: ${globalMetadata.muons_sampled ?? "—"}
  muon_triggers: ${globalMetadata.muons_vetoed ?? "—"}
  dangerous_muons: ${globalMetadata.dangerous_muons ?? "—"}
  dangerous_muons_vetoed: ${globalMetadata.dangerous_muons_vetoed ?? "—"}
  ge_77_triggers: ${globalMetadata.ge_77_vetoed ?? "—"}
  ge_77_creating_muons: ${globalMetadata.ge_77_creating_muons ?? "—"}
  ge_77_creating_muons_vetoed: ${globalMetadata.ge_77_creating_muons_vetoed ?? "—"}
  total_detected_neutrons: ${globalMetadata.total_detected_neutrons ?? "—"}
  total_captured_neutrons: ${globalMetadata.total_captured_neutrons ?? "—"}
  simulated_time: ${simulatedHours !== null ? simulatedHours.toFixed(2) + " hours" : "—"}
`;

  const summarySimTitle = document.getElementById("summary-sim-title");
  summarySimTitle.textContent = `Current time in simulation:\n${endTimeText ?? "-"}`;
}

function waitForBuffer(video, targetSeconds = 3) {
  return new Promise((resolve) => {
    const check = () => {
      if (video.buffered.length > 0) {
        const bufferedEnd = video.buffered.end(video.buffered.length - 1);
        const current = video.currentTime;

        if (bufferedEnd - current >= targetSeconds) {
          resolve();
          return;
        }
      }

      requestAnimationFrame(check);
    };

    check();
  });
}

function handleEvent(msg) {
  switch (msg.source) {
    case "Scheduler":
      updateScheduler(`[${msg.source}] ${msg.message}`);
      break;
    case "Jeff":
      updateJeff(`[${msg.source}] ${msg.message}`);
      break;
    case "Jeb":
      updateJeb(`[${msg.source}] ${msg.message}`);
      break;
    case "Jenny":
      updateJenny(`[${msg.source}] ${msg.message}`);
      break;
    default:
      log(`[${msg.source}] ${msg.message}`);
  }
}

async function fetchNextAndPlay() {
  try {
    const res = await fetch("/playlist/next");
    const data = await res.json();

    const globalRes = await fetch("/global_metadata");
    const globalMetadata = await globalRes.json();
    updateGlobalMetadata(globalMetadata);

    if (!data.item) {
      updateStatus("IDLE");
      updateEventPopup("Waiting for new events...", "");

      // retry after delay instead of stopping
      setTimeout(fetchNextAndPlay, 2000);
      return;
    }

    playEvent(data.item);
  } catch (err) {
    log("Fetch error: " + err);

    updateStatus("ERROR");

    // retry even on network failure (with backoff optional)
    setTimeout(fetchNextAndPlay, 3000);
  }
}

// -----------------------------
// Core: play event
// -----------------------------
function playEvent(path) {
  const player = document.getElementById("player");

  let eventName = path.split("/").pop();
  eventName = eventName.replace(".mp4", "");

  const parts = eventName.split("_");
  eventName = `${parts[0]}_${parts[2]}.${parts[1]}`;

  document.getElementById("event-id").innerText = eventName;

  updateEventPopup(eventName, "Loading metadata...");

  player.pause();
  player.removeAttribute("src");
  player.load();
  player.src = path + "?t=" + Date.now();

  player.oncanplay = async () => {
    log("Buffering event: " + eventName);

    // wait until we have at least x seconds buffered
    await waitForBuffer(player, 10);

    log("Playing event: " + eventName);
    updateStatus("PLAYING");
    player.play();
  };

  updateEventPopup(eventName, "Showing event visualization...");

  player.onerror = () => {
    log("ERROR loading: " + path);
    updateStatus("VIDEO ERROR");
  };

  player.onended = () => {
    fetchNextAndPlay();
  };
}

// -----------------------------
// Heartbeat (UI liveliness)
// -----------------------------
function heartbeat() {
  updateSystem("LIVE • CurOS version 0.1 • " + new Date().toLocaleTimeString());
}

// -----------------------------
// Main loop
// -----------------------------
setInterval(heartbeat, 1000);

// initial load
fetchNextAndPlay();
heartbeat();

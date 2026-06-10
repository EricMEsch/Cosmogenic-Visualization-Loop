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
  line.textContent =
    "[" + new Date().toLocaleTimeString() + "] " + msg;

  box.appendChild(line);

  // optional: auto-scroll like log
  box.scrollTop = box.scrollHeight;
}

function updateJeff(msg) {
  const box = document.getElementById("jeff");

  const line = document.createElement("div");
  line.textContent =
    "[" + new Date().toLocaleTimeString() + "] " + msg;

  box.appendChild(line);

  // optional: auto-scroll like log
  box.scrollTop = box.scrollHeight;
}

function updateJeb(msg) {
  const box = document.getElementById("jeb");

  const line = document.createElement("div");
  line.textContent =
    "[" + new Date().toLocaleTimeString() + "] " + msg;

  box.appendChild(line);

  // optional: auto-scroll like log
  box.scrollTop = box.scrollHeight;
}

function updateJenny(msg) {
  const box = document.getElementById("jenny");

  const line = document.createElement("div");
  line.textContent =
    "[" + new Date().toLocaleTimeString() + "] " + msg;

  box.appendChild(line);

  // optional: auto-scroll like log
  box.scrollTop = box.scrollHeight;
}

function updateCurOS(msg) {
  const box = document.getElementById("curos");

  const line = document.createElement("div");
  line.textContent =
    "[" + new Date().toLocaleTimeString() + "] " + msg;

  box.appendChild(line);

  // optional: auto-scroll like log
  box.scrollTop = box.scrollHeight;
}

function updateMonitor(msg) {
  const box = document.getElementById("hardware");
  box.textContent = msg;
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
    case "CurOS":
      updateCurOS(`[${msg.source}] ${msg.message}`);
      break;
    case "Monitor":
      updateMonitor(`${msg.message}`);
    default:
      log(`[${msg.source}] ${msg.message}`);
  }
}

// -----------------------------
// Event overlay
// -----------------------------

function showIdleOverlay() {
  overlay.classList.remove("disappear");
  overlay.classList.add("show");
  overlay.classList.remove("hidden");

  const seconds = ((Date.now() - lastMuonTime) / 1000).toFixed(0);

  stateEl.textContent = "Waiting for next muon...";
  metaEl.textContent = `time since last muon: ${seconds}s`;
}

function showNewEventOverlay(callback) {
  overlay.classList.remove("hidden", "disappear");
  overlay.classList.add("show");

  stateEl.textContent = "New muon event detected";
  metaEl.textContent = "Starting visualization...";
  setTimeout(() => {
    triggerGlitch(overlay);
    overlay.classList.add("disappear");

    setTimeout(() => {
      overlay.classList.remove("show");
      overlay.classList.add("hidden");

      if (callback) callback();
    }, 1000); // must match CSS transition duration
  }, 4000);
}

function triggerGlitch(el) {
  el.classList.remove("glitch"); // reset

  // force reflow so animation restarts
  void el.offsetWidth;

  el.classList.add("glitch");
}

// -----------------------------
// Core: play event
// -----------------------------

function idleplay(time) {
  updateStatus("IDLE");
  showIdleOverlay();
  setTimeout(fetchNextAndPlay, time);
}

async function fetchNextAndPlay() {
  try {
    const res = await fetch("/playlist/next");
    const data = await res.json();

    const globalRes = await fetch("/global_metadata");
    const globalMetadata = await globalRes.json();
    updateGlobalMetadata(globalMetadata);

    if (!data.item) {
      idleplay(2000);
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

async function playEvent(path) {
  const player = document.getElementById("player");

  let eventName = path.split("/").pop();
  eventName = eventName.replace(".mp4", "");

  const parts = eventName.split("_");
  eventName = `${parts[0]}_${parts[2]}.${parts[1]}`;

  log("Buffering event: " + eventName);

  // Reset video
  player.pause();
  player.autoplay = false;
  player.removeAttribute("src");
  player.load();

  // Load next video while overlay is visible
  player.src = path + "?t=" + Date.now();

  player.onerror = () => {
    log("ERROR loading: " + path);
    updateStatus("VIDEO ERROR");
  };

  await new Promise((resolve) => {
    player.onloadeddata = () => {
      const check = () => {
        if (player.readyState >= 4) {
          resolve();
        } else {
          requestAnimationFrame(check);
        }
      };
      check();
    };
  });

  // Overlay animation
  showNewEventOverlay(async () => {
    log("Playing event: " + eventName);
    updateStatus("PLAYING");

    // Fade video in
    player.classList.add("visible");
    triggerGlitch(player);
    setTimeout(async () => {
      try {
        await player.play();
      } catch (err) {
        console.error(err);
      }
    }, 1000);
  });

  player.onended = () => {
    lastMuonTime = Date.now();
    triggerGlitch(player);
    player.classList.remove("visible");

    setTimeout(() => {
      idleplay(10000);
    }, 1400);
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

let lastMuonTime = Date.now();
const overlay = document.getElementById("event-overlay");
const stateEl = document.getElementById("event-state");
const metaEl = document.getElementById("event-meta");
setInterval(heartbeat, 1000);
// Update idle overlay timer every 1s if visible
setInterval(() => {
  if (overlay.classList.contains("show") && !overlay.classList.contains("disappear") && (stateEl.textContent === "Waiting for next muon...")) {
    const seconds = ((Date.now() - lastMuonTime) / 1000).toFixed(1);
    metaEl.textContent = `time since last muon: ${seconds}s`;
  }
}, 1000);

// initial load
fetchNextAndPlay();
heartbeat();

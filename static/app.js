/* PronounceCoach frontend.
 * All state lives in this tab; nothing is persisted anywhere.
 * Duration limits (30-45s target, 45s hard cap) are enforced here
 * AND re-verified server-side from the decoded audio duration.
 */

const MAX_SEC = 45;
const MIN_SEC = 3;

const $ = (id) => document.getElementById(id);
const cards = ["consent-card", "capture-card", "loading-card", "results-card"];
const show = (id) => cards.forEach((c) => $(c).classList.toggle("hidden", c !== id));

let currentBlob = null;
let currentName = "recording.webm";

/* ---------- consent gate ---------- */
$("consent-check").addEventListener("change", (e) => ($("consent-btn").disabled = !e.target.checked));
$("consent-btn").addEventListener("click", () => show("capture-card"));

/* ---------- recording ---------- */
let mediaRecorder = null;
let chunks = [];
let timerInterval = null;
let startTime = 0;
let audioCtx = null, analyser = null, levelRAF = null, stream = null;

function fmt(s) { return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }

function stopLevelMeter() {
  cancelAnimationFrame(levelRAF);
  if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
  $("level-bar").style.width = "0";
}

function startLevelMeter(s) {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  audioCtx.createMediaStreamSource(s).connect(analyser);
  const buf = new Uint8Array(analyser.frequencyBinCount);
  (function loop() {
    analyser.getByteTimeDomainData(buf);
    let peak = 0;
    for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
    $("level-bar").style.width = `${Math.min(100, (peak / 128) * 160)}%`;
    levelRAF = requestAnimationFrame(loop);
  })();
}

async function startRecording() {
  hideError();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    return showError("Microphone access was denied. You can upload a file instead.");
  }
  chunks = [];
  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
  mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  mediaRecorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    stopLevelMeter();
    const elapsed = (Date.now() - startTime) / 1000;
    clearInterval(timerInterval);
    $("record-btn").classList.remove("recording");
    $("record-btn").textContent = "● Start recording";
    if (elapsed < MIN_SEC) return showError("That was too short — aim for 30-45 seconds.");
    currentBlob = new Blob(chunks, { type: mime || "audio/webm" });
    currentName = "recording.webm";
    showPreview();
  };
  mediaRecorder.start();
  startLevelMeter(stream);
  startTime = Date.now();
  $("record-btn").classList.add("recording");
  $("record-btn").textContent = "■ Stop recording";
  $("preview").classList.add("hidden");
  timerInterval = setInterval(() => {
    const s = (Date.now() - startTime) / 1000;
    $("timer").firstChild.textContent = fmt(s) + " ";
    if (s >= MAX_SEC) mediaRecorder.stop(); // hard cap
  }, 200);
}

$("record-btn").addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
  else startRecording();
});

/* ---------- file upload ---------- */
$("upload-btn").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", async (e) => {
  hideError();
  const file = e.target.files[0];
  if (!file) return;
  const dur = await audioDuration(file).catch(() => null);
  if (dur !== null && dur > MAX_SEC + 1) {
    e.target.value = "";
    return showError(`That file is ${Math.round(dur)}s long — the limit is 45 seconds.`);
  }
  if (dur !== null && dur < MIN_SEC) {
    e.target.value = "";
    return showError("That file is too short — aim for 30-45 seconds.");
  }
  currentBlob = file;
  currentName = file.name;
  showPreview();
});

function audioDuration(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const a = new Audio(url);
    a.addEventListener("loadedmetadata", () => { URL.revokeObjectURL(url); resolve(a.duration); });
    a.addEventListener("error", () => { URL.revokeObjectURL(url); reject(); });
  });
}

/* ---------- preview + analyze ---------- */
function showPreview() {
  $("player").src = URL.createObjectURL(currentBlob);
  $("preview").classList.remove("hidden");
}

$("discard-btn").addEventListener("click", resetCapture);

function resetCapture() {
  currentBlob = null;
  $("file-input").value = "";
  $("player").removeAttribute("src");
  $("preview").classList.add("hidden");
  $("timer").firstChild.textContent = "0:00 ";
  hideError();
}

function showError(msg) { const el = $("capture-error"); el.textContent = msg; el.classList.remove("hidden"); }
function hideError() { $("capture-error").classList.add("hidden"); }

$("analyze-btn").addEventListener("click", async () => {
  if (!currentBlob) return;
  show("loading-card");
  const phrases = ["Transcribing your speech…", "Measuring acoustic confidence…", "Getting AI feedback on each word…"];
  let pi = 0;
  const phraseTimer = setInterval(() => { pi = Math.min(pi + 1, phrases.length - 1); $("loading-text").textContent = phrases[pi]; }, 3500);

  const form = new FormData();
  form.append("audio", currentBlob, currentName);
  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: form });
    const body = await resp.json();
    clearInterval(phraseTimer);
    if (!resp.ok) {
      show("capture-card");
      return showError(body.detail || "Something went wrong. Please try again.");
    }
    renderResults(body);
  } catch {
    clearInterval(phraseTimer);
    show("capture-card");
    showError("Network error — please try again.");
  }
});

/* ---------- results ---------- */
const CIRC = 2 * Math.PI * 52;

function scoreColor(v) { return v >= 75 ? "var(--ok)" : v >= 50 ? "var(--warn)" : "var(--bad)"; }
function sevClass(cat) { return cat === "attention" || cat === "hesitation" ? "warn" : "bad"; }

function renderResults(r) {
  show("results-card");

  const ring = $("ring-fg");
  ring.style.stroke = scoreColor(r.overall);
  requestAnimationFrame(() => (ring.style.strokeDashoffset = CIRC * (1 - r.overall / 100)));
  $("score-value").textContent = r.overall;

  for (const k of ["clarity", "accuracy", "fluency"]) {
    const v = r.subscores[k];
    $(`val-${k}`).textContent = v;
    const bar = $(`bar-${k}`);
    bar.style.background = scoreColor(v);
    requestAnimationFrame(() => (bar.style.width = v + "%"));
  }

  $("summary").textContent = r.summary ||
    (r.overall >= 75
      ? "Solid, intelligible speech overall. Review the highlighted words below to polish the rough edges."
      : "Understandable, but several words were unclear. Work through the highlights below, then re-record.");

  const issueByIndex = Object.fromEntries(r.issues.map((i) => [i.index, i]));
  $("transcript").innerHTML = r.words
    .map((w) => {
      const issue = issueByIndex[w.index];
      if (!issue) return `<span class="w">${esc(w.word)}</span>`;
      return `<span class="w ${sevClass(issue.category)}" tabindex="0">${esc(w.word)}<span class="tip"><strong>${esc(issue.category)}</strong>: ${esc(issue.note || "")}</span></span>`;
    })
    .join(" ");

  const list = $("issues-list");
  if (r.issues.length) {
    list.innerHTML = r.issues
      .map((i) => `<li><span class="cat ${esc(i.category)}">${esc(i.category)}</span><span class="word-label">${esc(i.word)}</span><span class="time">at ${i.start}s</span><br>${esc(i.note || "")}</li>`)
      .join("");
    $("issues-block").classList.remove("hidden");
  } else {
    list.innerHTML = "<li>No specific problems detected — nice work! 🎉</li>";
  }

  if (r.tips && r.tips.length) {
    $("tips-list").innerHTML = r.tips.map((t) => `<li>💡 ${esc(t)}</li>`).join("");
    $("tips-block").classList.remove("hidden");
  } else {
    $("tips-block").classList.add("hidden");
  }
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

$("again-btn").addEventListener("click", () => { resetCapture(); show("capture-card"); });

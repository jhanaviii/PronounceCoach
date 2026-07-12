"""PronounceCoach — pronunciation scoring for English speech.

DPDP-by-design notes (see ARCHITECTURE.md for the full posture):
- Audio is processed entirely in memory; it is never written to disk,
  never persisted to any database, and discarded when the request ends.
- The only third-party transfers are the STT/LLM API calls needed to
  deliver the service the user consented to.
- Logs contain request metadata only — never transcripts or audio.
"""

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .llm import llm_review
from .scoring import build_acoustic_report, merge_llm_verdicts
from .stt import STTError, transcribe

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # ~45s is well under this for any sane codec
MAX_DURATION_SEC = 46.0             # assessment cap: 45s (+1s codec tolerance)
MIN_DURATION_SEC = 3.0
MIN_WORDS = 5

ALLOWED_TYPES = ("audio/", "video/webm", "application/octet-stream")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pronouncecoach")

app = FastAPI(title="PronounceCoach", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    request_id = uuid.uuid4().hex[:8]
    t0 = time.monotonic()

    content_type = (audio.content_type or "").lower()
    if content_type and not content_type.startswith(ALLOWED_TYPES):
        raise HTTPException(415, "Please upload an audio file.")

    data = await audio.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large. Recordings must be 45 seconds or less.")
    if len(data) < 1000:
        raise HTTPException(400, "That file looks empty. Please record or upload 30-45 seconds of speech.")

    try:
        stt = await transcribe(data, audio.filename, content_type)
    except STTError as e:
        log.error("req=%s stt_error=%s", request_id, e)
        raise HTTPException(502, "Transcription service is unavailable right now. Please try again.")
    finally:
        del data  # release the buffer promptly; audio is never stored

    duration = float(stt.get("duration") or 0.0)
    if duration > MAX_DURATION_SEC:
        raise HTTPException(400, f"Recording is {duration:.0f}s — the limit is 45 seconds. Please trim it.")
    if duration < MIN_DURATION_SEC or len(stt.get("words") or []) < MIN_WORDS:
        raise HTTPException(400, "We couldn't detect enough English speech. Aim for 30-45 seconds of clear talking.")

    report = build_acoustic_report(stt)
    llm = await llm_review(report)
    result = merge_llm_verdicts(report, llm)

    # Metadata only — no transcript, no audio, no user identifiers.
    log.info(
        "req=%s duration=%.1fs words=%d score=%d llm_judge=%s took=%.1fs",
        request_id, duration, len(result["words"]), result["overall"],
        result["llm_judge"], time.monotonic() - t0,
    )
    return result


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

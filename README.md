# 🎙️ PronounceCoach

A web app that scores English pronunciation from a 30–45 second audio clip and
highlights exactly which words went wrong — built as the Livo AI SWE assessment.

**Live demo:** _add your Render URL here after deploying (see below)_

## What it does

1. Record in the browser (auto-stops at 45s) or upload an audio file.
2. Whisper `large-v3` transcribes with word timestamps and decoder log-probabilities.
3. A scoring engine converts acoustic confidence + fluency (speaking rate, pauses)
   into an overall score and Clarity / Accuracy / Fluency sub-scores.
4. An LLM judge (Claude, with Llama fallback) reviews the evidence and produces
   per-word verdicts — *mispronounced*, *unclear*, *hesitation* — with specific,
   learner-friendly fixes and practice tips.
5. The transcript renders with color-coded word highlights and tooltips.

Full design rationale, scoring math, and DPDP compliance posture: [ARCHITECTURE.md](ARCHITECTURE.md).

## Privacy (DPDP Act 2023)

Audio is processed **entirely in memory** and discarded when the request completes.
No database, no file storage, no accounts, no cookies. Explicit consent is collected
before any recording. Logs contain metadata only (duration, score, latency) — never
audio or transcripts.

## Deploy (Render, ~3 minutes)

1. Get a free API key at [console.groq.com](https://console.groq.com) (powers STT
   and the fallback LLM judge). Optionally also an Anthropic key for Claude feedback.
2. On [Render](https://render.com): **New → Blueprint**, point it at this repo —
   `render.yaml` configures everything.
3. When prompted, paste `GROQ_API_KEY` (and optionally `ANTHROPIC_API_KEY`).
4. Deploy. Health check: `https://<your-app>.onrender.com/api/health`.

> Free-tier note: Render free instances sleep after inactivity; the first request
> after a while takes ~30s to cold-start. Hit the URL once before sharing it.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --reload
# open http://localhost:8000
```

Docker: `docker build -t pronouncecoach . && docker run -p 8000:8000 -e GROQ_API_KEY=... pronouncecoach`

## Tests

```bash
pip install pytest && python -m pytest tests/ -v
```

Covers the scoring engine: score bounds, flagging behavior on clean vs. degraded
input, LLM-merge overrides, and adjustment clamping.

## Project layout

```
app/
  main.py      FastAPI app: upload validation, duration enforcement, orchestration
  stt.py       Whisper (Groq) client — transcript + word timestamps + log-probs
  scoring.py   Acoustic + fluency scoring, heuristic word flagging
  llm.py       LLM judge with provider fallback (Claude → Llama → heuristics-only)
static/        Vanilla JS frontend: recorder, consent gate, results UI
tests/         Unit tests for the scoring engine
render.yaml    One-click Render blueprint
Dockerfile     Portable alternative (Fly.io, Railway, anywhere)
```

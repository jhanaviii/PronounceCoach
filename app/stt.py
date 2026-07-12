"""Speech-to-text via Groq-hosted Whisper large-v3.

Returns the raw verbose_json payload: full transcript, per-segment
avg_logprob / no_speech_prob (our acoustic confidence signal), and
word-level timestamps (our alignment for highlighting).
"""

import os

import httpx

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_MODEL = "whisper-large-v3"


class STTError(Exception):
    pass


async def transcribe(audio_bytes: bytes, filename: str, content_type: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise STTError("GROQ_API_KEY is not configured on the server.")

    files = {"file": (filename or "audio.webm", audio_bytes, content_type or "application/octet-stream")}
    data = {
        "model": STT_MODEL,
        "language": "en",
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
        "temperature": "0",
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files=files,
        )
    if resp.status_code != 200:
        raise STTError(f"Transcription failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()

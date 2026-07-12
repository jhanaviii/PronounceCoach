"""LLM judge: turns acoustic evidence into learner-friendly feedback.

Provider chain (first available wins):
  1. Anthropic Claude (ANTHROPIC_API_KEY) — best judgment quality
  2. Groq Llama 3.3 70B (GROQ_API_KEY)   — free tier, same key as STT
  3. None — the app degrades gracefully to heuristics-only

The judge sees the transcript with per-word acoustic confidence and
returns per-word verdicts (mispronounced / unclear / hesitation / ok),
a summary, and practice tips, as strict JSON.
"""

import json
import os

import httpx

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"

SYSTEM = """You are a pronunciation coach analyzing evidence from a speech recognizer.
You receive an English transcript as numbered words, each with an acoustic confidence
(0-1, from Whisper decoder log-probabilities) and heuristic flags (unclear/attention/hesitation).

Low confidence means the audio was ambiguous to the recognizer — usually unclear or
mispronounced speech. Also use the text itself: recognizer artifacts like wrong homophones,
oddly split words, or nonsense words in otherwise fluent context usually indicate
mispronunciation of the intended word.

Return ONLY a JSON object, no markdown fences, with this shape:
{
  "words": [{"index": <int>, "verdict": "mispronounced"|"unclear"|"hesitation"|"ok", "note": "<short, specific, learner-friendly tip>"}],
  "summary": "<2-3 sentence overall assessment addressed to the learner>",
  "tips": ["<up to 3 concrete practice tips>"],
  "score_adjustment": <int between -10 and 10, your correction to the acoustic score>
}
Only include words with a non-ok verdict in "words". Be selective: flag genuine problems,
not every imperfection. Notes must say WHAT went wrong and HOW to fix it
(e.g. "sounded like 'sheep' — lengthen the vowel in 'ship' /ɪ/ vs /iː/")."""


def _payload_for_judge(report: dict) -> str:
    lines = []
    for w in report["words"]:
        flag = report["heuristic_flags"].get(w["index"], "")
        lines.append(f'{w["index"]}: "{w["word"]}" conf={w["confidence"]}{" [" + flag + "]" if flag else ""}')
    return (
        f'Transcript: {report["transcript"]}\n'
        f'Duration: {report["duration"]}s, {len(report["words"])} words\n'
        f'Sub-scores: {report["subscores"]}\n'
        f'Words:\n' + "\n".join(lines)
    )


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


async def _anthropic(prompt: str, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            ANTHROPIC_URL,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1500,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


async def _groq(prompt: str, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_CHAT_MODEL,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            },
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def llm_review(report: dict) -> dict | None:
    """Best-effort LLM pass. Returns parsed verdicts or None (never raises)."""
    prompt = _payload_for_judge(report)
    for provider, key in (
        (_anthropic, os.environ.get("ANTHROPIC_API_KEY")),
        (_groq, os.environ.get("GROQ_API_KEY")),
    ):
        if not key:
            continue
        try:
            return _parse_json(await provider(prompt, key))
        except Exception:
            continue
    return None

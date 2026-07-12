"""Acoustic + fluency scoring over Whisper verbose_json output.

Signals used (all derived from the STT pass, no second model call):

1. Acoustic confidence — exp(avg_logprob) per segment. Whisper's decoder
   log-probability drops when audio is ambiguous to the acoustic model,
   which correlates strongly with unclear or mispronounced speech.
2. no_speech_prob — segments the model suspects aren't speech at all.
3. Fluency — speaking rate vs. a typical conversational band
   (1.8–3.2 words/sec) and long inter-word pauses (> 1.2 s).

Each transcript word gets a confidence inherited from its segment and a
heuristic flag. The LLM judge (llm.py) refines these flags; the final
report merges both.
"""

import math

# Confidence thresholds (on exp(avg_logprob), roughly "P(token) geometric mean")
CONF_LOW = 0.45   # below: likely problem
CONF_MID = 0.62   # below: worth attention
PAUSE_GAP_SEC = 1.2
RATE_LOW, RATE_HIGH = 1.8, 3.2  # words per second


def _segment_confidence(seg: dict) -> float:
    conf = math.exp(min(0.0, seg.get("avg_logprob", -1.0)))
    # Punish segments Whisper thinks may not contain speech.
    conf *= 1.0 - 0.6 * seg.get("no_speech_prob", 0.0)
    return max(0.0, min(1.0, conf))


def _words_with_confidence(stt: dict) -> list[dict]:
    segments = stt.get("segments") or []
    words = stt.get("words") or []
    out = []
    for i, w in enumerate(words):
        mid = (w.get("start", 0.0) + w.get("end", 0.0)) / 2
        conf = 0.5
        for seg in segments:
            if seg.get("start", 0.0) - 0.05 <= mid <= seg.get("end", 0.0) + 0.05:
                conf = _segment_confidence(seg)
                break
        out.append({
            "index": i,
            "word": (w.get("word") or "").strip(),
            "start": round(w.get("start", 0.0), 2),
            "end": round(w.get("end", 0.0), 2),
            "confidence": round(conf, 3),
        })
    return out


def _fluency(words: list[dict], duration: float) -> tuple[float, list[int]]:
    """Returns (fluency score 0..1, indices of words preceded by a long pause)."""
    if not words or duration <= 0:
        return 0.5, []
    rate = len(words) / duration
    if RATE_LOW <= rate <= RATE_HIGH:
        rate_score = 1.0
    else:
        # Linear falloff: 1 wps off the band costs ~0.45
        dist = (RATE_LOW - rate) if rate < RATE_LOW else (rate - RATE_HIGH)
        rate_score = max(0.0, 1.0 - 0.45 * dist)

    pause_indices = [
        words[i]["index"]
        for i in range(1, len(words))
        if words[i]["start"] - words[i - 1]["end"] > PAUSE_GAP_SEC
    ]
    pause_score = max(0.0, 1.0 - 0.12 * len(pause_indices))
    return 0.6 * rate_score + 0.4 * pause_score, pause_indices


def build_acoustic_report(stt: dict) -> dict:
    """Heuristic pass: per-word confidence, candidate flags, sub-scores."""
    words = _words_with_confidence(stt)
    duration = float(stt.get("duration") or (words[-1]["end"] if words else 0.0))
    fluency_score, pause_indices = _fluency(words, duration)

    flags = {}
    for w in words:
        if w["confidence"] < CONF_LOW:
            flags[w["index"]] = "unclear"
        elif w["confidence"] < CONF_MID:
            flags[w["index"]] = "attention"
    for idx in pause_indices:
        flags.setdefault(idx, "hesitation")

    clarity = sum(w["confidence"] for w in words) / len(words) if words else 0.0
    flagged_hard = sum(1 for v in flags.values() if v == "unclear")
    accuracy = 1.0 - (flagged_hard / len(words)) if words else 0.0

    overall = 100 * (0.45 * clarity + 0.30 * accuracy + 0.25 * fluency_score)
    return {
        "transcript": (stt.get("text") or "").strip(),
        "duration": round(duration, 2),
        "words": words,
        "heuristic_flags": flags,  # index -> unclear | attention | hesitation
        "subscores": {
            "clarity": round(100 * clarity),
            "accuracy": round(100 * accuracy),
            "fluency": round(100 * fluency_score),
        },
        "overall": round(overall),
    }


def merge_llm_verdicts(report: dict, llm: dict | None) -> dict:
    """Overlay LLM per-word verdicts on the heuristic flags.

    The LLM can escalate (ok -> mispronounced), refine a heuristic flag
    with a category + note, or clear a false positive (only 'attention'
    level flags — hard acoustic evidence is kept).
    """
    issues = []
    llm_by_index = {}
    if llm:
        for v in llm.get("words", []):
            if isinstance(v.get("index"), int):
                llm_by_index[v["index"]] = v

    for w in report["words"]:
        idx = w["index"]
        heur = report["heuristic_flags"].get(idx)
        verdict = llm_by_index.get(idx)
        category, note = None, None
        if verdict and verdict.get("verdict") not in (None, "ok"):
            category = verdict["verdict"]
            note = verdict.get("note")
        elif heur in ("unclear", "hesitation"):
            category = heur
            note = ("This segment was hard to make out — try saying it more slowly and distinctly."
                    if heur == "unclear" else "Long pause before this word.")
        elif heur == "attention" and not llm:
            category = "attention"
            note = "Slightly unclear to the recognizer."
        if category:
            issues.append({
                "index": idx, "word": w["word"], "start": w["start"], "end": w["end"],
                "category": category, "note": note,
            })

    result = {
        "overall": report["overall"],
        "subscores": report["subscores"],
        "transcript": report["transcript"],
        "duration": report["duration"],
        "words": report["words"],
        "issues": issues,
        "summary": (llm or {}).get("summary"),
        "tips": (llm or {}).get("tips") or [],
        "llm_judge": bool(llm),
    }
    # Let strong LLM signal nudge the score (bounded so acoustics still dominate).
    if llm and isinstance(llm.get("score_adjustment"), (int, float)):
        result["overall"] = max(0, min(100, round(report["overall"] + max(-10, min(10, llm["score_adjustment"])))))
    return result

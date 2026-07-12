"""Unit tests for the scoring engine (pure functions, no network)."""

from app.scoring import build_acoustic_report, merge_llm_verdicts


def make_stt(segments, words, text="hello world this is a test", duration=None):
    return {
        "text": text,
        "duration": duration or (words[-1]["end"] if words else 0),
        "segments": segments,
        "words": words,
    }


def fluent_stt():
    """~2.4 words/sec, high confidence."""
    words = [
        {"word": w, "start": i * 0.4, "end": i * 0.4 + 0.35}
        for i, w in enumerate("hello world this is a clear test of speech scoring".split())
    ]
    segments = [{"start": 0, "end": 4.5, "avg_logprob": -0.15, "no_speech_prob": 0.01}]
    return make_stt(segments, words, duration=4.5)


def poor_stt():
    """Low confidence segment + a long pause."""
    words = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "wrld", "start": 0.5, "end": 0.9},
        {"word": "this", "start": 3.0, "end": 3.3},  # 2.1s pause before
        {"word": "is", "start": 3.4, "end": 3.7},
        {"word": "fine", "start": 3.8, "end": 4.2},
    ]
    segments = [
        {"start": 0, "end": 1.0, "avg_logprob": -1.4, "no_speech_prob": 0.05},
        {"start": 3.0, "end": 4.2, "avg_logprob": -0.2, "no_speech_prob": 0.01},
    ]
    return make_stt(segments, words, duration=8.0)


def test_fluent_speech_scores_high():
    report = build_acoustic_report(fluent_stt())
    assert report["overall"] >= 80
    assert report["subscores"]["clarity"] >= 80
    assert not any(v == "unclear" for v in report["heuristic_flags"].values())


def test_poor_speech_scores_low_and_flags_words():
    report = build_acoustic_report(poor_stt())
    assert report["overall"] < build_acoustic_report(fluent_stt())["overall"] - 20
    assert "unclear" in report["heuristic_flags"].values()
    assert "hesitation" in report["heuristic_flags"].values()


def test_scores_bounded():
    for stt in (fluent_stt(), poor_stt()):
        report = build_acoustic_report(stt)
        assert 0 <= report["overall"] <= 100
        for v in report["subscores"].values():
            assert 0 <= v <= 100


def test_merge_without_llm_uses_heuristics():
    report = build_acoustic_report(poor_stt())
    result = merge_llm_verdicts(report, None)
    assert result["llm_judge"] is False
    assert len(result["issues"]) > 0
    assert all(i["note"] for i in result["issues"])


def test_merge_with_llm_overrides_and_adjusts():
    report = build_acoustic_report(poor_stt())
    llm = {
        "words": [{"index": 1, "verdict": "mispronounced", "note": "sounded like 'wrld' — round the vowel in 'world'"}],
        "summary": "Mostly clear with a few slips.",
        "tips": ["Slow down on multi-syllable words."],
        "score_adjustment": 5,
    }
    result = merge_llm_verdicts(report, llm)
    assert result["llm_judge"] is True
    slip = next(i for i in result["issues"] if i["index"] == 1)
    assert slip["category"] == "mispronounced"
    assert result["overall"] == min(100, report["overall"] + 5)
    assert result["tips"] == llm["tips"]


def test_llm_adjustment_is_clamped():
    report = build_acoustic_report(fluent_stt())
    result = merge_llm_verdicts(report, {"words": [], "score_adjustment": 50})
    assert result["overall"] <= min(100, report["overall"] + 10)


def test_empty_words_handled():
    report = build_acoustic_report({"text": "", "duration": 0, "segments": [], "words": []})
    assert report["overall"] == 0 or isinstance(report["overall"], int)

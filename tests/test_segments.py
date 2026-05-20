from omnivoice_translator.segments import (
    SpeechSegment,
    normalize_asr_segments,
    resolve_segment_overlaps,
)


def test_normalize_asr_segments_uses_timestamped_chunks():
    result = {
        "text": "Hello world",
        "chunks": [
            {"timestamp": (0.2, 1.4), "text": " Hello "},
            {"timestamp": (1.4, None), "text": " world"},
            {"timestamp": (2.0, 2.4), "text": "   "},
        ],
    }

    assert normalize_asr_segments(result, fallback_duration=2.0) == [
        SpeechSegment(start=0.2, end=1.4, text="Hello"),
        SpeechSegment(start=1.4, end=2.0, text="world"),
    ]


def test_normalize_asr_segments_falls_back_to_whole_text():
    result = {"text": "Hello from the full transcript"}

    assert normalize_asr_segments(result, fallback_duration=3.5) == [
        SpeechSegment(start=0.0, end=3.5, text="Hello from the full transcript")
    ]


def test_normalize_asr_segments_resolves_overlapping_chunks():
    result = {
        "chunks": [
            {"timestamp": (0.0, 1.2), "text": "one"},
            {"timestamp": (1.0, 2.0), "text": "two"},
        ],
    }

    segments = normalize_asr_segments(result, fallback_duration=2.0)

    assert segments == [
        SpeechSegment(start=0.0, end=1.1, text="one"),
        SpeechSegment(start=1.1, end=2.0, text="two"),
    ]


def test_resolve_segment_overlaps_drops_tiny_segments_after_adjustment():
    segments = resolve_segment_overlaps(
        [
            SpeechSegment(start=0.0, end=1.0, text="one"),
            SpeechSegment(start=0.1, end=0.12, text="tiny"),
        ]
    )

    assert segments == [SpeechSegment(start=0.0, end=0.55, text="one")]

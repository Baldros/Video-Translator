from omnivoice_translator.segment_planner import merge_speech_segments
from omnivoice_translator.segments import SpeechSegment


def test_merge_speech_segments_combines_short_adjacent_segments():
    chunks = merge_speech_segments(
        [
            SpeechSegment(0.0, 0.8, "hello"),
            SpeechSegment(0.9, 1.6, "there"),
            SpeechSegment(2.4, 3.0, "later"),
        ],
        min_duration=2.0,
        max_duration=4.0,
        max_gap=0.25,
    )

    assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [
        (0.0, 1.6, "hello there"),
        (2.4, 3.0, "later"),
    ]


def test_merge_speech_segments_respects_max_duration():
    chunks = merge_speech_segments(
        [
            SpeechSegment(0.0, 1.8, "one"),
            SpeechSegment(1.9, 3.8, "two"),
        ],
        min_duration=5.0,
        max_duration=3.0,
        max_gap=0.25,
    )

    assert [chunk.text for chunk in chunks] == ["one", "two"]

import numpy as np
import soundfile as sf

from omnivoice_translator.audio_segments import (
    SegmentAudio,
    fit_audio_to_duration,
    prepare_segment_audio,
    render_segment_timeline,
)
from omnivoice_translator.segments import SpeechSegment


def test_fit_audio_to_duration_pads_and_trims():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    padded = fit_audio_to_duration(audio, sample_rate=10, duration=0.5)
    trimmed = fit_audio_to_duration(audio, sample_rate=10, duration=0.2)

    assert np.allclose(padded, [1.0, 2.0, 3.0, 0.0, 0.0])
    assert np.allclose(trimmed, [1.0, 2.0])


def test_fit_audio_to_duration_can_time_stretch():
    audio = np.sin(np.linspace(0, np.pi * 2, 4096, dtype=np.float32))

    fitted = fit_audio_to_duration(
        audio,
        sample_rate=4096,
        duration=0.5,
        strategy="stretch",
        max_stretch_ratio=3.0,
    )

    assert len(fitted) == 2048


def test_prepare_segment_audio_trims_fades_normalizes_and_limits_peak():
    sample_rate = 1000
    audio = np.concatenate(
        [
            np.zeros(120, dtype=np.float32),
            np.full(760, 0.08, dtype=np.float32),
            np.zeros(120, dtype=np.float32),
        ]
    )

    prepared = prepare_segment_audio(
        audio,
        sample_rate=sample_rate,
        trim_silence=True,
        trim_top_db=35.0,
        fade_ms=20.0,
        target_rms=0.2,
        max_gain_db=12.0,
        peak_limit=0.18,
    )

    assert prepared.shape[0] < audio.shape[0]
    assert abs(float(np.max(np.abs(prepared))) - 0.18) < 1e-4
    assert abs(float(prepared[0])) < 1e-5
    assert abs(float(prepared[-1])) < 1e-5


def test_render_segment_timeline_places_audio_on_source_timeline(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "timeline.wav"

    sf.write(first, np.ones(4, dtype=np.float32), 10)
    sf.write(second, np.full(4, 0.5, dtype=np.float32), 10)

    render_segment_timeline(
        [
            SegmentAudio(
                segment=SpeechSegment(start=0.2, end=0.6, text="one"),
                audio_path=first,
            ),
            SegmentAudio(
                segment=SpeechSegment(start=1.0, end=1.2, text="two"),
                audio_path=second,
            ),
        ],
        total_duration=1.5,
        output_path=output,
        condition_audio=False,
    )

    data, sample_rate = sf.read(output)

    assert sample_rate == 10
    assert len(data) == 15
    assert np.allclose(data[:2], 0.0)
    assert np.allclose(data[2:6], 1.0, atol=1e-4)
    assert np.allclose(data[6:10], 0.0)
    assert np.allclose(data[10:12], 0.5, atol=1e-4)
    assert np.allclose(data[12:], 0.0)


def test_render_segment_timeline_trims_overlaps_instead_of_layering(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "timeline.wav"

    sf.write(first, np.ones(6, dtype=np.float32), 10)
    sf.write(second, np.full(5, 0.5, dtype=np.float32), 10)

    render_segment_timeline(
        [
            SegmentAudio(
                segment=SpeechSegment(start=0.0, end=0.6, text="one"),
                audio_path=first,
            ),
            SegmentAudio(
                segment=SpeechSegment(start=0.4, end=0.9, text="two"),
                audio_path=second,
            ),
        ],
        total_duration=1.0,
        output_path=output,
        condition_audio=False,
    )

    data, sample_rate = sf.read(output)

    assert sample_rate == 10
    assert np.allclose(data[:6], 1.0, atol=1e-4)
    assert np.allclose(data[6:9], 0.5, atol=1e-4)
    assert np.allclose(data[9:], 0.0, atol=1e-4)
    assert np.max(data) <= 1.0


def test_render_segment_timeline_can_condition_tts_segments(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "timeline.wav"
    sample_rate = 100

    sf.write(first, np.full(40, 0.02, dtype=np.float32), sample_rate)
    sf.write(second, np.full(40, 0.2, dtype=np.float32), sample_rate)

    render_segment_timeline(
        [
            SegmentAudio(
                segment=SpeechSegment(start=0.0, end=0.4, text="one"),
                audio_path=first,
            ),
            SegmentAudio(
                segment=SpeechSegment(start=0.5, end=0.9, text="two"),
                audio_path=second,
            ),
        ],
        total_duration=1.0,
        output_path=output,
        condition_audio=True,
        trim_silence=False,
        fade_ms=0.0,
        target_rms=0.1,
        max_gain_db=20.0,
        peak_limit=0.95,
    )

    data, sample_rate = sf.read(output, dtype="float32")

    assert sample_rate == 100
    assert np.isclose(np.sqrt(np.mean(data[:40] ** 2)), 0.1, atol=2e-3)
    assert np.isclose(np.sqrt(np.mean(data[50:90] ** 2)), 0.1, atol=2e-3)

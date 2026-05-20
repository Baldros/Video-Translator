import numpy as np
import soundfile as sf

from omnivoice_translator.segment_refinement import refine_segments_with_audio_energy
from omnivoice_translator.segments import SpeechSegment


def test_refine_segments_with_audio_energy_snaps_broad_asr_edges(tmp_path):
    sample_rate = 1000
    audio = np.zeros(2200, dtype=np.float32)
    t = np.arange(1000, dtype=np.float32) / sample_rate
    audio[500:1500] = 0.2 * np.sin(2 * np.pi * 180.0 * t)
    source = tmp_path / "source.wav"
    sf.write(source, audio, sample_rate)

    refined = refine_segments_with_audio_energy(
        [SpeechSegment(start=0.2, end=1.8, text="hello")],
        source,
        max_shift=0.45,
        padding=0.02,
        top_db=35.0,
        frame_length=128,
        hop_length=32,
    )

    assert len(refined) == 1
    assert 0.42 <= refined[0].start <= 0.55
    assert 1.45 <= refined[0].end <= 1.60
    assert refined[0].text == "hello"


def test_refine_segments_with_audio_energy_keeps_segments_when_no_energy_found(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(1000, dtype=np.float32), 1000)
    original = SpeechSegment(start=0.2, end=0.8, text="silence")

    refined = refine_segments_with_audio_energy([original], source)

    assert refined == [original]


def test_refine_segments_with_audio_energy_resolves_expanded_overlaps(tmp_path):
    sample_rate = 1000
    t = np.arange(2600, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 180.0 * t)
    source = tmp_path / "continuous.wav"
    sf.write(source, audio, sample_rate)

    refined = refine_segments_with_audio_energy(
        [
            SpeechSegment(start=0.0, end=1.3, text="first"),
            SpeechSegment(start=1.3, end=2.6, text="second"),
        ],
        source,
        max_shift=0.3,
        padding=0.05,
        top_db=35.0,
        frame_length=128,
        hop_length=32,
    )

    assert len(refined) == 2
    assert refined[0].end <= refined[1].start
    assert refined[0].duration > 0.8
    assert refined[1].duration > 0.8

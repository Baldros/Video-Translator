from pathlib import Path

import numpy as np
import soundfile as sf

from omnivoice_translator.audio_quality import (
    AudioSegmentInterval,
    analyze_audio,
    assess_audio_quality,
    compare_audio,
    compare_audio_prosody,
)


def test_analyze_audio_reports_librosa_spectral_metrics(tmp_path):
    sample_rate = 16000
    audio_path = tmp_path / "tone.wav"
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(audio_path, tone.astype(np.float32), sample_rate)

    metrics = analyze_audio(audio_path, sample_rate=sample_rate)

    assert metrics.duration_s == 1.0
    assert 0.19 < metrics.peak_amplitude < 0.21
    assert metrics.clipping_ratio == 0.0
    assert metrics.mel_band_count == 80
    assert metrics.mel_frame_count > 0
    assert metrics.spectral_centroid_mean_hz > 300.0
    assert metrics.spectral_rolloff_mean_hz > metrics.spectral_centroid_mean_hz


def test_assess_audio_quality_flags_clipping(tmp_path):
    sample_rate = 16000
    audio_path = tmp_path / "clipped.wav"
    clipped = np.ones(sample_rate, dtype=np.float32)
    sf.write(audio_path, clipped, sample_rate)

    metrics = analyze_audio(audio_path, sample_rate=sample_rate)
    issues = assess_audio_quality(metrics)

    assert any("clipping" in issue.lower() for issue in issues)


def test_analyze_audio_surfaces_delayed_copy_pattern(tmp_path):
    sample_rate = 16000
    clean_path = tmp_path / "clean.wav"
    doubled_path = tmp_path / "doubled.wav"
    clean = _speech_like_signal(sample_rate=sample_rate, duration_s=2.0)
    delay = int(0.12 * sample_rate)
    doubled = clean.copy()
    doubled[delay:] += 0.65 * clean[:-delay]
    doubled /= max(1.0, float(np.max(np.abs(doubled))))

    sf.write(clean_path, clean, sample_rate)
    sf.write(doubled_path, doubled, sample_rate)

    clean_metrics = analyze_audio(clean_path, sample_rate=sample_rate)
    doubled_metrics = analyze_audio(doubled_path, sample_rate=sample_rate)

    assert doubled_metrics.delayed_similarity_peak > clean_metrics.delayed_similarity_peak
    assert abs(doubled_metrics.delayed_similarity_lag_s - 0.12) < 0.05


def test_compare_audio_reports_envelope_alignment_and_mel_distance(tmp_path):
    sample_rate = 16000
    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference = _speech_like_signal(sample_rate=sample_rate, duration_s=2.0)
    delay = int(0.1 * sample_rate)
    candidate = np.pad(reference, (delay, 0))[: reference.shape[0]]

    sf.write(reference_path, reference, sample_rate)
    sf.write(candidate_path, candidate, sample_rate)

    comparison = compare_audio(
        reference_path,
        candidate_path,
        sample_rate=sample_rate,
    )

    assert comparison.envelope_correlation > 0.85
    assert abs(comparison.envelope_lag_s - 0.1) < 0.05
    assert comparison.log_mel_distance_db >= 0.0


def test_compare_audio_prosody_scores_segment_timing(tmp_path):
    sample_rate = 16000
    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference = _speech_like_signal(sample_rate=sample_rate, duration_s=2.0)
    candidate = reference.copy()
    candidate[int(1.0 * sample_rate) :] = 0.0

    sf.write(reference_path, reference, sample_rate)
    sf.write(candidate_path, candidate, sample_rate)

    comparison = compare_audio_prosody(
        reference_path,
        candidate_path,
        segments=[AudioSegmentInterval(start=0.0, end=2.0)],
        sample_rate=sample_rate,
    )

    assert comparison.segment_count == 1
    assert comparison.speech_activity_recall < 0.75
    assert comparison.speech_activity_f1 < 0.85
    assert comparison.energy_dtw_distance >= 0.0


def test_compare_audio_prosody_reports_metadata_overlap(tmp_path):
    sample_rate = 16000
    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    metadata_path = tmp_path / "metadata.json"
    audio = _speech_like_signal(sample_rate=sample_rate, duration_s=2.0)

    sf.write(reference_path, audio, sample_rate)
    sf.write(candidate_path, audio, sample_rate)
    metadata_path.write_text(
        """
        {
          "segments": [
            {"start": 0.0, "end": 1.2},
            {"start": 1.0, "end": 2.0}
          ]
        }
        """,
        encoding="utf-8",
    )

    comparison = compare_audio_prosody(
        reference_path,
        candidate_path,
        metadata_path=metadata_path,
        sample_rate=sample_rate,
    )

    assert comparison.metadata_overlap_s == 0.2
    assert comparison.metadata_max_overlap_s == 0.2


def _speech_like_signal(*, sample_rate: int, duration_s: float) -> np.ndarray:
    rng = np.random.default_rng(123)
    samples = int(sample_rate * duration_s)
    t = np.arange(samples) / sample_rate
    carrier = (
        np.sin(2 * np.pi * 180.0 * t)
        + 0.5 * np.sin(2 * np.pi * 360.0 * t)
        + 0.25 * np.sin(2 * np.pi * 720.0 * t)
    )
    control_points = rng.uniform(0.05, 1.0, size=18)
    control_x = np.linspace(0, samples - 1, num=control_points.size)
    envelope = np.interp(np.arange(samples), control_x, control_points)
    envelope = np.maximum(envelope, 0.03)
    return (0.25 * envelope * carrier).astype(np.float32)

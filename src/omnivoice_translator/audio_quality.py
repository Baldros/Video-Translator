from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioQualityMetrics:
    path: str
    sample_rate: int
    duration_s: float
    peak_amplitude: float
    clipping_ratio: float
    rms_mean: float
    rms_p95: float
    silence_ratio: float
    spectral_centroid_mean_hz: float
    spectral_bandwidth_mean_hz: float
    spectral_flatness_mean: float
    spectral_rolloff_mean_hz: float
    zero_crossing_rate_mean: float
    delayed_similarity_peak: float
    delayed_similarity_lag_s: float
    mel_band_count: int
    mel_frame_count: int


@dataclass(frozen=True)
class AudioComparisonMetrics:
    reference_path: str
    candidate_path: str
    duration_delta_s: float
    envelope_correlation: float
    envelope_lag_s: float
    log_mel_distance_db: float


@dataclass(frozen=True)
class AudioSegmentInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SegmentProsodyMetrics:
    index: int
    start: float
    end: float
    duration_s: float
    reference_active_ratio: float
    candidate_active_ratio: float
    speech_activity_precision: float
    speech_activity_recall: float
    speech_activity_f1: float
    energy_dtw_distance: float
    pitch_rmse_semitones: float | None
    pitch_correlation: float | None


@dataclass(frozen=True)
class ProsodyComparisonMetrics:
    reference_path: str
    candidate_path: str
    segment_count: int
    duration_delta_s: float
    metadata_overlap_s: float
    metadata_max_overlap_s: float
    speech_activity_precision: float
    speech_activity_recall: float
    speech_activity_f1: float
    energy_dtw_distance: float
    pitch_rmse_semitones: float | None
    pitch_correlation: float | None
    worst_segment_indices: list[int]
    segments: list[SegmentProsodyMetrics]


@dataclass(frozen=True)
class AudioQualityThresholds:
    peak_warning: float = 0.98
    clipping_ratio_warning: float = 0.0001
    rms_low_warning: float = 0.003
    silence_ratio_warning: float = 0.55
    delayed_similarity_warning: float = 0.90
    duration_delta_warning_s: float = 0.25
    envelope_correlation_warning: float = 0.35
    metadata_overlap_warning_s: float = 0.05
    speech_activity_f1_warning: float = 0.70
    speech_activity_recall_warning: float = 0.75
    segment_speech_activity_f1_warning: float = 0.60
    segment_speech_activity_recall_warning: float = 0.65
    energy_dtw_warning: float = 0.45


@dataclass(frozen=True)
class AudioQualityReport:
    candidate: AudioQualityMetrics
    comparison: AudioComparisonMetrics | None
    prosody_comparison: ProsodyComparisonMetrics | None
    issues: list[str]


def analyze_audio(
    path: str | Path,
    *,
    sample_rate: int = 24000,
    frame_length: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
) -> AudioQualityMetrics:
    audio_path = Path(path)
    y, sr = _load_audio(audio_path, sample_rate=sample_rate)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    stft = np.abs(
        librosa.stft(
            y,
            n_fft=frame_length,
            hop_length=hop_length,
            center=True,
        )
    )
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    delayed_peak, delayed_lag_s = _delayed_similarity(
        rms,
        sample_rate=sr,
        hop_length=hop_length,
    )

    return AudioQualityMetrics(
        path=str(audio_path),
        sample_rate=sr,
        duration_s=_rounded(y.size / sr),
        peak_amplitude=_rounded(float(np.max(np.abs(y)))),
        clipping_ratio=_rounded(float(np.mean(np.abs(y) >= 0.999))),
        rms_mean=_rounded(float(np.mean(rms))),
        rms_p95=_rounded(float(np.percentile(rms, 95))),
        silence_ratio=_rounded(float(np.mean(rms < 1e-4))),
        spectral_centroid_mean_hz=_rounded(
            float(np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr)))
        ),
        spectral_bandwidth_mean_hz=_rounded(
            float(np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr)))
        ),
        spectral_flatness_mean=_rounded(
            float(np.mean(librosa.feature.spectral_flatness(S=stft)))
        ),
        spectral_rolloff_mean_hz=_rounded(
            float(np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr)))
        ),
        zero_crossing_rate_mean=_rounded(
            float(
                np.mean(
                    librosa.feature.zero_crossing_rate(
                        y,
                        frame_length=frame_length,
                        hop_length=hop_length,
                    )
                )
            )
        ),
        delayed_similarity_peak=_rounded(delayed_peak),
        delayed_similarity_lag_s=_rounded(delayed_lag_s),
        mel_band_count=int(mel.shape[0]),
        mel_frame_count=int(mel.shape[1]),
    )


def compare_audio(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    sample_rate: int = 24000,
    frame_length: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
    max_lag_s: float = 3.0,
) -> AudioComparisonMetrics:
    reference = Path(reference_path)
    candidate = Path(candidate_path)
    ref_y, sr = _load_audio(reference, sample_rate=sample_rate)
    cand_y, _ = _load_audio(candidate, sample_rate=sample_rate)
    if ref_y.size == 0 or cand_y.size == 0:
        raise ValueError("Reference and candidate audio must be non-empty.")

    ref_env = _rms_envelope(ref_y, frame_length=frame_length, hop_length=hop_length)
    cand_env = _rms_envelope(cand_y, frame_length=frame_length, hop_length=hop_length)
    envelope_corr, lag_frames = _best_lagged_correlation(
        ref_env,
        cand_env,
        max_lag=math.ceil(max_lag_s * sr / hop_length),
    )
    lag_samples = int(lag_frames * hop_length)
    aligned_ref, aligned_candidate = _align_by_lag(ref_y, cand_y, lag_samples)
    log_mel_distance = _log_mel_distance(
        aligned_ref,
        aligned_candidate,
        sample_rate=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    return AudioComparisonMetrics(
        reference_path=str(reference),
        candidate_path=str(candidate),
        duration_delta_s=_rounded((cand_y.size - ref_y.size) / sr),
        envelope_correlation=_rounded(envelope_corr),
        envelope_lag_s=_rounded(lag_frames * hop_length / sr),
        log_mel_distance_db=_rounded(log_mel_distance),
    )


def compare_audio_prosody(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    segments: Sequence[AudioSegmentInterval] | None = None,
    metadata_path: str | Path | None = None,
    sample_rate: int = 16000,
    frame_length: int = 1024,
    hop_length: int = 256,
    top_db: float = 35.0,
) -> ProsodyComparisonMetrics:
    reference = Path(reference_path)
    candidate = Path(candidate_path)
    ref_y, sr = _load_audio(reference, sample_rate=sample_rate)
    cand_y, _ = _load_audio(candidate, sample_rate=sample_rate)
    if ref_y.size == 0 or cand_y.size == 0:
        raise ValueError("Reference and candidate audio must be non-empty.")

    intervals = list(segments or [])
    if metadata_path is not None:
        intervals = _load_metadata_intervals(metadata_path)
    if not intervals:
        duration = min(ref_y.size, cand_y.size) / sr
        intervals = [AudioSegmentInterval(start=0.0, end=duration)]

    segment_metrics: list[SegmentProsodyMetrics] = []
    for index, interval in enumerate(intervals):
        if interval.duration <= 0.0:
            continue
        ref_segment = _slice_audio(ref_y, sample_rate=sr, start=interval.start, end=interval.end)
        cand_segment = _slice_audio(
            cand_y,
            sample_rate=sr,
            start=interval.start,
            end=interval.end,
        )
        metrics = _compare_segment_prosody(
            index=index,
            interval=interval,
            reference=ref_segment,
            candidate=cand_segment,
            sample_rate=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            top_db=top_db,
        )
        segment_metrics.append(metrics)

    weights = [max(metric.duration_s, 1e-6) for metric in segment_metrics]
    pitch_weights = [
        weight
        for weight, metric in zip(weights, segment_metrics)
        if metric.pitch_rmse_semitones is not None
    ]
    pitch_rmse_values = [
        metric.pitch_rmse_semitones
        for metric in segment_metrics
        if metric.pitch_rmse_semitones is not None
    ]
    pitch_corr_values = [
        metric.pitch_correlation
        for metric in segment_metrics
        if metric.pitch_correlation is not None
    ]
    metadata_overlap, metadata_max_overlap = _metadata_overlap(intervals)
    worst = sorted(
        segment_metrics,
        key=lambda metric: (
            metric.speech_activity_f1,
            -metric.energy_dtw_distance,
        ),
    )[:5]

    return ProsodyComparisonMetrics(
        reference_path=str(reference),
        candidate_path=str(candidate),
        segment_count=len(segment_metrics),
        duration_delta_s=_rounded((cand_y.size - ref_y.size) / sr),
        metadata_overlap_s=_rounded(metadata_overlap),
        metadata_max_overlap_s=_rounded(metadata_max_overlap),
        speech_activity_precision=_rounded(
            _weighted_average(
                [metric.speech_activity_precision for metric in segment_metrics],
                weights,
            )
        ),
        speech_activity_recall=_rounded(
            _weighted_average(
                [metric.speech_activity_recall for metric in segment_metrics],
                weights,
            )
        ),
        speech_activity_f1=_rounded(
            _weighted_average(
                [metric.speech_activity_f1 for metric in segment_metrics],
                weights,
            )
        ),
        energy_dtw_distance=_rounded(
            _weighted_average(
                [metric.energy_dtw_distance for metric in segment_metrics],
                weights,
            )
        ),
        pitch_rmse_semitones=_rounded_optional(
            _weighted_average(pitch_rmse_values, pitch_weights)
        ),
        pitch_correlation=_rounded_optional(
            _weighted_average(pitch_corr_values, pitch_weights)
        ),
        worst_segment_indices=[metric.index for metric in worst],
        segments=segment_metrics,
    )


def assess_audio_quality(
    metrics: AudioQualityMetrics,
    comparison: AudioComparisonMetrics | None = None,
    prosody_comparison: ProsodyComparisonMetrics | None = None,
    *,
    thresholds: AudioQualityThresholds | None = None,
) -> list[str]:
    limits = thresholds or AudioQualityThresholds()
    issues: list[str] = []

    if metrics.peak_amplitude >= limits.peak_warning:
        issues.append(
            f"Peak amplitude is close to clipping: {metrics.peak_amplitude:.4f}."
        )
    if metrics.clipping_ratio > limits.clipping_ratio_warning:
        issues.append(
            f"Clipping ratio is high: {metrics.clipping_ratio:.6f}."
        )
    if metrics.rms_mean < limits.rms_low_warning:
        issues.append(f"Average RMS is very low: {metrics.rms_mean:.6f}.")
    if metrics.silence_ratio > limits.silence_ratio_warning:
        issues.append(f"Silence ratio is high: {metrics.silence_ratio:.3f}.")
    if metrics.delayed_similarity_peak > limits.delayed_similarity_warning:
        issues.append(
            "Strong delayed envelope similarity detected, possible echo or doubled "
            f"voice around {metrics.delayed_similarity_lag_s:.3f}s."
        )

    if comparison is not None:
        if abs(comparison.duration_delta_s) > limits.duration_delta_warning_s:
            issues.append(
                "Candidate/reference duration mismatch: "
                f"{comparison.duration_delta_s:.3f}s."
            )
        if (
            prosody_comparison is None
            and comparison.envelope_correlation < limits.envelope_correlation_warning
        ):
            issues.append(
                "Candidate differs strongly from the reference envelope: "
                f"{comparison.envelope_correlation:.3f}."
            )

    if prosody_comparison is not None:
        if prosody_comparison.metadata_overlap_s > limits.metadata_overlap_warning_s:
            issues.append(
                "Segment metadata contains overlapping source slots: "
                f"{prosody_comparison.metadata_overlap_s:.3f}s total, "
                f"{prosody_comparison.metadata_max_overlap_s:.3f}s max."
            )
        if prosody_comparison.speech_activity_f1 < limits.speech_activity_f1_warning:
            issues.append(
                "Generated speech activity does not track the source timing: "
                f"F1={prosody_comparison.speech_activity_f1:.3f}."
            )
        if (
            prosody_comparison.speech_activity_recall
            < limits.speech_activity_recall_warning
        ):
            issues.append(
                "Generated audio misses source speech regions: "
                f"recall={prosody_comparison.speech_activity_recall:.3f}."
            )
        if prosody_comparison.energy_dtw_distance > limits.energy_dtw_warning:
            issues.append(
                "Generated energy contour differs from the source prosody: "
                f"DTW={prosody_comparison.energy_dtw_distance:.3f}."
            )
        weak_segments = [
            metric.index
            for metric in prosody_comparison.segments
            if (
                metric.speech_activity_f1
                < limits.segment_speech_activity_f1_warning
                or metric.speech_activity_recall
                < limits.segment_speech_activity_recall_warning
            )
        ]
        if weak_segments:
            shown = ", ".join(str(index) for index in weak_segments[:8])
            suffix = "" if len(weak_segments) <= 8 else ", ..."
            issues.append(
                "Generated audio has low speech coverage in segment(s): "
                f"{shown}{suffix}."
            )

    return issues


def audit_audio(
    candidate_path: str | Path,
    *,
    reference_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    sample_rate: int = 24000,
) -> AudioQualityReport:
    candidate = analyze_audio(candidate_path, sample_rate=sample_rate)
    comparison = (
        compare_audio(reference_path, candidate_path, sample_rate=sample_rate)
        if reference_path
        else None
    )
    prosody_comparison = (
        compare_audio_prosody(
            reference_path,
            candidate_path,
            metadata_path=metadata_path,
            sample_rate=min(sample_rate, 16000),
        )
        if reference_path and metadata_path
        else None
    )
    return AudioQualityReport(
        candidate=candidate,
        comparison=comparison,
        prosody_comparison=prosody_comparison,
        issues=assess_audio_quality(candidate, comparison, prosody_comparison),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit generated speech/audio quality with librosa features."
    )
    parser.add_argument("candidate", help="Generated audio or video file.")
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional comparable reference, e.g. translated_audio.wav before mux.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Optional video translation metadata JSON. When provided with "
            "--reference, also compares prosody/timing per source segment."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()

    report = audit_audio(
        args.candidate,
        reference_path=args.reference,
        metadata_path=args.metadata,
        sample_rate=args.sample_rate,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 1 if report.issues else 0


def _load_audio(path: Path, *, sample_rate: int) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32), int(sr)


def _rms_envelope(
    y: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    return librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0].astype(np.float64)


def _delayed_similarity(
    envelope: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    min_delay_s: float = 0.08,
    max_delay_s: float = 0.7,
) -> tuple[float, float]:
    centered = envelope.astype(np.float64) - float(np.mean(envelope))
    min_lag = max(1, math.ceil(min_delay_s * sample_rate / hop_length))
    max_lag = min(
        len(centered) - 1,
        math.floor(max_delay_s * sample_rate / hop_length),
    )
    if max_lag <= min_lag:
        return 0.0, 0.0

    best, lag = _best_positive_lag_correlation(
        centered,
        min_lag=min_lag,
        max_lag=max_lag,
    )
    return best, lag * hop_length / sample_rate


def _best_positive_lag_correlation(
    values: np.ndarray,
    *,
    min_lag: int,
    max_lag: int,
) -> tuple[float, int]:
    best = 0.0
    best_lag = min_lag
    for lag in range(min_lag, max_lag + 1):
        first = values[:-lag]
        second = values[lag:]
        score = _correlation(first, second)
        if score > best:
            best = score
            best_lag = lag
    return best, best_lag


def _best_lagged_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_lag: int,
) -> tuple[float, int]:
    ref = reference.astype(np.float64) - float(np.mean(reference))
    cand = candidate.astype(np.float64) - float(np.mean(candidate))
    best = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            first = ref[-lag:]
            second = cand[: len(first)]
        elif lag > 0:
            first = ref[:-lag]
            second = cand[lag : lag + len(first)]
        else:
            size = min(len(ref), len(cand))
            first = ref[:size]
            second = cand[:size]
        size = min(len(first), len(second))
        if size < 3:
            continue
        score = _correlation(first[:size], second[:size])
        if score > best:
            best = score
            best_lag = lag
    return best, best_lag


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def _align_by_lag(
    reference: np.ndarray,
    candidate: np.ndarray,
    lag_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag_samples < 0:
        reference = reference[-lag_samples:]
    elif lag_samples > 0:
        candidate = candidate[lag_samples:]
    size = min(len(reference), len(candidate))
    return reference[:size], candidate[:size]


def _log_mel_distance(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    n_mels: int,
) -> float:
    size = min(len(reference), len(candidate))
    if size <= frame_length:
        return 0.0
    ref_mel = librosa.feature.melspectrogram(
        y=reference[:size],
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    cand_mel = librosa.feature.melspectrogram(
        y=candidate[:size],
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    ref_db = librosa.power_to_db(ref_mel + 1e-12, ref=1.0)
    cand_db = librosa.power_to_db(cand_mel + 1e-12, ref=1.0)
    return float(np.mean(np.abs(ref_db - cand_db)))


def _load_metadata_intervals(path: str | Path) -> list[AudioSegmentInterval]:
    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    intervals: list[AudioSegmentInterval] = []
    for item in payload.get("segments", []):
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            intervals.append(AudioSegmentInterval(start=start, end=end))
    return intervals


def _metadata_overlap(
    intervals: Sequence[AudioSegmentInterval],
) -> tuple[float, float]:
    total = 0.0
    maximum = 0.0
    previous_end: float | None = None
    for interval in sorted(intervals, key=lambda item: (item.start, item.end)):
        if previous_end is not None and interval.start < previous_end:
            overlap = previous_end - interval.start
            total += overlap
            maximum = max(maximum, overlap)
        previous_end = max(previous_end or interval.end, interval.end)
    return total, maximum


def _slice_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    start: float,
    end: float,
) -> np.ndarray:
    start_sample = max(0, int(round(start * sample_rate)))
    end_sample = max(start_sample, int(round(end * sample_rate)))
    if end_sample <= audio.shape[0]:
        return np.asarray(audio[start_sample:end_sample], dtype=np.float32)

    output = np.zeros(end_sample - start_sample, dtype=np.float32)
    if start_sample < audio.shape[0]:
        available = np.asarray(audio[start_sample:], dtype=np.float32)
        output[: available.shape[0]] = available
    return output


def _compare_segment_prosody(
    *,
    index: int,
    interval: AudioSegmentInterval,
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    top_db: float,
) -> SegmentProsodyMetrics:
    ref_features = _prosody_features(
        reference,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        top_db=top_db,
    )
    cand_features = _prosody_features(
        candidate,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        top_db=top_db,
    )
    precision, recall, f1 = _activity_scores(
        ref_features["active"],
        cand_features["active"],
    )
    energy_dtw, path = _dtw_distance_and_path(
        ref_features["energy"],
        cand_features["energy"],
    )
    pitch_rmse, pitch_corr = _pitch_scores(
        ref_features["pitch"],
        cand_features["pitch"],
        path,
    )

    return SegmentProsodyMetrics(
        index=index,
        start=_rounded(interval.start),
        end=_rounded(interval.end),
        duration_s=_rounded(interval.duration),
        reference_active_ratio=_rounded(float(np.mean(ref_features["active"]))),
        candidate_active_ratio=_rounded(float(np.mean(cand_features["active"]))),
        speech_activity_precision=_rounded(precision),
        speech_activity_recall=_rounded(recall),
        speech_activity_f1=_rounded(f1),
        energy_dtw_distance=_rounded(energy_dtw),
        pitch_rmse_semitones=_rounded_optional(pitch_rmse),
        pitch_correlation=_rounded_optional(pitch_corr),
    )


def _prosody_features(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    top_db: float,
) -> dict[str, np.ndarray]:
    if audio.size == 0:
        return {
            "energy": np.zeros(1, dtype=np.float64),
            "active": np.zeros(1, dtype=bool),
            "pitch": np.full(1, np.nan, dtype=np.float64),
        }

    rms = _rms_envelope(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    active_threshold = max(
        float(np.max(rms)) * (10.0 ** (-top_db / 20.0)),
        1e-5,
    )
    active = rms > active_threshold
    scale = float(np.percentile(rms, 95)) if rms.size else 0.0
    if scale <= 1e-8:
        energy = np.zeros_like(rms, dtype=np.float64)
    else:
        energy = np.clip(rms / scale, 0.0, 2.0).astype(np.float64)

    pitch = _estimate_relative_pitch(
        audio,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    size = min(energy.shape[0], active.shape[0], pitch.shape[0])
    return {
        "energy": energy[:size],
        "active": active[:size],
        "pitch": pitch[:size],
    }


def _estimate_relative_pitch(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    if audio.size < frame_length:
        return np.full(1, np.nan, dtype=np.float64)
    try:
        f0, _, _ = librosa.pyin(
            audio.astype(np.float32),
            fmin=50.0,
            fmax=min(500.0, sample_rate / 2.0 - 1.0),
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        return np.full(1, np.nan, dtype=np.float64)

    pitch = np.asarray(f0, dtype=np.float64)
    voiced = np.isfinite(pitch) & (pitch > 0.0)
    if np.count_nonzero(voiced) < 3:
        return np.full_like(pitch, np.nan, dtype=np.float64)

    median = float(np.median(pitch[voiced]))
    if median <= 0.0:
        return np.full_like(pitch, np.nan, dtype=np.float64)

    relative = np.full_like(pitch, np.nan, dtype=np.float64)
    relative[voiced] = 12.0 * np.log2(pitch[voiced] / median)
    return relative


def _activity_scores(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float, float]:
    size = min(reference.shape[0], candidate.shape[0])
    if size == 0:
        return 0.0, 0.0, 0.0

    ref = reference[:size].astype(bool)
    cand = candidate[:size].astype(bool)
    true_positive = float(np.count_nonzero(ref & cand))
    false_positive = float(np.count_nonzero(~ref & cand))
    false_negative = float(np.count_nonzero(ref & ~cand))
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def _dtw_distance_and_path(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, np.ndarray]:
    if reference.size == 0 or candidate.size == 0:
        return 0.0, np.zeros((0, 2), dtype=np.int64)
    try:
        cost, path = librosa.sequence.dtw(
            X=reference.reshape(1, -1),
            Y=candidate.reshape(1, -1),
            metric="euclidean",
        )
    except Exception:
        size = min(reference.size, candidate.size)
        if size == 0:
            return 0.0, np.zeros((0, 2), dtype=np.int64)
        aligned_path = np.column_stack([np.arange(size), np.arange(size)])
        return float(np.mean(np.abs(reference[:size] - candidate[:size]))), aligned_path

    aligned_path = path[::-1]
    path_length = max(1, aligned_path.shape[0])
    return float(cost[-1, -1] / path_length), aligned_path


def _pitch_scores(
    reference_pitch: np.ndarray,
    candidate_pitch: np.ndarray,
    path: np.ndarray,
) -> tuple[float | None, float | None]:
    if path.size == 0:
        return None, None

    ref_values: list[float] = []
    cand_values: list[float] = []
    for ref_index, cand_index in path:
        if ref_index >= reference_pitch.shape[0] or cand_index >= candidate_pitch.shape[0]:
            continue
        ref_pitch = float(reference_pitch[ref_index])
        cand_pitch = float(candidate_pitch[cand_index])
        if math.isfinite(ref_pitch) and math.isfinite(cand_pitch):
            ref_values.append(ref_pitch)
            cand_values.append(cand_pitch)

    if len(ref_values) < 3:
        return None, None

    ref_array = np.asarray(ref_values, dtype=np.float64)
    cand_array = np.asarray(cand_values, dtype=np.float64)
    delta = ref_array - cand_array
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    correlation = _correlation(
        ref_array - float(np.mean(ref_array)),
        cand_array - float(np.mean(cand_array)),
    )
    return rmse, correlation


def _weighted_average(
    values: Sequence[float | None],
    weights: Sequence[float],
) -> float:
    pairs = [
        (float(value), float(weight))
        for value, weight in zip(values, weights)
        if value is not None and math.isfinite(float(value)) and weight > 0.0
    ]
    if not pairs:
        return 0.0

    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in pairs) / total_weight


def _rounded_optional(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return _rounded(value, digits=digits)


def _rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


if __name__ == "__main__":
    raise SystemExit(main())

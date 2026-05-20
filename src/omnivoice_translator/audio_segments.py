from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from omnivoice_translator.segments import SpeechSegment


@dataclass(frozen=True)
class SegmentAudio:
    segment: SpeechSegment
    audio_path: Path


def fit_audio_to_duration(
    audio: np.ndarray,
    *,
    sample_rate: int,
    duration: float,
    strategy: str = "pad_trim",
    max_stretch_ratio: float = 1.35,
) -> np.ndarray:
    target_samples = max(0, int(round(duration * sample_rate)))
    data = np.asarray(audio)

    if strategy == "stretch":
        data = _maybe_time_stretch(
            data,
            sample_rate=sample_rate,
            target_duration=duration,
            max_stretch_ratio=max_stretch_ratio,
        )
    elif strategy != "pad_trim":
        raise ValueError(f"Unknown audio fit strategy: {strategy}")

    if data.shape[0] > target_samples:
        return data[:target_samples]
    if data.shape[0] == target_samples:
        return data

    pad_width = [(0, target_samples - data.shape[0])]
    pad_width.extend((0, 0) for _ in data.shape[1:])
    return np.pad(data, pad_width, mode="constant")


def prepare_segment_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    trim_silence: bool = True,
    trim_top_db: float = 35.0,
    fade_ms: float = 18.0,
    target_rms: float | None = 0.045,
    max_gain_db: float = 8.0,
    peak_limit: float = 0.95,
) -> np.ndarray:
    data = _to_mono(np.asarray(audio, dtype=np.float32))
    if data.size == 0:
        return data

    if trim_silence:
        data = _trim_silence(data, top_db=trim_top_db)
    data = _normalize_rms(
        data,
        target_rms=target_rms,
        max_gain_db=max_gain_db,
    )
    data = _limit_peak(data, peak_limit=peak_limit)
    data = _apply_fade(data, sample_rate=sample_rate, fade_ms=fade_ms)
    data = _limit_peak(data, peak_limit=peak_limit)
    return np.asarray(data, dtype=np.float32)


def render_segment_timeline(
    segments: list[SegmentAudio],
    *,
    total_duration: float,
    output_path: str | Path,
    fit_strategy: str = "pad_trim",
    max_stretch_ratio: float = 1.35,
    condition_audio: bool = True,
    trim_silence: bool = True,
    trim_top_db: float = 35.0,
    fade_ms: float = 18.0,
    target_rms: float | None = 0.045,
    max_gain_db: float = 8.0,
    peak_limit: float = 0.95,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = _detect_sample_rate(segments)
    total_samples = max(
        int(round(total_duration * sample_rate)),
        *[
            int(round(segment.segment.end * sample_rate))
            for segment in segments
        ],
        0,
    )
    timeline = np.zeros(total_samples, dtype=np.float32)

    cursor = 0
    for item in sorted(segments, key=lambda segment: segment.segment.start):
        data, item_sample_rate = sf.read(item.audio_path, dtype="float32")
        if item_sample_rate != sample_rate:
            raise ValueError(
                "All segment audio files must use the same sample rate. "
                f"Expected {sample_rate}, got {item_sample_rate}: {item.audio_path}"
            )

        mono = _to_mono(data)
        if condition_audio:
            mono = prepare_segment_audio(
                mono,
                sample_rate=sample_rate,
                trim_silence=trim_silence,
                trim_top_db=trim_top_db,
                fade_ms=fade_ms,
                target_rms=target_rms,
                max_gain_db=max_gain_db,
                peak_limit=peak_limit,
            )
        fitted = fit_audio_to_duration(
            mono,
            sample_rate=sample_rate,
            duration=item.segment.duration,
            strategy=fit_strategy,
            max_stretch_ratio=max_stretch_ratio,
        )
        start = int(round(item.segment.start * sample_rate))
        if start < cursor:
            fitted = fitted[cursor - start :]
            start = cursor
        end = min(start + fitted.shape[0], timeline.shape[0])
        if end <= start:
            continue
        timeline[start:end] = fitted[: end - start]
        cursor = end

    sf.write(output, np.clip(timeline, -1.0, 1.0), sample_rate)
    return output


def _detect_sample_rate(segments: list[SegmentAudio]) -> int:
    if not segments:
        raise ValueError("At least one segment audio file is required.")
    return int(sf.info(segments[0].audio_path).samplerate)


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return np.mean(data, axis=1)


def _trim_silence(audio: np.ndarray, *, top_db: float) -> np.ndarray:
    if not np.any(audio):
        return audio

    import librosa

    frame_length = min(2048, max(32, audio.shape[0] // 8))
    hop_length = max(1, frame_length // 4)
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if trimmed.size == 0:
        return audio
    return np.asarray(trimmed, dtype=np.float32)


def _normalize_rms(
    audio: np.ndarray,
    *,
    target_rms: float | None,
    max_gain_db: float,
) -> np.ndarray:
    if target_rms is None or target_rms <= 0.0:
        return audio

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms <= 1e-8:
        return audio

    gain = target_rms / rms
    max_gain = 10.0 ** (max_gain_db / 20.0)
    gain = min(gain, max_gain)
    return np.asarray(audio * gain, dtype=np.float32)


def _limit_peak(audio: np.ndarray, *, peak_limit: float) -> np.ndarray:
    if peak_limit <= 0.0:
        return audio
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= peak_limit or peak <= 1e-8:
        return audio
    return np.asarray(audio * (peak_limit / peak), dtype=np.float32)


def _apply_fade(
    audio: np.ndarray,
    *,
    sample_rate: int,
    fade_ms: float,
) -> np.ndarray:
    fade_samples = int(round(sample_rate * max(0.0, fade_ms) / 1000.0))
    if fade_samples <= 0 or audio.size <= 1:
        return audio

    fade_samples = min(fade_samples, audio.shape[0] // 2)
    if fade_samples <= 0:
        return audio

    faded = np.array(audio, copy=True, dtype=np.float32)
    faded[:fade_samples] *= np.linspace(
        0.0,
        1.0,
        fade_samples,
        endpoint=True,
        dtype=np.float32,
    )
    faded[-fade_samples:] *= np.linspace(
        1.0,
        0.0,
        fade_samples,
        endpoint=True,
        dtype=np.float32,
    )
    return faded


def _maybe_time_stretch(
    audio: np.ndarray,
    *,
    sample_rate: int,
    target_duration: float,
    max_stretch_ratio: float,
) -> np.ndarray:
    source_duration = audio.shape[0] / sample_rate if sample_rate > 0 else 0.0
    if source_duration <= 0.0 or target_duration <= 0.0:
        return audio

    rate = source_duration / target_duration
    if rate <= 0.0:
        return audio

    min_rate = 1.0 / max_stretch_ratio
    if rate < min_rate or rate > max_stretch_ratio:
        return audio

    import librosa

    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
    return np.asarray(stretched, dtype=np.float32)

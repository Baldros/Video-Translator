from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from omnivoice_translator.segments import SpeechSegment, resolve_segment_overlaps


def refine_segments_with_audio_energy(
    segments: list[SpeechSegment],
    audio_path: str | Path,
    *,
    max_shift: float = 0.28,
    padding: float = 0.06,
    top_db: float = 35.0,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> list[SpeechSegment]:
    if not segments:
        return []

    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if not audio.any():
            return segments
        intervals = librosa.effects.split(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        return segments

    if len(intervals) == 0:
        return segments

    active_ranges = [
        (start / sample_rate, end / sample_rate)
        for start, end in intervals
        if end > start
    ]
    if not active_ranges:
        return segments

    duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    refined: list[SpeechSegment] = []
    for segment in segments:
        search_start = max(0.0, segment.start - max_shift)
        search_end = min(duration, segment.end + max_shift)
        candidates = [
            active
            for active in active_ranges
            if _overlaps(active, (search_start, search_end))
        ]
        if not candidates:
            refined.append(segment)
            continue

        active_start = min(start for start, _ in candidates)
        active_end = max(end for _, end in candidates)
        start = _clamp(
            active_start - padding,
            segment.start - max_shift,
            segment.start + max_shift,
        )
        end = _clamp(
            active_end + padding,
            segment.end - max_shift,
            segment.end + max_shift,
        )
        start = float(max(0.0, start))
        end = float(min(duration, max(start, end)))
        if end <= start:
            refined.append(segment)
            continue
        refined.append(SpeechSegment(start=start, end=end, text=segment.text))
    return resolve_segment_overlaps(refined)


def _overlaps(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

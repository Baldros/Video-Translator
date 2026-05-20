from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def normalize_asr_segments(
    asr_result: Any,
    *,
    fallback_duration: float | None = None,
) -> list[SpeechSegment]:
    if isinstance(asr_result, dict):
        chunks = asr_result.get("chunks") or []
        segments = [_segment_from_chunk(chunk, fallback_duration) for chunk in chunks]
        normalized = [segment for segment in segments if segment is not None]
        if normalized:
            return resolve_segment_overlaps(normalized)

        text = str(asr_result.get("text") or "").strip()
        if text:
            return [
                SpeechSegment(
                    start=0.0,
                    end=max(0.0, float(fallback_duration or 0.0)),
                    text=text,
                )
            ]
        return []

    text = str(asr_result or "").strip()
    if not text:
        return []
    return [
        SpeechSegment(
            start=0.0,
            end=max(0.0, float(fallback_duration or 0.0)),
            text=text,
        )
    ]


def resolve_segment_overlaps(
    segments: list[SpeechSegment],
    *,
    min_duration: float = 0.05,
) -> list[SpeechSegment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end))
    resolved: list[SpeechSegment] = []
    for segment in ordered:
        current = segment
        if current.end <= current.start:
            continue

        if resolved and current.start < resolved[-1].end:
            previous = resolved[-1]
            boundary = (previous.end + current.start) / 2.0
            previous_end = _bounded_boundary(
                boundary,
                lower=previous.start + min_duration,
                upper=previous.end,
            )
            current_start = _bounded_boundary(
                boundary,
                lower=current.start,
                upper=current.end - min_duration,
            )
            if current_start < previous_end:
                current_start = previous_end

            resolved[-1] = SpeechSegment(
                start=previous.start,
                end=max(previous.start, previous_end),
                text=previous.text,
            )
            current = SpeechSegment(
                start=max(0.0, current_start),
                end=current.end,
                text=current.text,
            )

        if current.end - current.start >= min_duration:
            resolved.append(current)
    return resolved


def _segment_from_chunk(
    chunk: Any,
    fallback_duration: float | None,
) -> SpeechSegment | None:
    if not isinstance(chunk, dict):
        return None

    text = str(chunk.get("text") or "").strip()
    if not text:
        return None

    timestamp = chunk.get("timestamp") or chunk.get("timestamps") or (0.0, None)
    try:
        start, end = timestamp
    except (TypeError, ValueError):
        start, end = 0.0, None

    start_value = max(0.0, float(start or 0.0))
    if end is None:
        end_value = float(fallback_duration if fallback_duration is not None else start_value)
    else:
        end_value = float(end)

    return SpeechSegment(
        start=start_value,
        end=max(start_value, end_value),
        text=text,
    )


def _bounded_boundary(value: float, *, lower: float, upper: float) -> float:
    if upper < lower:
        return lower
    return max(lower, min(upper, value))

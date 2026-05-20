from __future__ import annotations

from dataclasses import dataclass

from omnivoice_translator.segments import SpeechSegment


@dataclass(frozen=True)
class SpeechChunk:
    start: float
    end: float
    text: str
    source_segments: tuple[SpeechSegment, ...]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_segment(self) -> SpeechSegment:
        return SpeechSegment(start=self.start, end=self.end, text=self.text)


def merge_speech_segments(
    segments: list[SpeechSegment],
    *,
    min_duration: float = 2.4,
    max_duration: float = 7.5,
    max_gap: float = 0.45,
) -> list[SpeechChunk]:
    if not segments:
        return []

    chunks: list[SpeechChunk] = []
    current: list[SpeechSegment] = []

    for segment in segments:
        if not current:
            current = [segment]
            continue

        previous = current[-1]
        gap = max(0.0, segment.start - previous.end)
        merged_duration = segment.end - current[0].start
        current_duration = current[-1].end - current[0].start

        should_merge = (
            gap <= max_gap
            and merged_duration <= max_duration
            and current_duration < min_duration
        )

        if should_merge:
            current.append(segment)
        else:
            chunks.append(_chunk_from_segments(current))
            current = [segment]

    if current:
        chunks.append(_chunk_from_segments(current))
    return chunks


def _chunk_from_segments(segments: list[SpeechSegment]) -> SpeechChunk:
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return SpeechChunk(
        start=segments[0].start,
        end=segments[-1].end,
        text=text,
        source_segments=tuple(segments),
    )

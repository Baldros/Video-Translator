from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from omnivoice_translator.segment_planner import SpeechChunk


@dataclass(frozen=True)
class AdaptedText:
    text: str
    notes: str | None = None


class TextAdapter(Protocol):
    def adapt(
        self,
        text: str,
        *,
        chunk: SpeechChunk,
        target_language: str | None,
    ) -> AdaptedText:
        ...


class CleanDubbingTextAdapter:
    def adapt(
        self,
        text: str,
        *,
        chunk: SpeechChunk,
        target_language: str | None,
    ) -> AdaptedText:
        del chunk, target_language
        return AdaptedText(text=normalize_spoken_text(text))


def normalize_spoken_text(text: str) -> str:
    normalized = " ".join(text.replace("\n", " ").split())
    return normalized.strip()

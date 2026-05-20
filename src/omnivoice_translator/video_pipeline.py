from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import soundfile as sf

from omnivoice_translator.audio_segments import SegmentAudio, render_segment_timeline
from omnivoice_translator.media import FfmpegMediaAdapter
from omnivoice_translator.segment_planner import SpeechChunk, merge_speech_segments
from omnivoice_translator.segment_refinement import refine_segments_with_audio_energy
from omnivoice_translator.segments import SpeechSegment
from omnivoice_translator.text_refinement import CleanDubbingTextAdapter, TextAdapter


class VoicePipeline(Protocol):
    def transcribe_segments(self, audio_path: str | Path) -> list[SpeechSegment]:
        ...

    def translate(
        self,
        text: str,
        *,
        source_lang: str | None,
        target_lang: str,
    ) -> str:
        ...

    def synthesize(self, text: str, **kwargs) -> Path:
        ...


class LipSyncBackend(Protocol):
    def sync(
        self,
        face_video: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        ...


class MediaAdapter(Protocol):
    def extract_audio(
        self,
        video_path: str | Path,
        output_path: str | Path,
        *,
        sample_rate: int = 16000,
    ) -> Path:
        ...

    def probe_duration(self, media_path: str | Path) -> float:
        ...

    def probe_video_resolution(self, media_path: str | Path):
        ...


@dataclass(frozen=True)
class TranslatedVideoSegment:
    source: SpeechChunk
    draft_translation: str
    translated_text: str
    audio_path: Path
    target_duration_s: float
    generated_duration_s: float | None
    synthesis_retry: bool


@dataclass(frozen=True)
class VideoTranslationResult:
    source_audio_path: Path
    translated_audio_path: Path
    output_video_path: Path
    segments: list[TranslatedVideoSegment]


RenderTimeline = Callable[..., Path]


class VideoTranslationPipeline:
    def __init__(
        self,
        *,
        voice_pipeline: VoicePipeline,
        lip_sync: LipSyncBackend,
        media: MediaAdapter | None = None,
        render_timeline: RenderTimeline = render_segment_timeline,
        text_adapter: TextAdapter | None = None,
    ) -> None:
        self.voice_pipeline = voice_pipeline
        self.lip_sync = lip_sync
        self.media = media or FfmpegMediaAdapter()
        self.render_timeline = render_timeline
        self.text_adapter = text_adapter or CleanDubbingTextAdapter()

    def translate_video(
        self,
        *,
        input_video: str | Path,
        output_video: str | Path,
        source_lang: str | None,
        target_lang: str,
        target_language: str | None = None,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        voice_instruct: str | None = None,
        speed: float = 1.0,
        num_step: int = 32,
        work_dir: str | Path | None = None,
        segment_min_duration: float = 2.4,
        segment_max_duration: float = 7.5,
        segment_max_gap: float = 0.45,
        tts_duration_mode: str = "segment",
        timeline_fit_strategy: str = "pad_trim",
        max_stretch_ratio: float = 1.35,
        refine_segment_boundaries: bool = True,
        boundary_refine_max_shift: float = 0.28,
        boundary_refine_padding: float = 0.06,
        boundary_refine_top_db: float = 35.0,
        audio_conditioning: bool = True,
        audio_trim_silence: bool = True,
        audio_trim_top_db: float = 35.0,
        audio_fade_ms: float = 18.0,
        audio_target_rms: float | None = 0.045,
        audio_max_gain_db: float = 8.0,
        audio_peak_limit: float = 0.95,
        reference_audio_mode: str = "segment",
        reference_audio_padding: float = 0.12,
        retry_short_tts: bool = True,
        tts_min_duration_ratio: float = 0.85,
        preserve_resolution: bool = True,
    ) -> VideoTranslationResult:
        if reference_audio_mode not in {"segment", "source"}:
            raise ValueError(
                "reference_audio_mode must be either 'segment' or 'source'."
            )

        input_path = Path(input_video)
        output_path = Path(output_video)
        working_dir = Path(work_dir) if work_dir else _default_work_dir(output_path)
        segments_dir = working_dir / "segments"
        references_dir = working_dir / "references"
        source_audio = working_dir / "source_audio.wav"
        translated_audio = working_dir / "translated_audio.wav"

        working_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)

        self.media.extract_audio(input_path, source_audio, sample_rate=16000)
        video_duration = self.media.probe_duration(input_path)
        source_segments = self.voice_pipeline.transcribe_segments(source_audio)
        if not source_segments:
            raise RuntimeError("No speech segments were detected in the input video.")
        if refine_segment_boundaries:
            source_segments = refine_segments_with_audio_energy(
                source_segments,
                source_audio,
                max_shift=boundary_refine_max_shift,
                padding=boundary_refine_padding,
                top_db=boundary_refine_top_db,
            )
        speech_chunks = merge_speech_segments(
            source_segments,
            min_duration=segment_min_duration,
            max_duration=segment_max_duration,
            max_gap=segment_max_gap,
        )

        translated_segments: list[TranslatedVideoSegment] = []
        segment_audio: list[SegmentAudio] = []
        explicit_reference_audio = Path(ref_audio) if ref_audio else None
        if explicit_reference_audio is None and reference_audio_mode == "segment":
            references_dir.mkdir(parents=True, exist_ok=True)

        for index, chunk in enumerate(speech_chunks):
            draft_translation = self.voice_pipeline.translate(
                chunk.text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            adapted_text = self.text_adapter.adapt(
                draft_translation,
                chunk=chunk,
                target_language=target_language,
            ).text
            segment_output = segments_dir / f"segment_{index:04d}.wav"
            requested_duration = (
                chunk.duration if tts_duration_mode == "segment" else None
            )
            reference_audio = explicit_reference_audio
            reference_text = ref_text
            if reference_audio is None:
                if reference_audio_mode == "segment":
                    reference_audio = references_dir / f"reference_{index:04d}.wav"
                    _write_reference_clip(
                        source_audio,
                        reference_audio,
                        start=chunk.start,
                        end=chunk.end,
                        padding=reference_audio_padding,
                    )
                    reference_text = ref_text if ref_text is not None else chunk.text
                else:
                    reference_audio = source_audio

            self.voice_pipeline.synthesize(
                adapted_text,
                language=target_language,
                output_path=segment_output,
                ref_audio=reference_audio,
                ref_text=reference_text,
                voice_instruct=voice_instruct,
                speed=speed,
                num_step=num_step,
                duration=requested_duration,
            )
            generated_duration = _probe_audio_duration(segment_output)
            synthesis_retry = False
            if (
                retry_short_tts
                and requested_duration is None
                and chunk.duration > 0.0
                and generated_duration is not None
                and generated_duration < chunk.duration * tts_min_duration_ratio
            ):
                self.voice_pipeline.synthesize(
                    adapted_text,
                    language=target_language,
                    output_path=segment_output,
                    ref_audio=reference_audio,
                    ref_text=reference_text,
                    voice_instruct=voice_instruct,
                    speed=speed,
                    num_step=num_step,
                    duration=chunk.duration,
                )
                synthesis_retry = True
                generated_duration = _probe_audio_duration(segment_output)

            translated = TranslatedVideoSegment(
                source=chunk,
                draft_translation=draft_translation,
                translated_text=adapted_text,
                audio_path=segment_output,
                target_duration_s=chunk.duration,
                generated_duration_s=generated_duration,
                synthesis_retry=synthesis_retry,
            )
            translated_segments.append(translated)
            segment_audio.append(
                SegmentAudio(segment=chunk.as_segment(), audio_path=segment_output)
            )

        self.render_timeline(
            segment_audio,
            total_duration=video_duration,
            output_path=translated_audio,
            fit_strategy=timeline_fit_strategy,
            max_stretch_ratio=max_stretch_ratio,
            condition_audio=audio_conditioning,
            trim_silence=audio_trim_silence,
            trim_top_db=audio_trim_top_db,
            fade_ms=audio_fade_ms,
            target_rms=audio_target_rms,
            max_gain_db=audio_max_gain_db,
            peak_limit=audio_peak_limit,
        )
        final_video = self.lip_sync.sync(input_path, translated_audio, output_path)
        if preserve_resolution:
            _assert_matching_resolution(self.media, input_path, final_video)

        return VideoTranslationResult(
            source_audio_path=source_audio,
            translated_audio_path=translated_audio,
            output_video_path=final_video,
            segments=translated_segments,
        )


def _default_work_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_work"


def _assert_matching_resolution(
    media: MediaAdapter,
    input_video: Path,
    output_video: Path,
) -> None:
    input_resolution = media.probe_video_resolution(input_video)
    output_resolution = media.probe_video_resolution(output_video)
    if input_resolution != output_resolution:
        raise RuntimeError(
            "Lip sync backend changed video resolution: "
            f"{input_resolution} -> {output_resolution}"
        )


def _write_reference_clip(
    source_audio: Path,
    output_path: Path,
    *,
    start: float,
    end: float,
    padding: float,
) -> Path:
    data, sample_rate = sf.read(source_audio, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    source_duration = data.shape[0] / sample_rate if sample_rate > 0 else 0.0
    clip_start = max(0.0, start - max(0.0, padding))
    clip_end = min(source_duration, end + max(0.0, padding))
    start_sample = int(round(clip_start * sample_rate))
    end_sample = int(round(clip_end * sample_rate))
    clip = np.asarray(data[start_sample:end_sample], dtype=np.float32)
    if clip.size == 0:
        clip = np.zeros(1, dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, clip, sample_rate)
    return output_path


def _probe_audio_duration(audio_path: Path) -> float | None:
    try:
        info = sf.info(audio_path)
    except (OSError, RuntimeError):
        return None
    if info.samplerate <= 0:
        return None
    return float(info.frames) / float(info.samplerate)

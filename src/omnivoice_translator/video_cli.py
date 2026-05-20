from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnivoice_translator.lipsync import (
    LatentSyncConfig,
    LatentSyncRunner,
    MuseTalkConfig,
    MuseTalkRunner,
    Wav2LipConfig,
    Wav2LipRunner,
)
from omnivoice_translator.lipsync_box import estimate_lower_face_box
from omnivoice_translator.pipeline import VoiceTranslationPipeline
from omnivoice_translator.video_pipeline import VideoTranslationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate a video with ASR, MT, OmniVoice, and lip sync."
    )
    parser.add_argument("--input", required=True, help="Input video file.")
    parser.add_argument("--output", required=True, help="Output translated video.")
    parser.add_argument("--source-lang", default=None, help="NLLB source code.")
    parser.add_argument("--target-lang", required=True, help="NLLB target code.")
    parser.add_argument(
        "--target-language",
        default=None,
        help="Language name/code hint for OmniVoice, e.g. Portuguese or pt.",
    )
    parser.add_argument("--ref-audio", default=None, help="Reference voice audio/video.")
    parser.add_argument("--ref-text", default=None, help="Reference voice transcript.")
    parser.add_argument(
        "--voice-instruct",
        default=None,
        help='Voice design prompt, e.g. "female, low pitch, british accent".',
    )
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--mt-model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--tts-model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--lip-sync-backend",
        choices=["wav2lip", "latentsync", "musetalk"],
        default="wav2lip",
        help="Lip sync backend used for final video generation.",
    )
    parser.add_argument(
        "--segment-min-duration",
        type=float,
        default=2.4,
        help="Merge speech chunks until this duration when gaps are short.",
    )
    parser.add_argument(
        "--segment-max-duration",
        type=float,
        default=7.5,
        help="Maximum merged chunk duration for translation/TTS.",
    )
    parser.add_argument(
        "--segment-max-gap",
        type=float,
        default=0.45,
        help="Maximum silence gap that may be merged into one chunk.",
    )
    parser.add_argument(
        "--tts-duration-mode",
        choices=["segment", "natural"],
        default="segment",
        help="Use segment duration as TTS target or let TTS speak naturally.",
    )
    parser.add_argument(
        "--timeline-fit",
        choices=["pad_trim", "stretch"],
        default="pad_trim",
        help="How synthesized segment audio is fitted back to source timings.",
    )
    parser.add_argument(
        "--max-stretch-ratio",
        type=float,
        default=1.35,
        help="Maximum time-stretch ratio for --timeline-fit stretch.",
    )
    parser.add_argument(
        "--no-boundary-refine",
        action="store_true",
        help="Disable librosa energy-based refinement of ASR segment boundaries.",
    )
    parser.add_argument("--boundary-refine-max-shift", type=float, default=0.28)
    parser.add_argument("--boundary-refine-padding", type=float, default=0.06)
    parser.add_argument("--boundary-refine-top-db", type=float, default=35.0)
    parser.add_argument(
        "--no-audio-conditioning",
        action="store_true",
        help="Disable per-segment TTS trim/fade/RMS/peak conditioning.",
    )
    parser.add_argument(
        "--no-audio-trim-silence",
        action="store_true",
        help="Do not trim leading/trailing silence from generated TTS segments.",
    )
    parser.add_argument("--audio-trim-top-db", type=float, default=35.0)
    parser.add_argument("--audio-fade-ms", type=float, default=18.0)
    parser.add_argument("--audio-target-rms", type=float, default=0.045)
    parser.add_argument("--audio-max-gain-db", type=float, default=8.0)
    parser.add_argument("--audio-peak-limit", type=float, default=0.95)
    parser.add_argument(
        "--reference-audio-mode",
        choices=["segment", "source"],
        default="segment",
        help=(
            "When --ref-audio is omitted, use per-segment source clips or the "
            "full extracted source audio as the OmniVoice reference."
        ),
    )
    parser.add_argument(
        "--reference-audio-padding",
        type=float,
        default=0.12,
        help="Seconds of source-audio context added around per-segment references.",
    )
    parser.add_argument(
        "--no-tts-retry-short-segments",
        action="store_true",
        help=(
            "Disable retrying natural-duration TTS segments that are much "
            "shorter than their video slot."
        ),
    )
    parser.add_argument(
        "--tts-min-duration-ratio",
        type=float,
        default=0.85,
        help=(
            "Minimum generated/target duration ratio before a natural TTS "
            "segment is retried with explicit duration."
        ),
    )
    parser.add_argument(
        "--no-preserve-resolution-check",
        action="store_true",
        help="Do not fail if lip sync output resolution differs from input.",
    )
    parser.add_argument("--wav2lip-repo", default=None, help="Path to Wav2Lip repo.")
    parser.add_argument(
        "--wav2lip-checkpoint",
        default=None,
        help="Path to Wav2Lip .pth checkpoint.",
    )
    parser.add_argument(
        "--wav2lip-python",
        default=None,
        help="Python executable used to run Wav2Lip. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--wav2lip-extra-arg",
        action="append",
        default=[],
        help="Extra raw argument passed to Wav2Lip. Repeat for multiple values.",
    )
    parser.add_argument(
        "--wav2lip-box",
        nargs=4,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        default=None,
        help="Fixed Wav2Lip face box. Useful when automatic face detection fails.",
    )
    parser.add_argument(
        "--wav2lip-auto-box",
        action="store_true",
        help="Estimate a stable lower-face box automatically before running Wav2Lip.",
    )
    parser.add_argument("--latentsync-repo", default=None, help="Path to LatentSync repo.")
    parser.add_argument(
        "--latentsync-checkpoint",
        default=None,
        help="Path to LatentSync checkpoint.",
    )
    parser.add_argument(
        "--latentsync-unet-config",
        default=None,
        help="Path to LatentSync UNet config YAML.",
    )
    parser.add_argument("--latentsync-steps", type=int, default=20)
    parser.add_argument("--latentsync-guidance-scale", type=float, default=1.5)
    parser.add_argument(
        "--latentsync-disable-deepcache",
        action="store_true",
        help="Disable LatentSync DeepCache acceleration.",
    )
    parser.add_argument("--musetalk-repo", default=None, help="Path to MuseTalk repo.")
    parser.add_argument(
        "--musetalk-python",
        default=None,
        help="Python executable used to run MuseTalk. Defaults to this interpreter.",
    )
    parser.add_argument("--musetalk-version", default="v15", choices=["v1", "v15"])
    parser.add_argument("--musetalk-unet-model", default=None)
    parser.add_argument("--musetalk-unet-config", default=None)
    parser.add_argument("--musetalk-whisper-dir", default=None)
    parser.add_argument("--musetalk-batch-size", type=int, default=8)
    parser.add_argument(
        "--musetalk-no-float16",
        action="store_true",
        help="Disable MuseTalk float16 inference.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON file with segment metadata.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    voice = VoiceTranslationPipeline.from_pretrained(
        asr_model=args.asr_model,
        mt_model=args.mt_model,
        tts_model=args.tts_model,
        device=args.device,
    )
    wav2lip_extra_args = list(args.wav2lip_extra_arg)
    if args.wav2lip_box and args.wav2lip_auto_box:
        raise ValueError("Use only one of --wav2lip-box or --wav2lip-auto-box.")
    if args.wav2lip_auto_box:
        box = estimate_lower_face_box(args.input)
        wav2lip_extra_args.extend(["--box", *box.as_cli_args()])
    elif args.wav2lip_box:
        wav2lip_extra_args.extend(["--box", *args.wav2lip_box])

    lip_sync = build_lip_sync_backend(args, tuple(wav2lip_extra_args))
    result = VideoTranslationPipeline(
        voice_pipeline=voice,
        lip_sync=lip_sync,
    ).translate_video(
        input_video=args.input,
        output_video=args.output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        target_language=args.target_language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        voice_instruct=args.voice_instruct,
        speed=args.speed,
        num_step=args.num_step,
        work_dir=args.work_dir,
        segment_min_duration=args.segment_min_duration,
        segment_max_duration=args.segment_max_duration,
        segment_max_gap=args.segment_max_gap,
        tts_duration_mode=args.tts_duration_mode,
        timeline_fit_strategy=args.timeline_fit,
        max_stretch_ratio=args.max_stretch_ratio,
        refine_segment_boundaries=not args.no_boundary_refine,
        boundary_refine_max_shift=args.boundary_refine_max_shift,
        boundary_refine_padding=args.boundary_refine_padding,
        boundary_refine_top_db=args.boundary_refine_top_db,
        audio_conditioning=not args.no_audio_conditioning,
        audio_trim_silence=not args.no_audio_trim_silence,
        audio_trim_top_db=args.audio_trim_top_db,
        audio_fade_ms=args.audio_fade_ms,
        audio_target_rms=args.audio_target_rms,
        audio_max_gain_db=args.audio_max_gain_db,
        audio_peak_limit=args.audio_peak_limit,
        reference_audio_mode=args.reference_audio_mode,
        reference_audio_padding=args.reference_audio_padding,
        retry_short_tts=not args.no_tts_retry_short_segments,
        tts_min_duration_ratio=args.tts_min_duration_ratio,
        preserve_resolution=not args.no_preserve_resolution_check,
    )

    payload = {
        "source_audio_path": str(result.source_audio_path),
        "translated_audio_path": str(result.translated_audio_path),
        "output_video_path": str(result.output_video_path),
        "segments": [
            {
                "start": segment.source.start,
                "end": segment.source.end,
                "source_text": segment.source.text,
                "draft_translation": segment.draft_translation,
                "translated_text": segment.translated_text,
                "audio_path": str(segment.audio_path),
                "target_duration_s": segment.target_duration_s,
                "generated_duration_s": segment.generated_duration_s,
                "synthesis_retry": segment.synthesis_retry,
            }
            for segment in result.segments
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.metadata:
        metadata = Path(args.metadata)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


def build_lip_sync_backend(args, wav2lip_extra_args: tuple[str, ...]):
    if args.lip_sync_backend != "wav2lip" and (
        args.wav2lip_box or args.wav2lip_auto_box
    ):
        raise ValueError("Wav2Lip box options only apply to --lip-sync-backend wav2lip.")

    if args.lip_sync_backend == "wav2lip":
        _require_args(args, "wav2lip_repo", "wav2lip_checkpoint")
        return Wav2LipRunner(
            Wav2LipConfig(
                repo_path=Path(args.wav2lip_repo),
                checkpoint_path=Path(args.wav2lip_checkpoint),
                python_executable=args.wav2lip_python
                or Wav2LipConfig.python_executable,
                extra_args=wav2lip_extra_args,
            )
        )

    if args.lip_sync_backend == "latentsync":
        _require_args(
            args,
            "latentsync_repo",
            "latentsync_checkpoint",
            "latentsync_unet_config",
        )
        return LatentSyncRunner(
            LatentSyncConfig(
                repo_path=Path(args.latentsync_repo),
                checkpoint_path=Path(args.latentsync_checkpoint),
                unet_config_path=Path(args.latentsync_unet_config),
                inference_steps=args.latentsync_steps,
                guidance_scale=args.latentsync_guidance_scale,
                enable_deepcache=not args.latentsync_disable_deepcache,
            )
        )

    if args.lip_sync_backend == "musetalk":
        _require_args(args, "musetalk_repo")
        return MuseTalkRunner(
            MuseTalkConfig(
                repo_path=Path(args.musetalk_repo),
                python_executable=args.musetalk_python
                or MuseTalkConfig.python_executable,
                version=args.musetalk_version,
                unet_model_path=Path(args.musetalk_unet_model)
                if args.musetalk_unet_model
                else None,
                unet_config_path=Path(args.musetalk_unet_config)
                if args.musetalk_unet_config
                else None,
                whisper_dir=Path(args.musetalk_whisper_dir)
                if args.musetalk_whisper_dir
                else None,
                batch_size=args.musetalk_batch_size,
                use_float16=not args.musetalk_no_float16,
            )
        )

    raise ValueError(f"Unsupported lip sync backend: {args.lip_sync_backend}")


def _require_args(args, *names: str) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required argument(s): {', '.join(missing)}")


if __name__ == "__main__":
    raise SystemExit(main())

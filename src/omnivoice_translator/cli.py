from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnivoice_translator.pipeline import VoiceTranslationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate speech to speech using ASR, MT, and OmniVoice."
    )
    parser.add_argument("--input", required=True, help="Input audio file.")
    parser.add_argument("--output", required=True, help="Output WAV file.")
    parser.add_argument("--source-lang", default=None, help="NLLB source code.")
    parser.add_argument("--target-lang", required=True, help="NLLB target code.")
    parser.add_argument(
        "--target-language",
        default=None,
        help="Language name/code hint for OmniVoice, e.g. Portuguese or pt.",
    )
    parser.add_argument("--ref-audio", default=None, help="Reference voice audio.")
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
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON file with transcript and translation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    translator = VoiceTranslationPipeline.from_pretrained(
        asr_model=args.asr_model,
        mt_model=args.mt_model,
        tts_model=args.tts_model,
        device=args.device,
    )
    result = translator.translate_file(
        input_path=args.input,
        output_path=args.output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        target_language=args.target_language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        voice_instruct=args.voice_instruct,
        speed=args.speed,
        num_step=args.num_step,
    )

    payload = {
        "source_text": result.source_text,
        "translated_text": result.translated_text,
        "output_path": str(result.output_path),
        "sampling_rate": result.sampling_rate,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.metadata:
        Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metadata).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

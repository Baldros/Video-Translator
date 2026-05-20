from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from omnivoice import OmniVoice

from omnivoice_translator.device import get_best_device, get_default_dtype


NLLB_TO_OMNIVOICE_LANGUAGE = {
    "eng_Latn": "English",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "zho_Hans": "Chinese",
    "zho_Hant": "Chinese",
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
}


def resolve_language(language: str | None, target_lang: str | None) -> str | None:
    if language:
        return language
    if target_lang:
        return NLLB_TO_OMNIVOICE_LANGUAGE.get(target_lang)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate speech directly with OmniVoice."
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ref-audio", "--ref_audio", dest="ref_audio", default=None)
    parser.add_argument("--ref-text", "--ref_text", dest="ref_text", default=None)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--target-lang",
        default=None,
        help="Optional NLLB-style code used only to infer --language.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-step", "--num_step", dest="num_step", type=int, default=32)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--guidance-scale", "--guidance_scale", dest="guidance_scale", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = args.device or get_best_device()
    dtype = get_default_dtype(device)
    language = resolve_language(args.language, args.target_lang)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
    )
    audio = model.generate(
        text=args.text,
        language=language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        instruct=args.instruct,
        num_step=args.num_step,
        speed=args.speed,
        duration=args.duration,
        guidance_scale=args.guidance_scale,
    )
    sf.write(output, audio[0], model.sampling_rate)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

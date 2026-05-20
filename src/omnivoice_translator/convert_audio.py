from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def convert_wav_to_mp4(
    input_path: str | Path,
    output_path: str | Path,
    *,
    bitrate: str = "192k",
    overwrite: bool = False,
) -> Path:
    source = Path(input_path)
    target = Path(output_path)

    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    if source.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav input file, got: {source}")
    if target.suffix.lower() not in {".mp4", ".m4a"}:
        raise ValueError(f"Expected a .mp4 or .m4a output file, got: {target}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite: {target}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found in PATH.")

    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        str(target),
    ]
    subprocess.run(command, check=True)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert WAV audio to MP4/M4A.")
    parser.add_argument("input", help="Input .wav file.")
    parser.add_argument("output", help="Output .mp4 or .m4a file.")
    parser.add_argument("--bitrate", default="192k", help="AAC audio bitrate.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = convert_wav_to_mp4(
        args.input,
        args.output,
        bitrate=args.bitrate,
        overwrite=args.overwrite,
    )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoResolution:
    width: int
    height: int


class FfmpegMediaAdapter:
    def extract_audio(
        self,
        video_path: str | Path,
        output_path: str | Path,
        *,
        sample_rate: int = 16000,
    ) -> Path:
        source = Path(video_path)
        output = Path(output_path)
        if not source.exists():
            raise FileNotFoundError(f"Input video not found: {source}")

        ffmpeg = _require_command("ffmpeg")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(output),
        ]
        subprocess.run(command, check=True)
        return output

    def probe_duration(self, media_path: str | Path) -> float:
        source = Path(media_path)
        if not source.exists():
            raise FileNotFoundError(f"Media file not found: {source}")

        ffprobe = _require_command("ffprobe")
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return float(completed.stdout.strip())

    def probe_video_resolution(self, media_path: str | Path) -> VideoResolution:
        source = Path(media_path)
        if not source.exists():
            raise FileNotFoundError(f"Media file not found: {source}")

        ffprobe = _require_command("ffprobe")
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(source),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        width_text, height_text = completed.stdout.strip().split("x", maxsplit=1)
        return VideoResolution(width=int(width_text), height=int(height_text))


def _require_command(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"{name} was not found in PATH.")
    return resolved

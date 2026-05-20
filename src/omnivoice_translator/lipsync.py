from __future__ import annotations

import os
import subprocess
import sys
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Wav2LipConfig:
    repo_path: Path
    checkpoint_path: Path
    python_executable: str = sys.executable
    extra_args: tuple[str, ...] = field(default_factory=tuple)


class Wav2LipRunner:
    def __init__(self, config: Wav2LipConfig) -> None:
        self.config = config

    def sync(
        self,
        face_video: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        repo = Path(self.config.repo_path)
        inference = repo / "inference.py"
        checkpoint = Path(self.config.checkpoint_path)
        face = Path(face_video)
        audio = Path(audio_path)
        output = Path(output_path)

        if not inference.exists():
            raise FileNotFoundError(f"Wav2Lip inference.py not found: {inference}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Wav2Lip checkpoint not found: {checkpoint}")
        if not face.exists():
            raise FileNotFoundError(f"Face video not found: {face}")
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.python_executable,
            str(inference),
            "--checkpoint_path",
            str(checkpoint),
            "--face",
            str(face),
            "--audio",
            str(audio),
            "--outfile",
            str(output),
            *self.config.extra_args,
        ]
        subprocess.run(command, check=True, cwd=str(repo))
        return output


@dataclass(frozen=True)
class LatentSyncConfig:
    repo_path: Path
    checkpoint_path: Path
    unet_config_path: Path
    python_executable: str = sys.executable
    inference_steps: int = 20
    guidance_scale: float = 1.5
    enable_deepcache: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)


class LatentSyncRunner:
    def __init__(self, config: LatentSyncConfig) -> None:
        self.config = config

    def sync(
        self,
        face_video: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        repo = Path(self.config.repo_path)
        checkpoint = Path(self.config.checkpoint_path)
        unet_config = Path(self.config.unet_config_path)
        face = Path(face_video)
        audio = Path(audio_path)
        output = Path(output_path)

        if not repo.exists():
            raise FileNotFoundError(f"LatentSync repo not found: {repo}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"LatentSync checkpoint not found: {checkpoint}")
        if not unet_config.exists():
            raise FileNotFoundError(f"LatentSync UNet config not found: {unet_config}")
        if not face.exists():
            raise FileNotFoundError(f"Face video not found: {face}")
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.python_executable,
            "-m",
            "scripts.inference",
            "--unet_config_path",
            str(unet_config),
            "--inference_ckpt_path",
            str(checkpoint),
            "--inference_steps",
            str(self.config.inference_steps),
            "--guidance_scale",
            str(self.config.guidance_scale),
            "--video_path",
            str(face),
            "--audio_path",
            str(audio),
            "--video_out_path",
            str(output),
        ]
        if self.config.enable_deepcache:
            command.append("--enable_deepcache")
        command.extend(self.config.extra_args)
        subprocess.run(command, check=True, cwd=str(repo))
        return output


@dataclass(frozen=True)
class MuseTalkConfig:
    repo_path: Path
    python_executable: str = sys.executable
    version: str = "v15"
    unet_model_path: Path | None = None
    unet_config_path: Path | None = None
    whisper_dir: Path | None = None
    batch_size: int = 8
    use_float16: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)


class MuseTalkRunner:
    def __init__(self, config: MuseTalkConfig) -> None:
        self.config = config

    def sync(
        self,
        face_video: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        repo = Path(self.config.repo_path)
        inference = repo / "scripts" / "inference.py"
        face = Path(face_video)
        audio = Path(audio_path)
        output = Path(output_path)

        if not repo.exists():
            raise FileNotFoundError(
                f"MuseTalk repo not found: {repo}. "
                "Clone/install MuseTalk and pass its root with --musetalk-repo."
            )
        if not inference.exists():
            raise FileNotFoundError(
                f"MuseTalk inference.py not found: {inference}. "
                "--musetalk-repo must point to the MuseTalk repository root, "
                "the folder that contains scripts\\inference.py."
            )
        if not face.exists():
            raise FileNotFoundError(f"Face video not found: {face}")
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        target_duration = _max_duration(
            _probe_format_duration(face),
            _probe_format_duration(audio),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = output.parent / f"{output.stem}_musetalk_inputs"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_face = staging_dir / f"input{face.suffix.lower()}"
        staged_audio = staging_dir / f"audio{audio.suffix.lower()}"
        shutil.copy2(face, staged_face)
        shutil.copy2(audio, staged_audio)

        config_path = output.parent / f"{output.stem}_musetalk.yaml"
        _write_musetalk_config(config_path, staged_face, staged_audio, output.name)

        command = [
            self.config.python_executable,
            "-m",
            "scripts.inference",
            "--inference_config",
            str(config_path),
            "--result_dir",
            str(output.parent),
            "--output_vid_name",
            output.name,
            "--version",
            self.config.version,
            "--batch_size",
            str(self.config.batch_size),
        ]
        unet_model_path = self.config.unet_model_path or _default_musetalk_unet_model(
            repo, self.config.version
        )
        unet_config_path = self.config.unet_config_path or _default_musetalk_unet_config(
            repo, self.config.version
        )
        if unet_model_path:
            command.extend(["--unet_model_path", str(unet_model_path)])
        if unet_config_path:
            command.extend(["--unet_config", str(unet_config_path)])
        if self.config.whisper_dir:
            command.extend(["--whisper_dir", str(self.config.whisper_dir)])
        if self.config.use_float16:
            command.append("--use_float16")
        command.extend(self.config.extra_args)

        subprocess.run(command, check=True, cwd=str(repo), env=_utf8_subprocess_env())
        nested_output = output.parent / self.config.version / output.name
        if not output.exists() and nested_output.exists():
            shutil.move(str(nested_output), str(output))
        if not output.exists():
            raise RuntimeError(
                "MuseTalk finished without creating the expected output: "
                f"{output}. Check MuseTalk logs for a swallowed inference error."
            )
        _pad_short_video_stream(output, target_duration)
        return output


def _write_musetalk_config(
    path: Path,
    face_video: Path,
    audio_path: Path,
    output_name: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "task_0:",
                f"  video_path: {str(face_video)!r}",
                f"  audio_path: {str(audio_path)!r}",
                f"  result_name: {output_name!r}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _default_musetalk_unet_model(repo: Path, version: str) -> Path | None:
    if version == "v15":
        return repo / "models" / "musetalkV15" / "unet.pth"
    if version == "v1":
        return repo / "models" / "musetalk" / "pytorch_model.bin"
    return None


def _default_musetalk_unet_config(repo: Path, version: str) -> Path | None:
    if version == "v15":
        return repo / "models" / "musetalkV15" / "musetalk.json"
    if version == "v1":
        return repo / "models" / "musetalk" / "musetalk.json"
    return None


def _utf8_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _max_duration(*durations: float | None) -> float | None:
    measured = [duration for duration in durations if duration is not None]
    if not measured:
        return None
    return max(measured)


def _probe_format_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(completed.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    if duration <= 0:
        return None
    return duration


def _probe_video_stream_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(completed.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    if duration <= 0:
        return None
    return duration


def _pad_short_video_stream(
    path: Path,
    target_duration: float | None,
    *,
    tolerance: float = 0.25,
) -> None:
    if target_duration is None:
        return
    current_duration = _probe_video_stream_duration(path)
    if current_duration is None or current_duration + tolerance >= target_duration:
        return

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found in PATH; cannot pad MuseTalk output.")

    temp_output = path.with_name(f"{path.stem}_duration_fixed{path.suffix}")
    pad_duration = target_duration - current_duration
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        f"tpad=stop_mode=clone:stop_duration={pad_duration:.3f}",
        "-t",
        f"{target_duration:.3f}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(temp_output),
    ]
    subprocess.run(command, check=True)
    temp_output.replace(path)

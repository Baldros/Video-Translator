from pathlib import Path

import pytest

from omnivoice_translator import lipsync
from omnivoice_translator.lipsync import (
    LatentSyncConfig,
    LatentSyncRunner,
    MuseTalkConfig,
    MuseTalkRunner,
    Wav2LipConfig,
    Wav2LipRunner,
)


def test_wav2lip_runner_builds_expected_command(monkeypatch, tmp_path):
    repo = tmp_path / "Wav2Lip"
    checkpoint = repo / "checkpoints" / "wav2lip_gan.pth"
    face = tmp_path / "input.mp4"
    audio = tmp_path / "translated.wav"
    output = tmp_path / "out" / "translated.mp4"
    calls = []

    repo.mkdir()
    (repo / "inference.py").write_text("# fake", encoding="utf-8")
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    face.write_bytes(b"video")
    audio.write_bytes(b"audio")

    monkeypatch.setattr(
        lipsync.subprocess,
        "run",
        lambda command, check, cwd: calls.append((command, cwd)),
    )

    runner = Wav2LipRunner(
        Wav2LipConfig(
            repo_path=repo,
            checkpoint_path=checkpoint,
            python_executable="python",
            extra_args=("--box", "60", "390", "45", "305"),
        )
    )

    result = runner.sync(face, audio, output)

    assert result == output
    assert output.parent.exists()
    assert calls == [
        (
            [
                "python",
                str(repo / "inference.py"),
                "--checkpoint_path",
                str(checkpoint),
                "--face",
                str(face),
                "--audio",
                str(audio),
                "--outfile",
                str(output),
                "--box",
                "60",
                "390",
                "45",
                "305",
            ],
            str(repo),
        )
    ]


def test_wav2lip_runner_requires_inference_script(tmp_path):
    repo = tmp_path / "Wav2Lip"
    checkpoint = tmp_path / "wav2lip.pth"
    face = tmp_path / "input.mp4"
    audio = tmp_path / "translated.wav"

    repo.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    face.write_bytes(b"video")
    audio.write_bytes(b"audio")

    runner = Wav2LipRunner(
        Wav2LipConfig(repo_path=repo, checkpoint_path=checkpoint)
    )

    with pytest.raises(FileNotFoundError, match="Wav2Lip inference.py not found"):
        runner.sync(face, audio, tmp_path / "translated.mp4")


def test_latentsync_runner_builds_expected_command(monkeypatch, tmp_path):
    repo = tmp_path / "LatentSync"
    checkpoint = repo / "checkpoints" / "latentsync_unet.pt"
    unet_config = repo / "configs" / "unet" / "stage2_512.yaml"
    face = tmp_path / "input.mp4"
    audio = tmp_path / "translated.wav"
    output = tmp_path / "translated.mp4"
    calls = []

    unet_config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir()
    repo.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    unet_config.write_text("fake: true", encoding="utf-8")
    face.write_bytes(b"video")
    audio.write_bytes(b"audio")

    monkeypatch.setattr(
        lipsync.subprocess,
        "run",
        lambda command, check, cwd: calls.append((command, cwd)),
    )

    result = LatentSyncRunner(
        LatentSyncConfig(
            repo_path=repo,
            checkpoint_path=checkpoint,
            unet_config_path=unet_config,
            python_executable="python",
            inference_steps=25,
            guidance_scale=1.7,
        )
    ).sync(face, audio, output)

    assert result == output
    assert calls == [
        (
            [
                "python",
                "-m",
                "scripts.inference",
                "--unet_config_path",
                str(unet_config),
                "--inference_ckpt_path",
                str(checkpoint),
                "--inference_steps",
                "25",
                "--guidance_scale",
                "1.7",
                "--video_path",
                str(face),
                "--audio_path",
                str(audio),
                "--video_out_path",
                str(output),
                "--enable_deepcache",
            ],
            str(repo),
        )
    ]


def test_musetalk_runner_writes_config_and_builds_expected_command(
    monkeypatch, tmp_path
):
    repo = tmp_path / "MuseTalk"
    inference = repo / "scripts" / "inference.py"
    face = tmp_path / "input.mp4"
    audio = tmp_path / "translated.wav"
    output = tmp_path / "out" / "translated.mp4"
    calls = []

    inference.parent.mkdir(parents=True)
    inference.write_text("# fake", encoding="utf-8")
    (repo / "models" / "musetalkV15").mkdir(parents=True)
    face.write_bytes(b"video")
    audio.write_bytes(b"audio")

    def fake_run(command, check, cwd, env=None):
        calls.append((command, cwd, env))
        nested = output.parent / "v15"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / output.name).write_bytes(b"video")

    monkeypatch.setattr(lipsync.subprocess, "run", fake_run)
    monkeypatch.setattr(lipsync, "_probe_format_duration", lambda path: None)

    result = MuseTalkRunner(
        MuseTalkConfig(
            repo_path=repo,
            python_executable="python",
            batch_size=4,
            use_float16=False,
        )
    ).sync(face, audio, output)

    config_path = output.parent / "translated_musetalk.yaml"
    assert result == output
    assert "video_path" in config_path.read_text(encoding="utf-8")
    assert output.exists()
    assert (output.parent / "translated_musetalk_inputs" / "input.mp4").exists()
    assert (output.parent / "translated_musetalk_inputs" / "audio.wav").exists()
    assert len(calls) == 1
    command, cwd, env = calls[0]
    assert command == [
        "python",
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config_path),
        "--result_dir",
        str(output.parent),
        "--output_vid_name",
        output.name,
        "--version",
        "v15",
        "--batch_size",
        "4",
        "--unet_model_path",
        str(repo / "models" / "musetalkV15" / "unet.pth"),
        "--unet_config",
        str(repo / "models" / "musetalkV15" / "musetalk.json"),
    ]
    assert cwd == str(repo)
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"


def test_musetalk_runner_pads_output_when_video_stream_is_short(
    monkeypatch, tmp_path
):
    repo = tmp_path / "MuseTalk"
    inference = repo / "scripts" / "inference.py"
    face = tmp_path / "input.mp4"
    audio = tmp_path / "translated.wav"
    output = tmp_path / "out" / "translated.mp4"
    calls = []

    inference.parent.mkdir(parents=True)
    inference.write_text("# fake", encoding="utf-8")
    (repo / "models" / "musetalkV15").mkdir(parents=True)
    face.write_bytes(b"video")
    audio.write_bytes(b"audio")

    def fake_run(command, check, cwd=None, env=None, **kwargs):
        calls.append(command)
        if command[:3] == ["python", "-m", "scripts.inference"]:
            nested = output.parent / "v15"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / output.name).write_bytes(b"short video")
            return
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"padded video")

    monkeypatch.setattr(lipsync.subprocess, "run", fake_run)
    monkeypatch.setattr(lipsync, "_probe_format_duration", lambda path: 10.0)
    monkeypatch.setattr(lipsync, "_probe_video_stream_duration", lambda path: 6.0)
    monkeypatch.setattr(
        lipsync.shutil,
        "which",
        lambda name: name if name in {"ffmpeg", "ffprobe"} else None,
    )

    result = MuseTalkRunner(
        MuseTalkConfig(
            repo_path=repo,
            python_executable="python",
            use_float16=False,
        )
    ).sync(face, audio, output)

    assert result == output
    assert output.read_bytes() == b"padded video"
    ffmpeg_call = calls[-1]
    assert ffmpeg_call[0] == "ffmpeg"
    assert "tpad=stop_mode=clone:stop_duration=4.000" in ffmpeg_call
    assert "-t" in ffmpeg_call
    assert "10.000" in ffmpeg_call


def test_musetalk_runner_explains_missing_repo(tmp_path):
    runner = MuseTalkRunner(MuseTalkConfig(repo_path=tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="MuseTalk repo not found"):
        runner.sync(
            tmp_path / "input.mp4",
            tmp_path / "translated.wav",
            tmp_path / "translated.mp4",
        )


def test_musetalk_runner_explains_wrong_repo_root(tmp_path):
    repo = tmp_path / "MuseTalk"
    face = tmp_path / "input.mp4"
    audio = tmp_path / "translated.wav"
    repo.mkdir()
    face.write_bytes(b"video")
    audio.write_bytes(b"audio")
    runner = MuseTalkRunner(MuseTalkConfig(repo_path=repo))

    with pytest.raises(FileNotFoundError, match="repository root"):
        runner.sync(face, audio, tmp_path / "translated.mp4")

from types import SimpleNamespace

from omnivoice_translator import media
from omnivoice_translator.media import FfmpegMediaAdapter, VideoResolution


def test_extract_audio_builds_ffmpeg_command(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    output = tmp_path / "nested" / "source.wav"
    video.write_bytes(b"fake video")
    calls = []

    monkeypatch.setattr(media.shutil, "which", lambda name: f"C:/bin/{name}.exe")
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda command, check, **kwargs: calls.append(command),
    )

    result = FfmpegMediaAdapter().extract_audio(video, output, sample_rate=16000)

    assert result == output
    assert output.parent.exists()
    assert calls == [
        [
            "C:/bin/ffmpeg.exe",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ]
    ]


def test_probe_duration_reads_ffprobe_stdout(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video")
    calls = []

    monkeypatch.setattr(media.shutil, "which", lambda name: f"C:/bin/{name}.exe")

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return SimpleNamespace(stdout="12.34\n")

    monkeypatch.setattr(media.subprocess, "run", fake_run)

    duration = FfmpegMediaAdapter().probe_duration(video)

    assert duration == 12.34
    assert calls[0][0] == "C:/bin/ffprobe.exe"
    assert str(video) in calls[0]


def test_probe_video_resolution_reads_ffprobe_stdout(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video")

    monkeypatch.setattr(media.shutil, "which", lambda name: f"C:/bin/{name}.exe")
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda command, check, capture_output, text: SimpleNamespace(stdout="360x640\n"),
    )

    assert FfmpegMediaAdapter().probe_video_resolution(video) == VideoResolution(
        width=360,
        height=640,
    )

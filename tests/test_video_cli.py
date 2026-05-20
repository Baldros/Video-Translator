from argparse import Namespace

import pytest

from omnivoice_translator.lipsync import (
    LatentSyncRunner,
    MuseTalkRunner,
    Wav2LipRunner,
)
from omnivoice_translator.video_cli import build_lip_sync_backend


def base_args(**overrides):
    values = {
        "lip_sync_backend": "wav2lip",
        "wav2lip_repo": None,
        "wav2lip_checkpoint": None,
        "wav2lip_python": None,
        "wav2lip_box": None,
        "wav2lip_auto_box": False,
        "latentsync_repo": None,
        "latentsync_checkpoint": None,
        "latentsync_unet_config": None,
        "latentsync_steps": 20,
        "latentsync_guidance_scale": 1.5,
        "latentsync_disable_deepcache": False,
        "musetalk_repo": None,
        "musetalk_python": None,
        "musetalk_version": "v15",
        "musetalk_unet_model": None,
        "musetalk_unet_config": None,
        "musetalk_whisper_dir": None,
        "musetalk_batch_size": 8,
        "musetalk_no_float16": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_lip_sync_backend_requires_wav2lip_paths():
    with pytest.raises(ValueError, match="--wav2lip-repo"):
        build_lip_sync_backend(base_args(), ())


def test_build_lip_sync_backend_builds_wav2lip():
    backend = build_lip_sync_backend(
        base_args(wav2lip_repo="repo", wav2lip_checkpoint="model.pth"),
        ("--box", "1", "2", "3", "4"),
    )

    assert isinstance(backend, Wav2LipRunner)
    assert backend.config.extra_args == ("--box", "1", "2", "3", "4")


def test_build_lip_sync_backend_builds_latentsync():
    backend = build_lip_sync_backend(
        base_args(
            lip_sync_backend="latentsync",
            latentsync_repo="repo",
            latentsync_checkpoint="model.pt",
            latentsync_unet_config="config.yaml",
            latentsync_steps=30,
        ),
        (),
    )

    assert isinstance(backend, LatentSyncRunner)
    assert backend.config.inference_steps == 30


def test_build_lip_sync_backend_builds_musetalk():
    backend = build_lip_sync_backend(
        base_args(
            lip_sync_backend="musetalk",
            musetalk_repo="repo",
            musetalk_python="python-musetalk",
        ),
        (),
    )

    assert isinstance(backend, MuseTalkRunner)
    assert backend.config.python_executable == "python-musetalk"

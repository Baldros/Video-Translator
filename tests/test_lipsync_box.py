import pytest

from omnivoice_translator.lipsync_box import (
    Wav2LipBox,
    estimate_lower_face_box,
    lower_face_box_from_detections,
)


def test_lower_face_box_from_detections_uses_median_lower_face_crop():
    box = lower_face_box_from_detections(
        [
            (100, 120, 180, 180),
            (110, 130, 170, 170),
            (105, 125, 175, 175),
        ],
        frame_width=360,
        frame_height=640,
    )

    assert box == Wav2LipBox(top=174, bottom=314, left=91, right=294)
    assert box.as_cli_args() == ["174", "314", "91", "294"]


def test_lower_face_box_from_detections_requires_detection():
    with pytest.raises(ValueError, match="At least one face detection"):
        lower_face_box_from_detections([], frame_width=360, frame_height=640)


def test_estimate_lower_face_box_rejects_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        estimate_lower_face_box(tmp_path / "missing.mp4")

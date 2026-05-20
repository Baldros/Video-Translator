from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Wav2LipBox:
    top: int
    bottom: int
    left: int
    right: int

    def as_cli_args(self) -> list[str]:
        return [
            str(self.top),
            str(self.bottom),
            str(self.left),
            str(self.right),
        ]


def lower_face_box_from_detections(
    detections: list[tuple[int, int, int, int]],
    *,
    frame_width: int,
    frame_height: int,
    top_ratio: float = 0.28,
    bottom_ratio: float = 1.08,
    side_padding_ratio: float = 0.08,
) -> Wav2LipBox:
    if not detections:
        raise ValueError("At least one face detection is required.")

    values = np.asarray(detections, dtype=np.float32)
    x, y, width, height = np.median(values, axis=0)
    side_padding = width * side_padding_ratio

    top = int(round(y + height * top_ratio))
    bottom = int(round(y + height * bottom_ratio))
    left = int(round(x - side_padding))
    right = int(round(x + width + side_padding))

    return Wav2LipBox(
        top=_clamp(top, 0, frame_height),
        bottom=_clamp(bottom, 0, frame_height),
        left=_clamp(left, 0, frame_width),
        right=_clamp(right, 0, frame_width),
    )


def estimate_lower_face_box(
    video_path: str | Path,
    *,
    sample_count: int = 12,
) -> Wav2LipBox:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(
            f"Video file not found for --wav2lip-auto-box: {video}"
        )

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for --wav2lip-auto-box."
        ) from exc

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for face box estimation: {video}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
            raise RuntimeError(f"Could not read video metadata: {video}")

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            raise RuntimeError("OpenCV Haar face cascade could not be loaded.")

        detections: list[tuple[int, int, int, int]] = []
        sample_indexes = np.linspace(0, frame_count - 1, sample_count, dtype=int)
        for frame_index in sample_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(50, 50),
            )
            if len(faces) == 0:
                continue

            x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
            detections.append((int(x), int(y), int(width), int(height)))

        if not detections:
            raise RuntimeError(
                "Could not estimate a Wav2Lip face box automatically. "
                "Use --wav2lip-box as a fallback."
            )

        return lower_face_box_from_detections(
            detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    finally:
        capture.release()


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))

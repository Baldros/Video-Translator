from pathlib import Path

import numpy as np
import soundfile as sf

from omnivoice_translator.segments import SpeechSegment
from omnivoice_translator.video_pipeline import VideoTranslationPipeline
from omnivoice_translator.media import VideoResolution


class FakeVoicePipeline:
    def __init__(self):
        self.synthesis_calls = []

    def transcribe_segments(self, audio_path):
        self.transcribed_audio = Path(audio_path)
        return [
            SpeechSegment(start=0.0, end=1.0, text="hello"),
            SpeechSegment(start=2.0, end=3.0, text="bye"),
        ]

    def translate(self, text, *, source_lang, target_lang):
        assert source_lang == "eng_Latn"
        assert target_lang == "por_Latn"
        return {"hello": "ola", "bye": "tchau"}[text]

    def synthesize(self, text, **kwargs):
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(text.encode("utf-8"))
        self.synthesis_calls.append({"text": text, **kwargs})
        return output


class FakeMedia:
    def __init__(self, output_resolution=None):
        self.output_resolution = output_resolution or VideoResolution(320, 240)

    def extract_audio(self, video_path, output_path, sample_rate=16000):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, np.zeros(sample_rate * 4, dtype=np.float32), sample_rate)
        self.extracted = (Path(video_path), output, sample_rate)
        return output

    def probe_duration(self, media_path):
        self.probed = Path(media_path)
        return 4.0

    def probe_video_resolution(self, media_path):
        path = Path(media_path)
        if path.name.startswith("translated") or path.name == "translated.mp4":
            return self.output_resolution
        return VideoResolution(320, 240)


class FakeLipSync:
    def sync(self, face_video, audio_path, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"translated video")
        self.synced = (Path(face_video), Path(audio_path), output)
        return output


def test_video_translation_pipeline_orchestrates_segmented_flow(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "out" / "translated.mp4"
    work_dir = tmp_path / "work"
    input_video.write_bytes(b"video")
    voice = FakeVoicePipeline()
    media = FakeMedia()
    lip_sync = FakeLipSync()
    rendered = {}

    def fake_render_timeline(segments, *, total_duration, output_path, **kwargs):
        output = Path(output_path)
        output.write_bytes(b"translated audio")
        rendered["segments"] = segments
        rendered["total_duration"] = total_duration
        rendered["output_path"] = output
        rendered["kwargs"] = kwargs
        return output

    result = VideoTranslationPipeline(
        voice_pipeline=voice,
        lip_sync=lip_sync,
        media=media,
        render_timeline=fake_render_timeline,
    ).translate_video(
        input_video=input_video,
        output_video=output_video,
        work_dir=work_dir,
        source_lang="eng_Latn",
        target_lang="por_Latn",
        target_language="Portuguese",
        num_step=8,
    )

    assert result.output_video_path == output_video
    assert result.source_audio_path == work_dir / "source_audio.wav"
    assert result.translated_audio_path == work_dir / "translated_audio.wav"
    assert [segment.translated_text for segment in result.segments] == [
        "ola",
        "tchau",
    ]
    assert [segment.draft_translation for segment in result.segments] == [
        "ola",
        "tchau",
    ]
    assert [call["duration"] for call in voice.synthesis_calls] == [1.0, 1.0]
    assert [call["ref_audio"] for call in voice.synthesis_calls] == [
        work_dir / "references" / "reference_0000.wav",
        work_dir / "references" / "reference_0001.wav",
    ]
    assert [call["ref_text"] for call in voice.synthesis_calls] == [
        "hello",
        "bye",
    ]
    assert (work_dir / "references" / "reference_0000.wav").exists()
    assert media.extracted == (input_video, work_dir / "source_audio.wav", 16000)
    assert rendered["total_duration"] == 4.0
    assert rendered["kwargs"] == {
        "fit_strategy": "pad_trim",
        "max_stretch_ratio": 1.35,
        "condition_audio": True,
        "trim_silence": True,
        "trim_top_db": 35.0,
        "fade_ms": 18.0,
        "target_rms": 0.045,
        "max_gain_db": 8.0,
        "peak_limit": 0.95,
    }
    assert [item.audio_path for item in rendered["segments"]] == [
        work_dir / "segments" / "segment_0000.wav",
        work_dir / "segments" / "segment_0001.wav",
    ]
    assert lip_sync.synced == (
        input_video,
        work_dir / "translated_audio.wav",
        output_video,
    )


def test_video_translation_pipeline_can_merge_short_segments(tmp_path):
    class ShortSegmentVoice(FakeVoicePipeline):
        def transcribe_segments(self, audio_path):
            return [
                SpeechSegment(start=0.0, end=0.8, text="hello"),
                SpeechSegment(start=0.9, end=1.6, text="there"),
            ]

        def translate(self, text, *, source_lang, target_lang):
            return {"hello there": "ola ai"}[text]

    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "translated.mp4"
    input_video.write_bytes(b"video")
    voice = ShortSegmentVoice()

    VideoTranslationPipeline(
        voice_pipeline=voice,
        lip_sync=FakeLipSync(),
        media=FakeMedia(),
        render_timeline=lambda segments, **kwargs: Path(kwargs["output_path"]).write_bytes(b"x")
        or Path(kwargs["output_path"]),
    ).translate_video(
        input_video=input_video,
        output_video=output_video,
        work_dir=tmp_path / "work",
        source_lang="eng_Latn",
        target_lang="por_Latn",
        segment_min_duration=2.0,
        segment_max_gap=0.25,
    )

    assert [call["text"] for call in voice.synthesis_calls] == ["ola ai"]
    assert [call["duration"] for call in voice.synthesis_calls] == [1.6]


def test_video_translation_pipeline_can_use_natural_tts_duration(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "translated.mp4"
    input_video.write_bytes(b"video")
    voice = FakeVoicePipeline()

    VideoTranslationPipeline(
        voice_pipeline=voice,
        lip_sync=FakeLipSync(),
        media=FakeMedia(),
        render_timeline=lambda segments, **kwargs: Path(kwargs["output_path"]).write_bytes(b"x")
        or Path(kwargs["output_path"]),
    ).translate_video(
        input_video=input_video,
        output_video=output_video,
        work_dir=tmp_path / "work",
        source_lang="eng_Latn",
        target_lang="por_Latn",
        tts_duration_mode="natural",
    )

    assert [call["duration"] for call in voice.synthesis_calls] == [None, None]


def test_video_translation_pipeline_retries_short_natural_tts(tmp_path):
    class ShortAudioVoice(FakeVoicePipeline):
        def transcribe_segments(self, audio_path):
            return [SpeechSegment(start=0.0, end=4.0, text="hello")]

        def translate(self, text, *, source_lang, target_lang):
            return "ola"

        def synthesize(self, text, **kwargs):
            output = Path(kwargs["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            sample_rate = 1000
            duration = kwargs["duration"] if kwargs["duration"] is not None else 1.0
            sf.write(
                output,
                np.ones(int(sample_rate * duration), dtype=np.float32) * 0.05,
                sample_rate,
            )
            self.synthesis_calls.append({"text": text, **kwargs})
            return output

    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "translated.mp4"
    input_video.write_bytes(b"video")
    voice = ShortAudioVoice()

    result = VideoTranslationPipeline(
        voice_pipeline=voice,
        lip_sync=FakeLipSync(),
        media=FakeMedia(),
        render_timeline=lambda segments, **kwargs: Path(kwargs["output_path"]).write_bytes(b"x")
        or Path(kwargs["output_path"]),
    ).translate_video(
        input_video=input_video,
        output_video=output_video,
        work_dir=tmp_path / "work",
        source_lang="eng_Latn",
        target_lang="por_Latn",
        tts_duration_mode="natural",
        tts_min_duration_ratio=0.8,
    )

    assert [call["duration"] for call in voice.synthesis_calls] == [None, 4.0]
    assert result.segments[0].synthesis_retry is True
    assert result.segments[0].generated_duration_s == 4.0


def test_video_translation_pipeline_honors_explicit_reference_audio(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "translated.mp4"
    reference = tmp_path / "reference.wav"
    input_video.write_bytes(b"video")
    reference.write_bytes(b"reference")
    voice = FakeVoicePipeline()

    VideoTranslationPipeline(
        voice_pipeline=voice,
        lip_sync=FakeLipSync(),
        media=FakeMedia(),
        render_timeline=lambda segments, **kwargs: Path(kwargs["output_path"]).write_bytes(b"x")
        or Path(kwargs["output_path"]),
    ).translate_video(
        input_video=input_video,
        output_video=output_video,
        work_dir=tmp_path / "work",
        source_lang="eng_Latn",
        target_lang="por_Latn",
        ref_audio=reference,
    )

    assert [call["ref_audio"] for call in voice.synthesis_calls] == [
        reference,
        reference,
    ]


def test_video_translation_pipeline_can_use_full_source_reference_audio(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "translated.mp4"
    work_dir = tmp_path / "work"
    input_video.write_bytes(b"video")
    voice = FakeVoicePipeline()

    VideoTranslationPipeline(
        voice_pipeline=voice,
        lip_sync=FakeLipSync(),
        media=FakeMedia(),
        render_timeline=lambda segments, **kwargs: Path(kwargs["output_path"]).write_bytes(b"x")
        or Path(kwargs["output_path"]),
    ).translate_video(
        input_video=input_video,
        output_video=output_video,
        work_dir=work_dir,
        source_lang="eng_Latn",
        target_lang="por_Latn",
        reference_audio_mode="source",
    )

    assert [call["ref_audio"] for call in voice.synthesis_calls] == [
        work_dir / "source_audio.wav",
        work_dir / "source_audio.wav",
    ]


def test_video_translation_pipeline_rejects_resolution_change(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "translated.mp4"
    input_video.write_bytes(b"video")

    pipeline = VideoTranslationPipeline(
        voice_pipeline=FakeVoicePipeline(),
        lip_sync=FakeLipSync(),
        media=FakeMedia(output_resolution=VideoResolution(640, 360)),
        render_timeline=lambda segments, **kwargs: Path(kwargs["output_path"]).write_bytes(b"x")
        or Path(kwargs["output_path"]),
    )

    import pytest

    with pytest.raises(RuntimeError, match="changed video resolution"):
        pipeline.translate_video(
            input_video=input_video,
            output_video=output_video,
            work_dir=tmp_path / "work",
            source_lang="eng_Latn",
            target_lang="por_Latn",
        )

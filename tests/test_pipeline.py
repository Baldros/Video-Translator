from pathlib import Path

import numpy as np

from omnivoice_translator.pipeline import VoiceTranslationPipeline


class FakeTTS:
    sampling_rate = 24000

    def generate(self, **kwargs):
        return [np.zeros(240, dtype=np.float32)]


class RecordingTTS:
    sampling_rate = 10

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [np.zeros(10, dtype=np.float32)]


class FakeSegmentASR:
    def __call__(self, audio_path, *, return_timestamps):
        assert return_timestamps is True
        return {
            "chunks": [
                {"timestamp": (0.0, 0.4), "text": " first "},
                {"timestamp": (0.4, None), "text": "second"},
            ]
        }


def test_synthesize_creates_output_parent(tmp_path):
    pipeline = VoiceTranslationPipeline(
        asr=None,
        mt_model=None,
        mt_tokenizer=None,
        tts=FakeTTS(),
        device="cpu",
    )

    output = tmp_path / "nested" / "voice.wav"
    result = pipeline.synthesize(
        "teste",
        language="Portuguese",
        output_path=output,
    )

    assert result == output
    assert Path(output).exists()


def test_synthesize_forwards_duration_to_tts(tmp_path):
    tts = RecordingTTS()
    pipeline = VoiceTranslationPipeline(
        asr=None,
        mt_model=None,
        mt_tokenizer=None,
        tts=tts,
        device="cpu",
    )

    pipeline.synthesize(
        "teste",
        language="Portuguese",
        output_path=tmp_path / "voice.wav",
        duration=1.2,
        guidance_scale=2.0,
    )

    assert tts.kwargs["duration"] == 1.2
    assert tts.kwargs["guidance_scale"] == 2.0


def test_transcribe_segments_uses_asr_timestamps_and_audio_duration(tmp_path):
    audio = tmp_path / "source.wav"
    sf_data = np.zeros(10, dtype=np.float32)
    import soundfile as sf

    sf.write(audio, sf_data, 10)
    pipeline = VoiceTranslationPipeline(
        asr=FakeSegmentASR(),
        mt_model=None,
        mt_tokenizer=None,
        tts=FakeTTS(),
        device="cpu",
    )

    segments = pipeline.transcribe_segments(audio)

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0.0, 0.4, "first"),
        (0.4, 1.0, "second"),
    ]

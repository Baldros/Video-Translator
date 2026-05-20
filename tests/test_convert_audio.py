import pytest

from omnivoice_translator.convert_audio import convert_wav_to_mp4


def test_rejects_non_wav_input(tmp_path):
    source = tmp_path / "audio.mp3"
    source.write_bytes(b"not real audio")

    with pytest.raises(ValueError, match="Expected a .wav input"):
        convert_wav_to_mp4(source, tmp_path / "audio.mp4")


def test_rejects_non_mp4_or_m4a_output(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"not real audio")

    with pytest.raises(ValueError, match="Expected a .mp4 or .m4a output"):
        convert_wav_to_mp4(source, tmp_path / "audio.wav")

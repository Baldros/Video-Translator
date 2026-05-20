from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from omnivoice_translator.device import get_best_device, get_default_dtype
from omnivoice_translator.segments import SpeechSegment, normalize_asr_segments


@dataclass(frozen=True)
class TranslationResult:
    source_text: str
    translated_text: str
    output_path: Path
    sampling_rate: int


class VoiceTranslationPipeline:
    def __init__(
        self,
        asr: Any,
        mt_model: Any,
        mt_tokenizer: Any,
        tts: Any,
        device: str,
    ) -> None:
        self.asr = asr
        self.mt_model = mt_model
        self.mt_tokenizer = mt_tokenizer
        self.tts = tts
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        asr_model: str = "openai/whisper-large-v3-turbo",
        mt_model: str = "facebook/nllb-200-distilled-600M",
        tts_model: str = "k2-fsa/OmniVoice",
        device: str | None = None,
    ) -> "VoiceTranslationPipeline":
        import torch
        from omnivoice import OmniVoice
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        resolved_device = device or get_best_device()
        torch_dtype = get_default_dtype(resolved_device)
        hf_device = 0 if resolved_device == "cuda" else -1

        asr = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            torch_dtype=torch_dtype,
            device=hf_device,
        )
        mt_tokenizer = AutoTokenizer.from_pretrained(mt_model)
        mt = AutoModelForSeq2SeqLM.from_pretrained(mt_model).to(resolved_device)
        tts = OmniVoice.from_pretrained(
            tts_model,
            device_map=resolved_device,
            dtype=torch_dtype,
        )

        return cls(
            asr=asr,
            mt_model=mt,
            mt_tokenizer=mt_tokenizer,
            tts=tts,
            device=resolved_device,
        )

    def transcribe(self, audio_path: str | Path) -> str:
        result = self.asr(str(audio_path), return_timestamps=False)
        text = result["text"] if isinstance(result, dict) else str(result)
        return text.strip()

    def transcribe_segments(self, audio_path: str | Path) -> list[SpeechSegment]:
        result = self.asr(str(audio_path), return_timestamps=True)
        return normalize_asr_segments(
            result,
            fallback_duration=_audio_duration(audio_path),
        )

    def translate(
        self,
        text: str,
        *,
        source_lang: str | None,
        target_lang: str,
        max_new_tokens: int = 512,
    ) -> str:
        if source_lang:
            self.mt_tokenizer.src_lang = source_lang

        forced_bos_token_id = self.mt_tokenizer.convert_tokens_to_ids(target_lang)
        inputs = self.mt_tokenizer(text, return_tensors="pt", truncation=True).to(
            self.device
        )
        outputs = self.mt_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
        )
        return self.mt_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def synthesize(
        self,
        text: str,
        *,
        language: str | None,
        output_path: str | Path,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        voice_instruct: str | None = None,
        speed: float = 1.0,
        num_step: int = 32,
        duration: float | None = None,
        guidance_scale: float | None = None,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        generate_kwargs = {
            "text": text,
            "language": language,
            "ref_audio": str(ref_audio) if ref_audio else None,
            "ref_text": ref_text,
            "instruct": voice_instruct,
            "speed": speed,
            "num_step": num_step,
        }
        if duration is not None:
            generate_kwargs["duration"] = duration
        if guidance_scale is not None:
            generate_kwargs["guidance_scale"] = guidance_scale

        audio = self.tts.generate(**generate_kwargs)
        sf.write(output, audio[0], self.tts.sampling_rate)
        return output

    def translate_file(
        self,
        *,
        input_path: str | Path,
        output_path: str | Path,
        source_lang: str | None,
        target_lang: str,
        target_language: str | None = None,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        voice_instruct: str | None = None,
        speed: float = 1.0,
        num_step: int = 32,
    ) -> TranslationResult:
        source_text = self.transcribe(input_path)
        translated_text = self.translate(
            source_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        output = self.synthesize(
            translated_text,
            language=target_language,
            output_path=output_path,
            ref_audio=ref_audio,
            ref_text=ref_text,
            voice_instruct=voice_instruct,
            speed=speed,
            num_step=num_step,
        )
        return TranslationResult(
            source_text=source_text,
            translated_text=translated_text,
            output_path=output,
            sampling_rate=self.tts.sampling_rate,
        )


def _audio_duration(audio_path: str | Path) -> float | None:
    try:
        info = sf.info(audio_path)
    except (RuntimeError, OSError):
        return None
    if info.samplerate <= 0:
        return None
    return float(info.frames) / float(info.samplerate)

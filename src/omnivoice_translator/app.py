from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr

from omnivoice_translator.pipeline import VoiceTranslationPipeline


COMMON_NLLB_LANGS = {
    "English": "eng_Latn",
    "Portuguese": "por_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Chinese Simplified": "zho_Hans",
}


def build_demo(translator: VoiceTranslationPipeline) -> gr.Blocks:
    def run(
        input_audio,
        source_lang_name,
        target_lang_name,
        ref_audio,
        ref_text,
        voice_instruct,
        speed,
        num_step,
    ):
        if not input_audio:
            return None, "Envie um audio de entrada.", "", ""

        source_lang = COMMON_NLLB_LANGS.get(source_lang_name) if source_lang_name else None
        target_lang = COMMON_NLLB_LANGS[target_lang_name]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = Path(handle.name)

        result = translator.translate_file(
            input_path=input_audio,
            output_path=output_path,
            source_lang=source_lang,
            target_lang=target_lang,
            target_language=target_lang_name,
            ref_audio=ref_audio,
            ref_text=ref_text or None,
            voice_instruct=voice_instruct or None,
            speed=float(speed),
            num_step=int(num_step),
        )
        return (
            str(result.output_path),
            "Concluido.",
            result.source_text,
            result.translated_text,
        )

    with gr.Blocks(title="OmniVoice Translator") as demo:
        gr.Markdown("# OmniVoice Translator")
        gr.Markdown(
            "Traducao de voz baseada em ASR + traducao NLLB + OmniVoice TTS."
        )
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(label="Audio de entrada", type="filepath")
                source_lang = gr.Dropdown(
                    ["Auto"] + list(COMMON_NLLB_LANGS),
                    value="Auto",
                    label="Idioma de origem",
                )
                target_lang = gr.Dropdown(
                    list(COMMON_NLLB_LANGS),
                    value="Portuguese",
                    label="Idioma alvo",
                )
                ref_audio = gr.Audio(
                    label="Audio de referencia para clonagem de voz",
                    type="filepath",
                )
                ref_text = gr.Textbox(label="Texto do audio de referencia", lines=2)
                voice_instruct = gr.Textbox(
                    label="Voice design",
                    placeholder='Ex: "female, low pitch"',
                )
                speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Velocidade")
                num_step = gr.Slider(4, 64, value=32, step=1, label="Passos TTS")
                submit = gr.Button("Traduzir", variant="primary")
            with gr.Column():
                output_audio = gr.Audio(label="Audio traduzido", type="filepath")
                status = gr.Textbox(label="Status")
                source_text = gr.Textbox(label="Transcricao", lines=5)
                translated_text = gr.Textbox(label="Traducao", lines=5)

        submit.click(
            run,
            inputs=[
                input_audio,
                source_lang,
                target_lang,
                ref_audio,
                ref_text,
                voice_instruct,
                speed,
                num_step,
            ],
            outputs=[output_audio, status, source_text, translated_text],
        )

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the OmniVoice translator UI.")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--mt-model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--tts-model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    translator = VoiceTranslationPipeline.from_pretrained(
        asr_model=args.asr_model,
        mt_model=args.mt_model,
        tts_model=args.tts_model,
        device=args.device,
    )
    build_demo(translator).queue().launch(
        server_name=args.ip,
        server_port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

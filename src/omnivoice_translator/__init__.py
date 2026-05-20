__all__ = ["VoiceTranslationPipeline"]


def __getattr__(name: str):
    if name == "VoiceTranslationPipeline":
        from omnivoice_translator.pipeline import VoiceTranslationPipeline

        return VoiceTranslationPipeline
    raise AttributeError(name)

from omnivoice_translator.speak import resolve_language


def test_resolve_language_prefers_explicit_language():
    assert resolve_language("Russian", "por_Latn") == "Russian"


def test_resolve_language_maps_nllb_code():
    assert resolve_language(None, "rus_Cyrl") == "Russian"
    assert resolve_language(None, "zho_Hans") == "Chinese"

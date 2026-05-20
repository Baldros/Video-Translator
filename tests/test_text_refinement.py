from omnivoice_translator.text_refinement import normalize_spoken_text


def test_normalize_spoken_text_collapses_whitespace():
    assert normalize_spoken_text(" Hello\n\n   world  ") == "Hello world"

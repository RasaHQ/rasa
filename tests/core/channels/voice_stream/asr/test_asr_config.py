import pytest

from rasa.core.channels.voice_stream.asr.asr_engine import (
    ASREngineConfig,
    ASRLanguageMapEntry,
)


def test_asr_language_field_emits_deprecation_warning():
    with pytest.warns(FutureWarning, match="language_map"):
        ASREngineConfig(language="en-US")


def test_asr_model_field_emits_deprecation_warning():
    with pytest.warns(FutureWarning, match="language_map"):
        ASREngineConfig(model="nova-2-general")


def test_asr_deprecated_language_config_is_usable_after_apply():
    with pytest.warns(FutureWarning):
        cfg = ASREngineConfig(language="en-US")
    result = cfg.apply_deprecated_fields("en")
    assert result.language_map is not None
    assert "en" in result.language_map


def test_asr_config_without_deprecated_fields_unchanged():
    cfg = ASREngineConfig(language_map={"en": ASRLanguageMapEntry(language="en-US")})
    result = cfg.apply_deprecated_fields("en")
    assert result is cfg  # no-op returns same object


def test_asr_merge_preserves_non_none_values():
    base = ASREngineConfig(
        language_map={"en": ASRLanguageMapEntry(language="en-US")},
        keep_alive_interval=10,
    )
    override = ASREngineConfig(keep_alive_interval=20)
    merged = base.merge(override)
    assert merged.keep_alive_interval == 20
    assert merged.language_map == base.language_map


def test_asr_language_folded_into_language_map():
    with pytest.warns(FutureWarning):
        cfg = ASREngineConfig(language="en-US")
    result = cfg.apply_deprecated_fields("en")
    assert result.language is None
    assert result.language_map["en"].language == "en-US"


def test_asr_language_and_model_folded_into_language_map():
    with pytest.warns(FutureWarning):
        cfg = ASREngineConfig(language="en-US", model="nova-2-general")
    result = cfg.apply_deprecated_fields("en")
    assert result.language is None
    assert result.model is None
    assert result.language_map["en"].language == "en-US"
    assert result.language_map["en"].model == "nova-2-general"


def test_asr_language_map_dicts_coerced_to_entries():
    cfg = ASREngineConfig(language_map={"en": {"language": "en-US", "model": "nova-2"}})
    assert isinstance(cfg.language_map["en"], ASRLanguageMapEntry)
    assert cfg.language_map["en"].language == "en-US"


def test_asr_engine_config_field_defaults():
    cfg = ASREngineConfig()
    assert cfg.keep_alive_interval == 5
    assert cfg.language_map is None
    assert cfg.language is None
    assert cfg.model is None


def test_asr_azure_config_static_defaults():
    from rasa.core.channels.voice_stream.asr.azure import AzureASRConfig

    cfg = AzureASRConfig(language_map={"en": ASRLanguageMapEntry(language="en-US")})
    assert cfg.speech_region is None


def test_asr_unknown_field_emits_warning():
    with pytest.warns(UserWarning, match="'lang'"):
        cfg = ASREngineConfig(lang="en")
    assert cfg.model_extra == {"lang": "en"}


def test_asr_multiple_unknown_fields_emit_warning():
    with pytest.warns(UserWarning, match="Unknown ASR config"):
        cfg = ASREngineConfig(foo="bar", baz=123)
    assert "foo" in cfg.model_extra
    assert "baz" in cfg.model_extra


def test_asr_unknown_fields_silent_during_merge():
    """Unknown fields should not trigger a warning during internal merge."""
    base = ASREngineConfig(language_map={"en": ASRLanguageMapEntry(language="en-US")})
    # merge uses context={"merging": True}, so no warning should fire
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base.merge(None)  # should not raise


def test_asr_deepgram_config_static_defaults():
    from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR

    cfg = DeepgramASR.get_default_config("en")
    assert cfg.language_map is not None
    assert "en" in cfg.language_map
    assert cfg.language_map["en"].language == "en"
    assert cfg.language_map["en"].model == "nova-3"


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "language": "en-US",
            "language_map": {"en": ASRLanguageMapEntry(language="en-US")},
        },
        {
            "model": "nova-2",
            "language_map": {"en": ASRLanguageMapEntry(language="en-US")},
        },
        {
            "language": "en-US",
            "model": "nova-2",
            "language_map": {"en": ASRLanguageMapEntry(language="en-US")},
        },
    ],
)
def test_asr_config_raises_when_deprecated_fields_and_language_map_both_set(kwargs):
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError, match="language.*model.*language_map|language_map"
    ):
        ASREngineConfig(**kwargs)

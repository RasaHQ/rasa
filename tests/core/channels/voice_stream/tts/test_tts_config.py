import pytest
import structlog

from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSEngineConfig,
    TTSLanguageMapEntry,
)


def test_tts_language_field_emits_deprecation_warning():
    with pytest.warns(FutureWarning, match="language_map"):
        TTSEngineConfig(language="en-US")


def test_tts_voice_field_emits_deprecation_warning():
    with pytest.warns(FutureWarning, match="language_map"):
        TTSEngineConfig(voice="en-US-JennyNeural")


def test_tts_model_field_emits_deprecation_warning():
    with pytest.warns(FutureWarning, match="language_map"):
        TTSEngineConfig(model="some-model")


def test_tts_deprecated_config_is_usable_after_apply():
    with pytest.warns(FutureWarning):
        cfg = TTSEngineConfig(language="en-US", voice="en-US-JennyNeural")
    result = cfg.apply_deprecated_fields("en")
    assert result.language_map is not None
    assert "en" in result.language_map


def test_tts_config_without_deprecated_fields_unchanged():
    cfg = TTSEngineConfig(
        language_map={"en": TTSLanguageMapEntry(language="en-US", voice="Jenny")}
    )
    result = cfg.apply_deprecated_fields("en")
    assert result is cfg  # no-op returns the same object


def test_tts_merge_preserves_non_none_values():
    base = TTSEngineConfig(
        language_map={"en": TTSLanguageMapEntry(language="en-US", voice="Jenny")},
        timeout=60,
    )
    override = TTSEngineConfig(timeout=90)
    merged = base.merge(override)
    assert merged.timeout == 90
    assert merged.language_map == base.language_map


def test_tts_language_and_voice_folded_into_language_map():
    with pytest.warns(FutureWarning):
        cfg = TTSEngineConfig(language="en-US", voice="en-US-JennyNeural")
    result = cfg.apply_deprecated_fields("en")
    assert result.language is None
    assert result.voice is None
    assert result.language_map["en"].language == "en-US"
    assert result.language_map["en"].voice == "en-US-JennyNeural"


def test_tts_unknown_field_emits_warning():
    with pytest.warns(UserWarning, match="'langauge'"):
        cfg = TTSEngineConfig(langauge="en-US")
    assert cfg.model_extra == {"langauge": "en-US"}


def test_tts_multiple_unknown_fields_emit_warning():
    with pytest.warns(UserWarning, match="Unknown TTS config"):
        cfg = TTSEngineConfig(foo="bar", baz=123)
    assert "foo" in cfg.model_extra
    assert "baz" in cfg.model_extra


def test_tts_unknown_fields_silent_during_merge():
    """Unknown fields should not trigger a warning during internal merge."""
    base = TTSEngineConfig(
        language_map={"en": TTSLanguageMapEntry(language="en-US", voice="Jenny")}
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base.merge(None)  # should not raise


def test_tts_language_map_dicts_coerced_to_entries():
    cfg = TTSEngineConfig(language_map={"en": {"language": "en-US", "voice": "Jenny"}})
    assert isinstance(cfg.language_map["en"], TTSLanguageMapEntry)
    assert cfg.language_map["en"].language == "en-US"


def test_tts_engine_config_field_defaults():
    cfg = TTSEngineConfig()
    assert cfg.timeout == 30
    assert cfg.language_map is None
    assert cfg.language is None
    assert cfg.voice is None
    assert cfg.model is None


def test_tts_azure_config_speech_region_default():
    from rasa.core.channels.voice_stream.tts.azure import AzureTTSConfig

    cfg = AzureTTSConfig(
        language_map={"en": TTSLanguageMapEntry(language="en-US", voice="Jenny")}
    )
    assert cfg.speech_region == "eastus"
    assert cfg.timeout == 30


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "language": "en-US",
            "language_map": {"en": TTSLanguageMapEntry(language="en-US")},
        },
        {
            "voice": "Jenny",
            "language_map": {"en": TTSLanguageMapEntry(language="en-US")},
        },
        {
            "model": "some-model",
            "language_map": {"en": TTSLanguageMapEntry(language="en-US")},
        },
        {
            "language": "en-US",
            "voice": "Jenny",
            "model": "some-model",
            "language_map": {"en": TTSLanguageMapEntry(language="en-US")},
        },
    ],
)
def test_tts_config_raises_when_deprecated_fields_and_language_map_both_set(kwargs):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="language_map"):
        TTSEngineConfig(**kwargs)


def test_tts_deepgram_config_defaults():
    from rasa.core.channels.voice_stream.tts.deepgram import DeepgramTTS

    cfg = DeepgramTTS.get_default_config("en")
    assert cfg.language_map is not None
    assert "en" in cfg.language_map
    assert cfg.language_map["en"].model is not None


def test_tts_validate_language_map_keys_warns_when_additional_language_missing():
    """Tests that warning is raised when additional language keys are missing."""
    cfg = TTSEngineConfig(
        language_map={"en": TTSLanguageMapEntry(language="en-US", voice="Jenny")}
    )
    with structlog.testing.capture_logs() as cap:
        cfg.validate_language_map_keys(rasa_language="en", additional_languages=["de"])

    warning_events = [
        log
        for log in cap
        if log.get("log_level") == "warning"
        and log.get("event") == "tts_engine_config.missing_language_map_keys"
    ]
    assert len(warning_events) == 1
    assert "de" in warning_events[0]["message"]


def test_tts_validate_language_map_keys_warns_for_each_missing_additional_language():
    """Tests that warning is raised when additional language keys are missing."""
    cfg = TTSEngineConfig(
        language_map={"en": TTSLanguageMapEntry(language="en-US", voice="Jenny")}
    )
    with structlog.testing.capture_logs() as cap:
        cfg.validate_language_map_keys(
            rasa_language="en", additional_languages=["de", "fr"]
        )

    warning_events = [
        log
        for log in cap
        if log.get("log_level") == "warning"
        and log.get("event") == "tts_engine_config.missing_language_map_keys"
    ]
    assert len(warning_events) == 1
    assert "de" in warning_events[0]["message"]
    assert "fr" in warning_events[0]["message"]


def test_tts_validate_language_map_keys_no_warning_when_all_keys_present():
    """Tests that there is no warning when all keys are present."""
    cfg = TTSEngineConfig(
        language_map={
            "en": TTSLanguageMapEntry(language="en-US", voice="Jenny"),
            "de": TTSLanguageMapEntry(language="de-DE", voice="Hedda"),
        }
    )
    with structlog.testing.capture_logs() as cap:
        cfg.validate_language_map_keys(rasa_language="en", additional_languages=["de"])

    warning_events = [
        log
        for log in cap
        if log.get("log_level") == "warning"
        and log.get("event") == "tts_engine_config.missing_language_map_keys"
    ]
    assert warning_events == []


def test_tts_validate_language_map_keys_no_warning_when_no_additional_languages():
    """Tests that there is no warning when all keys are present."""
    cfg = TTSEngineConfig(
        language_map={
            "en": TTSLanguageMapEntry(language="en-US", voice="Jenny"),
        }
    )
    with structlog.testing.capture_logs() as cap:
        cfg.validate_language_map_keys(rasa_language="en", additional_languages=None)

    warning_events = [
        log
        for log in cap
        if log.get("log_level") == "warning"
        and log.get("event") == "tts_engine_config.missing_language_map_keys"
    ]
    assert warning_events == []


def test_tts_validate_language_map_keys_no_warning_when_empty_additional_languages():
    """Tests that there is no warning when all keys are present."""
    cfg = TTSEngineConfig(
        language_map={
            "en": TTSLanguageMapEntry(language="en-US", voice="Jenny"),
        }
    )
    with structlog.testing.capture_logs() as cap:
        cfg.validate_language_map_keys(rasa_language="en", additional_languages=[])

    warning_events = [
        log
        for log in cap
        if log.get("log_level") == "warning"
        and log.get("event") == "tts_engine_config.missing_language_map_keys"
    ]
    assert warning_events == []


def test_tts_validate_language_map_keys_warns_only_for_absent_additional_languages():
    """Tests that warning is raised when absent additional language keys are missing."""
    cfg = TTSEngineConfig(
        language_map={
            "en": TTSLanguageMapEntry(language="en-US", voice="Jenny"),
            "de": TTSLanguageMapEntry(language="de-DE", voice="Hedda"),
        }
    )
    with structlog.testing.capture_logs() as cap:
        cfg.validate_language_map_keys(
            rasa_language="en", additional_languages=["de", "fr"]
        )

    warning_events = [
        log
        for log in cap
        if log.get("log_level") == "warning"
        and log.get("event") == "tts_engine_config.missing_language_map_keys"
    ]
    assert len(warning_events) == 1
    assert "fr" in warning_events[0]["message"]
    assert "de" not in warning_events[0]["message"]
    assert "en" not in warning_events[0]["message"]

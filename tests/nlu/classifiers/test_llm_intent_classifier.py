from pathlib import Path
from typing import Any, Generator, List, Optional
from unittest.mock import Mock, patch

import pytest
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.docstore.document import Document
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms.fake import FakeListLLM
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from rasa.nlu.classifiers.llm_intent_classifier import (
    RASA_PRO_BETA_LLM_INTENT,
    LLMIntentClassifier,
)


@pytest.fixture()
def default_patched_llm_intent_classifier(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[LLMIntentClassifier, None, None]:
    monkeypatch.setenv(RASA_PRO_BETA_LLM_INTENT, "true")

    with patch(
        "rasa.nlu.classifiers.llm_intent_classifier.llm_factory",
        Mock(return_value=FakeListLLM(responses=["greet"])),
    ):
        with patch(
            "rasa.nlu.classifiers.llm_intent_classifier.embedder_factory",
            Mock(return_value=FakeEmbeddings(size=100)),
        ):
            yield LLMIntentClassifier.create(
                LLMIntentClassifier.get_default_config(),
                default_model_storage,
                Resource("llmintentclassifier"),
                default_execution_context,
            )


@pytest.mark.skip(
    reason=(
        "LLMIntentClassifier is marked for removal in the following ticket:"
        "https://rasahq.atlassian.net/browse/ENG-1199"
    )
)
def test_persist_and_load(
    default_patched_llm_intent_classifier: LLMIntentClassifier,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> None:
    training_data = TrainingData(
        training_examples=[
            Message(data={TEXT: "hello", INTENT: "greet"}),
            Message(data={TEXT: "goodbye", INTENT: "bye"}),
        ]
    )

    resource = default_patched_llm_intent_classifier.train(training_data)

    loaded = LLMIntentClassifier.load(
        LLMIntentClassifier.get_default_config(),
        default_model_storage,
        resource,
        default_execution_context,
    )
    assert loaded is not None
    assert loaded.example_docsearch is not None
    assert loaded.intent_docsearch is not None
    assert loaded.available_intents == {"greet", "bye"}


def test_loading_from_storage_fail(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RASA_PRO_BETA_LLM_INTENT, "true")

    with patch(
        "rasa.nlu.classifiers.llm_intent_classifier.embedder_factory",
        Mock(return_value=FakeEmbeddings(size=100)),
    ):
        loaded = LLMIntentClassifier.load(
            LLMIntentClassifier.get_default_config(),
            default_model_storage,
            Resource("test"),
            default_execution_context,
        )
        assert isinstance(loaded, LLMIntentClassifier)
        assert loaded.example_docsearch is None


@pytest.mark.skip(
    reason=(
        "LLMIntentClassifier is marked for removal in the following ticket:"
        "https://rasahq.atlassian.net/browse/ENG-1199"
    )
)
def test_find_closest_examples(
    default_patched_llm_intent_classifier: LLMIntentClassifier,
) -> None:
    training_data = TrainingData(
        training_examples=[
            Message(data={TEXT: "hello", INTENT: "greet"}),
            Message(data={TEXT: "goodbye", INTENT: "bye"}),
        ]
    )

    default_patched_llm_intent_classifier.train(training_data)

    examples = default_patched_llm_intent_classifier.select_few_shot_examples(
        Message(data={TEXT: "hello"})
    )

    assert len(examples) == 2
    assert (
        Document(page_content="goodbye", metadata={INTENT: "bye", TEXT: "goodbye"})
        in examples
    )
    assert (
        Document(page_content="hello", metadata={INTENT: "greet", TEXT: "hello"})
        in examples
    )


def test_find_closest_examples_with_no_examples(
    default_patched_llm_intent_classifier: LLMIntentClassifier,
) -> None:
    examples = default_patched_llm_intent_classifier.select_few_shot_examples(
        Message(data={TEXT: "hello"})
    )

    assert len(examples) == 0


@pytest.mark.skip(
    reason=(
        "LLMIntentClassifier is marked for removal in the following ticket:"
        "https://rasahq.atlassian.net/browse/ENG-1199"
    )
)
async def test_process_sets_intent(
    default_patched_llm_intent_classifier: LLMIntentClassifier,
) -> None:
    training_data = TrainingData(
        training_examples=[
            Message(data={TEXT: "hello", INTENT: "greet"}),
            Message(data={TEXT: "goodbye", INTENT: "bye"}),
        ]
    )

    default_patched_llm_intent_classifier.train(training_data)

    message = Message(data={TEXT: "hello"})
    await default_patched_llm_intent_classifier.process([message])

    assert message.get(INTENT) == {
        "confidence": 1.0,
        "metadata": {"llm_intent": "greet"},
        "name": "greet",
    }


def test_process_sets_no_intent_with_no_examples(
    default_patched_llm_intent_classifier: LLMIntentClassifier,
) -> None:
    message = Message(data={TEXT: "hello"})
    default_patched_llm_intent_classifier.prompt_template = "test_template"
    default_patched_llm_intent_classifier.process([message])

    assert message.get(INTENT) is None


@pytest.mark.skip(
    reason=(
        "LLMIntentClassifier is marked for removal in the following ticket:"
        "https://rasahq.atlassian.net/browse/ENG-1199"
    )
)
async def test_process_sends_default_prompt(
    default_patched_llm_intent_classifier: LLMIntentClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AssertingLLM(FakeListLLM):
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            """First try to lookup in queries, else return 'foo' or 'bar'."""
            # check that there are some instructions
            assert "The intent should be one of the following" in prompt
            # check that the examples are there
            assert "Message: hello" in prompt
            assert "Intent: greet" in prompt
            assert "Message: goodbye" in prompt
            assert "Intent: bye" in prompt
            # check that the intents are listed
            assert "- bye" in prompt
            assert "- greet" in prompt
            # check that the current message is there
            assert "Message: howdy" in prompt
            return super()._call(prompt, stop, run_manager)

    monkeypatch.setenv(RASA_PRO_BETA_LLM_INTENT, "true")
    training_data = TrainingData(
        training_examples=[
            Message(data={TEXT: "hello", INTENT: "greet"}),
            Message(data={TEXT: "goodbye", INTENT: "bye"}),
        ]
    )
    mock = Mock(return_value=AssertingLLM(responses=["greet"]))

    with patch("rasa.nlu.classifiers.llm_intent_classifier.llm_factory", mock):
        default_patched_llm_intent_classifier.train(training_data)

        message = Message(data={TEXT: "howdy"})
        await default_patched_llm_intent_classifier.process([message])  #

    mock.assert_called_once()


@pytest.mark.skip(
    reason=(
        "LLMIntentClassifier is marked for removal in the following ticket:"
        "https://rasahq.atlassian.net/browse/ENG-1199"
    )
)
async def test_llm_intent_classification_prompt_init_custom(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RASA_PRO_BETA_LLM_INTENT, "true")
    intent_classifier = LLMIntentClassifier(
        {"prompt": "data/prompt_templates/test_prompt.jinja2"},
        default_model_storage,
        Resource("llmintentclassifier"),
        default_execution_context,
    )
    assert intent_classifier.prompt_template.startswith(
        "Identify the user's message intent"
    )

    training_data = TrainingData(
        training_examples=[
            Message(data={TEXT: "hello", INTENT: "greet"}),
        ]
    )
    with patch(
        "rasa.nlu.classifiers.llm_intent_classifier.llm_factory",
        Mock(return_value=FakeListLLM(responses=["greet"])),
    ):
        with patch(
            "rasa.nlu.classifiers.llm_intent_classifier.embedder_factory",
            Mock(return_value=FakeEmbeddings(size=100)),
        ):
            resource = intent_classifier.train(training_data)
            loaded = LLMIntentClassifier.load(
                LLMIntentClassifier.get_default_config(),
                default_model_storage,
                resource,
                default_execution_context,
            )
    assert loaded.prompt_template.startswith("Identify the user's message")


@pytest.mark.skip(
    reason=(
        "LLMIntentClassifier is marked for removal in the following ticket:"
        "https://rasahq.atlassian.net/browse/ENG-1199"
    )
)
async def test_llm_intent_classification_prompt_init_default(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RASA_PRO_BETA_LLM_INTENT, "true")
    intent_classifier = LLMIntentClassifier(
        {},
        default_model_storage,
        Resource("llmintentclassifier"),
        default_execution_context,
    )
    assert intent_classifier.prompt_template.startswith("Label a users message")

    training_data = TrainingData(
        training_examples=[
            Message(data={TEXT: "hello", INTENT: "greet"}),
        ]
    )
    with patch(
        "rasa.nlu.classifiers.llm_intent_classifier.llm_factory",
        Mock(return_value=FakeListLLM(responses=["greet"])),
    ):
        with patch(
            "rasa.nlu.classifiers.llm_intent_classifier.embedder_factory",
            Mock(return_value=FakeEmbeddings(size=100)),
        ):
            resource = intent_classifier.train(training_data)
            loaded = LLMIntentClassifier.load(
                LLMIntentClassifier.get_default_config(),
                default_model_storage,
                resource,
                default_execution_context,
            )
    assert loaded.prompt_template.startswith("Label a users message")


async def test_llm_intent_classifier_fingerprint_addon_diff_in_prompt_template(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RASA_PRO_BETA_LLM_INTENT, "true")
    prompt_dir = Path(tmp_path) / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "llm_intent_classifier_prompt.jinja2"
    prompt_file.write_text("This is a test prompt")

    config = {"prompt": str(prompt_file)}

    intent_classifier = LLMIntentClassifier(
        config,
        default_model_storage,
        Resource("llmintentclassifier"),
        default_execution_context,
    )

    fingerprint_1 = intent_classifier.fingerprint_addon(config)

    prompt_file.write_text("This is a test prompt. It has been changed.")

    fingerprint_2 = intent_classifier.fingerprint_addon(config)
    assert fingerprint_1 != fingerprint_2


async def test_llm_intent_classifier_fingerprint_addon_no_diff_in_prompt_template(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RASA_PRO_BETA_LLM_INTENT, "true")
    prompt_dir = Path(tmp_path) / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "llm_intent_classifier_prompt.jinja2"
    prompt_file.write_text("This is a test prompt")

    config = {"prompt": str(prompt_file)}

    intent_classifier = LLMIntentClassifier(
        config,
        default_model_storage,
        Resource("llmintentclassifier"),
        default_execution_context,
    )

    fingerprint_1 = intent_classifier.fingerprint_addon(config)
    fingerprint_2 = intent_classifier.fingerprint_addon(config)
    assert fingerprint_1 is not None
    assert fingerprint_1 == fingerprint_2


async def test_llm_intent_classifier_fingerprint_addon_default_prompt_template(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RASA_PRO_BETA_LLM_INTENT, "true")
    intent_classifier = LLMIntentClassifier(
        {},
        default_model_storage,
        Resource("llmintentclassifier"),
        default_execution_context,
    )
    fingerprint_1 = intent_classifier.fingerprint_addon({})
    fingerprint_2 = intent_classifier.fingerprint_addon({})
    assert fingerprint_1 is not None
    assert fingerprint_1 == fingerprint_2

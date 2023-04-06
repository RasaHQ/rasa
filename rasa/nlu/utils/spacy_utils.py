from __future__ import annotations

import dataclasses
import typing
import logging
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.constants import DENSE_FEATURIZABLE_ATTRIBUTES, SPACY_DOCS
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.model import InvalidModelError
from rasa.shared.constants import DOCS_URL_COMPONENTS

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc


@dataclasses.dataclass
class SpacyModel:
    """Wraps `SpacyNLP` output to make it fingerprintable."""

    model: Language
    model_name: Text

    def fingerprint(self) -> Text:
        """Fingerprints the model name.

        Use a static fingerprint as we assume this only changes if the model name
        changes and want to avoid investigating the model in greater detail for now.

        Returns:
            Fingerprint for model.
        """
        return str(self.model_name)


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.MODEL_LOADER,
        DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER,
    ],
    is_trainable=False,
    model_from="SpacyNLP",
)
class SpacyNLP(GraphComponent):
    """Component which provides the common loaded SpaCy model to others.

    This is used to avoid loading the SpaCy model multiple times. Instead the Spacy
    model is only loaded once and then shared by depending components.
    """

    def __init__(self, model: SpacyModel, config: Dict[Text, Any]) -> None:
        """Initializes a `SpacyNLP`."""
        self._model = model
        self._config = config

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Default config."""
        return {
            # when retrieving word vectors, this will decide if the casing
            # of the word is relevant. E.g. `hello` and `Hello` will
            # retrieve the same vector, if set to `False`. For some
            # applications and models it makes sense to differentiate
            # between these two words, therefore setting this to `True`.
            "case_sensitive": False
        }

    @staticmethod
    def load_model(spacy_model_name: Text) -> SpacyModel:
        """Try loading the model, catching the OSError if missing."""
        import spacy

        if not spacy_model_name:
            raise InvalidModelError(
                f"Missing model configuration for `SpacyNLP` in `config.yml`.\n"
                f"You must pass a model to the `SpacyNLP` component explicitly.\n"
                f"For example:\n"
                f"- name: SpacyNLP\n"
                f"  model: en_core_web_md\n"
                f"More information can be found on {DOCS_URL_COMPONENTS}#spacynlp"
            )

        try:
            language = spacy.load(spacy_model_name, disable=["parser"])
            spacy_runtime_version = spacy.about.__version__
            spacy_model_info = spacy.info(spacy_model_name)
            spacy_model_version_req = (
                spacy_model_info.get("spacy_version")
                if isinstance(spacy_model_info, dict)
                else ""
            )
            if not spacy.util.is_compatible_version(
                spacy_runtime_version, spacy_model_version_req
            ):
                raise InvalidModelError(
                    f"The specified model - {spacy_model_name} requires a spaCy "
                    f"runtime version {spacy_model_version_req} and is not compatible "
                    f"with the current spaCy runtime version {spacy_runtime_version}"
                )
            return SpacyModel(model=language, model_name=spacy_model_name)
        except OSError:
            raise InvalidModelError(
                f"Please confirm that {spacy_model_name} is an available spaCy model. "
                f"You need to download one upfront. For example:\n"
                f"python -m spacy download en_core_web_md\n"
                f"More information can be found on {DOCS_URL_COMPONENTS}#spacynlp"
            )

    @staticmethod
    def required_packages() -> List[Text]:
        """Lists required dependencies (see parent class for full docstring)."""
        return ["spacy"]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> SpacyNLP:
        """Creates component (see parent class for full docstring)."""
        spacy_model_name = config.get("model")

        logger.info(f"Trying to load SpaCy model with name '{spacy_model_name}'.")

        model = cls.load_model(spacy_model_name)

        cls.ensure_proper_language_model(model.model)
        return cls(model, {**cls.get_default_config(), **config})

    @staticmethod
    def ensure_proper_language_model(nlp: Optional[Language]) -> None:
        """Checks if the SpaCy language model is properly loaded.

        Raises an exception if the model is invalid.
        """
        if nlp is None:
            raise Exception(
                "Failed to load SpaCy language model. "
                "Loading the model returned 'None'."
            )
        if nlp.path is None:
            # Spacy sets the path to `None` if
            # it did not load the model from disk.
            # In this case `nlp` is an unusable stub.
            raise Exception(
                f"Failed to load SpaCy language model for "
                f"lang '{nlp.lang}'. Make sure you have downloaded the "
                f"correct model (https://spacy.io/docs/usage/)."
                ""
            )

    def provide(self) -> SpacyModel:
        """Provides the loaded SpaCy model."""
        return self._model

    def _doc_for_text(self, model: Language, text: Text) -> Doc:
        """Makes a SpaCy doc object from a string of text."""
        return model(self._preprocess_text(text))

    def _preprocess_text(self, text: Optional[Text]) -> Text:
        """Processes the text before it is handled by SpaCy."""
        if text is None:
            # converted to empty string so that it can still be passed to spacy.
            # Another option could be to neglect tokenization of the attribute of
            # this example, but since we are processing in batch mode, it would
            # get complex to collect all processed and neglected examples.
            text = ""
        if self._config.get("case_sensitive"):
            return text
        else:
            return text.lower()

    def _get_text(self, example: Dict[Text, Any], attribute: Text) -> Text:
        return self._preprocess_text(example.get(attribute))

    @staticmethod
    def _merge_content_lists(
        indexed_training_samples: List[Tuple[int, Text]],
        doc_lists: List[Tuple[int, Doc]],
    ) -> List[Tuple[int, Doc]]:
        """Merge lists with processed Docs back into their original order."""
        dct = dict(indexed_training_samples)
        dct.update(doc_lists)
        return sorted(dct.items())

    @staticmethod
    def _filter_training_samples_by_content(
        indexed_training_samples: List[Tuple[int, Text]]
    ) -> Tuple[List[Tuple[int, Text]], List[Tuple[int, Text]]]:
        """Separates empty training samples from content bearing ones."""
        docs_to_pipe = list(
            filter(
                lambda training_sample: training_sample[1] != "",
                indexed_training_samples,
            )
        )
        empty_docs = list(
            filter(
                lambda training_sample: training_sample[1] == "",
                indexed_training_samples,
            )
        )
        return docs_to_pipe, empty_docs

    @staticmethod
    def _process_content_bearing_samples(
        model: Language, samples_to_pipe: List[Tuple[int, Text]]
    ) -> List[Tuple[int, Doc]]:
        """Sends content bearing training samples to SpaCy's pipe."""
        docs = [
            (to_pipe_sample[0], doc)
            for to_pipe_sample, doc in zip(
                samples_to_pipe,
                [
                    doc
                    for doc in model.pipe(
                        [txt for _, txt in samples_to_pipe], batch_size=50
                    )
                ],
            )
        ]
        return docs

    @staticmethod
    def _process_non_content_bearing_samples(
        model: Language, empty_samples: List[Tuple[int, Text]]
    ) -> List[Tuple[int, Doc]]:
        """Creates empty Doc-objects from zero-lengthed training samples strings."""
        from spacy.tokens import Doc

        n_docs = [
            (empty_sample[0], doc)
            for empty_sample, doc in zip(
                empty_samples, [Doc(model.vocab) for doc in empty_samples]
            )
        ]
        return n_docs

    def _docs_for_training_data(
        self, model: Language, training_data: TrainingData
    ) -> Dict[Text, List[Any]]:
        attribute_docs = {}
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

            texts = [
                self._get_text(e, attribute) for e in training_data.training_examples
            ]
            # Index and freeze indices of the training samples for preserving the order
            # after processing the data.
            indexed_training_samples = [(idx, text) for idx, text in enumerate(texts)]

            samples_to_pipe, empty_samples = self._filter_training_samples_by_content(
                indexed_training_samples
            )

            content_bearing_docs = self._process_content_bearing_samples(
                model, samples_to_pipe
            )

            non_content_bearing_docs = self._process_non_content_bearing_samples(
                model, empty_samples
            )

            attribute_document_list = self._merge_content_lists(
                indexed_training_samples,
                content_bearing_docs + non_content_bearing_docs,
            )

            # Since we only need the training samples strings,
            # we create a list to get them out of the tuple.
            attribute_docs[attribute] = [doc for _, doc in attribute_document_list]
        return attribute_docs

    def process_training_data(
        self, training_data: TrainingData, model: SpacyModel
    ) -> TrainingData:
        """Adds SpaCy tokens and features to training data messages."""
        attribute_docs = self._docs_for_training_data(model.model, training_data)

        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

            for idx, example in enumerate(training_data.training_examples):
                example_attribute_doc = attribute_docs[attribute][idx]
                if len(example_attribute_doc):
                    # If length is 0, that means the initial text feature
                    # was None and was replaced by ''
                    # in preprocess method
                    example.set(SPACY_DOCS[attribute], example_attribute_doc)

        return training_data

    def process(self, messages: List[Message], model: SpacyModel) -> List[Message]:
        """Adds SpaCy tokens and features to messages."""
        for message in messages:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                if message.get(attribute):
                    message.set(
                        SPACY_DOCS[attribute],
                        self._doc_for_text(model.model, message.get(attribute)),
                    )

        return messages

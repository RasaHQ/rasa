from __future__ import annotations
import logging
from rasa.nlu.tokenizers.tokenizer import Tokenizer
import typing
from typing import Any, Dict, List, Optional, Text, Type

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_CONFIDENCE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    TEXT,
    ENTITIES,
)
from rasa.nlu.utils.mitie_utils import MitieModel, MitieNLP
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
import rasa.shared.utils.io
from rasa.shared.exceptions import InvalidConfigException

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import mitie


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    is_trainable=True,
    model_from="MitieNLP",
)
class MitieEntityExtractor(GraphComponent, EntityExtractorMixin):
    """A Mitie Entity Extractor (which is a thin wrapper around `Dlib-ml`)."""

    MITIE_RESOURCE_FILE = "mitie_ner.dat"

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [MitieNLP, Tokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["mitie"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {"num_threads": 1}

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        ner: Optional["mitie.named_entity_extractor"] = None,
    ) -> None:
        """Creates a new instance.

        Args:
            config: The configuration.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            ner: Mitie named entity extractor
        """
        self._config = config
        self._model_storage = model_storage
        self._resource = resource
        self.validate_config(self._config)
        self._ner = ner

    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Checks whether the given configuration is valid.

        Args:
          config: a configuration for a Mitie entity extractor component
        """
        num_threads = config.get("num_threads")
        if num_threads is None or num_threads <= 0:
            raise InvalidConfigException(
                f"Expected `num_threads` to be some value >= 1 (default: 1)."
                f"but received {num_threads}"
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new `MitieEntityExtractor`.

        Args:
            config: This config overrides the `default_config`.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run. Unused.

        Returns: An instantiated `MitieEntityExtractor`.
        """
        return cls(config, model_storage, resource)

    def train(self, training_data: TrainingData, model: MitieModel) -> Resource:
        """Trains a MITIE named entity recognizer.

        Args:
            training_data: the training data
            model: a MitieModel
        Returns:
            resource for loading the trained model
        """
        import mitie

        trainer = mitie.ner_trainer(str(model.model_path))
        trainer.num_threads = self._config["num_threads"]

        # check whether there are any (not pre-trained) entities in the training data
        found_one_entity = False

        # filter out pre-trained entity examples
        filtered_entity_examples = self.filter_trainable_entities(
            training_data.nlu_examples
        )

        for example in filtered_entity_examples:
            sample = self._prepare_mitie_sample(example)

            found_one_entity = sample.num_entities > 0 or found_one_entity
            trainer.add(sample)

        # Mitie will fail to train if there is not a single entity tagged
        if found_one_entity:
            self._ner = trainer.train()
        else:
            rasa.shared.utils.io.raise_warning(
                f"{self.__class__.__name__} could not be trained because no trainable "
                f"entities where found in the given training data. Please add some "
                f"NLU training examples that include entities where the `extractor` "
                f"is either `None` or '{self.__class__.__name__}'."
            )

        self.persist()
        return self._resource

    @staticmethod
    def _prepare_mitie_sample(training_example: Message) -> Any:
        """Prepare a message so that it can be passed to a MITIE trainer."""
        import mitie

        text = training_example.get(TEXT)
        tokens = training_example.get(TOKENS_NAMES[TEXT])
        sample = mitie.ner_training_instance([t.text for t in tokens])
        for ent in training_example.get(ENTITIES, []):
            try:
                # if the token is not aligned an exception will be raised
                start, end = MitieEntityExtractor.find_entity(ent, text, tokens)
            except ValueError as e:
                rasa.shared.utils.io.raise_warning(
                    f"Failed to use example '{text}' to train MITIE "
                    f"entity extractor. Example will be skipped."
                    f"Error: {e}"
                )
                continue
            try:
                # mitie will raise an exception on malicious
                # input - e.g. on overlapping entities
                sample.add_entity(list(range(start, end)), ent["entity"])
            except Exception as e:
                rasa.shared.utils.io.raise_warning(
                    f"Failed to add entity example "
                    f"'{str(e)}' of sentence '{str(text)}'. "
                    f"Example will be ignored. Reason: "
                    f"{e}"
                )
                continue
        return sample

    def process(self, messages: List[Message], model: MitieModel) -> List[Message]:
        """Extracts entities from messages and appends them to the attribute.

        If no patterns where found during training, then the given messages will not
        be modified. In particular, if no `ENTITIES` attribute exists yet, then
        it will *not* be created.

        If no pattern can be found in the given message, then no entities will be
        added to any existing list of entities. However, if no `ENTITIES` attribute
        exists yet, then an `ENTITIES` attribute will be created.

        Returns:
           the given list of messages that have been modified
        """
        if not self._ner:
            return messages

        for message in messages:
            entities = self._extract_entities(message, mitie_model=model)
            extracted = self.add_extractor_name(entities)
            message.set(
                ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True
            )
        return messages

    def _extract_entities(
        self, message: Message, mitie_model: MitieModel
    ) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message.

        Args:
            message: a user message
            mitie_model: MitieModel containing a `mitie.total_word_feature_extractor`

        Returns:
            a list of dictionaries describing the entities
        """
        text = message.get(TEXT)
        tokens = message.get(TOKENS_NAMES[TEXT])

        entities = []
        token_texts = [token.text for token in tokens]
        if self._ner is None:
            mitie_entities = []
        else:
            mitie_entities = self._ner.extract_entities(
                token_texts, mitie_model.word_feature_extractor
            )
        for e in mitie_entities:
            if len(e[0]):
                start = tokens[e[0][0]].start
                end = tokens[e[0][-1]].end

                entities.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: e[1],
                        ENTITY_ATTRIBUTE_VALUE: text[start:end],
                        ENTITY_ATTRIBUTE_START: start,
                        ENTITY_ATTRIBUTE_END: end,
                        ENTITY_ATTRIBUTE_CONFIDENCE: None,
                    }
                )
        return entities

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> MitieEntityExtractor:
        """Loads trained component (see parent class for full docstring)."""
        import mitie

        try:
            with model_storage.read_from(resource) as model_path:
                ner_file = model_path / cls.MITIE_RESOURCE_FILE
                if not ner_file.exists():
                    raise FileNotFoundError(
                        f"Expected a MITIE extractor file at {ner_file}."
                    )
                ner = mitie.named_entity_extractor(str(ner_file))
                return cls(config, model_storage, resource, ner=ner)

        except (FileNotFoundError, ValueError) as e:
            logger.debug(
                f"Failed to load {cls.__name__} from model storage. "
                f"This can happen if the model could not be trained because regexes "
                f"could not be extracted from the given training data - and hence "
                f"could not be persisted. Error: {e}."
            )
            return cls(config, model_storage, resource)

    def persist(self) -> None:
        """Persist this model."""
        if not self._ner:
            return
        with self._model_storage.write_to(self._resource) as model_path:
            ner_file = model_path / self.MITIE_RESOURCE_FILE
            self._ner.save_to_disk(str(ner_file), pure_model=True)

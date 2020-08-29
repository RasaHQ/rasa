import logging
import os
from pathlib import Path
import random
from collections import Counter, OrderedDict
import copy
from os.path import relpath
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Callable

from rasa import data
import rasa.nlu.utils
from rasa.utils.common import raise_warning, lazy_property
from rasa.nlu.constants import (
    RESPONSE,
    NO_ENTITY_TAG,
    INTENT_RESPONSE_KEY,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    INTENT,
    ENTITIES,
    TEXT,
)
from rasa.nlu.training_data.message import Message
from rasa.nlu.training_data.util import check_duplicate_synonym
from rasa.nlu.utils import list_to_str

DEFAULT_TRAINING_DATA_OUTPUT_PATH = "training_data.json"

logger = logging.getLogger(__name__)


class TrainingData:
    """Holds loaded intent and entity training data."""

    # Validation will ensure and warn if these lower limits are not met
    MIN_EXAMPLES_PER_INTENT = 2
    MIN_EXAMPLES_PER_ENTITY = 2

    def __init__(
        self,
        training_examples: Optional[List[Message]] = None,
        entity_synonyms: Optional[Dict[Text, Text]] = None,
        regex_features: Optional[List[Dict[Text, Text]]] = None,
        lookup_tables: Optional[List[Dict[Text, Any]]] = None,
        responses: Optional[Dict[Text, List[Dict[Text, Any]]]] = None,
    ) -> None:

        if training_examples:
            self.training_examples = self.sanitize_examples(training_examples)
        else:
            self.training_examples = []
        self.entity_synonyms = entity_synonyms or {}
        self.regex_features = regex_features or []
        self.sort_regex_features()
        self.lookup_tables = lookup_tables or []
        self.responses = responses or {}

        self._fill_response_phrases()

    def merge(self, *others: "TrainingData") -> "TrainingData":
        """Return merged instance of this data with other training data."""

        training_examples = copy.deepcopy(self.training_examples)
        entity_synonyms = self.entity_synonyms.copy()
        regex_features = copy.deepcopy(self.regex_features)
        lookup_tables = copy.deepcopy(self.lookup_tables)
        responses = copy.deepcopy(self.responses)
        others = [other for other in others if other]

        for o in others:
            training_examples.extend(copy.deepcopy(o.training_examples))
            regex_features.extend(copy.deepcopy(o.regex_features))
            lookup_tables.extend(copy.deepcopy(o.lookup_tables))

            for text, syn in o.entity_synonyms.items():
                check_duplicate_synonym(
                    entity_synonyms, text, syn, "merging training data"
                )

            entity_synonyms.update(o.entity_synonyms)
            responses.update(o.responses)

        return TrainingData(
            training_examples, entity_synonyms, regex_features, lookup_tables, responses
        )

    def filter_training_examples(
        self, condition: Callable[[Message], bool]
    ) -> "TrainingData":
        """Filter training examples.

        Args:
            condition: A function that will be applied to filter training examples.

        Returns:
            TrainingData: A TrainingData with filtered training examples.
        """

        return TrainingData(
            list(filter(condition, self.training_examples)),
            self.entity_synonyms,
            self.regex_features,
            self.lookup_tables,
            self.responses,
        )

    def __hash__(self) -> int:
        from rasa.core import utils as core_utils

        stringified = self.nlu_as_json() + self.nlg_as_markdown()
        text_hash = core_utils.get_text_hash(stringified)

        return int(text_hash, 16)

    @staticmethod
    def sanitize_examples(examples: List[Message]) -> List[Message]:
        """Makes sure the training data is clean.

        Remove trailing whitespaces from intent and response annotations and drop
        duplicate examples.
        """

        for ex in examples:
            if ex.get(INTENT):
                ex.set(INTENT, ex.get(INTENT).strip())

            if ex.get(RESPONSE):
                ex.set(RESPONSE, ex.get(RESPONSE).strip())

        return list(OrderedDict.fromkeys(examples))

    @lazy_property
    def intent_examples(self) -> List[Message]:
        return [ex for ex in self.training_examples if ex.get(INTENT)]

    @lazy_property
    def response_examples(self) -> List[Message]:
        return [ex for ex in self.training_examples if ex.get(RESPONSE)]

    @lazy_property
    def entity_examples(self) -> List[Message]:
        return [ex for ex in self.training_examples if ex.get(ENTITIES)]

    @lazy_property
    def intents(self) -> Set[Text]:
        """Returns the set of intents in the training data."""
        return {ex.get(INTENT) for ex in self.training_examples} - {None}

    @lazy_property
    def retrieval_intents(self) -> Set[Text]:
        """Returns the total number of response types in the training data"""
        return {
            ex.get(INTENT)
            for ex in self.training_examples
            if ex.get(RESPONSE) is not None
        }

    @lazy_property
    def number_of_examples_per_intent(self) -> Dict[Text, int]:
        """Calculates the number of examples per intent."""
        intents = [ex.get(INTENT) for ex in self.training_examples]
        return dict(Counter(intents))

    @lazy_property
    def number_of_examples_per_response(self) -> Dict[Text, int]:
        """Calculates the number of examples per response."""
        responses = [
            ex.get(RESPONSE) for ex in self.training_examples if ex.get(RESPONSE)
        ]
        return dict(Counter(responses))

    @lazy_property
    def entities(self) -> Set[Text]:
        """Returns the set of entity types in the training data."""
        entity_types = [e.get(ENTITY_ATTRIBUTE_TYPE) for e in self.sorted_entities()]
        return set(entity_types)

    @lazy_property
    def entity_roles(self) -> Set[Text]:
        """Returns the set of entity roles in the training data."""
        entity_types = [
            e.get(ENTITY_ATTRIBUTE_ROLE)
            for e in self.sorted_entities()
            if ENTITY_ATTRIBUTE_ROLE in e
        ]
        return set(entity_types) - {NO_ENTITY_TAG}

    @lazy_property
    def entity_groups(self) -> Set[Text]:
        """Returns the set of entity groups in the training data."""
        entity_types = [
            e.get(ENTITY_ATTRIBUTE_GROUP)
            for e in self.sorted_entities()
            if ENTITY_ATTRIBUTE_GROUP in e
        ]
        return set(entity_types) - {NO_ENTITY_TAG}

    def entity_roles_groups_used(self) -> bool:
        entity_groups_used = (
            self.entity_groups is not None and len(self.entity_groups) > 0
        )
        entity_roles_used = self.entity_roles is not None and len(self.entity_roles) > 0

        return entity_groups_used or entity_roles_used

    @lazy_property
    def number_of_examples_per_entity(self) -> Dict[Text, int]:
        """Calculates the number of examples per entity."""

        entities = []

        def _append_entity(entity: Dict[Text, Any], attribute: Text) -> None:
            if attribute in entity:
                _value = entity.get(attribute)
                if _value is not None and _value != NO_ENTITY_TAG:
                    entities.append(f"{attribute} '{_value}'")

        for entity in self.sorted_entities():
            _append_entity(entity, ENTITY_ATTRIBUTE_TYPE)
            _append_entity(entity, ENTITY_ATTRIBUTE_ROLE)
            _append_entity(entity, ENTITY_ATTRIBUTE_GROUP)

        return dict(Counter(entities))

    def sort_regex_features(self) -> None:
        """Sorts regex features lexicographically by name+pattern"""
        self.regex_features = sorted(
            self.regex_features, key=lambda e: "{}+{}".format(e["name"], e["pattern"])
        )

    def _fill_response_phrases(self) -> None:
        """Set response phrase for all examples by looking up NLG stories"""
        for example in self.training_examples:
            # if intent_response_key is None, that means the corresponding intent is not a
            # retrieval intent and hence no response text needs to be fetched.
            # If intent_response_key is set, fetch the corresponding response text
            if example.get(INTENT_RESPONSE_KEY) is None:
                continue

            # look for corresponding bot utterance
            story_lookup_intent = example.get_full_intent()
            assistant_utterances = self.responses.get(story_lookup_intent, [])
            if assistant_utterances:

                # Use the first response text as training label if needed downstream
                for assistant_utterance in assistant_utterances:
                    if assistant_utterance.get(TEXT):
                        example.set(RESPONSE, assistant_utterance[TEXT])

                # If no text attribute was found use the key for training
                if not example.get(RESPONSE):
                    example.set(RESPONSE, story_lookup_intent)

    def nlu_as_json(self, **kwargs: Any) -> Text:
        """Represent this set of training examples as json."""
        from rasa.nlu.training_data.formats import (  # pytype: disable=pyi-error
            RasaWriter,
        )

        return RasaWriter().dumps(self, **kwargs)

    def nlg_as_markdown(self) -> Text:
        """Generates the markdown representation of the response phrases (NLG) of
        TrainingData."""

        from rasa.nlu.training_data.formats import (  # pytype: disable=pyi-error
            NLGMarkdownWriter,
        )

        return NLGMarkdownWriter().dumps(self)

    def nlg_as_yaml(self) -> Text:
        """Generates yaml representation of the response phrases (NLG) of TrainingData.

        Returns:
            responses in yaml format as a string
        """
        from rasa.nlu.training_data.formats.rasa_yaml import (  # pytype: disable=pyi-error
            RasaYAMLWriter,
        )

        # only dump responses. at some point it might make sense to remove the
        # differentiation between dumping NLU and dumping responses. but we
        # can't do that until after we remove markdown support.
        return RasaYAMLWriter().dumps(TrainingData(responses=self.responses))

    def nlu_as_markdown(self) -> Text:
        """Generates the markdown representation of the NLU part of TrainingData."""
        from rasa.nlu.training_data.formats import (  # pytype: disable=pyi-error
            MarkdownWriter,
        )

        return MarkdownWriter().dumps(self)

    def nlu_as_yaml(self) -> Text:
        from rasa.nlu.training_data.formats.rasa_yaml import (  # pytype: disable=pyi-error
            RasaYAMLWriter,
        )

        # avoid dumping NLG data (responses). this is a workaround until we
        # can remove the distinction between nlu & nlg when converting to a string
        # (so until after we remove markdown support)
        no_responses_training_data = copy.copy(self)
        no_responses_training_data.responses = {}

        return RasaYAMLWriter().dumps(no_responses_training_data)

    def persist_nlu(self, filename: Text = DEFAULT_TRAINING_DATA_OUTPUT_PATH) -> None:

        if data.is_likely_json_file(filename):
            rasa.nlu.utils.write_to_file(filename, self.nlu_as_json(indent=2))
        elif data.is_likely_markdown_file(filename):
            rasa.nlu.utils.write_to_file(filename, self.nlu_as_markdown())
        elif data.is_likely_yaml_file(filename):
            rasa.nlu.utils.write_to_file(filename, self.nlu_as_yaml())
        else:
            ValueError(
                "Unsupported file format detected. Supported file formats are 'json' "
                "and 'md'."
            )

    def persist_nlg(self, filename: Text) -> None:
        if data.is_likely_yaml_file(filename):
            rasa.nlu.utils.write_to_file(filename, self.nlg_as_yaml())
        elif data.is_likely_markdown_file(filename):
            nlg_serialized_data = self.nlg_as_markdown()
            if nlg_serialized_data:
                rasa.nlu.utils.write_to_file(filename, nlg_serialized_data)
        else:
            ValueError(
                "Unsupported file format detected. Supported file formats are 'md' "
                "and 'yml'."
            )

    @staticmethod
    def get_nlg_persist_filename(nlu_filename: Text) -> Text:

        extension = Path(nlu_filename).suffix
        if data.is_likely_json_file(nlu_filename):
            # backwards compatibility: previously NLG was always dumped as md. now
            # we are going to dump in the same format as the NLU data. unfortunately
            # there is a special case: NLU is in json format, in this case we use
            # md as we do not have a NLG json format
            extension = "md"
        # Add nlg_ as prefix and change extension to .md
        filename = (
            Path(nlu_filename)
            .with_name("nlg_" + Path(nlu_filename).name)
            .with_suffix("." + extension)
        )
        return str(filename)

    def persist(
        self, dir_name: Text, filename: Text = DEFAULT_TRAINING_DATA_OUTPUT_PATH
    ) -> Dict[Text, Any]:
        """Persists this training data to disk and returns necessary
        information to load it again."""

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        nlu_data_file = os.path.join(dir_name, filename)
        self.persist_nlu(nlu_data_file)
        self.persist_nlg(self.get_nlg_persist_filename(nlu_data_file))

        return {"training_data": relpath(nlu_data_file, dir_name)}

    def sorted_entities(self) -> List[Any]:
        """Extract all entities from examples and sorts them by entity type."""

        entity_examples = [
            entity for ex in self.entity_examples for entity in ex.get("entities")
        ]
        return sorted(entity_examples, key=lambda e: e["entity"])

    def sorted_intent_examples(self) -> List[Message]:
        """Sorts the intent examples by the name of the intent and then response"""

        return sorted(
            self.intent_examples, key=lambda e: (e.get(INTENT), e.get(RESPONSE))
        )

    def validate(self) -> None:
        """Ensures that the loaded training data is valid.

        Checks that the data has a minimum of certain training examples."""

        logger.debug("Validating training data...")
        if "" in self.intents:
            raise_warning(
                "Found empty intent, please check your "
                "training data. This may result in wrong "
                "intent predictions."
            )

        if "" in self.responses:
            raise_warning(
                "Found empty response, please check your "
                "training data. This may result in wrong "
                "response predictions."
            )

        # emit warnings for intents with only a few training samples
        for intent, count in self.number_of_examples_per_intent.items():
            if count < self.MIN_EXAMPLES_PER_INTENT:
                raise_warning(
                    f"Intent '{intent}' has only {count} training examples! "
                    f"Minimum is {self.MIN_EXAMPLES_PER_INTENT}, training may fail."
                )

        # emit warnings for entities with only a few training samples
        for entity, count in self.number_of_examples_per_entity.items():
            if count < self.MIN_EXAMPLES_PER_ENTITY:
                raise_warning(
                    f"Entity {entity} has only {count} training examples! "
                    f"The minimum is {self.MIN_EXAMPLES_PER_ENTITY}, because of "
                    f"this the training may fail."
                )

        # emit warnings for response intents without a response template
        for example in self.training_examples:
            if example.get(INTENT_RESPONSE_KEY) and not example.get(RESPONSE):
                raise_warning(
                    f"Your training data contains an example '{example.text[:20]}...' "
                    f"for the {example.get_full_intent()} intent. "
                    f"You either need to add a response phrase or correct the "
                    f"intent for this example in your training data. "
                    f"If you intend to use Response Selector in the pipeline, the training ."
                )

    def train_test_split(
        self, train_frac: float = 0.8, random_seed: Optional[int] = None
    ) -> Tuple["TrainingData", "TrainingData"]:
        """Split into a training and test dataset,
        preserving the fraction of examples per intent."""

        # collect all nlu data
        test, train = self.split_nlu_examples(train_frac, random_seed)

        # collect all nlg stories
        test_responses = self._needed_responses_for_examples(test)
        train_responses = self._needed_responses_for_examples(train)

        data_train = TrainingData(
            train,
            entity_synonyms=self.entity_synonyms,
            regex_features=self.regex_features,
            lookup_tables=self.lookup_tables,
            responses=train_responses,
        )

        data_test = TrainingData(
            test,
            entity_synonyms=self.entity_synonyms,
            regex_features=self.regex_features,
            lookup_tables=self.lookup_tables,
            responses=test_responses,
        )

        return data_train, data_test

    def _needed_responses_for_examples(
        self, examples: List[Message]
    ) -> Dict[Text, List[Dict[Text, Any]]]:
        """Get all responses used in any of the examples.

        Args:
            examples: messages to select responses by.

        Returns:
            All responses that appear at least once in the list of examples.
        """

        responses = {}
        for ex in examples:
            if ex.get(INTENT_RESPONSE_KEY) and ex.get(RESPONSE):
                key = ex.get_full_intent()
                responses[key] = self.responses[key]
        return responses

    def split_nlu_examples(
        self, train_frac: float, random_seed: Optional[int] = None
    ) -> Tuple[list, list]:
        """Split the training data into a train and test set.

        Args:
            train_frac: percentage of examples to add to the training set.
            random_seed: random seed

        Returns:
            Test and training examples.
        """
        train, test = [], []
        training_examples = set(self.training_examples)

        def _split(_examples: List[Message], _count: int) -> None:
            if random_seed is not None:
                random.Random(random_seed).shuffle(_examples)
            else:
                random.shuffle(_examples)

            n_train = int(_count * train_frac)
            train.extend(_examples[:n_train])
            test.extend(_examples[n_train:])

        # to make sure we have at least one example per response and intent in the
        # training/test data, we first go over the response examples and then go over
        # intent examples

        for response, count in self.number_of_examples_per_response.items():
            examples = [
                e
                for e in training_examples
                if RESPONSE in e.data and e.data[RESPONSE] == response
            ]
            _split(examples, count)
            training_examples = training_examples - set(examples)

        for intent, count in self.number_of_examples_per_intent.items():
            examples = [
                e
                for e in training_examples
                if INTENT in e.data and e.data[INTENT] == intent
            ]
            _split(examples, count)
            training_examples = training_examples - set(examples)

        return test, train

    def print_stats(self) -> None:
        number_of_examples_for_each_intent = []
        for intent_name, example_count in self.number_of_examples_per_intent.items():
            number_of_examples_for_each_intent.append(
                f"intent: {intent_name}, training examples: {example_count}   "
            )
        newline = "\n"

        logger.info("Training data stats:")
        logger.info(
            f"Number of intent examples: {len(self.intent_examples)} "
            f"({len(self.intents)} distinct intents)"
            "\n"
        )
        # log the number of training examples per intent

        logger.debug(f"{newline.join(number_of_examples_for_each_intent)}")

        if self.intents:
            logger.info(f"  Found intents: {list_to_str(self.intents)}")
        logger.info(
            f"Number of response examples: {len(self.response_examples)} "
            f"({len(self.responses)} distinct responses)"
        )
        logger.info(
            f"Number of entity examples: {len(self.entity_examples)} "
            f"({len(self.entities)} distinct entities)"
        )
        if self.entities:
            logger.info(f"  Found entity types: {list_to_str(self.entities)}")
        if self.entity_roles:
            logger.info(f"  Found entity roles: {list_to_str(self.entity_roles)}")
        if self.entity_groups:
            logger.info(f"  Found entity groups: {list_to_str(self.entity_groups)}")

    def is_empty(self) -> bool:
        """Checks if any training data was loaded."""

        lists_to_check = [
            self.training_examples,
            self.entity_synonyms,
            self.regex_features,
            self.lookup_tables,
        ]
        return not any([len(lst) > 0 for lst in lists_to_check])

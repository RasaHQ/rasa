import logging
import os
from pathlib import Path
import random
from collections import Counter, OrderedDict
import copy
from os.path import relpath
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Callable
import operator

import rasa.shared.data
from rasa.shared.utils.common import lazy_property
import rasa.shared.utils.io
from rasa.shared.nlu.constants import (
    RESPONSE,
    INTENT_RESPONSE_KEY,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    NO_ENTITY_TAG,
    INTENT,
    ENTITIES,
    TEXT,
    ACTION_NAME,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data import util


DEFAULT_TRAINING_DATA_OUTPUT_PATH = "training_data.yml"

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

    @staticmethod
    def _load_lookup_table(lookup_table: Dict[Text, Any]) -> Dict[Text, Any]:
        """Loads the actual lookup table from file if there is a file specified.

        Checks if the specified lookup table contains a filename in
        `elements` and replaces it with actual elements from the file.
        Returns the unchanged lookup table otherwise.
        It works with JSON training data.

        Params:
            lookup_table: A lookup table.

        Returns:
            Updated lookup table where filenames are replaced with the contents of
            these files.
        """
        elements = lookup_table["elements"]
        potential_file = elements if isinstance(elements, str) else elements[0]

        if Path(potential_file).is_file():
            try:
                lookup_table["elements"] = rasa.shared.utils.io.read_file(
                    potential_file
                )
                return lookup_table
            except (FileNotFoundError, UnicodeDecodeError):
                return lookup_table

        return lookup_table

    def fingerprint(self) -> Text:
        """Fingerprint the training data.

        Returns:
            hex string as a fingerprint of the training data.
        """
        relevant_attributes = {
            "training_examples": list(
                sorted(e.fingerprint() for e in self.training_examples)
            ),
            "entity_synonyms": self.entity_synonyms,
            "regex_features": self.regex_features,
            "lookup_tables": [
                self._load_lookup_table(table) for table in self.lookup_tables
            ],
            "responses": self.responses,
        }
        return rasa.shared.utils.io.deep_container_fingerprint(relevant_attributes)

    def label_fingerprint(self) -> Text:
        """Fingerprints the labels in the training data.

        Returns:
            hex string as a fingerprint of the training data labels.
        """
        labels = {
            "intents": sorted(self.intents),
            "entities": sorted(self.entities),
            "entity_groups": sorted(self.entity_groups),
            "entity_roles": sorted(self.entity_roles),
            "actions": sorted(self.action_names),
        }
        return rasa.shared.utils.io.deep_container_fingerprint(labels)

    def merge(self, *others: Optional["TrainingData"]) -> "TrainingData":
        """Return merged instance of this data with other training data.

        Args:
            others: other training data instances to merge this one with

        Returns:
            Merged training data object. Merging is not done in place, this
            will be a new instance.
        """
        training_examples = copy.deepcopy(self.training_examples)
        entity_synonyms = self.entity_synonyms.copy()
        regex_features = copy.deepcopy(self.regex_features)
        lookup_tables = copy.deepcopy(self.lookup_tables)
        responses = copy.deepcopy(self.responses)

        for o in others:
            if not o:
                continue

            training_examples.extend(copy.deepcopy(o.training_examples))
            regex_features.extend(copy.deepcopy(o.regex_features))
            lookup_tables.extend(copy.deepcopy(o.lookup_tables))

            for text, syn in o.entity_synonyms.items():
                util.check_duplicate_synonym(
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
        """Calculate hash for the training data object.

        Returns:
            Hash of the training data object.
        """
        return int(self.fingerprint(), 16)

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
    def nlu_examples(self) -> List[Message]:
        """Return examples which have come from NLU training data.

        E.g. If the example came from a story or domain it is not included.

        Returns:
            List of NLU training examples.
        """
        return [
            ex for ex in self.training_examples if not ex.is_core_or_domain_message()
        ]

    @lazy_property
    def intent_examples(self) -> List[Message]:
        """Returns the list of examples that have intent."""
        return [ex for ex in self.nlu_examples if ex.get(INTENT)]

    @lazy_property
    def response_examples(self) -> List[Message]:
        """Returns the list of examples that have response."""
        return [ex for ex in self.nlu_examples if ex.get(INTENT_RESPONSE_KEY)]

    @lazy_property
    def entity_examples(self) -> List[Message]:
        """Returns the list of examples that have entities."""
        return [ex for ex in self.nlu_examples if ex.get(ENTITIES)]

    @lazy_property
    def intents(self) -> Set[Text]:
        """Returns the set of intents in the training data."""
        return {ex.get(INTENT) for ex in self.training_examples} - {None}

    @lazy_property
    def action_names(self) -> Set[Text]:
        """Returns the set of action names in the training data."""
        return {ex.get(ACTION_NAME) for ex in self.training_examples} - {None}

    @lazy_property
    def retrieval_intents(self) -> Set[Text]:
        """Returns the total number of response types in the training data."""
        return {
            ex.get(INTENT)
            for ex in self.training_examples
            if ex.get(INTENT_RESPONSE_KEY)
        }

    @lazy_property
    def number_of_examples_per_intent(self) -> Dict[Text, int]:
        """Calculates the number of examples per intent."""
        intents = [ex.get(INTENT) for ex in self.nlu_examples]
        return dict(Counter(intents))

    @lazy_property
    def number_of_examples_per_response(self) -> Dict[Text, int]:
        """Calculates the number of examples per response."""
        responses = [
            ex.get(INTENT_RESPONSE_KEY)
            for ex in self.training_examples
            if ex.get(INTENT_RESPONSE_KEY)
        ]
        return dict(Counter(responses))

    @lazy_property
    def entities(self) -> Set[Text]:
        """Returns the set of entity types in the training data."""
        return {e.get(ENTITY_ATTRIBUTE_TYPE) for e in self.sorted_entities()}

    @lazy_property
    def entity_roles(self) -> Set[Text]:
        """Returns the set of entity roles in the training data."""
        entity_types = {
            e.get(ENTITY_ATTRIBUTE_ROLE)
            for e in self.sorted_entities()
            if ENTITY_ATTRIBUTE_ROLE in e
        }
        return entity_types - {NO_ENTITY_TAG}

    @lazy_property
    def entity_groups(self) -> Set[Text]:
        """Returns the set of entity groups in the training data."""
        entity_types = {
            e.get(ENTITY_ATTRIBUTE_GROUP)
            for e in self.sorted_entities()
            if ENTITY_ATTRIBUTE_GROUP in e
        }
        return entity_types - {NO_ENTITY_TAG}

    def entity_roles_groups_used(self) -> bool:
        """Checks if any entity roles or groups are used in the training data."""
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
        """Sorts regex features lexicographically by name+pattern."""
        self.regex_features = sorted(
            self.regex_features, key=lambda e: "{}+{}".format(e["name"], e["pattern"])
        )

    def _fill_response_phrases(self) -> None:
        """Set response phrase for all examples by looking up NLG stories."""
        for example in self.training_examples:
            # if intent_response_key is None, that means the corresponding intent is
            # not a retrieval intent and hence no response text needs to be fetched.
            # If intent_response_key is set, fetch the corresponding response text
            if example.get(INTENT_RESPONSE_KEY) is None:
                continue

            # look for corresponding bot utterance
            story_lookup_key = util.intent_response_key_to_template_key(
                example.get_full_intent()
            )
            assistant_utterances = self.responses.get(story_lookup_key, [])
            if assistant_utterances:
                # Use the first response text as training label if needed downstream
                for assistant_utterance in assistant_utterances:
                    if assistant_utterance.get(TEXT):
                        example.set(RESPONSE, assistant_utterance[TEXT])

                # If no text attribute was found use the key for training
                if not example.get(RESPONSE):
                    example.set(RESPONSE, story_lookup_key)

    def nlu_as_json(self, **kwargs: Any) -> Text:
        """Represent this set of training examples as json."""
        from rasa.shared.nlu.training_data.formats import RasaWriter

        return RasaWriter().dumps(self, **kwargs)

    def nlg_as_yaml(self) -> Text:
        """Generates yaml representation of the response phrases (NLG) of TrainingData.

        Returns:
            responses in yaml format as a string
        """
        from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter

        # only dump responses. at some point it might make sense to remove the
        # differentiation between dumping NLU and dumping responses. but we
        # can't do that until after we remove markdown support.
        return RasaYAMLWriter().dumps(TrainingData(responses=self.responses))

    def nlu_as_yaml(self) -> Text:
        """Generates YAML representation of NLU of TrainingData.

        Returns:
            data in YAML format as a string
        """
        from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter

        # avoid dumping NLG data (responses). this is a workaround until we
        # can remove the distinction between nlu & nlg when converting to a string
        # (so until after we remove markdown support)
        no_responses_training_data = copy.copy(self)
        no_responses_training_data.responses = {}

        return RasaYAMLWriter().dumps(no_responses_training_data)

    def persist_nlu(self, filename: Text = DEFAULT_TRAINING_DATA_OUTPUT_PATH) -> None:
        """Saves NLU to a file."""
        if rasa.shared.data.is_likely_json_file(filename):
            rasa.shared.utils.io.write_text_file(self.nlu_as_json(indent=2), filename)
        elif rasa.shared.data.is_likely_yaml_file(filename):
            rasa.shared.utils.io.write_text_file(self.nlu_as_yaml(), filename)
        else:
            raise ValueError(
                "Unsupported file format detected. "
                "Supported file formats are 'json', 'yml' "
                "and 'md'."
            )

    def persist_nlg(self, filename: Text) -> None:
        """Saves NLG to a file."""
        if rasa.shared.data.is_likely_yaml_file(filename):
            rasa.shared.utils.io.write_text_file(self.nlg_as_yaml(), filename)
        else:
            raise ValueError(
                "Unsupported file format detected. 'yml' is the only "
                "supported file format."
            )

    @staticmethod
    def get_nlg_persist_filename(nlu_filename: Text) -> Text:
        """Returns the full filename to persist NLG data."""
        extension = Path(nlu_filename).suffix
        if rasa.shared.data.is_likely_json_file(nlu_filename):
            # backwards compatibility: previously NLG was always dumped as md. now
            # we are going to dump in the same format as the NLU data. unfortunately
            # there is a special case: NLU is in json format, in this case we use
            # YAML as we do not have a NLG json format
            extension = rasa.shared.data.yaml_file_extension()
        # Add nlg_ as prefix and change extension to the correct one
        filename = (
            Path(nlu_filename)
            .with_name("nlg_" + Path(nlu_filename).name)
            .with_suffix(extension)
        )
        return str(filename)

    def persist(
        self, dir_name: Text, filename: Text = DEFAULT_TRAINING_DATA_OUTPUT_PATH
    ) -> Dict[Text, Any]:
        """Persists this training data to disk and returns necessary
        information to load it again.
        """
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

    def validate(self) -> None:
        """Ensures that the loaded training data is valid.

        Checks that the data has a minimum of certain training examples.
        """
        logger.debug("Validating training data...")
        if "" in self.intents:
            rasa.shared.utils.io.raise_warning(
                "Found empty intent, please check your "
                "training data. This may result in wrong "
                "intent predictions."
            )

        if "" in self.responses:
            rasa.shared.utils.io.raise_warning(
                "Found empty response, please check your "
                "training data. This may result in wrong "
                "response predictions."
            )

        # emit warnings for intents with only a few training samples
        for intent, count in self.number_of_examples_per_intent.items():
            if count < self.MIN_EXAMPLES_PER_INTENT:
                rasa.shared.utils.io.raise_warning(
                    f"Intent '{intent}' has only {count} training examples! "
                    f"Minimum is {self.MIN_EXAMPLES_PER_INTENT}, training may fail."
                )

        # emit warnings for entities with only a few training samples
        for entity, count in self.number_of_examples_per_entity.items():
            if count < self.MIN_EXAMPLES_PER_ENTITY:
                rasa.shared.utils.io.raise_warning(
                    f"Entity {entity} has only {count} training examples! "
                    f"The minimum is {self.MIN_EXAMPLES_PER_ENTITY}, because of "
                    f"this the training may fail."
                )

        # emit warnings for response intents without a response template
        for example in self.training_examples:
            if example.get(INTENT_RESPONSE_KEY) and not example.get(RESPONSE):
                rasa.shared.utils.io.raise_warning(
                    f"Your training data contains an example "
                    f"'{example.get(TEXT)[:20]}...' "
                    f"for the '{example.get_full_intent()}' intent. "
                    f"You either need to add a response phrase or correct the "
                    f"intent for this example in your training data. "
                    f"If you intend to use Response Selector in the pipeline, the "
                    f"training may fail."
                )

    def train_test_split(
        self, train_frac: float = 0.8, random_seed: Optional[int] = None
    ) -> Tuple["TrainingData", "TrainingData"]:
        """Split into a training and test dataset,
        preserving the fraction of examples per intent.
        """
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
                key = util.intent_response_key_to_template_key(ex.get_full_intent())
                responses[key] = self.responses[key]
        return responses

    def split_nlu_examples(
        self, train_frac: float, random_seed: Optional[int] = None
    ) -> Tuple[list, list]:
        """Split the training data into a train and test set.

        Args:
            train_frac: percentage of examples to add to the training set.
            random_seed: random seed used to shuffle examples.

        Returns:
            Test and training examples.
        """
        self.validate()

        # Stratified split: both test and train should have (approximately) the
        # same class distribution as the original data. We also require that
        # each class is represented in both splits.

        # First check that there is enough data to split at the requested
        # rate: we must be able to include one example per class in both
        # test and train, so num_classes is the minimum size of either.
        smaller_split_frac = train_frac if train_frac < 0.5 else (1.0 - train_frac)
        num_classes = (
            len(self.number_of_examples_per_intent.items())
            - len(self.retrieval_intents)
            + len(self.number_of_examples_per_response)
        )
        num_examples = sum(self.number_of_examples_per_intent.values())

        if int(smaller_split_frac * num_examples) + 1 < num_classes:
            rasa.shared.utils.io.raise_warning(
                f"There aren't enough intent examples in your data to include "
                f"an example of each class in both test and train splits and "
                f"also reserve {train_frac} of the data for training. "
                f"The output training fraction will differ."
            )

        # Now simulate traversing the sorted examples, sampling at a rate
        # of train_frac, so that after traversing k examples (for all k), we
        # have sampled int(k * train_frac) of them for training.
        # Corner case that makes this approximate: we require at least one sample
        # in test, and at least one in train, so proportions will be less exact
        # when classes have few examples, e.g. when a class has only 2 examples
        # but the user requests an 80% / 20% split.

        train, test = [], []

        # helper to simulate the traversal of all examples in a single class
        def _split_class(
            _examples: List[Message], _running_count: int, _running_train_count: int
        ) -> Tuple[int, int]:
            if random_seed is not None:
                random.Random(random_seed).shuffle(_examples)
            else:
                random.shuffle(_examples)

            # first determine how many samples we should have in training after
            # traversing the examples in this class, if sampling train_frac of
            # them. Then adjust so there's at least one example in test and train.
            # Adjustment can accumulate until we encounter a frequent class.
            exact_train_count = (
                int((_running_count + len(_examples)) * train_frac)
                - _running_train_count
            )
            approx_train_count = min(len(_examples) - 1, max(1, exact_train_count))

            train.extend(_examples[:approx_train_count])
            test.extend(_examples[approx_train_count:])

            return (
                _running_count + len(_examples),
                _running_train_count + approx_train_count,
            )

        training_examples = set(self.training_examples)
        running_count = 0
        running_train_count = 0

        # Sort by class frequency so we first handle the tail of the distribution,
        # where the percentages in the split are most approximate. Items from
        # more frequent classes can then be over/ undersampled as needed to
        # meet the requested train_frac. First for responses:
        for response, _ in sorted(
            self.number_of_examples_per_response.items(), key=operator.itemgetter(1)
        ):
            examples = [
                e
                for e in training_examples
                if e.get(INTENT_RESPONSE_KEY) and e.get(INTENT_RESPONSE_KEY) == response
            ]
            running_count, running_train_count = _split_class(
                examples, running_count, running_train_count
            )
            training_examples = training_examples - set(examples)

        # Again for intents:
        for intent, _ in sorted(
            self.number_of_examples_per_intent.items(), key=operator.itemgetter(1)
        ):
            examples = [
                e
                for e in training_examples
                if INTENT in e.data and e.data[INTENT] == intent
            ]
            if len(examples) > 0:  # will be 0 for retrieval intents
                running_count, running_train_count = _split_class(
                    examples, running_count, running_train_count
                )
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

    def contains_no_pure_nlu_data(self) -> bool:
        """Checks if any NLU training data was loaded."""
        lists_to_check = [
            self.nlu_examples,
            self.entity_synonyms,
            self.regex_features,
            self.lookup_tables,
        ]
        return not any([len(lst) > 0 for lst in lists_to_check])

    def has_e2e_examples(self) -> bool:
        """Checks if there are any training examples from e2e stories."""
        return any(message.is_e2e_message() for message in self.training_examples)


def list_to_str(lst: List[Text], delim: Text = ", ", quote: Text = "'") -> Text:
    """Converts list to a string.

    Args:
        lst: The list to convert.
        delim: The delimiter that is used to separate list inputs.
        quote: The quote that is used to wrap list inputs.

    Returns:
        The string.
    """
    return delim.join([quote + e + quote for e in lst])

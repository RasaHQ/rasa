import logging
import os
import random
from collections import Counter, OrderedDict
from copy import deepcopy
from os.path import relpath
from typing import Any, Dict, List, Optional, Set, Text, Tuple

import rasa.nlu.utils
from rasa.utils.common import raise_warning, lazy_property
from rasa.nlu.constants import RESPONSE, RESPONSE_KEY_ATTRIBUTE
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
        lookup_tables: Optional[List[Dict[Text, Text]]] = None,
        nlg_stories: Optional[Dict[Text, List[Text]]] = None,
    ) -> None:

        if training_examples:
            self.training_examples = self.sanitize_examples(training_examples)
        else:
            self.training_examples = []
        self.entity_synonyms = entity_synonyms if entity_synonyms else {}
        self.regex_features = regex_features if regex_features else []
        self.sort_regex_features()
        self.lookup_tables = lookup_tables if lookup_tables else []
        self.nlg_stories = nlg_stories if nlg_stories else {}

    def merge(self, *others: "TrainingData") -> "TrainingData":
        """Return merged instance of this data with other training data."""

        training_examples = deepcopy(self.training_examples)
        entity_synonyms = self.entity_synonyms.copy()
        regex_features = deepcopy(self.regex_features)
        lookup_tables = deepcopy(self.lookup_tables)
        nlg_stories = deepcopy(self.nlg_stories)
        others = [other for other in others if other]

        for o in others:
            training_examples.extend(deepcopy(o.training_examples))
            regex_features.extend(deepcopy(o.regex_features))
            lookup_tables.extend(deepcopy(o.lookup_tables))

            for text, syn in o.entity_synonyms.items():
                check_duplicate_synonym(
                    entity_synonyms, text, syn, "merging training data"
                )

            entity_synonyms.update(o.entity_synonyms)
            nlg_stories.update(o.nlg_stories)

        return TrainingData(
            training_examples,
            entity_synonyms,
            regex_features,
            lookup_tables,
            nlg_stories,
        )

    def filter_by_intent(self, intent: Text):
        """Filter training examples """

        training_examples = []
        for ex in self.training_examples:
            if ex.get("intent") == intent:
                training_examples.append(ex)

        return TrainingData(
            training_examples,
            self.entity_synonyms,
            self.regex_features,
            self.lookup_tables,
        )

    def __hash__(self) -> int:
        from rasa.core import utils as core_utils

        stringified = self.nlu_as_json() + self.nlg_as_markdown()
        text_hash = core_utils.get_text_hash(stringified)

        return int(text_hash, 16)

    @staticmethod
    def sanitize_examples(examples: List[Message]) -> List[Message]:
        """Makes sure the training data is clean.

        Remove trailing whitespaces from intent and response annotations and drop duplicate examples."""

        for ex in examples:
            if ex.get("intent"):
                ex.set("intent", ex.get("intent").strip())

            if ex.get("response"):
                ex.set("response", ex.get("response").strip())

        return list(OrderedDict.fromkeys(examples))

    @lazy_property
    def intent_examples(self) -> List[Message]:
        return [ex for ex in self.training_examples if ex.get("intent")]

    @lazy_property
    def response_examples(self) -> List[Message]:
        return [ex for ex in self.training_examples if ex.get("response")]

    @lazy_property
    def entity_examples(self) -> List[Message]:
        return [ex for ex in self.training_examples if ex.get("entities")]

    @lazy_property
    def intents(self) -> Set[Text]:
        """Returns the set of intents in the training data."""
        return {ex.get("intent") for ex in self.training_examples} - {None}

    @lazy_property
    def responses(self) -> Set[Text]:
        """Returns the set of responses in the training data."""
        return {ex.get("response") for ex in self.training_examples} - {None}

    @lazy_property
    def retrieval_intents(self) -> Set[Text]:
        """Returns the total number of response types in the training data"""
        return {
            ex.get("intent")
            for ex in self.training_examples
            if ex.get("response") is not None
        }

    @lazy_property
    def examples_per_intent(self) -> Dict[Text, int]:
        """Calculates the number of examples per intent."""
        intents = [ex.get("intent") for ex in self.training_examples]
        return dict(Counter(intents))

    @lazy_property
    def examples_per_response(self) -> Dict[Text, int]:
        """Calculates the number of examples per response."""
        return dict(Counter(self.responses))

    @lazy_property
    def entities(self) -> Set[Text]:
        """Returns the set of entity types in the training data."""
        entity_types = [e.get("entity") for e in self.sorted_entities()]
        return set(entity_types)

    @lazy_property
    def examples_per_entity(self) -> Dict[Text, int]:
        """Calculates the number of examples per entity."""
        entity_types = [e.get("entity") for e in self.sorted_entities()]
        return dict(Counter(entity_types))

    def sort_regex_features(self) -> None:
        """Sorts regex features lexicographically by name+pattern"""
        self.regex_features = sorted(
            self.regex_features, key=lambda e: "{}+{}".format(e["name"], e["pattern"])
        )

    def fill_response_phrases(self) -> None:
        """Set response phrase for all examples by looking up NLG stories"""
        for example in self.training_examples:
            response_key = example.get(RESPONSE_KEY_ATTRIBUTE)
            # if response_key is None, that means the corresponding intent is not a retrieval intent
            # and hence no response text needs to be fetched.
            # If response_key is set, fetch the corresponding response text
            if response_key:
                # look for corresponding bot utterance
                story_lookup_intent = example.get_combined_intent_response_key()
                assistant_utterances = self.nlg_stories.get(story_lookup_intent, [])
                if assistant_utterances:
                    # selecting only first assistant utterance for now
                    example.set(RESPONSE, assistant_utterances[0])
                else:
                    raise ValueError(
                        "No response phrases found for {}. Check training data "
                        "files for a possible wrong intent name in NLU/NLG file".format(
                            story_lookup_intent
                        )
                    )

    def nlu_as_json(self, **kwargs: Any) -> Text:
        """Represent this set of training examples as json."""
        from rasa.nlu.training_data.formats import (  # pytype: disable=pyi-error
            RasaWriter,
        )

        return RasaWriter().dumps(self, **kwargs)

    def as_json(self) -> Text:

        raise_warning(
            "Function 'as_json()' is deprecated and will be removed "
            "in future versions. Use 'nlu_as_json()' instead.",
            DeprecationWarning,
        )

        return self.nlu_as_json()

    def nlg_as_markdown(self) -> Text:
        """Generates the markdown representation of the response phrases(NLG) of
        TrainingData."""

        from rasa.nlu.training_data.formats import (  # pytype: disable=pyi-error
            NLGMarkdownWriter,
        )

        return NLGMarkdownWriter().dumps(self)

    def nlu_as_markdown(self) -> Text:
        """Generates the markdown representation of the NLU part of TrainingData."""
        from rasa.nlu.training_data.formats import (  # pytype: disable=pyi-error
            MarkdownWriter,
        )

        return MarkdownWriter().dumps(self)

    def as_markdown(self) -> Text:

        raise_warning(
            "Function 'as_markdown()' is deprecated and will be removed "
            "in future versions. Use 'nlu_as_markdown()' and 'nlg_as_markdown()' "
            "instead.",
            DeprecationWarning,
        )

        return self.nlu_as_markdown()

    def persist_nlu(self, filename: Text = DEFAULT_TRAINING_DATA_OUTPUT_PATH):

        if filename.endswith("json"):
            rasa.nlu.utils.write_to_file(filename, self.nlu_as_json(indent=2))
        elif filename.endswith("md"):
            rasa.nlu.utils.write_to_file(filename, self.nlu_as_markdown())
        else:
            ValueError(
                "Unsupported file format detected. Supported file formats are 'json' "
                "and 'md'."
            )

    def persist_nlg(self, filename: Text) -> None:

        nlg_serialized_data = self.nlg_as_markdown()
        if nlg_serialized_data == "":
            return

        rasa.nlu.utils.write_to_file(filename, self.nlg_as_markdown())

    @staticmethod
    def get_nlg_persist_filename(nlu_filename: Text) -> Text:

        # Add nlg_ as prefix and change extension to .md
        filename = os.path.join(
            os.path.dirname(nlu_filename),
            "nlg_" + os.path.splitext(os.path.basename(nlu_filename))[0] + ".md",
        )
        return filename

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
            self.intent_examples, key=lambda e: (e.get("intent"), e.get("response"))
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
        for intent, count in self.examples_per_intent.items():
            if count < self.MIN_EXAMPLES_PER_INTENT:
                raise_warning(
                    f"Intent '{intent}' has only {count} training examples! "
                    f"Minimum is {self.MIN_EXAMPLES_PER_INTENT}, training may fail."
                )

        # emit warnings for entities with only a few training samples
        for entity_type, count in self.examples_per_entity.items():
            if count < self.MIN_EXAMPLES_PER_ENTITY:
                raise_warning(
                    f"Entity '{entity_type}' has only {count} training examples! "
                    f"The minimum is {self.MIN_EXAMPLES_PER_ENTITY}, because of "
                    f"this the training may fail."
                )

    def train_test_split(
        self, train_frac: float = 0.8, random_seed: Optional[int] = None
    ) -> Tuple["TrainingData", "TrainingData"]:
        """Split into a training and test dataset,
        preserving the fraction of examples per intent."""

        # collect all nlu data
        test, train = self.split_nlu_examples(train_frac, random_seed)

        # collect all nlg stories
        test_nlg_stories, train_nlg_stories = self.split_nlg_responses(test, train)

        data_train = TrainingData(
            train,
            entity_synonyms=self.entity_synonyms,
            regex_features=self.regex_features,
            lookup_tables=self.lookup_tables,
            nlg_stories=train_nlg_stories,
        )
        data_train.fill_response_phrases()

        data_test = TrainingData(
            test,
            entity_synonyms=self.entity_synonyms,
            regex_features=self.regex_features,
            lookup_tables=self.lookup_tables,
            nlg_stories=test_nlg_stories,
        )
        data_test.fill_response_phrases()

        return data_train, data_test

    def split_nlg_responses(
        self, test, train
    ) -> Tuple[Dict[Text, list], Dict[Text, list]]:

        train_nlg_stories = self.build_nlg_stories_from_examples(train)
        test_nlg_stories = self.build_nlg_stories_from_examples(test)
        return test_nlg_stories, train_nlg_stories

    @staticmethod
    def build_nlg_stories_from_examples(examples) -> Dict[Text, list]:

        nlg_stories = {}
        for ex in examples:
            if ex.get(RESPONSE_KEY_ATTRIBUTE) and ex.get(RESPONSE):
                nlg_stories[ex.get_combined_intent_response_key()] = [ex.get(RESPONSE)]
        return nlg_stories

    def split_nlu_examples(
        self, train_frac: float, random_seed: Optional[int] = None
    ) -> Tuple[list, list]:
        train, test = [], []
        for intent, count in self.examples_per_intent.items():
            ex = [e for e in self.intent_examples if e.data["intent"] == intent]
            if random_seed is not None:
                random.Random(random_seed).shuffle(ex)
            else:
                random.shuffle(ex)

            n_train = int(count * train_frac)
            train.extend(ex[:n_train])
            test.extend(ex[n_train:])
        return test, train

    def print_stats(self) -> None:
        logger.info(
            "Training data stats: \n"
            + "\t- intent examples: {} ({} distinct intents)\n".format(
                len(self.intent_examples), len(self.intents)
            )
            + "\t- Found intents: {}\n".format(list_to_str(self.intents))
            + "\t- Number of response examples: {} ({} distinct response)\n".format(
                len(self.response_examples), len(self.responses)
            )
            + "\t- entity examples: {} ({} distinct entities)\n".format(
                len(self.entity_examples), len(self.entities)
            )
            + "\t- found entities: {}\n".format(list_to_str(self.entities))
        )

    def is_empty(self) -> bool:
        """Checks if any training data was loaded."""

        lists_to_check = [
            self.training_examples,
            self.entity_synonyms,
            self.regex_features,
            self.lookup_tables,
        ]
        return not any([len(l) > 0 for l in lists_to_check])

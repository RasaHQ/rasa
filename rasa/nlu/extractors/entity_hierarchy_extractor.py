import os
import string
import io
import re
from typing import Any, Dict, List, Optional, Text, Type, Union
from glob import glob
import rasa.shared.utils.io


from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.utils import write_json_to_file
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_CONFIDENCE,
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
)
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import logging
from rasa.shared.utils.io import read_yaml_file

logger = logging.getLogger(f"{__name__}")

ENTITY_ATTRIBUTE_PROCESSORS = "processors"

# CONSTANTS
TARGET_VALUE_KEY = "value"
EXAMPLES = "examples"
EXAMPLE_TEXT = "text"
EXAMPLE_REF = "ref"
EXAMPLE_COMPOSITE = "composite"
EXAMPLE_TEXT_ALT_SPELLING = "alternatives"
DONT_CREATE_ENTITY = "structure_only"
MAPPINGS_KEY = "mappings"
CONFIG_KEY = "config"

logger = logging.getLogger(__file__)


def _walkthrough(dict_of_lists: dict) -> list:
    """List generator for GridSearch-like parameter searches.

    Given a dict of lists it returns a list of dicts.


    >>>walkthrough({'a':[1],'b':5, 'c':[3,6],'d':[1,2,3,4]})
        [{'a': 1, 'b': 5, 'c': 3, 'd': 1},
        {'a': 1, 'b': 5, 'c': 3, 'd': 2},
        {'a': 1, 'b': 5, 'c': 3, 'd': 3},
        {'a': 1, 'b': 5, 'c': 3, 'd': 4},
        {'a': 1, 'b': 5, 'c': 6, 'd': 1},
        {'a': 1, 'b': 5, 'c': 6, 'd': 2},
        {'a': 1, 'b': 5, 'c': 6, 'd': 3},
        {'a': 1, 'b': 5, 'c': 6, 'd': 4}]

    Args:
        dict_of_lists (dict): The dictionary with lists as values.
        (Scalars will be converted)

    Returns:
        list: A list of dictionaries. For each dict another combination
        from the lists of the original dict is used.
    """

    def __iterate(the_list: list, __cur=[]):
        par = the_list[0][0]
        pvm = []
        if not isinstance(
            the_list[0][1], list
        ):  # avoid single items breaking th e tool
            the_list[0][1] = [the_list[0][1]]

        if len(the_list) > 1:  # there are more parameters to come
            for pv in the_list[0][1]:
                pvm.extend(
                    __iterate(the_list[1:], __cur + [(par, pv)])
                )  # go through the others
        else:
            # lowest level
            for pv in the_list[0][1]:
                pvm.append(__cur + [(par, pv)])
        return pvm

    # logger.debug(f"Iterating {crossmul(dict_of_lists)} item combinations")
    tups = [[k, v] for (k, v) in dict_of_lists.items()]
    return [dict(a) for a in __iterate(tups)]


def _topdownparser(data: Dict[str, list]) -> dict:
    """Parse a top-down type entity hierarchy dictionary.

    The format is
    ```
    (target-entity):
      config:
        structure_only: False
        mappings:
          - value: (target-value)  [value key is optional]
            examples:
              - text: (source-string) ...
              - ref: (another target entity) ...
              - composite: (text {target entity name} text)...
    ```

    Args:
        data (Dict[str,dict]): top down hierarchy loaded from file(s)

    Returns:
        dict: bottom up hierarchy for fast replacements of values
    """
    target_mapping = {}
    alternatives_mapping = {}

    def create_text_entry(
        ent_source_value: Text,
        ent_target: Union[str, None],
        val_target: Any = True,
        # ent_restriction: str = ANY_SOURCE_ENTITY_KEY,
    ) -> None:
        if not ent_target:
            return
        if not target_mapping.get(ent_source_value):
            target_mapping[ent_source_value] = {}
        target_mapping[ent_source_value][ent_target] = val_target

    def collect_composite(keyword: str) -> List[str]:
        """Return example texts as list of strings, starting with the given keyword.

        Local target values are ignored.
        'ref' and 'composite' references are followed recursively.

        Args:
            keyword (str): The entity value to look for examples

        Returns:
            List[str]: A list of strings, containing each text
            example found under keyword.
        """
        # collect all strings from the examples as list
        # if example is ref, include those as well
        # if example is composite, recurse
        refdata: dict = data.get(keyword)
        if not refdata:
            raise ValueError(f"Missing entry for {keyword}")
        start: list = refdata.get(MAPPINGS_KEY)
        # ignore value
        # take only example keys
        results = []
        for entry in start:
            for empl in entry.get(EXAMPLES):
                text = empl.get(EXAMPLE_TEXT)
                ref = empl.get(EXAMPLE_REF)
                if text:
                    results.append(text)
                    alts = empl.get(EXAMPLE_TEXT_ALT_SPELLING)
                    if alts:
                        results.extend(alts)
                if ref:
                    results.extend(collect_composite(ref))
                comp = empl.get(EXAMPLE_COMPOSITE)
                if comp:
                    results.extend(process_composite(comp))
        return results

    def process_composite(composite_text: str) -> List[str]:
        """Search for all valid combinations.

        Search for all valid combinations that are possible given a
        f-string-like text, such as "{handy}-vertrag". It will look for the
        handy-keyword and return all the text examples below, plus following
        all the 'ref' examples, plus also recursively resolving all other
        'composite' examples themselves.

        Args:
            composite_text (str): f-string-like text, such as
            "static {reference}"

        Returns:
            List[str]: List of composed text strings with all placeholders
            replaced
        """
        # take one f-string and return a list of populated strings
        compounds = re.findall(r"{(.*?)}", composite_text)
        # logger.debug(compounds)
        replacers = {}
        # collect the lists of replacements
        for placeholder in compounds:
            replacers[placeholder] = collect_composite(placeholder)
        # iterate the combinations
        results = []
        for one_combination in _walkthrough(replacers):
            results.append(composite_text.format(**one_combination))
        return results

    def parse_one(
        target_entity: Union[str, None],
        e_data_lst: list,
        e_config: dict,
        parent_target_value: Any = None,
    ):
        """Parse one entity section fro the YAML-derived object.

        Args:
            target_entity (Union[str, None]): literal name of the entity to be
                defined
            e_data_lst (list): List object containing examples (and optional
                values)
            e_config (dict): Configuration dict defined with the entity
            parent_target_value (Any, optional): If recursively called,
                the optional fixed value of th eparent entity.
                Defaults to None.
        """
        for e_dict in e_data_lst:

            # the dictionary is supposed to have up to two keys:
            # value (optional) - string
            # examples - list

            if not isinstance(e_dict, dict):
                raise ValueError(
                    f"Target entity {target_entity} malformed for entity "
                    "hierarchy parser. Object expected."
                )

            targ_val = parent_target_value or e_dict.get(TARGET_VALUE_KEY)

            for example in e_dict.get(EXAMPLES, []):

                text = example.get(EXAMPLE_TEXT)
                ref = example.get(EXAMPLE_REF)
                composite: str = example.get(EXAMPLE_COMPOSITE)

                if text:  # literal text entry
                    create_text_entry(
                        ent_source_value=text,
                        ent_target=target_entity,
                        val_target=targ_val or text,
                    )
                    alts = example.get(
                        EXAMPLE_TEXT_ALT_SPELLING, []
                    )  # alternative spellings
                    for alternative in alts:
                        alternatives_mapping.update({alternative: text})

                if ref:
                    parse_one(
                        target_entity=target_entity,
                        e_data_lst=data[ref].get(MAPPINGS_KEY, []),
                        e_config=data[ref].get(
                            CONFIG_KEY, {}
                        ),  # do _not_ pass parent config.
                        # For future detailed augmentation
                        parent_target_value=targ_val,  # pass parent value
                    )

                if composite:
                    # restrictions: composites do only pull all
                    # texts (also recursive) from mentioned
                    # entity-values
                    cmp_list = process_composite(composite)
                    for word in cmp_list:
                        # create an entry per returned string
                        create_text_entry(
                            ent_source_value=word,
                            ent_target=target_entity,
                            val_target=targ_val or word,
                        )

    for target_entity, entity_dict in data.items():
        e_config = entity_dict.get(CONFIG_KEY, {})
        e_data_lst = entity_dict.get(MAPPINGS_KEY, [])
        if not e_config.get(DONT_CREATE_ENTITY, False):
            parse_one(target_entity, e_data_lst, e_config)
    return {"entities": target_mapping, "alternatives": alternatives_mapping}


# subclass EntityExtractor to skip featurize_message() in
# rasa.nlu.model.Interpreter
class EntityHierarchyExtractor(EntityExtractor):
    """EntityHierarchy is a multiple entity per token extractor."""

    # Which components are required by this component.
    # Listed components should appear before the component itself in the
    # pipeline.
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""
        return [EntityExtractor]

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    # entityfile: Single yaml file name or GLOB pattern,
    # such as ./entity/**/*.yml
    defaults: dict = {
        "entityfile": None,
        "case_sensitive": False,
        "include_repeated_entities": False,  # if true the same entity
        # will only return its first occurrence
        "non_word_boundaries": "_öäüÖÄÜß-",  # include German Umlauts and
        # hyphen
    }

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    supported_language_list = None

    # Defines what language(s) this component can NOT handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    not_supported_language_list = None  # TODO add non-whitespace tokenizable languages

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        entityhierarchy: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Create a EntityHierachy which is a multiple entity per token extractor.

        Args:
            component_config (Optional[Dict[Text, Any]], optional):
                Defaults to None.
            entityhierarchy (Optional[Dict[Text, Any]], optional):
                Defaults to None.
        """
        super().__init__(component_config)
        if not component_config:
            component_config = self.defaults
        self.keyword_processor = _KeywordProcessor(
            case_sensitive=self.component_config["case_sensitive"]
        )
        for non_word_boundary in self.component_config["non_word_boundaries"]:
            self.keyword_processor.add_non_word_boundary(non_word_boundary)
        self._entityfile = component_config.get("entityfile", None)
        self.include_repeated_entities = component_config.get(
            "include_repeated_entities", False
        )

        if entityhierarchy:
            logger.debug("restore entity hierarchy")
            self._entityhierarchy = entityhierarchy
            self._parse_prepared_hierarchies()
        else:
            self._entityhierarchy = {}

    def _parse_prepared_hierarchies(self):
        """Train the flashtext component with prepared hierarchies."""
        for keyword, ent_dict in self._entityhierarchy.get("entities", {}).items():
            # keyword is the full text to be found, the dict contains
            # entity:value pairs to be set as flashtext can store ANY
            # python object to be returned, we'll use the full dict as
            # return value
            self.keyword_processor.add_keyword(keyword, ent_dict)

        lookups = self.keyword_processor.get_all_keywords()
        if len(lookups.keys()) == 0:
            rasa.shared.utils.io.raise_warning(
                "No entity hierarchies defined in the training data that "
                "have text examples to use for the extractor"
            )
        # populate the secondary alternatives dictionary too
        # alternative spellings are a two step process: return string and
        # lookup string in dictionary
        for keyword, clean_name in self._entityhierarchy.get(
            "alternatives", {}
        ).items():
            self.keyword_processor.add_keyword(keyword, clean_name)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one.
        """
        self._entityhierarchy = {}
        # read the YAML file(s)
        if not self._entityfile:
            rasa.shared.utils.io.raise_warning(
                "EntityHierarchy is in the pipeline but no entity file name"
                "is defined in config."
            )
            return
        filelist = glob(self._entityfile, recursive=True)

        raw_hierarchy = {}
        if filelist:
            # read each file and merge the results
            for fn in filelist:
                logger.debug(f"reading file {fn}")
                filecontent = read_yaml_file(fn)
                if isinstance(filecontent, dict):
                    if [k for k in filecontent if k in raw_hierarchy]:
                        raise ValueError(
                            "Duplicate key(s) "
                            f"{[k for k in filecontent if k in raw_hierarchy]}"
                            f" found in file {fn}"
                        )  # TODO replace by proper RasaException
                        # for invalid keys in YAML
                    raw_hierarchy.update(filecontent)
                    logger.info(f"Processed file {fn}")
                else:
                    logger.warn(
                        f"{fn} invalid file format: must be a " "dictionary in YAML"
                    )
        else:
            rasa.shared.utils.io.raise_warning(
                "EntityHierarchy is in the pipeline but no entity file name"
                "is loadable from definition in config."
            )
            return
        self._entityhierarchy = _topdownparser(raw_hierarchy)

        self._parse_prepared_hierarchies()

    # process from flashE
    def process(self, message: Message, **kwargs: Any) -> None:
        """Process a message with EntityHierarchy.

        Args:
            message (Message): Rasa Message (Training, Inference)
        """
        extracted_entities = self._extract_entities(message)
        extracted_entities = self.add_extractor_name(extracted_entities)
        entities = self._extent_entities(
            original_entities=message.get(ENTITIES, []),
            new_entities=extracted_entities,
        )

        message.set(ENTITIES, entities, add_to_output=True)

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        if len(self.keyword_processor) == 0:
            return []
        matches_ = self.keyword_processor.extract_keywords(
            message.get(TEXT), span_info=True
        )
        # matches looks like
        # [
        # ({"festnetz": true,"internet": "wlan","wlan": "wlan",
        #   "topic": "festnetz"}, 39, 54),
        # ({'festnetz': True}, 63, 72)},
        # ('somethingfixed',100,112)
        # ]
        # if match[0] is a string it was an alternative spelling hit
        #
        matches = []
        for match in matches_:
            match = list(match)  # convert tuple to list to make it mutable
            # do the lookup of alternative spellings first
            if isinstance(match[0], (str, int, float)):
                # look it up and replace it
                match[0] = self._entityhierarchy.get("entities", {}).get(match[0], {})
            matches.append(match)
        # if duplicates are to be ignored, sort the list and remove duplicates
        if not self.include_repeated_entities:
            matches.sort(
                key=lambda e: e[1]
            )  # sort by first occurrence in the message text

        extracted_entities = []
        name_cache = []

        for match in matches:
            for entity_type, entity_value in match[0].items():
                if entity_type not in name_cache:
                    if not self.include_repeated_entities:
                        name_cache.append(entity_type)
                    extracted_entities += [
                        {
                            ENTITY_ATTRIBUTE_TYPE: entity_type,
                            ENTITY_ATTRIBUTE_START: match[1],
                            ENTITY_ATTRIBUTE_END: match[2],
                            ENTITY_ATTRIBUTE_VALUE: entity_value,
                            ENTITY_ATTRIBUTE_CONFIDENCE: 1.0,
                        }
                    ]
        return extracted_entities

    def _extent_entities(
        self,
        original_entities: List[Dict[Text, Any]],
        new_entities: List[Dict[Text, Any]],
    ) -> List[Dict[Text, Any]]:
        """Add new_entities to original_entities and returns the complete list.

           Respects setting of self._ignore_repeated_entities

        Args:
            original_entities (List[Dict[Text, Any]]): The entities already
                contained in the message, to be altered in_place
            new_entities (List[Dict[Text, Any]]): The entities to add to
                the message
        """
        entities = original_entities[:]
        entity_keys = [e.get(ENTITY_ATTRIBUTE_TYPE) for e in entities]
        for ent in new_entities:
            if (
                self.include_repeated_entities
                or ent.get(ENTITY_ATTRIBUTE_TYPE) not in entity_keys
            ):
                self.add_extractor_name([ent])
                entities.append(ent)
            else:
                # change value of passed entity
                pos = entity_keys.index(ent.get(ENTITY_ATTRIBUTE_TYPE))
                entity = entities[pos]
                entity.update({ENTITY_ATTRIBUTE_VALUE: ent.get(ENTITY_ATTRIBUTE_VALUE)})
                self.add_processor_name(entity)
        # entities.extend(new_ents)
        return entities

    # SAFE and LOAD methods
    #
    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        if self._entityhierarchy:
            file_name = file_name + ".json"
            entity_files = os.path.join(model_dir, file_name)
            write_json_to_file(entity_files, self._entityhierarchy)

            return {"file": file_name}
        else:
            return {"file": None}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""
        file_name = meta.get("file")
        if not file_name:
            enthier = None
            return cls(meta, enthier)

        entities_file = os.path.join(model_dir, file_name)
        if os.path.isfile(entities_file):
            enthier = rasa.shared.utils.io.read_json_file(entities_file)
        else:
            enthier = None
        return cls(meta, enthier)


#################
# modified version of flashtext
# https://github.com/vi3k6i5/flashtext
#
# MIT License
# Copyright (c) 2017 Vikash Singh (vikash.duliajan@gmail.com)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################


class _KeywordProcessor(object):
    """KeywordProcessor.

    Attributes:
        _keyword (str): Used as key to store keywords in trie dictionary.
            Defaults to '_keyword_'
        non_word_boundaries (set(str)): Characters that will determine if
            the word is continuing. Defaults to set([A-Za-z0-9_])
        keyword_trie_dict (dict): Trie dict built character by character,
            that is used for lookup. Defaults to empty dictionary
        case_sensitive (boolean): if the search algorithm should be case
            sensitive or not. Defaults to False

    Note:
        * loosely based on `Aho-Corasick algorithm
          <https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm>`_.
        * Idea came from this `Stack Overflow Question
          <https://stackoverflow.com/questions/44178449/regex-replace-is-taking-time-for-millions-of-documents-how-to-make-it-faster>`_.
    """

    def __init__(self, case_sensitive=False):
        """
        Create a KeywordProcessor.

        Args:
            case_sensitive (boolean): Keyword search should be case sensitive
                set or not.
                Defaults to False
        """
        self._keyword = "_keyword_"
        self._white_space_chars = set([".", "\t", "\n", "\a", " ", ","])
        if case_sensitive:
            # python 3.x
            self.non_word_boundaries = set(string.digits + string.ascii_letters + "_")
        else:
            self.non_word_boundaries = set(string.digits + string.ascii_lowercase + "_")
        self.keyword_trie_dict = dict()
        self.case_sensitive = case_sensitive
        self._terms_in_trie = 0

    def __len__(self):
        """Return number of terms present in the keyword_trie_dict.

        Returns:
            length : int
                Count of number of distinct terms in trie dictionary.

        """
        return self._terms_in_trie

    def __contains__(self, word):
        """Check if word is present in the keyword_trie_dict.

        Args:
            word : string
                word that you want to check

        Returns:
            status : bool
                If word is present as it is in keyword_trie_dict then
                we return True, else False

        Examples:
            >>> keyword_processor.add_keyword('Big Apple')
            >>> 'Big Apple' in keyword_processor
            >>> # True

        """
        if not self.case_sensitive:
            word = word.lower()
        current_dict = self.keyword_trie_dict
        len_covered = 0
        for char in word:
            if char in current_dict:
                current_dict = current_dict[char]
                len_covered += 1
            else:
                break
        return self._keyword in current_dict and len_covered == len(word)

    def __getitem__(self, word):
        """If word is present in keyword_trie_dict return the clean name for it.

        Args:
            word : string
                word that you want to check

        Returns:
            keyword : string
                If word is present as it is in keyword_trie_dict then we return
                keyword mapped to it.

        Examples:
            >>> keyword_processor.add_keyword('Big Apple', 'New York')
            >>> keyword_processor['Big Apple']
            >>> # New York
        """
        if not self.case_sensitive:
            word = word.lower()
        current_dict = self.keyword_trie_dict
        len_covered = 0
        for char in word:
            if char in current_dict:
                current_dict = current_dict[char]
                len_covered += 1
            else:
                break
        if self._keyword in current_dict and len_covered == len(word):
            return current_dict[self._keyword]

    def __setitem__(self, keyword, clean_name=None):
        """Add keyword to the dictionary.

        Pass the keyword and the clean name it maps to.

        Args:
            keyword : string
                keyword that you want to identify

            clean_name : string
                clean term for that keyword that you would want to get back in
                return or replace
                if not provided, keyword will be used as the clean name also.

        Examples:
            >>> keyword_processor['Big Apple'] = 'New York'
        """
        status = False
        if not clean_name and keyword:
            clean_name = keyword

        if keyword and clean_name:
            if not self.case_sensitive:
                keyword = keyword.lower()
            current_dict = self.keyword_trie_dict
            for letter in keyword:
                current_dict = current_dict.setdefault(letter, {})
            if self._keyword not in current_dict:
                status = True
                self._terms_in_trie += 1
            current_dict[self._keyword] = clean_name
        return status

    def __delitem__(self, keyword):
        """Remove keyword from the dictionary.

        Pass the keyword to be deleted.

        Args:
            keyword : string
                keyword that you want to remove if it's present

        Examples:
            >>> keyword_processor.add_keyword('Big Apple')
            >>> del keyword_processor['Big Apple']
        """
        status = False
        if keyword:
            if not self.case_sensitive:
                keyword = keyword.lower()
            current_dict = self.keyword_trie_dict
            character_trie_list = []
            for letter in keyword:
                if letter in current_dict:
                    character_trie_list.append((letter, current_dict))
                    current_dict = current_dict[letter]
                else:
                    # if character is not found, break out of the loop
                    current_dict = None
                    break
            # remove the characters from trie dict if there are no other
            # keywords with them
            if current_dict and self._keyword in current_dict:
                # we found a complete match for input keyword.
                character_trie_list.append((self._keyword, current_dict))
                character_trie_list.reverse()

                for key_to_remove, dict_pointer in character_trie_list:
                    if len(dict_pointer.keys()) == 1:
                        dict_pointer.pop(key_to_remove)
                    else:
                        # more than one key means more than 1 path.
                        # Delete not required path and keep the other
                        dict_pointer.pop(key_to_remove)
                        break
                # successfully removed keyword
                status = True
                self._terms_in_trie -= 1
        return status

    def __iter__(self):
        """Disabled iteration as get_all_keywords() is the right way to iterate.

        Deprecated.
        """
        raise NotImplementedError("Please use get_all_keywords() instead")

    def set_non_word_boundaries(self, non_word_boundaries):
        """Set of characters that will be considered as part of word.

        Args:
            non_word_boundaries (set(str)):
                Set of characters that will be considered as part of word.
        """
        self.non_word_boundaries = non_word_boundaries

    def add_non_word_boundary(self, character):
        """Add a character that will be considered as part of word.

        Args:
            character (char):
                Character that will be considered as part of word.

        """
        self.non_word_boundaries.add(character)

    def add_keyword(self, keyword, clean_name=None):
        """Add one or more keywords to the dictionary.

        pass the keyword and the clean name it maps to.

        Args:
            keyword : string
                keyword that you want to identify

            clean_name : string
                clean term for that keyword that you would want to get back in
                return or replace
                if not provided, keyword will be used as the clean name also.

        Returns:
            status : bool
                The return value. True for success, False otherwise.

        Examples:
            >>> keyword_processor.add_keyword('Big Apple', 'New York')
            >>> # This case 'Big Apple' will return 'New York'
            >>> # OR
            >>> keyword_processor.add_keyword('Big Apple')
            >>> # This case 'Big Apple' will return 'Big Apple'
        """
        return self.__setitem__(keyword, clean_name)

    def remove_keyword(self, keyword):
        """Remove one or more keywords from the dictionary.

        Args:
            keyword : string
                keyword that you want to remove if it's present

        Returns:
            status : bool
                The return value. True for success, False otherwise.

        Examples:
            >>> keyword_processor.add_keyword('Big Apple')
            >>> keyword_processor.remove_keyword('Big Apple')
            >>> # Returns True
            >>> # This case 'Big Apple' will no longer be a recognized keyword
            >>> keyword_processor.remove_keyword('Big Apple')
            >>> # Returns False

        """
        return self.__delitem__(keyword)

    def get_keyword(self, word):
        """If word is present in keyword_trie_dict return the clean name for it.

        Args:
            word : string
                word that you want to check

        Returns:
            keyword : string
                If word is present as it is in keyword_trie_dict then we return
                keyword mapped to it.

        Examples:
            >>> keyword_processor.add_keyword('Big Apple', 'New York')
            >>> keyword_processor.get('Big Apple')
            >>> # New York
        """
        return self.__getitem__(word)

    def add_keyword_from_file(self, keyword_file, encoding="utf-8"):
        """Add keywords from a file.

        Args:
            keyword_file : path to keywords file
            encoding : specify the encoding of the file

        Examples:
            keywords file format can be like:

            >>> # Option 1: keywords.txt content
            >>> # java_2e=>java
            >>> # java programing=>java
            >>> # product management=>product management
            >>> # product management techniques=>product management

            >>> # Option 2: keywords.txt content
            >>> # java
            >>> # python
            >>> # c++

            >>> keyword_processor.add_keyword_from_file('keywords.txt')

        Raises:
            IOError: If `keyword_file` path is not valid

        """
        if not os.path.isfile(keyword_file):
            raise IOError("Invalid file path {}".format(keyword_file))
        with io.open(keyword_file, encoding=encoding) as f:
            for line in f:
                if "=>" in line:
                    keyword, clean_name = line.split("=>")
                    self.add_keyword(keyword, clean_name.strip())
                else:
                    keyword = line.strip()
                    self.add_keyword(keyword)

    def add_keywords_from_dict(self, keyword_dict):
        """Add keywords from a dictionary.

        Args:
            keyword_dict (dict):
            A dictionary with `str` key and (list `str`) as value

        Examples:
            >>> keyword_dict = {
                    "java": ["java_2e", "java programing"],
                    "product management": ["PM", "product manager"]
                }
            >>> keyword_processor.add_keywords_from_dict(keyword_dict)

        Raises:
            AttributeError: If value for a key in `keyword_dict` is not a list.

        """
        for clean_name, keywords in keyword_dict.items():
            if not isinstance(keywords, list):
                raise AttributeError(
                    "Value of key {} should be a list".format(clean_name)
                )

            for keyword in keywords:
                self.add_keyword(keyword, clean_name)

    def remove_keywords_from_dict(self, keyword_dict):
        """Remove keywords from a dictionary.

        Args:
            keyword_dict (dict):
            A dictionary with `str` key and (list `str`) as value

        Examples:
            >>> keyword_dict = {
                    "java": ["java_2e", "java programing"],
                    "product management": ["PM", "product manager"]
                }
            >>> keyword_processor.remove_keywords_from_dict(keyword_dict)

        Raises:
            AttributeError: If value for a key in `keyword_dict` is not a list.

        """
        for clean_name, keywords in keyword_dict.items():
            if not isinstance(keywords, list):
                raise AttributeError(
                    "Value of key {} should be a list".format(clean_name)
                )

            for keyword in keywords:
                self.remove_keyword(keyword)

    def add_keywords_from_list(self, keyword_list):
        """Add keywords from a list.

        Args:
            keyword_list (list(str)): List of keywords to add

        Examples:
            >>> keyword_processor.add_keywords_from_list(["java", "python"]})
        Raises:
            AttributeError: If `keyword_list` is not a list.

        """
        if not isinstance(keyword_list, list):
            raise AttributeError("keyword_list should be a list")

        for keyword in keyword_list:
            self.add_keyword(keyword)

    def remove_keywords_from_list(self, keyword_list):
        """Remove keywords present in list.

        Args:
            keyword_list (list(str)): List of keywords to remove

        Raises:
            AttributeError: If `keyword_list` is not a list.

        """
        if not isinstance(keyword_list, list):
            raise AttributeError("keyword_list should be a list")

        for keyword in keyword_list:
            self.remove_keyword(keyword)

    def get_all_keywords(self, term_so_far="", current_dict=None):
        """Recursively builds a dictionary of keywords.

        And the clean name mapped to those keywords.

        Args:
            term_so_far : string
                term built so far by adding all previous characters
            current_dict : dict
                current recursive position in dictionary

        Returns:
            terms_present : dict
                A map of key and value where each key is a term in the
                keyword_trie_dict.
                And value mapped to it is the clean name mapped to it.

        Examples:
            >>> keyword_processor = KeywordProcessor()
            >>> keyword_processor.add_keyword('j2ee', 'Java')
            >>> keyword_processor.add_keyword('Python', 'Python')
            >>> keyword_processor.get_all_keywords()
            >>> {'j2ee': 'Java', 'python': 'Python'}
            >>> # NOTE: for case_insensitive all keys will be lowercased.
        """
        terms_present = {}
        if not term_so_far:
            term_so_far = ""
        if current_dict is None:
            current_dict = self.keyword_trie_dict
        for key in current_dict:
            if key == "_keyword_":
                terms_present[term_so_far] = current_dict[key]
            else:
                sub_values = self.get_all_keywords(term_so_far + key, current_dict[key])
                for key in sub_values:
                    terms_present[key] = sub_values[key]
        return terms_present

    def extract_keywords(self, sentence, span_info=False):
        """Search in the string for all keywords present in corpus.

        Keywords present are added to a list `keywords_extracted`
        and returned.

        Args:
            sentence (str): Line of text where we will search
            for keywords

        Returns:
            keywords_extracted (list(str)): List of terms/keywords
            found in sentence that match our corpus
        """
        keywords_extracted = []
        if not sentence:
            # if sentence is empty or none just return empty list
            return keywords_extracted
        if not self.case_sensitive:
            sentence = sentence.lower()
        current_dict = self.keyword_trie_dict
        sequence_start_pos = 0
        sequence_end_pos = 0
        reset_current_dict = False
        idx = 0
        sentence_len = len(sentence)
        while idx < sentence_len:
            char = sentence[idx]
            # when we reach a character that might denote word end
            if char not in self.non_word_boundaries:

                # if end is present in current_dict
                if self._keyword in current_dict or char in current_dict:
                    # update longest sequence found
                    sequence_found = None
                    longest_sequence_found = None
                    is_longer_seq_found = False
                    if self._keyword in current_dict:
                        sequence_found = current_dict[self._keyword]
                        longest_sequence_found = current_dict[self._keyword]
                        sequence_end_pos = idx

                    # re look for longest_sequence from this position
                    if char in current_dict:
                        current_dict_continued = current_dict[char]

                        idy = idx + 1
                        while idy < sentence_len:
                            inner_char = sentence[idy]
                            if (
                                inner_char not in self.non_word_boundaries
                                and self._keyword in current_dict_continued
                            ):
                                # update longest sequence found
                                longest_sequence_found = current_dict_continued[
                                    self._keyword
                                ]
                                sequence_end_pos = idy
                                is_longer_seq_found = True
                            if inner_char in current_dict_continued:
                                current_dict_continued = current_dict_continued[
                                    inner_char
                                ]
                            else:
                                break
                            idy += 1
                        else:
                            # end of sentence reached.
                            if self._keyword in current_dict_continued:
                                # update longest sequence found
                                longest_sequence_found = current_dict_continued[
                                    self._keyword
                                ]
                                sequence_end_pos = idy
                                is_longer_seq_found = True
                        if is_longer_seq_found:
                            idx = sequence_end_pos
                    current_dict = self.keyword_trie_dict
                    if longest_sequence_found:
                        keywords_extracted.append(
                            (longest_sequence_found, sequence_start_pos, idx)
                        )
                    reset_current_dict = True
                else:
                    # we reset current_dict
                    current_dict = self.keyword_trie_dict
                    reset_current_dict = True
            elif char in current_dict:
                # we can continue from this char
                current_dict = current_dict[char]
            else:
                # we reset current_dict
                current_dict = self.keyword_trie_dict
                reset_current_dict = True
                # skip to end of word
                idy = idx + 1
                while idy < sentence_len:
                    char = sentence[idy]
                    if char not in self.non_word_boundaries:
                        break
                    idy += 1
                idx = idy
            # if we are end of sentence and have a sequence discovered
            if idx + 1 >= sentence_len:
                if self._keyword in current_dict:
                    sequence_found = current_dict[self._keyword]
                    keywords_extracted.append(
                        (sequence_found, sequence_start_pos, sentence_len)
                    )
            idx += 1
            if reset_current_dict:
                reset_current_dict = False
                sequence_start_pos = idx
        if span_info:
            return keywords_extracted
        return [value[0] for value in keywords_extracted]

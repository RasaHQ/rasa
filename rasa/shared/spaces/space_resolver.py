import os
import tempfile
from typing import Text, Tuple, Optional, List, Dict

from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.spaces.story_resolver import StoryResolver
from rasa.shared.spaces.yaml_spaces_reader import Space
from rasa.shared.spaces.domain_resolver import DomainResolver
from rasa.shared.spaces.nlu_resolver import NLUResolver
import rasa.shared.utils.io

MAIN_SPACE = "main"


class SpaceResolver:

    @classmethod
    def resolve_space(cls, space: Space) -> Tuple[Dict, TrainingData, Dict]:
        """Loads a space and prefixes its data."""
        if space.name != MAIN_SPACE:
            domain_yaml, domain_info = \
                DomainResolver.load_and_resolve(space.domain_path, space.name)
            training_data = \
                NLUResolver.load_and_resolve(space.nlu_path, space.name, domain_info)
            stories_yaml = \
                StoryResolver.load_and_resolve(space.stories_path, space.name, domain_info)
        else:
            domain_yaml = DomainResolver.load_domain_yaml(space.domain_path)
            training_data = NLUResolver.load_nlu(space.nlu_path)
            stories_yaml = StoryResolver.load(space.stories_path)
        return domain_yaml, training_data, stories_yaml

    @classmethod
    def resolve_spaces(cls, spaces: List[Space],
                       target_folder: Optional[Text] = None) -> Text:
        """Loads multiple spaces, prefixes their data and joins their files."""
        if target_folder:
            os.makedirs(target_folder, exist_ok=True)
        else:
            target_folder = tempfile.mkdtemp()

        domain_folder = os.path.join(target_folder, "domain")
        nlu_folder = os.path.join(target_folder, "nlu")
        stories_folder = os.path.join(target_folder, "stories")
        os.makedirs(domain_folder, exist_ok=True)
        os.makedirs(nlu_folder, exist_ok=True)
        os.makedirs(stories_folder, exist_ok=True)

        for space in spaces:
            domain_yaml, training_data, stories_yaml = cls.resolve_space(space)
            domain_yaml_path = os.path.join(domain_folder, f"{space.name}_domain.yml")
            rasa.shared.utils.io.write_yaml(domain_yaml, domain_yaml_path, True)

            nlu_yaml_path = os.path.join(nlu_folder, f"{space.name}_nlu.yml")
            rasa_yaml_writer = RasaYAMLWriter()
            rasa_yaml_writer.dump(nlu_yaml_path, training_data)

            stories_yaml_path = os.path.join(stories_folder,
                                             f"{space.name}_stories_and_rules.yml")
            rasa.shared.utils.io.write_yaml(stories_yaml, stories_yaml_path, True)

        return target_folder

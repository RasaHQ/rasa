import os
import tempfile
from typing import Text, Tuple, Optional, List, Dict

from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.spaces.yaml_spaces_reader import Space
from rasa.shared.spaces.domain_resolver import DomainResolver, DomainInfo
from rasa.shared.spaces.nlu_resolver import NLUResolver
import rasa.shared.utils.io

MAIN_SPACE = "main"


class SpaceResolver:

    @classmethod
    def resolve_nlu(cls, nlu_path: Optional[Text], prefix: Text,
                    domain_info: DomainInfo) -> TrainingData:
        return NLUResolver.load_and_resolve(nlu_path, prefix, domain_info)

    @classmethod
    def resolve_domain(cls, domain_path: Text, prefix: Text) -> Tuple[Dict, DomainInfo]:
        return DomainResolver.load_and_resolve(domain_path, prefix)

    @classmethod
    def resolve_space(cls, space: Space) -> Tuple[Dict, TrainingData]:
        """Loads a space and prefixes its data."""
        if space.name != MAIN_SPACE:
            domain_yaml, domain_info = cls.resolve_domain(space.domain_path, space.name)
            training_data = cls.resolve_nlu(space.nlu_path, space.name, domain_info)
        else:
            domain_yaml = DomainResolver.load_domain_yaml(space.domain_path)
            training_data = NLUResolver.load_nlu(space.nlu_path)
        return domain_yaml, training_data

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
        os.makedirs(domain_folder, exist_ok=True)
        os.makedirs(nlu_folder, exist_ok=True)

        for space in spaces:
            domain_yaml, training_data = cls.resolve_space(space)
            domain_yaml_path = os.path.join(domain_folder, f"{space.name}_domain.yml")
            rasa.shared.utils.io.write_yaml(domain_yaml, domain_yaml_path, True)
            nlu_yaml_path = os.path.join(nlu_folder, f"{space.name}_nlu.yml")
            rasa_yaml_writer = RasaYAMLWriter()
            rasa_yaml_writer.dump(nlu_yaml_path, training_data)

        return target_folder

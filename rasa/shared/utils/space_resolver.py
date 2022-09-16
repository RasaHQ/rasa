import copy
import functools
from typing import Text, Tuple, Optional, List

from rasa.shared.core.domain import Domain
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.constants import INTENT
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.yaml_spaces_reader import Space

MAIN_SPACE = "main"

class SpaceResolver:

    @classmethod
    def prefix_training_data(cls, prefix: Text, domain: Domain,
                             training_data: TrainingData) -> TrainingData:
        """Prefix training data according to the prefix and domain."""
        for msg in training_data.training_examples:
            intent = msg.get(INTENT)
            if intent and intent in domain.intents:
                msg.set(INTENT, f"{prefix}!{intent}")
        # TODO: entities, lookups, regexes, responses
        return training_data

    @classmethod
    def prefix_domain(cls, prefix: Text, domain: Domain) -> Domain:
        domain_as_dict = copy.deepcopy(domain.as_dict())
        domain_as_dict["intents"] = [f"{prefix}!{intent}"
                                     for intent in domain_as_dict["intents"]]
        return Domain.from_dict(domain_as_dict)


    @classmethod
    def resolve_space(cls, space: Space) -> Tuple[Domain, Optional[TrainingData]]:
        """Loads a space an prefixes its data."""
        domain = Domain.from_path(space.domain_path)
        if space.nlu_path:
            importer = RasaFileImporter(training_data_paths=space.nlu_path)
            training_data = importer.get_nlu_data()
            if space.name != MAIN_SPACE:
                cls.prefix_training_data(space.name, domain, training_data)
        else:
            training_data = None

        if space.name != MAIN_SPACE:
            domain = cls.prefix_domain(space.name, domain)

        return domain, training_data

    @classmethod
    def resolve_spaces(cls, spaces: List[Space]) -> Tuple[Domain,
                                                          Optional[TrainingData]]:
        """Loads multiple spaces, prefixes their data and joins their files."""
        domains, training_data_instances = \
            zip(*[cls.resolve_space(space) for space in spaces])

        domain = functools.reduce(lambda d1, d2: d1.merge(d2), domains, Domain.empty())
        training_data = functools.reduce(lambda t1, t2: t1.merge(t2),
                                         training_data_instances, TrainingData())

        return domain, training_data

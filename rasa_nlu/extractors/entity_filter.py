from rasa_nlu.components import Component
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class EntityFilter(Component):
    """
    Simple component that filters entities that were not in the
    training set.

    The idea is that entities NOT in the training set are removed from
    the prediction. Therefore, this component is useful when one wants to
    handle only a finite number of entities.

    The class of entities that should be filtered must be defined in
    the pipeline definition. Furthermore, this component should come
    at the end of the pipeline definition (i.e. after the entities
    are extracted and, possibly, after the synonym component)

    """
    name = "bot.EntityFilter"

    provides = ["entities"]

    defaults = {
        "fields": []
    }

    FILENAME = "entity_filter.pkl"

    def __init__(self, component_config=None):
        super().__init__(component_config=component_config)

        self.seen_values = {}

    def train(self, training_data, cfg, **kwargs):
        self.component_config = cfg.for_component(self.name, self.defaults)

        if not self.component_config["fields"]:
            cls_name = self.__class__.__name__
            logger.warning("Component {} was called without any field to check".format(cls_name))
            return

        seen_values = {f: set() for f in self.component_config["fields"]}

        for example in training_data.entity_examples:
            for entity in example.get("entities", []):
                seen_set = seen_values.get(entity.get("entity"), None)

                if seen_set is not None:
                    seen_set.add(entity["value"])

        self.seen_values = seen_values

    def _keep_entity(self, entity):
        entity_cls = entity.get("entity", None)
        possible_values = self.seen_values.get(entity_cls)

        # Keep any element for which we do not track its value
        if possible_values is None:
            return True

        return entity.get("value") in possible_values

    def process(self, message, **kwargs):
        current_entities = message.get("entities", [])
        keep_entities = list(filter(self._keep_entity, current_entities))

        message.set("entities", keep_entities, add_to_output=True)

    def persist(self, model_dir):
        file = os.path.join(model_dir, self.FILENAME)

        with open(file, 'wb') as f:
            pickle.dump(self.seen_values, f)

        return {"entity_filter_file": file}

    @classmethod
    def load(cls,
             model_dir=None,   # type: Optional[Text]
             model_metadata=None,   # type: Optional[Metadata]
             cached_component=None,   # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        if cached_component:
            return cached_component

        component_config = model_metadata.for_component(cls.name)
        self = cls(component_config)
        file = os.path.join(model_dir, cls.FILENAME)

        with open(file, 'rb') as f:
            self.seen_values = pickle.load(f)

        return self

from rasa.nlu.components.component import Component
from typing import Any, Dict, Text, Optional, List


class ComponentIterator:
    """ Iterator class """

    def __init__(self, pipeline: ComponentPipeline):
        self._components = pipeline.get_all_components()
        self._index = 0

    def __next__(self):
        """'Returns the next value from pipeline lists """
        if self._index < (len(self._components)):
            result = (self._components[self._index])
            self._index += 1
            return result

        raise StopIteration


class ComponentPipeline:
    def __init__(self, components: Optional[List[Component]] = None):
        if components is None:
            self.__components = list()
        else:
            self.__components = components

    def __iter__(self):
        return ComponentIterator(self)

    def add_component(self, component: Component) -> None:
        """Add component to the pipeline."""

        if component is not None:
            self.__components.append(component)

    def validate(self, context: Dict[Text, Any], allow_empty_pipeline: bool = False, ) -> None:
        """Validates a pipeline before it is run. Ensures, that all
           arguments are present to train the pipeline."""

        # Ensure the pipeline is not empty
        if not allow_empty_pipeline and len(self.__components) == 0:
            raise ValueError(
                "Cannot train an empty pipeline. "
                "Make sure to specify a proper pipeline in "
                "the configuration using the `pipeline` key."
                + "The `backend` configuration key is "
                  "NOT supported anymore."
            )

        provided_properties = set(context.keys())

        for component in self.__components:
            for r in component.requires:
                if r not in provided_properties:
                    raise Exception(
                        "Failed to validate at component "
                        "'{}'. Missing property: '{}'"
                        "".format(component.name, r)
                    )
            provided_properties.update(component.provides)

    def get_all_components(self):
        return self.__components

    def get_component(self, index):
        return self.__components[index]
import warnings
from typing import Optional, Union, Text, Any, Dict

from rasa.core.domain import Domain
from rasa.utils.endpoints import EndpointConfig
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.common import class_from_module_path


class NaturalLanguageGenerator:
    """Generate bot utterances based on a dialogue state."""

    async def generate(
        self,
        template_name: Text,
        tracker: "DialogueStateTracker",
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested template.

        There are a lot of different methods to implement this, e.g. the
        generation can be based on templates or be fully ML based by feeding
        the dialogue state into a machine learning NLG model."""
        raise NotImplementedError

    @staticmethod
    def create(
        obj: Union["NaturalLanguageGenerator", EndpointConfig, None],
        domain: Optional[Domain],
    ) -> "NaturalLanguageGenerator":
        """Factory to create a generator."""

        custom_nlg = None
        if isinstance(obj, EndpointConfig) and obj.type:
            custom_nlg = NaturalLanguageGenerator.load_nlg_from_module_string(obj)

        if isinstance(obj, NaturalLanguageGenerator):
            return obj
        elif custom_nlg:
            return custom_nlg(domain=domain, endpoint_config=obj)
        elif isinstance(obj, EndpointConfig) and obj.url:
            from rasa.core.nlg import (  # pytype: disable=pyi-error
                CallbackNaturalLanguageGenerator,
            )

            return CallbackNaturalLanguageGenerator(obj)
        elif obj is None:
            from rasa.core.nlg import (  # pytype: disable=pyi-error
                TemplatedNaturalLanguageGenerator,
            )

            templates = domain.templates if domain else []
            return TemplatedNaturalLanguageGenerator(templates)
        else:
            raise Exception(
                "Cannot create a NaturalLanguageGenerator "
                "based on the passed object. Type: `{}`"
                "".format(type(obj))
            )

    @staticmethod
    def load_nlg_from_module_string(nlg: EndpointConfig,) -> "NaturalLanguageGenerator":
        custom_nlg = None
        try:
            custom_nlg = class_from_module_path(nlg.type)
        except (AttributeError, ImportError):
            warnings.warn(
                f"NLG type '{nlg.type}' not found. "
                "Using CallbackNaturalLanguageGenerator "
                "or TemplatedNaturalLanguageGenerator instead"
            )
        return custom_nlg

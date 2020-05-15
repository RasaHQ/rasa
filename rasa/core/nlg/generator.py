import logging
from typing import Optional, Union, Text, Any, Dict

from rasa.core.domain import Domain
from rasa.utils import common
from rasa.utils.endpoints import EndpointConfig
from rasa.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


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

        if isinstance(obj, NaturalLanguageGenerator):
            return obj
        else:
            return _create_from_endpoint_config(obj, domain)


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None, domain: Optional[Domain] = None
) -> "NaturalLanguageGenerator":
    """Given an endpoint configuration, create a proper NLG object."""

    domain = domain or Domain.empty()

    if endpoint_config is None:
        from rasa.core.nlg import (  # pytype: disable=pyi-error
            TemplatedNaturalLanguageGenerator,
        )

        # this is the default type if no endpoint config is set
        nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    elif endpoint_config.type is None or endpoint_config.type.lower() == "callback":
        from rasa.core.nlg import (  # pytype: disable=pyi-error
            CallbackNaturalLanguageGenerator,
        )

        # this is the default type if no nlg type is set
        nlg = CallbackNaturalLanguageGenerator(endpoint_config=endpoint_config)
    elif endpoint_config.type.lower() == "template":
        from rasa.core.nlg import (  # pytype: disable=pyi-error
            TemplatedNaturalLanguageGenerator,
        )

        nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    else:
        nlg = _load_from_module_string(endpoint_config, domain)

    logger.debug(f"Instantiated NLG to '{nlg.__class__.__name__}'.")
    return nlg


def _load_from_module_string(
    endpoint_config: EndpointConfig, domain: Domain
) -> "NaturalLanguageGenerator":
    """Initializes a custom natural language generator.

    Args:
        domain: defines the universe in which the assistant operates
        endpoint_config: the specific natural language generator
    """

    try:
        nlg_class = common.class_from_module_path(endpoint_config.type)
        return nlg_class(endpoint_config=endpoint_config, domain=domain)
    except (AttributeError, ImportError) as e:
        raise Exception(
            f"Could not find a class based on the module path "
            f"'{endpoint_config.type}'. Failed to create a "
            f"`NaturalLanguageGenerator` instance. Error: {e}"
        )

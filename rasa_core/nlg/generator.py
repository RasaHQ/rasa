from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Optional, Union

from rasa_core.utils import EndpointConfig

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain


class NaturalLanguageGenerator(object):
    """Generate bot utterances based on a dialogue state."""

    def generate(self, template_name, tracker, output_channel, **kwargs):
        """Generate a response for the requested template.

        There are a lot of different methods to implement this, e.g. the
        generation can be based on templates or be fully ML based by feeding
        the dialogue state into a machine learning NLG model."""
        raise NotImplementedError

    @staticmethod
    def create(
            obj,  # type: Union[NaturalLanguageGenerator, EndpointConfig, None]
            domain  # type: Optional[Domain]
    ):
        # type: (...) -> NaturalLanguageGenerator
        """Factory to create a generator."""

        if isinstance(obj, NaturalLanguageGenerator):
            return obj
        elif isinstance(obj, EndpointConfig):
            from rasa_core.nlg import CallbackNaturalLanguageGenerator
            return CallbackNaturalLanguageGenerator(obj)
        elif obj is None:
            from rasa_core.nlg import TemplatedNaturalLanguageGenerator
            templates = domain.templates if domain else []
            return TemplatedNaturalLanguageGenerator(templates)
        else:
            raise Exception("Cannot create a NaturalLanguageGenerator "
                            "based on the passed object. Type: `{}`"
                            "".format(type(obj)))

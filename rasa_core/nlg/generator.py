from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.utils import EndpointConfig


class NaturalLanguageGenerator(object):
    def generate(self, template_name, tracker, output_channel, **kwargs):
        pass

    @staticmethod
    def create(obj, domain):
        if isinstance(obj, NaturalLanguageGenerator):
            return obj
        elif isinstance(obj, EndpointConfig):
            from rasa_core.nlg.callback import \
                CallbackNaturalLanguageGenerator
            return CallbackNaturalLanguageGenerator(obj)
        elif obj is None:
            from rasa_core.nlg.template import \
                TemplatedNaturalLanguageGenerator
            return TemplatedNaturalLanguageGenerator(domain.templates)
        else:
            raise Exception("Invalid nlg")

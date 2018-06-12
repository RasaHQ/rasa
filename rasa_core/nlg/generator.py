from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.actions.action import EndpointConfig


class NaturalLanguageGenerator(object):
    def generate(self, template_name, tracker, output_channel, **kwargs):
        pass

    @staticmethod
    def create(obj, domain):
        if isinstance(obj, NaturalLanguageGenerator):
            return obj
        elif isinstance(obj, dict):
            # TODO: TB - check if we should really do it this way
            if obj.get("type") == "template":
                from rasa_core.nlg.template import \
                    TemplatedNaturalLanguageGenerator

                return TemplatedNaturalLanguageGenerator(domain.templates)
            elif obj.get("type") == "http":
                from rasa_core.nlg.callback import \
                    CallbackNaturalLanguageGenerator

                endpoint = EndpointConfig.from_dict(obj)
                return CallbackNaturalLanguageGenerator(endpoint)
        else:
            return None  # TODO: TB - default generator

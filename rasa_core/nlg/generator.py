from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


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
        else:
            return None  # TODO: TB - default generator

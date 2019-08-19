from rasa.nlu.components import Component

# How many intents are at max put into the output intent
# ranking, everything else will be cut off
INTENT_RANKING_LENGTH = 10


class Classifier(Component):
    """ Abstract class for Classifier objects."""
    pass

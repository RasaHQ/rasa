from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.classifiers.diet_classifier import DIETClassifier


@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
     DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
     DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER], is_trainable=True
)
class DIETFeaturizer(DIETClassifier):

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # classify and add the attributes to the messages on the training data
        return training_data

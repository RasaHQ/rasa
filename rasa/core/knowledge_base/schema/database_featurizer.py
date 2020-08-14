from typing import List
import numpy as np

from rasa.nlu.constants import TOKENS_NAMES, TEXT, FEATURE_TYPE_SENTENCE
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.training_data import Message, TrainingData
from rasa.core.knowledge_base.schema.database_schema import DatabaseSchema


class DatabaseSchemaFeaturizer:
    @staticmethod
    def featurize(database_schema: DatabaseSchema) -> np.ndarray:
        messages = DatabaseSchemaFeaturizer._convert_to_messages(database_schema)

        training_data = TrainingData(messages)

        featurizer = CountVectorsFeaturizer()
        featurizer.train(training_data)

        all_features = []
        for message in training_data.training_examples:
            sentence_features = [
                f.features for f in message.features if f.type == FEATURE_TYPE_SENTENCE
            ][0]
            all_features.append(sentence_features.toarray())

        return np.array(all_features)

    @staticmethod
    def _convert_to_messages(database_schema: DatabaseSchema) -> List[Message]:
        texts = [
            f"{column.refer_table.name}_{column.name}_{column.column_type}"
            for column in database_schema.columns
        ]
        tokens = [
            Tokenizer._convert_words_to_tokens(text.split("_"), text) for text in texts
        ]

        return [
            Message(text, data={TOKENS_NAMES[TEXT]: token})
            for text, token in zip(texts, tokens)
        ]

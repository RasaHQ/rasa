import logging
from typing import Any, Dict, List, Text, Tuple, Optional

from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.tokenizer import Token
import rasa.utils.train_utils as train_utils
import numpy as np

from rasa.nlu.constants import (
    TEXT,
    LANGUAGE_MODEL_DOCS,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    TOKEN_IDS,
    TOKENS,
    SENTENCE_FEATURES,
    SEQUENCE_FEATURES,
)

logger = logging.getLogger(__name__)


class HFTransformersNLP(Component):
    """Utility Component for interfacing between Transformers library and Rasa OS.

    The transformers(https://github.com/huggingface/transformers) library
    is used to load pre-trained language models like BERT, GPT-2, etc.
    The component also tokenizes and featurizes dense featurizable attributes of each
    message.
    """

    defaults = {
        # name of the language model to load.
        "model_name": "bert",
        # Pre-Trained weights to be loaded(string)
        "model_weights": None,
        # an optional path to a specific directory to download
        # and cache the pre-trained model weights.
        "cache_dir": None,
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super(HFTransformersNLP, self).__init__(component_config)

        self._load_model()
        self.whitespace_tokenizer = WhitespaceTokenizer()

    def _load_model(self) -> None:
        """Try loading the model"""

        from rasa.nlu.utils.hugging_face.registry import (
            model_class_dict,
            model_weights_defaults,
            model_tokenizer_dict,
        )

        self.model_name = self.component_config["model_name"]

        if self.model_name not in model_class_dict:
            raise KeyError(
                f"'{self.model_name}' not a valid model name. Choose from "
                f"{str(list(model_class_dict.keys()))}or create"
                f"a new class inheriting from this class to support your model."
            )

        self.model_weights = self.component_config["model_weights"]
        self.cache_dir = self.component_config["cache_dir"]

        if not self.model_weights:
            logger.info(
                f"Model weights not specified. Will choose default model weights: "
                f"{model_weights_defaults[self.model_name]}"
            )
            self.model_weights = model_weights_defaults[self.model_name]

        logger.debug(f"Loading Tokenizer and Model for {self.model_name}")

        self.tokenizer = model_tokenizer_dict[self.model_name].from_pretrained(
            self.model_weights, cache_dir=self.cache_dir
        )
        self.model = model_class_dict[self.model_name].from_pretrained(
            self.model_weights, cache_dir=self.cache_dir
        )

        # Use a universal pad token since all transformer architectures do not have a
        # consistent token. Instead of pad_token_id we use unk_token_id because
        # pad_token_id is not set for all architectures. We can't add a new token as
        # well since vocabulary resizing is not yet supported for TF classes.
        # Also, this does not hurt the model predictions since we use an attention mask
        # while feeding input.
        self.pad_token_id = self.tokenizer.unk_token_id

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["transformers"]

    def _lm_tokenize(self, text: Text) -> Tuple[List[int], List[Text]]:
        """
        Pass the text through the tokenizer of the language model.

        Args:
            text: Text to be tokenized.

        Returns:
            List of token ids and token strings.

        """
        split_token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        split_token_strings = self.tokenizer.convert_ids_to_tokens(split_token_ids)

        return split_token_ids, split_token_strings

    def _add_lm_specific_special_tokens(
        self, token_ids: List[List[int]]
    ) -> List[List[int]]:
        """Add language model specific special tokens which were used during their training.

        Args:
            token_ids: List of token ids for each example in the batch.

        Returns:
            Augmented list of token ids for each example in the batch.
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_special_tokens_pre_processors,
        )

        augmented_tokens = [
            model_special_tokens_pre_processors[self.model_name](example_token_ids)
            for example_token_ids in token_ids
        ]
        return augmented_tokens

    def _lm_specific_token_cleanup(
        self, split_token_ids: List[int], token_strings: List[Text]
    ) -> Tuple[List[int], List[Text]]:
        """Clean up special chars added by tokenizers of language models.

        Many language models add a special char in front/back of (some) words. We clean up those chars as they are not
        needed once the features are already computed.

        Args:
            split_token_ids: List of token ids received as output from the language model specific tokenizer.
            token_strings: List of token strings received as output from the language model specific tokenizer.

        Returns:
            Cleaned up token ids and token strings.
        """
        from rasa.nlu.utils.hugging_face.registry import model_tokens_cleaners

        return model_tokens_cleaners[self.model_name](split_token_ids, token_strings)

    def _post_process_sequence_embeddings(
        self, sequence_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sentence level representations and sequence level representations for relevant tokens.

        Args:
            sequence_embeddings: Sequence level dense features received as output from language model.

        Returns:
            Sentence and sequence level representations.
        """

        from rasa.nlu.utils.hugging_face.registry import (
            model_embeddings_post_processors,
        )

        sentence_embeddings = []
        post_processed_sequence_embeddings = []

        for example_embedding in sequence_embeddings:
            (
                example_sentence_embedding,
                example_post_processed_embedding,
            ) = model_embeddings_post_processors[self.model_name](example_embedding)

            sentence_embeddings.append(example_sentence_embedding)
            post_processed_sequence_embeddings.append(example_post_processed_embedding)

        return (
            np.array(sentence_embeddings),
            np.array(post_processed_sequence_embeddings),
        )

    def _tokenize_example(
        self, message: Message, attribute: Text
    ) -> Tuple[List[Token], List[int]]:
        """Tokenize a single message example.

        Many language models add a special char in front of (some) words and split words into
        sub-words. To ensure the entity start and end values matches the token values,
        tokenize the text first using the whitespace tokenizer. If individual tokens
        are split up into multiple tokens, we make sure that the start and end value
        of the first and last respective tokens stay the same.

        Args:
            message: Single message object to be processed.
            attribute: Property of message to be processed, one of ``TEXT`` or ``RESPONSE``.

        Returns:
            List of token strings and token ids for the corresponding attribute of the message.
        """

        tokens_in = self.whitespace_tokenizer.tokenize(message, attribute)

        tokens_out = []

        token_ids_out = []

        for token in tokens_in:
            # use lm specific tokenizer to further tokenize the text
            split_token_ids, split_token_strings = self._lm_tokenize(token.text)

            split_token_ids, split_token_strings = self._lm_specific_token_cleanup(
                split_token_ids, split_token_strings
            )

            token_ids_out += split_token_ids

            tokens_out += train_utils.align_tokens(
                split_token_strings, token.end, token.start
            )

        return tokens_out, token_ids_out

    def _get_token_ids_for_batch(
        self, batch_examples: List[Message], attribute: Text
    ) -> Tuple[List[List[Token]], List[List[int]]]:
        """Compute token ids and token strings for each example in batch.

        A token id is the id of that token in the vocabulary of the language model.
        Args:
            batch_examples: Batch of message objects for which tokens need to be computed.
            attribute: Property of message to be processed, one of ``TEXT`` or ``RESPONSE``.

        Returns:
            List of token strings and token ids for each example in the batch.
        """

        batch_token_ids = []
        batch_tokens = []
        for example in batch_examples:

            example_tokens, example_token_ids = self._tokenize_example(
                example, attribute
            )
            batch_tokens.append(example_tokens)
            batch_token_ids.append(example_token_ids)

        return batch_tokens, batch_token_ids

    @staticmethod
    def _compute_attention_mask(actual_sequence_lengths: List[int]) -> np.ndarray:
        """Compute a mask for padding tokens.

        This mask will be used by the language model so that it does not attend to padding tokens.

        Args:
            actual_sequence_lengths: List of length of each example without any padding

        Returns:
            Computed attention mask, 0 for padding and 1 for non-padding tokens.
        """

        attention_mask = []
        max_seq_length = max(actual_sequence_lengths)
        for actual_sequence_length in actual_sequence_lengths:
            # add 1s for present tokens, fill up the remaining space up to max
            # sequence length with 0s (non-existing tokens)
            padded_sequence = [1] * actual_sequence_length + [0] * (
                max_seq_length - actual_sequence_length
            )
            attention_mask.append(padded_sequence)

        attention_mask = np.array(attention_mask).astype(np.float32)

        return attention_mask

    def _add_padding_to_batch(
        self, batch_token_ids: List[List[int]]
    ) -> Tuple[List[int], List[List[int]]]:
        """Add padding so that all examples in the batch are of the same length.

        Args:
            batch_token_ids: Batch of examples where each example is a non-padded list of token ids.

        Returns:
            Padded batch with all examples of the same length.
        """
        padded_token_ids = []
        # Compute max length across examples
        max_seq_len = 0
        actual_sequence_lengths = []

        for example_token_ids in batch_token_ids:
            actual_sequence_lengths.append(len(example_token_ids))
            max_seq_len = max(max_seq_len, len(example_token_ids))

        # Add padding according to max_seq_len
        # Some models don't contain pad token, we use unknown token as padding token.
        # This doesn't affect the computation since we compute an attention mask
        # anyways.
        for example_token_ids in batch_token_ids:
            padded_token_ids.append(
                example_token_ids
                + [self.pad_token_id] * (max_seq_len - len(example_token_ids))
            )
        return actual_sequence_lengths, padded_token_ids

    @staticmethod
    def _extract_nonpadded_embeddings(
        embeddings: np.ndarray, actual_sequence_lengths: List[int]
    ) -> np.ndarray:
        """Use pre-computed non-padded lengths of each example to extract embeddings for non-padding tokens.

        Args:
            embeddings: sequence level representations for each example of the batch.
            actual_sequence_lengths: non-padded lengths of each example of the batch.

        Returns:
            Sequence level embeddings for only non-padding tokens of the batch.
        """
        nonpadded_sequence_embeddings = []
        for index, embedding in enumerate(embeddings):
            unmasked_embedding = embedding[: actual_sequence_lengths[index]]
            nonpadded_sequence_embeddings.append(unmasked_embedding)

        return np.array(nonpadded_sequence_embeddings)

    def _compute_batch_sequence_features(
        self, batch_attention_mask: np.ndarray, padded_token_ids: List[List[int]]
    ) -> np.ndarray:
        """Feed the padded batch to the language model.

        Args:
            batch_attention_mask: Mask of 0s and 1s which indicate whether the token is a padding token or not.
            padded_token_ids: Batch of token ids for each example. The batch is padded and hence can be fed at once.

        Returns:
            Sequence level representations from the language model.
        """
        model_outputs = self.model(
            np.array(padded_token_ids), attention_mask=np.array(batch_attention_mask)
        )

        # sequence hidden states is always the first output from all models
        sequence_hidden_states = model_outputs[0]

        sequence_hidden_states = sequence_hidden_states.numpy()
        return sequence_hidden_states

    def _get_model_features_for_batch(
        self, batch_token_ids: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dense features of each example in the batch.

        We first add the special tokens corresponding to each language model. Next, we add appropriate padding
        and compute a mask for that padding so that it doesn't affect the feature computation. The padded batch is next
        fed to the language model and token level embeddings are computed. Using the pre-computed mask, embeddings for
        non-padding tokens are extracted and subsequently sentence level embeddings are computed.

        Args:
            batch_token_ids: List of token ids of each example in the batch.

        Returns:
            Sentence and token level dense representations.
        """
        # Let's first add tokenizer specific special tokens to all examples
        batch_token_ids_augmented = self._add_lm_specific_special_tokens(
            batch_token_ids
        )

        # Let's first add padding so that whole batch can be fed to the model
        actual_sequence_lengths, padded_token_ids = self._add_padding_to_batch(
            batch_token_ids_augmented
        )

        # Compute attention mask based on actual_sequence_length
        batch_attention_mask = self._compute_attention_mask(actual_sequence_lengths)

        # Get token level features from the model
        sequence_hidden_states = self._compute_batch_sequence_features(
            batch_attention_mask, padded_token_ids
        )

        # Extract features for only non-padding tokens
        sequence_nonpadded_embeddings = self._extract_nonpadded_embeddings(
            sequence_hidden_states, actual_sequence_lengths
        )

        # Extract sentence level and post-processed features
        (
            sentence_embeddings,
            sequence_final_embeddings,
        ) = self._post_process_sequence_embeddings(sequence_nonpadded_embeddings)

        return sentence_embeddings, sequence_final_embeddings

    def _get_docs_for_batch(
        self, batch_examples: List[Message], attribute: Text
    ) -> List[Dict[Text, Any]]:
        """Compute language model docs for all examples in the batch.

        Args:
            batch_examples: Batch of message objects for which language model docs need to be computed.
            attribute: Property of message to be processed, one of ``TEXT`` or ``RESPONSE``.

        Returns:
            List of language model docs for each message in batch.
        """

        batch_tokens, batch_token_ids = self._get_token_ids_for_batch(
            batch_examples, attribute
        )

        (
            batch_sentence_features,
            batch_sequence_features,
        ) = self._get_model_features_for_batch(batch_token_ids)

        # A doc consists of
        # {'token_ids': ..., 'tokens': ..., 'sequence_features': ..., 'sentence_features': ...}
        batch_docs = []
        for index in range(len(batch_examples)):
            doc = {
                TOKEN_IDS: batch_token_ids[index],
                TOKENS: batch_tokens[index],
                SEQUENCE_FEATURES: batch_sequence_features[index],
                SENTENCE_FEATURES: np.reshape(batch_sentence_features[index], (1, -1)),
            }
            batch_docs.append(doc)

        return batch_docs

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Compute tokens and dense features for each message in training data.

        Args:
            training_data: NLU training data to be tokenized and featurized
            config: NLU pipeline config consisting of all components.

        """

        batch_size = 64

        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

            non_empty_examples = list(
                filter(lambda x: x.get(attribute), training_data.training_examples)
            )

            batch_start_index = 0

            while batch_start_index < len(non_empty_examples):

                batch_end_index = min(
                    batch_start_index + batch_size, len(non_empty_examples)
                )
                # Collect batch examples
                batch_messages = non_empty_examples[batch_start_index:batch_end_index]

                # Construct a doc with relevant features extracted(tokens, dense_features)
                batch_docs = self._get_docs_for_batch(batch_messages, attribute)

                for index, ex in enumerate(batch_messages):

                    ex.set(LANGUAGE_MODEL_DOCS[attribute], batch_docs[index])

                batch_start_index += batch_size

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message by computing its tokens and dense features.

        Args:
            message: Incoming message object
        """

        message.set(
            LANGUAGE_MODEL_DOCS[TEXT],
            self._get_docs_for_batch([message], attribute=TEXT)[0],
        )

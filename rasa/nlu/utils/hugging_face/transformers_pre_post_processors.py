from typing import List, Tuple, Text
import numpy as np


def cleanup_tokens(
    token_ids_string: List[Tuple[int, Text]], delimiter: Text
) -> Tuple[List[int], List[Text]]:
    """Utility method to apply delimiter based cleanup on list of tokens.

    Args:
        token_ids_string: List of tuples with each tuple containing
            (token id, token string).
        delimiter: character/string to be cleaned from token strings.

    Returns:
        Token ids and Token strings unpacked.
    """
    token_ids_string = [
        (id, string.replace(delimiter, "")) for id, string in token_ids_string
    ]

    # remove empty strings
    token_ids_string = [(id, string) for id, string in token_ids_string if string]

    # return as individual token ids and token strings
    token_ids, token_strings = zip(*token_ids_string)
    return token_ids, token_strings


def bert_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add BERT style special tokens(CLS and SEP).

    Args:
        token_ids: List of token ids without any special tokens.

    Returns:
        List of token ids augmented with special tokens.
    """
    BERT_CLS_ID = 101
    BERT_SEP_ID = 102

    processed_tokens = token_ids

    processed_tokens.insert(0, BERT_CLS_ID)
    processed_tokens.append(BERT_SEP_ID)

    return processed_tokens


def gpt_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add GPT style special tokens(None).

    Args:
        token_ids: List of token ids without any special tokens.

    Returns:
        List of token ids augmented with special tokens.
    """

    return token_ids


def xlnet_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add XLNET style special tokens.

    Args:
        token_ids: List of token ids without any special tokens.

    Returns:
        List of token ids augmented with special tokens.
    """
    XLNET_CLS_ID = 3
    XLNET_SEP_ID = 4

    token_ids.append(XLNET_SEP_ID)
    token_ids.append(XLNET_CLS_ID)

    return token_ids


def roberta_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add RoBERTa style special tokens.

    Args:
        token_ids: List of token ids without any special tokens.

    Returns:
        List of token ids augmented with special tokens.
    """
    ROBERTA_BEG_ID = 0
    ROBERTA_END_ID = 2

    token_ids.insert(0, ROBERTA_BEG_ID)
    token_ids.append(ROBERTA_END_ID)

    return token_ids


def xlm_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add XLM style special tokens.

    Args:
        token_ids: List of token ids without any special tokens.

    Returns:
        List of token ids augmented with special tokens.
    """
    XLM_SEP_ID = 1

    token_ids.insert(0, XLM_SEP_ID)
    token_ids.append(XLM_SEP_ID)

    return token_ids


def bert_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-process embeddings from BERT.

    by removing CLS and SEP embeddings and returning CLS token embedding as
    sentence representation.

    Args:
        sequence_embeddings: Sequence of token level embeddings received as output from
            BERT.

    Returns:
        sentence level embedding and post-processed sequence level embedding.
    """
    sentence_embedding = sequence_embeddings[0]
    post_processed_embedding = sequence_embeddings[1:-1]

    return sentence_embedding, post_processed_embedding


def gpt_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-process embeddings from GPT models.

    by taking a mean over sequence embeddings and returning that as sentence
    representation.

    Args:
        sequence_embeddings: Sequence of token level embeddings received as output from
            GPT.

    Returns:
        sentence level embedding and post-processed sequence level embedding.
    """
    sentence_embedding = np.mean(sequence_embeddings, axis=0)
    post_processed_embedding = sequence_embeddings

    return sentence_embedding, post_processed_embedding


def xlnet_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-process embeddings from XLNet models.

    by taking a mean over sequence embeddings and returning that as sentence
    representation. Remove last two time steps corresponding
    to special tokens from the sequence embeddings.

    Args:
        sequence_embeddings: Sequence of token level embeddings received as output from
            XLNet.

    Returns:
        sentence level embedding and post-processed sequence level embedding.
    """
    post_processed_embedding = sequence_embeddings[:-2]
    sentence_embedding = np.mean(post_processed_embedding, axis=0)

    return sentence_embedding, post_processed_embedding


def roberta_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post process embeddings from Roberta models.

    by taking a mean over sequence embeddings and returning that as sentence
    representation. Remove first and last time steps
    corresponding to special tokens from the sequence embeddings.

    Args:
        sequence_embeddings: Sequence of token level embeddings received as output from
            Roberta

    Returns:
        sentence level embedding and post-processed sequence level embedding
    """

    post_processed_embedding = sequence_embeddings[1:-1]
    sentence_embedding = np.mean(post_processed_embedding, axis=0)

    return sentence_embedding, post_processed_embedding


def xlm_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post process embeddings from XLM models

    by taking a mean over sequence embeddings and returning that as sentence
    representation. Remove first and last time steps
    corresponding to special tokens from the sequence embeddings.

    Args:
        sequence_embeddings: Sequence of token level embeddings received as output from
            XLM

    Returns:
        sentence level embedding and post-processed sequence level embedding
    """
    post_processed_embedding = sequence_embeddings[1:-1]
    sentence_embedding = np.mean(post_processed_embedding, axis=0)

    return sentence_embedding, post_processed_embedding


def bert_tokens_cleaner(
    token_ids: List[int], token_strings: List[Text]
) -> Tuple[List[int], List[Text]]:
    """Token cleanup method for BERT.

    Clean up tokens with the extra delimiters(##) BERT adds while breaking a token into
    sub-tokens.

    Args:
        token_ids: List of token ids received as output from BERT Tokenizer.
        token_strings: List of token strings received as output from BERT Tokenizer.

    Returns:
        Cleaned token ids and token strings.
    """
    return cleanup_tokens(list(zip(token_ids, token_strings)), "##")


def openaigpt_tokens_cleaner(
    token_ids: List[int], token_strings: List[Text]
) -> Tuple[List[int], List[Text]]:
    """Token cleanup method for GPT.

    Clean up tokens with the extra delimiters(</w>) OpenAIGPT adds while breaking a
    token into sub-tokens.

    Args:
        token_ids: List of token ids received as output from GPT Tokenizer.
        token_strings: List of token strings received as output from GPT Tokenizer.

    Returns:
        Cleaned token ids and token strings.
    """
    return cleanup_tokens(list(zip(token_ids, token_strings)), "</w>")


def gpt2_tokens_cleaner(
    token_ids: List[int], token_strings: List[Text]
) -> Tuple[List[int], List[Text]]:
    """Token cleanup method for GPT2.

    Clean up tokens with the extra delimiters(Ġ) GPT2 adds while breaking a token into
    sub-tokens.

    Args:
        token_ids: List of token ids received as output from GPT Tokenizer.
        token_strings: List of token strings received as output from GPT Tokenizer.

    Returns:
        Cleaned token ids and token strings.
    """
    return cleanup_tokens(list(zip(token_ids, token_strings)), "Ġ")


def xlnet_tokens_cleaner(
    token_ids: List[int], token_strings: List[Text]
) -> Tuple[List[int], List[Text]]:
    """Token cleanup method for XLNet.

    Clean up tokens with the extra delimiters(▁) XLNet adds while breaking a token into
    sub-tokens.

    Args:
        token_ids: List of token ids received as output from GPT Tokenizer.
        token_strings: List of token strings received as output from GPT Tokenizer.

    Returns:
        Cleaned token ids and token strings.
    """
    return cleanup_tokens(list(zip(token_ids, token_strings)), "▁")

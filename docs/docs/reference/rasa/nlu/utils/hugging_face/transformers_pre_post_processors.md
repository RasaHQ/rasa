---
sidebar_label: rasa.nlu.utils.hugging_face.transformers_pre_post_processors
title: rasa.nlu.utils.hugging_face.transformers_pre_post_processors
---
#### cleanup\_tokens

```python
def cleanup_tokens(token_ids_string: List[Tuple[int, Text]], delimiter: Text) -> Tuple[List[int], List[Text]]
```

Utility method to apply delimiter based cleanup on list of tokens.

**Arguments**:

- `token_ids_string` - List of tuples with each tuple containing
  (token id, token string).
- `delimiter` - character/string to be cleaned from token strings.
  

**Returns**:

  Token ids and Token strings unpacked.

#### bert\_tokens\_pre\_processor

```python
def bert_tokens_pre_processor(token_ids: List[int]) -> List[int]
```

Add BERT style special tokens(CLS and SEP).

**Arguments**:

- `token_ids` - List of token ids without any special tokens.
  

**Returns**:

  List of token ids augmented with special tokens.

#### gpt\_tokens\_pre\_processor

```python
def gpt_tokens_pre_processor(token_ids: List[int]) -> List[int]
```

Add GPT style special tokens(None).

**Arguments**:

- `token_ids` - List of token ids without any special tokens.
  

**Returns**:

  List of token ids augmented with special tokens.

#### xlnet\_tokens\_pre\_processor

```python
def xlnet_tokens_pre_processor(token_ids: List[int]) -> List[int]
```

Add XLNET style special tokens.

**Arguments**:

- `token_ids` - List of token ids without any special tokens.
  

**Returns**:

  List of token ids augmented with special tokens.

#### roberta\_tokens\_pre\_processor

```python
def roberta_tokens_pre_processor(token_ids: List[int]) -> List[int]
```

Add RoBERTa style special tokens.

**Arguments**:

- `token_ids` - List of token ids without any special tokens.
  

**Returns**:

  List of token ids augmented with special tokens.

#### xlm\_tokens\_pre\_processor

```python
def xlm_tokens_pre_processor(token_ids: List[int]) -> List[int]
```

Add XLM style special tokens.

**Arguments**:

- `token_ids` - List of token ids without any special tokens.
  

**Returns**:

  List of token ids augmented with special tokens.

#### bert\_embeddings\_post\_processor

```python
def bert_embeddings_post_processor(sequence_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Post-process embeddings from BERT.

by removing CLS and SEP embeddings and returning CLS token embedding as
sentence representation.

**Arguments**:

- `sequence_embeddings` - Sequence of token level embeddings received as output from
  BERT.
  

**Returns**:

  sentence level embedding and post-processed sequence level embedding.

#### gpt\_embeddings\_post\_processor

```python
def gpt_embeddings_post_processor(sequence_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Post-process embeddings from GPT models.

by taking a mean over sequence embeddings and returning that as sentence
representation.

**Arguments**:

- `sequence_embeddings` - Sequence of token level embeddings received as output from
  GPT.
  

**Returns**:

  sentence level embedding and post-processed sequence level embedding.

#### xlnet\_embeddings\_post\_processor

```python
def xlnet_embeddings_post_processor(sequence_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Post-process embeddings from XLNet models.

by taking a mean over sequence embeddings and returning that as sentence
representation. Remove last two time steps corresponding
to special tokens from the sequence embeddings.

**Arguments**:

- `sequence_embeddings` - Sequence of token level embeddings received as output from
  XLNet.
  

**Returns**:

  sentence level embedding and post-processed sequence level embedding.

#### roberta\_embeddings\_post\_processor

```python
def roberta_embeddings_post_processor(sequence_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Post process embeddings from Roberta models.

by taking a mean over sequence embeddings and returning that as sentence
representation. Remove first and last time steps
corresponding to special tokens from the sequence embeddings.

**Arguments**:

- `sequence_embeddings` - Sequence of token level embeddings received as output from
  Roberta
  

**Returns**:

  sentence level embedding and post-processed sequence level embedding

#### xlm\_embeddings\_post\_processor

```python
def xlm_embeddings_post_processor(sequence_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Post process embeddings from XLM models

by taking a mean over sequence embeddings and returning that as sentence
representation. Remove first and last time steps
corresponding to special tokens from the sequence embeddings.

**Arguments**:

- `sequence_embeddings` - Sequence of token level embeddings received as output from
  XLM
  

**Returns**:

  sentence level embedding and post-processed sequence level embedding

#### bert\_tokens\_cleaner

```python
def bert_tokens_cleaner(token_ids: List[int], token_strings: List[Text]) -> Tuple[List[int], List[Text]]
```

Token cleanup method for BERT.

Clean up tokens with the extra delimiters(##) BERT adds while breaking a token into
sub-tokens.

**Arguments**:

- `token_ids` - List of token ids received as output from BERT Tokenizer.
- `token_strings` - List of token strings received as output from BERT Tokenizer.
  

**Returns**:

  Cleaned token ids and token strings.

#### openaigpt\_tokens\_cleaner

```python
def openaigpt_tokens_cleaner(token_ids: List[int], token_strings: List[Text]) -> Tuple[List[int], List[Text]]
```

Token cleanup method for GPT.

Clean up tokens with the extra delimiters(&lt;/w&gt;) OpenAIGPT adds while breaking a
token into sub-tokens.

**Arguments**:

- `token_ids` - List of token ids received as output from GPT Tokenizer.
- `token_strings` - List of token strings received as output from GPT Tokenizer.
  

**Returns**:

  Cleaned token ids and token strings.

#### gpt2\_tokens\_cleaner

```python
def gpt2_tokens_cleaner(token_ids: List[int], token_strings: List[Text]) -> Tuple[List[int], List[Text]]
```

Token cleanup method for GPT2.

Clean up tokens with the extra delimiters(Ġ) GPT2 adds while breaking a token into
sub-tokens.

**Arguments**:

- `token_ids` - List of token ids received as output from GPT Tokenizer.
- `token_strings` - List of token strings received as output from GPT Tokenizer.
  

**Returns**:

  Cleaned token ids and token strings.

#### xlnet\_tokens\_cleaner

```python
def xlnet_tokens_cleaner(token_ids: List[int], token_strings: List[Text]) -> Tuple[List[int], List[Text]]
```

Token cleanup method for XLNet.

Clean up tokens with the extra delimiters(▁) XLNet adds while breaking a token into
sub-tokens.

**Arguments**:

- `token_ids` - List of token ids received as output from GPT Tokenizer.
- `token_strings` - List of token strings received as output from GPT Tokenizer.
  

**Returns**:

  Cleaned token ids and token strings.


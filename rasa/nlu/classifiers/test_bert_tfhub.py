import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import bert
from bert import tokenization

input_ids = tf.constant([[1, 2], [3, 4]])
input_mask = tf.constant([[1, 1], [1, 1]])
input_segment_ids = tf.constant([[0, 0], [0, 0]])

# bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
#                             trainable=True)

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

module = hub.Module(BERT_MODEL_HUB, trainable=True)


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run(
                [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
            )

    return tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case
    )


print("Type", input_ids.dtype)

tokenizer = create_tokenizer_from_hub_module()

print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))

bert_inputs = dict(
    input_ids=input_ids, input_mask=input_mask, segment_ids=input_segment_ids
)
bert_outputs = module(inputs=bert_inputs, signature="tokens", as_dict=True)

# print(tf.trainable_variables())
print("Layer initialized")

# pooled_output, sequence_output = bert_layer([input_ids, input_mask, input_segment_ids])
# pooled_output, sequence_output = module([input_ids, input_mask, input_segment_ids])

pooled_output = bert_outputs["pooled_output"]
sequence_output = bert_outputs["sequence_output"]

print(pooled_output.shape, sequence_output.shape)

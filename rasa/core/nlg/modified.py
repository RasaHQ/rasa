import copy
import logging
from collections import defaultdict

from rasa.core.trackers import DialogueStateTracker
from typing import Text, Any, Dict, Optional, List

from rasa.core.nlg.generator import NaturalLanguageGenerator
from rasa.core.nlg.interpolator import interpolate_text, interpolate
from rasa.core.nlg.template import TemplatedNaturalLanguageGenerator

import tensorflow_hub as tfhub
import tensorflow as tf
import tensorflow_text

logger = logging.getLogger(__name__)

class ModifiedTemplateGenerator(TemplatedNaturalLanguageGenerator):
    """Natural language generator that generates messages based on templates.

    The templates can use variables to customize the utterances based on the
    state of the dialogue."""

    def __init__(self, templates: Dict[Text, List[Dict[Text, Any]]]) -> None:
        self.templates = templates

    async def generate(
        self,
        template_name: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested template."""
        filled_slots = tracker.current_slot_values()
        return self.generate_from_slots(
            template_name, filled_slots, output_channel, **kwargs
        )


if __name__ == "__main__":
    sess = None

    if sess is not None:
        sess.close()


    sess = tf.InteractiveSession(graph=tf.Graph())

    module = tfhub.Module("http://models.poly-ai.com/convert/v1/model.tar.gz")

    text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
    context_encoding_tensor = module(text_placeholder, signature="encode_context")
    response_encoding_tensor = module(text_placeholder, signature="encode_response")

    encoding_dim = int(context_encoding_tensor.shape[1])
    print(f"ConveRT encodes contexts & responses to {encoding_dim}-dimensional vectors")

    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    def encode_contexts(texts):
        return sess.run(context_encoding_tensor, feed_dict={text_placeholder: texts})

    def encode_responses(texts):
        return sess.run(response_encoding_tensor, feed_dict={text_placeholder: texts})

    import numpy as np
    import random

    modifiers = ["Ok. ", "Actually, No.", "Yes, that's right"]

    responses = []
    fname = "/Users/alan/Developer/dialog/carbon-bot/convert/faq.txt"
    with open(fname) as f:
        for line in f:
            l = line.strip()
            responses.append(l)
            for m in modifiers:
                responses.append(f"{m} {l}")

    print("\n\t- ".join(["Three random facts:"] + random.sample(responses, 3)))

    # Encode the responses in batches of 64.
    batch_size = 64
    response_encodings = []
    for i in range(0, len(responses), batch_size):
        batch = responses[i:i + batch_size]
        response_encodings.append(encode_responses(batch))

    response_encodings = np.concatenate(response_encodings)
    print(f"Encoded {response_encodings.shape[0]} candidate responses.")

    context = "should I thank you?" # @param {type:"string"}
    if context:
        context_encoding = encode_contexts([context])
        scores = np.dot(response_encodings, context_encoding.T)
        top_index = np.argmax(scores)
        top_score = float(scores[top_index])
        print(f"[{top_score:.3f}] {responses[top_index]}")

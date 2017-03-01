from mitie import tokenize, ner_trainer, ner_training_instance

from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer


def find_entity(ent, text):
    tk = MITIETokenizer()
    tokens, offsets = tk.tokenize_with_offsets(text)
    if ent["start"] not in offsets:
        message = u"invalid entity {0} in example '{1}':".format(ent, text) + \
                  u" entities must span whole tokens"
        raise ValueError(message)
    start = offsets.index(ent["start"])
    _slice = text[ent["start"]:ent["end"]]
    val_tokens = tokenize(_slice)
    end = start + len(val_tokens)
    return start, end


def train_entity_extractor(entity_examples, fe_file, max_num_threads):
    trainer = ner_trainer(fe_file)
    trainer.num_threads = max_num_threads
    for example in entity_examples:
        text = example["text"]
        tokens = tokenize(text)
        sample = ner_training_instance(tokens)
        for ent in example["entities"]:
            start, end = find_entity(ent, text)
            sample.add_entity(xrange(start, end), ent["entity"])

        trainer.add(sample)
    return trainer.train()

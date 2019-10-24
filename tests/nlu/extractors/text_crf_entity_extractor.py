from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData, Message


def test_crf_extractor(spacy_nlp, ner_crf_pos_feature_config):
    from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor

    ext = CRFEntityExtractor(component_config=ner_crf_pos_feature_config)
    examples = [
        Message(
            "anywhere in the west",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 16, "end": 20, "value": "west", "entity": "location"}
                ],
                "spacy_doc": spacy_nlp("anywhere in the west"),
            },
        ),
        Message(
            "central indian restaurant",
            {
                "intent": "restaurant_search",
                "entities": [
                    {
                        "start": 0,
                        "end": 7,
                        "value": "central",
                        "entity": "location",
                        "extractor": "random_extractor",
                    },
                    {
                        "start": 8,
                        "end": 14,
                        "value": "indian",
                        "entity": "cuisine",
                        "extractor": "CRFEntityExtractor",
                    },
                ],
                "spacy_doc": spacy_nlp("central indian restaurant"),
            },
        ),
    ]

    # uses BILOU and the default features
    ext.train(TrainingData(training_examples=examples), RasaNLUModelConfig())
    sentence = "anywhere in the west"
    doc = {"spacy_doc": spacy_nlp(sentence)}
    crf_format = ext._from_text_to_crf(Message(sentence, doc))
    assert [word[0] for word in crf_format] == ["anywhere", "in", "the", "west"]
    feats = ext._sentence_to_features(crf_format)
    assert "BOS" in feats[0]
    assert "EOS" in feats[-1]
    assert feats[1]["0:low"] == "in"
    sentence = "anywhere in the west"
    ext.extract_entities(Message(sentence, {"spacy_doc": spacy_nlp(sentence)}))
    filtered = ext.filter_trainable_entities(examples)
    assert filtered[0].get("entities") == [
        {"start": 16, "end": 20, "value": "west", "entity": "location"}
    ], "Entity without extractor remains"
    assert filtered[1].get("entities") == [
        {
            "start": 8,
            "end": 14,
            "value": "indian",
            "entity": "cuisine",
            "extractor": "CRFEntityExtractor",
        }
    ], "Only CRFEntityExtractor entity annotation remains"
    assert examples[1].get("entities")[0] == {
        "start": 0,
        "end": 7,
        "value": "central",
        "entity": "location",
        "extractor": "random_extractor",
    }, "Original examples are not mutated"


def test_crf_json_from_BILOU(spacy_nlp, ner_crf_pos_feature_config):
    from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor

    ext = CRFEntityExtractor(component_config=ner_crf_pos_feature_config)
    sentence = "I need a home cleaning close-by"
    doc = {"spacy_doc": spacy_nlp(sentence)}
    r = ext._from_crf_to_json(
        Message(sentence, doc),
        [
            {"O": 1.0},
            {"O": 1.0},
            {"O": 1.0},
            {"B-what": 1.0},
            {"L-what": 1.0},
            {"B-where": 1.0},
            {"I-where": 1.0},
            {"L-where": 1.0},
        ],
    )
    assert len(r) == 2, "There should be two entities"

    assert r[0]["confidence"]  # confidence should exist
    del r[0]["confidence"]
    assert r[0] == {"start": 9, "end": 22, "value": "home cleaning", "entity": "what"}

    assert r[1]["confidence"]  # confidence should exist
    del r[1]["confidence"]
    assert r[1] == {"start": 23, "end": 31, "value": "close-by", "entity": "where"}


def test_crf_json_from_non_BILOU(spacy_nlp, ner_crf_pos_feature_config):
    from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor

    ner_crf_pos_feature_config.update({"BILOU_flag": False})
    ext = CRFEntityExtractor(component_config=ner_crf_pos_feature_config)
    sentence = "I need a home cleaning close-by"
    doc = {"spacy_doc": spacy_nlp(sentence)}
    rs = ext._from_crf_to_json(
        Message(sentence, doc),
        [
            {"O": 1.0},
            {"O": 1.0},
            {"O": 1.0},
            {"what": 1.0},
            {"what": 1.0},
            {"where": 1.0},
            {"where": 1.0},
            {"where": 1.0},
        ],
    )

    # non BILOU will split multi-word entities - hence 5
    assert len(rs) == 5, "There should be five entities"

    for r in rs:
        assert r["confidence"]  # confidence should exist
        del r["confidence"]

    assert rs[0] == {"start": 9, "end": 13, "value": "home", "entity": "what"}
    assert rs[1] == {"start": 14, "end": 22, "value": "cleaning", "entity": "what"}
    assert rs[2] == {"start": 23, "end": 28, "value": "close", "entity": "where"}
    assert rs[3] == {"start": 28, "end": 29, "value": "-", "entity": "where"}
    assert rs[4] == {"start": 29, "end": 31, "value": "by", "entity": "where"}


def test_crf_create_entity_dict(spacy_nlp):
    from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
    from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

    crf_extractor = CRFEntityExtractor()
    spacy_tokenizer = SpacyTokenizer()
    white_space_tokenizer = WhitespaceTokenizer()

    examples = [
        {
            "message": Message(
                "where is St. Michael's Hospital?",
                {
                    "intent": "search_location",
                    "entities": [
                        {
                            "start": 9,
                            "end": 31,
                            "value": "St. Michael's Hospital",
                            "entity": "hospital",
                            "SpacyTokenizer": {
                                "entity_start_token_idx": 2,
                                "entity_end_token_idx": 5,
                            },
                            "WhitespaceTokenizer": {
                                "entity_start_token_idx": 2,
                                "entity_end_token_idx": 5,
                            },
                        }
                    ],
                },
            )
        },
        {
            "message": Message(
                "where is Children's Hospital?",
                {
                    "intent": "search_location",
                    "entities": [
                        {
                            "start": 9,
                            "end": 28,
                            "value": "Children's Hospital",
                            "entity": "hospital",
                            "SpacyTokenizer": {
                                "entity_start_token_idx": 2,
                                "entity_end_token_idx": 4,
                            },
                            "WhitespaceTokenizer": {
                                "entity_start_token_idx": 2,
                                "entity_end_token_idx": 4,
                            },
                        }
                    ],
                },
            )
        },
    ]
    for ex in examples:
        # spacy tokenizers receives a Doc as input and whitespace tokenizer receives a text
        spacy_tokens = spacy_tokenizer.tokenize(spacy_nlp(ex["message"].text))
        white_space_tokens = white_space_tokenizer.tokenize(ex["message"].text)
        for tokenizer, tokens in [
            ("SpacyTokenizer", spacy_tokens),
            ("WhitespaceTokenizer", white_space_tokens),
        ]:
            for entity in ex["message"].get("entities"):
                parsed_entities = crf_extractor._create_entity_dict(
                    ex["message"],
                    tokens,
                    entity[tokenizer]["entity_start_token_idx"],
                    entity[tokenizer]["entity_end_token_idx"],
                    entity["entity"],
                    0.8,
                )
                assert parsed_entities == {
                    "start": entity["start"],
                    "end": entity["end"],
                    "value": entity["value"],
                    "entity": entity["entity"],
                    "confidence": 0.8,
                }


def test_crf_use_dense_features(ner_crf_pos_feature_config, spacy_nlp):
    from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer

    ner_crf_pos_feature_config["features"][1].append("text_dense_features")
    crf_extractor = CRFEntityExtractor(component_config=ner_crf_pos_feature_config)

    spacy_featurizer = SpacyFeaturizer()
    white_space_tokenizer = WhitespaceTokenizer({"use_cls_token": False})

    text = "Rasa is a company in Berlin"
    message = Message(text)
    message.set("spacy_doc", spacy_nlp(text))

    white_space_tokenizer.process(message)
    spacy_featurizer.process(message)

    text_data = crf_extractor._from_text_to_crf(message)
    features = crf_extractor._sentence_to_features(text_data)

    assert "0:text_dense_features" in features[0]
    for i in range(0, len(message.data.get("text_dense_features")[0])):
        assert (
            features[0]["0:text_dense_features"]["text_dense_features"][str(i)]
            == message.data.get("text_dense_features")[0][i]
        )

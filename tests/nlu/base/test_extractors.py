# coding=utf-8

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData, Message
from tests.nlu import utilities
from httpretty import httpretty


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


def test_duckling_entity_extractor(component_builder):
    httpretty.register_uri(
        httpretty.POST,
        "http://localhost:8000/parse",
        body="""[{"body":"Today","start":0,"value":{"values":[{
             "value":"2018-11-13T00:00:00.000-08:00","grain":"day",
             "type":"value"}],"value":"2018-11-13T00:00:00.000-08:00",
             "grain":"day","type":"value"},"end":5,
             "dim":"time","latent":false},{"body":"the 5th","start":9,
             "value":{"values":[{
             "value":"2018-12-05T00:00:00.000-08:00","grain":"day",
             "type":"value"},
             {"value":"2019-01-05T00:00:00.000-08:00","grain":"day",
             "type":"value"},
             {"value":"2019-02-05T00:00:00.000-08:00","grain":"day",
             "type":"value"}],
             "value":"2018-12-05T00:00:00.000-08:00","grain":"day",
             "type":"value"},"end":16,"dim":"time",
             "latent":false},{"body":"5th of May","start":13,"value":{
             "values":[{
             "value":"2019-05-05T00:00:00.000-07:00","grain":"day",
             "type":"value"},
             {"value":"2020-05-05T00:00:00.000-07:00","grain":"day",
             "type":"value"},
             {"value":"2021-05-05T00:00:00.000-07:00","grain":"day",
             "type":"value"}],
             "value":"2019-05-05T00:00:00.000-07:00","grain":"day",
             "type":"value"},"end":23,"dim":"time",
             "latent":false},{"body":"tomorrow","start":37,"value":{
             "values":[{
             "value":"2018-11-14T00:00:00.000-08:00","grain":"day",
             "type":"value"}],
             "value":"2018-11-14T00:00:00.000-08:00","grain":"day",
             "type":"value"},"end":45,"dim":"time",
             "latent":false}]""",
    )
    httpretty.enable()

    _config = RasaNLUModelConfig({"pipeline": [{"name": "DucklingHTTPExtractor"}]})
    _config.set_component_attr(
        0, dimensions=["time"], timezone="UTC", url="http://localhost:8000"
    )
    duckling = component_builder.create_component(_config.for_component(0), _config)
    message = Message("Today is the 5th of May. Let us meet tomorrow.")
    duckling.process(message)
    entities = message.get("entities")
    assert len(entities) == 4

    # Test duckling with a defined date

    httpretty.register_uri(
        httpretty.POST,
        "http://localhost:8000/parse",
        body="""[{"body":"tomorrow","start":12,"value":{"values":[{
             "value":"2013-10-13T00:00:00.000Z","grain":"day",
             "type":"value"}],"value":"2013-10-13T00:00:00.000Z",
             "grain":"day","type":"value"},"end":20,
             "dim":"time","latent":false}]""",
    )

    # 1381536182 == 2013/10/12 02:03:02
    message = Message("Let us meet tomorrow.", time="1381536182")
    duckling.process(message)
    entities = message.get("entities")
    assert len(entities) == 1
    assert entities[0]["text"] == "tomorrow"
    assert entities[0]["value"] == "2013-10-13T00:00:00.000Z"

    # Test dimension filtering includes only specified dimensions
    _config = RasaNLUModelConfig({"pipeline": [{"name": "DucklingHTTPExtractor"}]})
    _config.set_component_attr(0, dimensions=["number"], url="http://localhost:8000")
    duckling_number = component_builder.create_component(
        _config.for_component(0), _config
    )
    httpretty.register_uri(
        httpretty.POST,
        "http://localhost:8000/parse",
        body="""[{"body":"Yesterday","start":0,"value":{"values":[{
            "value":"2019-02-28T00:00:00.000+01:00","grain":"day",
            "type":"value"}],"value":"2019-02-28T00:00:00.000+01:00",
            "grain":"day","type":"value"},"end":9,"dim":"time"},
            {"body":"5","start":21,"value":{"value":5,"type":"value"},
            "end":22,"dim":"number"}]""",
    )

    message = Message("Yesterday there were 5 people in a room")
    duckling_number.process(message)
    entities = message.get("entities")

    assert len(entities) == 1
    assert entities[0]["text"] == "5"
    assert entities[0]["value"] == 5

    httpretty.disable()


def test_duckling_entity_extractor_and_synonyms(component_builder):
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "DucklingHTTPExtractor"},
                {"name": "EntitySynonymMapper"},
            ]
        }
    )
    _config.set_component_attr(0, dimensions=["number"])
    duckling = component_builder.create_component(_config.for_component(0), _config)
    synonyms = component_builder.create_component(_config.for_component(1), _config)
    message = Message("He was 6 feet away")
    duckling.process(message)
    # checks that the synonym processor
    # can handle entities that have int values
    synonyms.process(message)
    assert message is not None


def test_unintentional_synonyms_capitalized(component_builder):
    _config = utilities.base_test_conf("pretrained_embeddings_spacy")
    ner_syn = component_builder.create_component(_config.for_component(5), _config)
    examples = [
        Message(
            "Any Mexican restaurant will do",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 4, "end": 11, "value": "Mexican", "entity": "cuisine"}
                ],
            },
        ),
        Message(
            "I want Tacos!",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 7, "end": 12, "value": "Mexican", "entity": "cuisine"}
                ],
            },
        ),
    ]
    ner_syn.train(TrainingData(training_examples=examples), _config)
    assert ner_syn.synonyms.get("mexican") is None
    assert ner_syn.synonyms.get("tacos") == "Mexican"


def test_spacy_ner_extractor(component_builder, spacy_nlp):
    _config = RasaNLUModelConfig({"pipeline": [{"name": "SpacyEntityExtractor"}]})
    ext = component_builder.create_component(_config.for_component(0), _config)
    example = Message(
        "anywhere in the U.K.",
        {
            "intent": "restaurant_search",
            "entities": [],
            "spacy_doc": spacy_nlp("anywhere in the west"),
        },
    )

    ext.process(example, spacy_nlp=spacy_nlp)

    assert len(example.get("entities", [])) == 1
    assert example.get("entities")[0] == {
        "start": 16,
        "extractor": "SpacyEntityExtractor",
        "end": 20,
        "value": "U.K.",
        "entity": "GPE",
        "confidence": None,
    }

    # Test dimension filtering includes only specified dimensions

    example = Message(
        "anywhere in the West with Sebastian Thrun",
        {
            "intent": "example_intent",
            "entities": [],
            "spacy_doc": spacy_nlp("anywhere in the West with Sebastian Thrun"),
        },
    )
    _config = RasaNLUModelConfig({"pipeline": [{"name": "SpacyEntityExtractor"}]})

    _config.set_component_attr(0, dimensions=["PERSON"])
    ext = component_builder.create_component(_config.for_component(0), _config)
    ext.process(example, spacy_nlp=spacy_nlp)

    assert len(example.get("entities", [])) == 1
    assert example.get("entities")[0] == {
        "start": 26,
        "extractor": "SpacyEntityExtractor",
        "end": 41,
        "value": "Sebastian Thrun",
        "entity": "PERSON",
        "confidence": None,
    }

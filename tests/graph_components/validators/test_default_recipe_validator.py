import warnings
from pathlib import Path

import rasa.shared.utils.io
from rasa.core.featurizers.precomputation import CoreFeaturizationInputConverter
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource

from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from typing import Dict, List, Optional, Set, Text, Any, Tuple, Type
import re

import pytest
from _pytest.monkeypatch import MonkeyPatch
from unittest.mock import Mock
from rasa.engine.graph import GraphComponent, ExecutionContext, GraphSchema, SchemaNode

from rasa.graph_components.validators.default_recipe_validator import (
    POLICY_CLASSSES,
    DefaultV1RecipeValidator,
    TRAINABLE_EXTRACTORS,
    _types_to_str,
)
from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor
from rasa.nlu.extractors.crf_entity_extractor import (
    CRFEntityExtractor,
    CRFEntityExtractorOptions,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.policy import Policy
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.core.domain import KEY_FORMS, Domain, InvalidDomain
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.data import TrainingType
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT_RESPONSE_KEY,
    TEXT,
    INTENT,
    RESPONSE,
)
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.validation import YamlValidationException
import rasa.utils.common
from tests.conftest import filter_expected_warnings


class DummyImporter(TrainingDataImporter):
    def __init__(
        self,
        training_data: Optional[TrainingData] = None,
        config: Optional[Dict[Text, Any]] = None,
        domain: Optional[Domain] = None,
    ) -> None:
        self.training_data = training_data or TrainingData([])
        self.config = config or {}
        self.domain = domain or Domain.empty()

    def get_domain(self) -> Domain:
        return self.domain

    def get_nlu_data(self) -> TrainingData:
        return self.training_data

    def get_config(self) -> Dict[Text, Any]:
        return self.config

    def get_stories(self) -> StoryGraph:
        return StoryGraph([])

    def get_config_file_for_auto_config(self) -> Optional[Text]:
        return "config.yml"


def _test_validation_warnings_with_default_configs(
    training_data: TrainingData,
    component_types: List[Type],
    warnings: Optional[List[Text]] = None,
):
    dummy_importer = DummyImporter(training_data=training_data)
    graph_schema = GraphSchema(
        {
            f"{idx}": SchemaNode(
                needs={},
                uses=component_type,
                constructor_name="",
                fn="",
                config=component_type.get_default_config(),
            )
            for idx, component_type in enumerate(component_types)
        }
    )
    validator = DefaultV1RecipeValidator(graph_schema)
    if not warnings:
        with pytest.warns(None) as records:
            validator.validate(dummy_importer)
            assert len(records) == 0, [warning.message for warning in records.list]
    else:
        with pytest.warns(None) as records:
            validator.validate(dummy_importer)
        assert len(records) == len(warnings), ", ".join(
            warning.message.args[0] for warning in records
        )
        assert [
            re.match(warning.message.args[0], expected_warning)
            for warning, expected_warning in zip(records, warnings)
        ]


@pytest.mark.parametrize(
    "component_type, warns", [(ResponseSelector, False), (None, True)]
)
def test_nlu_warn_if_training_examples_with_intent_response_key_are_unused(
    component_type: Type[GraphComponent], warns: bool
):
    messages = [
        Message(
            {
                INTENT: "faq",
                INTENT_RESPONSE_KEY: "faq/dummy",
                TEXT: "hi",
                RESPONSE: "utter_greet",
            }
        ),
        Message(
            {
                INTENT: "faq",
                INTENT_RESPONSE_KEY: "faq/dummy",
                TEXT: "hi hi",
                RESPONSE: "utter_greet",
            }
        ),
    ]
    training_data = TrainingData(training_examples=messages)
    warnings = (
        (
            [
                "You have defined training data with examples "
                "for training a response selector, "
                "but your NLU configuration"
            ]
        )
        if warns
        else None
    )
    component_types = [WhitespaceTokenizer]
    if component_type:
        component_types.append(component_type)
    _test_validation_warnings_with_default_configs(
        training_data=training_data, component_types=component_types, warnings=warnings
    )


@pytest.mark.parametrize(
    "component_type, warns",
    [(extractor, False) for extractor in TRAINABLE_EXTRACTORS] + [(None, True)],
)
def test_nlu_warn_if_training_examples_with_entities_are_unused(
    component_type: Type[GraphComponent], warns: bool
):
    messages = [
        Message(
            {ENTITIES: [{ENTITY_ATTRIBUTE_TYPE: "dummy"}], INTENT: "dummy", TEXT: "hi"}
        ),
        Message(
            {
                ENTITIES: [{ENTITY_ATTRIBUTE_TYPE: "dummy"}],
                INTENT: "dummy",
                TEXT: "hi hi",
            }
        ),
    ]
    training_data = TrainingData(training_examples=messages)
    warnings = (
        (
            [
                "You have defined training data consisting of entity examples, "
                "but your NLU configuration"
            ]
        )
        if warns
        else None
    )
    component_types = [WhitespaceTokenizer]
    if component_type:
        component_types.append(component_type)
    _test_validation_warnings_with_default_configs(
        training_data=training_data, component_types=component_types, warnings=warnings
    )


@pytest.mark.parametrize(
    "component_type, role_instead_of_group, warns",
    [
        (
            extractor,
            role_instead_of_group,
            extractor not in {DIETClassifier, CRFEntityExtractor},
        )
        for extractor in TRAINABLE_EXTRACTORS
        for role_instead_of_group in [True, False]
    ],
)
def test_nlu_warn_if_training_examples_with_entity_roles_are_unused(
    component_type: Type[GraphComponent], role_instead_of_group: bool, warns: bool
):
    messages = [
        Message(
            {
                ENTITIES: [
                    {
                        ENTITY_ATTRIBUTE_TYPE: "dummy",
                        (
                            ENTITY_ATTRIBUTE_ROLE
                            if role_instead_of_group
                            else ENTITY_ATTRIBUTE_GROUP
                        ): "dummy-2",
                    }
                ],
                TEXT: f"hi{i}",
                INTENT: "dummy",
            }
        )
        for i in range(2)
    ]
    training_data = TrainingData(training_examples=messages)
    warnings = (
        [
            "You have defined training data with entities that have roles/groups, "
            "but your NLU configuration"
        ]
        if warns
        else []
    )
    component_types = [WhitespaceTokenizer]
    if component_type:
        component_types.append(component_type)
    _test_validation_warnings_with_default_configs(
        training_data=training_data, component_types=component_types, warnings=warnings
    )


@pytest.mark.parametrize(
    "component_type, warns",
    [(RegexFeaturizer, False), (RegexEntityExtractor, False), (None, True)],
)
def test_nlu_warn_if_regex_features_are_not_used(
    component_type: Type[GraphComponent], warns: bool
):
    training_data = TrainingData(
        training_examples=[Message({TEXT: "hi"}), Message({TEXT: "hi hi"})],
        regex_features=[{"name": "dummy", "pattern": "dummy"}],
    )
    component_types = [WhitespaceTokenizer]
    if component_type:
        component_types.append(component_type)
    warnings = (
        ["You have defined training data with regexes, but your NLU"] if warns else None
    )
    _test_validation_warnings_with_default_configs(
        training_data=training_data, component_types=component_types, warnings=warnings
    )


@pytest.mark.parametrize(
    "featurizer, consumer, warns_featurizer, warns_consumer",
    [(None, consumer, True, False) for consumer in [DIETClassifier, CRFEntityExtractor]]
    + [
        (RegexFeaturizer, None, False, True),
        # (None, RegexEntityExtractor, False, False), # does not work
        (None, RegexEntityExtractor, False, True),  # instead we get this
    ]
    + [
        (featurizer, consumer, False, False)
        for consumer in [DIETClassifier, CRFEntityExtractor]
        for featurizer in [RegexFeaturizer, RegexEntityExtractor]
    ],
)
def test_nlu_warn_if_lookup_table_is_not_used(
    featurizer: Type[GraphComponent],
    consumer: Type[GraphComponent],
    warns_featurizer: bool,
    warns_consumer: bool,
):
    training_data = TrainingData(
        training_examples=[Message({TEXT: "hi"}), Message({TEXT: "hi hi"})],
        lookup_tables=[{"elements": "this-is-no-file-and-that-does-not-matter"}],
    )
    assert training_data.lookup_tables is not None
    component_types = [WhitespaceTokenizer, featurizer, consumer]
    component_types = [type for type in component_types if type is not None]

    expected_warnings = []
    if warns_featurizer:
        warning = (
            "You have defined training data consisting of lookup tables, "
            "your NLU configuration does not include a featurizer using the "
            "lookup table."
        )
        expected_warnings.append(warning)
    if warns_consumer:
        warning = (
            "You have defined training data consisting of lookup tables, but "
            "your NLU configuration does not include any components "
            "that uses the features created from the lookup table. "
        )
        expected_warnings.append(warning)
    _test_validation_warnings_with_default_configs(
        training_data=training_data,
        component_types=component_types,
        warnings=expected_warnings,
    )


@pytest.mark.parametrize(
    "nodes, warns",
    [
        (
            [
                SchemaNode({}, WhitespaceTokenizer, "", "", {}),
                SchemaNode({}, RegexFeaturizer, "", "", {}),
                SchemaNode({}, CRFEntityExtractor, "", "", {"features": [["pos"]]}),
            ],
            True,
        ),
        (
            [
                SchemaNode({}, WhitespaceTokenizer, "", "", {}),
                SchemaNode({}, RegexFeaturizer, "", "", {}),
                SchemaNode(
                    {},
                    CRFEntityExtractor,
                    "",
                    "",
                    {"features": [["suffix1", "pattern"], ["pos"]]},
                ),
            ],
            False,
        ),
        (
            [
                SchemaNode({}, WhitespaceTokenizer, "", "", {}),
                SchemaNode({}, RegexFeaturizer, "", "", {}),
                SchemaNode({}, CRFEntityExtractor, "", "", {"features": [["pos"]]}),
                SchemaNode(
                    {},
                    CRFEntityExtractor,
                    "",
                    "",
                    {"features": [["suffix1", "pattern"], ["pos"]]},
                ),
            ],
            False,
        ),
    ],
)
def test_nlu_warn_if_lookup_table_and_crf_extractor_pattern_feature_mismatch(
    nodes: List[SchemaNode], warns: bool
):
    training_data = TrainingData(
        training_examples=[Message({TEXT: "hi"}), Message({TEXT: "hi hi"})],
        lookup_tables=[{"elements": "this-is-no-file-and-that-does-not-matter"}],
    )
    assert training_data.lookup_tables is not None
    importer = DummyImporter(training_data=training_data)

    graph_schema = GraphSchema({f"{idx}": node for idx, node in enumerate(nodes)})
    validator = DefaultV1RecipeValidator(graph_schema)

    if warns:
        match = (
            f"You have defined training data consisting of lookup tables, "
            f"but your NLU configuration's "
            f"'{CRFEntityExtractor.__name__}' does not include the "
            f"'{CRFEntityExtractorOptions.PATTERN}' feature"
        )

        with pytest.warns(UserWarning, match=match):
            validator.validate(importer)
    else:
        with pytest.warns(None) as records:
            validator.validate(importer)
            assert len(records) == 0


@pytest.mark.parametrize(
    "components, warns",
    [
        ([WhitespaceTokenizer, CRFEntityExtractor], True),
        ([WhitespaceTokenizer, CRFEntityExtractor, EntitySynonymMapper], False),
    ],
)
def test_nlu_warn_if_entity_synonyms_unused(
    components: List[GraphComponent], warns: bool
):
    training_data = TrainingData(
        training_examples=[Message({TEXT: "hi"}), Message({TEXT: "hi hi"})],
        entity_synonyms={"cat": "dog"},
    )
    assert training_data.entity_synonyms is not None
    importer = DummyImporter(training_data=training_data)

    graph_schema = GraphSchema(
        {
            f"{idx}": SchemaNode({}, component, "", "", {})
            for idx, component in enumerate(components)
        }
    )
    validator = DefaultV1RecipeValidator(graph_schema)

    if warns:
        match = (
            f"You have defined synonyms in your training data, but "
            f"your NLU configuration does not include an "
            f"'{EntitySynonymMapper.__name__}'. "
        )

        with pytest.warns(UserWarning, match=match):
            validator.validate(importer)
    else:
        with pytest.warns(None) as records:
            validator.validate(importer)
            assert len(records) == 0


@pytest.mark.parametrize(
    "nodes",
    [
        {
            # With end-to-end the tokenizer appears twice due to the Core featurization
            "end_to_end": SchemaNode({}, CoreFeaturizationInputConverter, "", "", {}),
            "a": SchemaNode({}, WhitespaceTokenizer, "", "", {}),
            "b": SchemaNode({}, WhitespaceTokenizer, "", "", {}),
            "c": SchemaNode({}, WhitespaceTokenizer, "", "", {}),
        },
        {
            "a": SchemaNode({}, WhitespaceTokenizer, "", "", {}),
            "b": SchemaNode({}, WhitespaceTokenizer, "", "", {}),
        },
    ],
)
def test_nlu_raise_if_more_than_one_tokenizer(nodes: Dict[Text, SchemaNode]):
    graph_schema = GraphSchema(nodes)
    importer = DummyImporter()
    validator = DefaultV1RecipeValidator(graph_schema)
    with pytest.raises(InvalidConfigException, match=".* more than one tokenizer"):
        validator.validate(importer)


def test_nlu_do_not_raise_if_two_tokenizers_with_end_to_end():
    config = rasa.shared.utils.io.read_yaml_file(
        "rasa/engine/recipes/config_files/default_config.yml"
    )
    graph_config = DefaultV1Recipe().graph_config_for_recipe(
        config, cli_parameters={}, training_type=TrainingType.END_TO_END
    )

    importer = DummyImporter()
    validator = DefaultV1RecipeValidator(graph_config.train_schema)

    # Does not raise
    validator.validate(importer)


def test_nlu_do_not_raise_if_trainable_tokenizer():
    config = rasa.shared.utils.io.read_yaml_file(
        "data/test_config/config_pretrained_embeddings_mitie_zh.yml"
    )
    graph_config = DefaultV1Recipe().graph_config_for_recipe(config, cli_parameters={})

    importer = DummyImporter()
    validator = DefaultV1RecipeValidator(graph_config.train_schema)

    # Does not raise
    validator.validate(importer)


@pytest.mark.parametrize(
    "component_types,should_warn",
    [
        (
            [
                WhitespaceTokenizer,
                LexicalSyntacticFeaturizer,
                CRFEntityExtractor,
                DIETClassifier,
            ],
            True,
        ),
        ([WhitespaceTokenizer, LexicalSyntacticFeaturizer, DIETClassifier], False),
    ],
)
def test_nlu_warn_of_competing_extractors(
    component_types: List[Type[GraphComponent]], should_warn: bool
):
    graph_schema = GraphSchema(
        {
            f"{idx}": SchemaNode({}, component_type, "", "", {})
            for idx, component_type in enumerate(component_types)
        }
    )
    importer = DummyImporter()
    nlu_validator = DefaultV1RecipeValidator(graph_schema)
    if should_warn:
        with pytest.warns(UserWarning, match=".*defined multiple entity extractors"):
            nlu_validator.validate(importer)
    else:
        with pytest.warns(None) as records:
            nlu_validator.validate(importer)
        assert len(records) == 0


@pytest.mark.parametrize(
    "component_types,data_path,should_warn",
    [
        (
            [
                WhitespaceTokenizer,
                LexicalSyntacticFeaturizer,
                RegexEntityExtractor,
                DIETClassifier,
            ],
            "data/test/overlapping_regex_entities.yml",
            True,
        ),
        (
            [WhitespaceTokenizer, LexicalSyntacticFeaturizer, RegexEntityExtractor],
            "data/test/overlapping_regex_entities.yml",
            False,
        ),
        (
            [WhitespaceTokenizer, LexicalSyntacticFeaturizer, DIETClassifier],
            "data/test/overlapping_regex_entities.yml",
            False,
        ),
        (
            [
                WhitespaceTokenizer,
                LexicalSyntacticFeaturizer,
                RegexEntityExtractor,
                DIETClassifier,
            ],
            "data/examples/rasa/demo-rasa.yml",
            False,
        ),
    ],
)
def test_nlu_warn_of_competition_with_regex_extractor(
    monkeypatch: MonkeyPatch,
    component_types: List[Dict[Text, Text]],
    data_path: Text,
    should_warn: bool,
):
    importer = TrainingDataImporter.load_from_dict(training_data_paths=[data_path])
    # there are no domain files for the above examples, so:
    monkeypatch.setattr(Domain, "check_missing_responses", lambda *args, **kwargs: None)

    graph_schema = GraphSchema(
        {
            f"{idx}": SchemaNode({}, component_type, "", "", {})
            for idx, component_type in enumerate(component_types)
        }
    )
    validator = DefaultV1RecipeValidator(graph_schema)
    monkeypatch.setattr(
        validator, "_warn_if_some_training_data_is_unused", lambda *args, **kwargs: None
    )

    if should_warn:
        with pytest.warns(
            UserWarning,
            match=(
                f"You have an overlap between the "
                f"'{RegexEntityExtractor.__name__}' and the statistical"
            ),
        ):
            validator.validate(importer)
    else:
        with warnings.catch_warnings() as records:
            validator.validate(importer)

        if records is not None:
            records = filter_expected_warnings(records)
            assert len(records) == 0


@pytest.mark.parametrize(
    "component_types_and_configs, should_raise",
    [
        (
            [
                (
                    "1",
                    LexicalSyntacticFeaturizer,
                    {FEATURIZER_CLASS_ALIAS: "different-class-same-name"},
                    "process_training_data",
                ),
                (
                    "2",
                    RegexFeaturizer,
                    {FEATURIZER_CLASS_ALIAS: "different-class-same-name"},
                    "process_training_data",
                ),
            ],
            True,
        ),
        (
            [
                (
                    "1",
                    RegexFeaturizer,
                    {FEATURIZER_CLASS_ALIAS: "same-class-other-name"},
                    "process_training_data",
                ),
                (
                    "2",
                    RegexFeaturizer,
                    {FEATURIZER_CLASS_ALIAS: "same-class-different-name"},
                    "process_training_data",
                ),
            ],
            False,
        ),
        (
            [
                (
                    "1",
                    RegexFeaturizer,
                    {FEATURIZER_CLASS_ALIAS: "same-class-same-name"},
                    "process_training_data",
                ),
                (
                    "2",
                    RegexFeaturizer,
                    {FEATURIZER_CLASS_ALIAS: "same-class-same-name"},
                    "train",
                ),
            ],
            False,
        ),
        (
            [
                (
                    "1",
                    RegexFeaturizer,
                    {FEATURIZER_CLASS_ALIAS: "same-class-same-name"},
                    "process_training_data",
                ),
                (
                    "e2e_1",
                    RegexFeaturizer,
                    {FEATURIZER_CLASS_ALIAS: "same-class-same-name"},
                    "process_training_data",
                ),
            ],
            False,
        ),
        (
            [
                ("1", RegexFeaturizer, {}, "process_training_data"),
                ("2", RegexFeaturizer, {}, "process_training_data"),
            ],
            False,
        ),
    ],
)
def test_nlu_raise_if_featurizers_are_not_compatible(
    component_types_and_configs: List[
        Tuple[Type[GraphComponent], Dict[Text, Any], Text]
    ],
    should_raise: bool,
):
    graph_schema = GraphSchema(
        {
            f"{node_name}": SchemaNode({}, component_type, "", fn, config)
            for (node_name, component_type, config, fn) in component_types_and_configs
        }
    )
    importer = DummyImporter()
    validator = DefaultV1RecipeValidator(graph_schema)
    if should_raise:
        with pytest.raises(InvalidConfigException):
            validator.validate(importer)
    else:
        validator.validate(importer)


@pytest.mark.parametrize("policy_type", [TEDPolicy, RulePolicy, MemoizationPolicy])
def test_core_warn_if_data_but_no_policy(
    monkeypatch: MonkeyPatch, policy_type: Optional[Type[Policy]]
):

    importer = TrainingDataImporter.load_from_dict(
        domain_path="data/test_e2ebot/domain.yml",
        training_data_paths=[
            "data/test_e2ebot/data/nlu.yml",
            "data/test_e2ebot/data/stories.yml",
        ],
    )

    nodes = {
        "tokenizer": SchemaNode({}, WhitespaceTokenizer, "", "", {}),
        "nlu-component": SchemaNode({}, DIETClassifier, "", "", {}),
    }
    if policy_type is not None:
        nodes["some-policy"] = SchemaNode({}, policy_type, "", "", {})
    graph_schema = GraphSchema(nodes)

    validator = DefaultV1RecipeValidator(graph_schema)
    monkeypatch.setattr(
        validator,
        "_raise_if_a_rule_policy_is_incompatible_with_domain",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(validator, "_warn_if_no_rule_policy_is_contained", lambda: None)
    monkeypatch.setattr(
        validator,
        "_warn_if_rule_based_data_is_unused_or_missing",
        lambda *args, **kwargs: None,
    )

    if policy_type is None:
        with pytest.warns(
            UserWarning, match="Found data for training policies but no policy"
        ) as records:
            validator.validate(importer)
        assert len(records) == 1
    else:
        with pytest.warns() as records:
            validator.validate(importer)
        records = filter_expected_warnings(records)
        assert len(records) == 0


@pytest.mark.parametrize(
    "policy_types, should_warn",
    [
        ([TEDPolicy], True),
        ([RulePolicy], False),
        ([MemoizationPolicy, RulePolicy], False),
    ],
)
def test_core_warn_if_no_rule_policy(
    monkeypatch: MonkeyPatch, policy_types: List[Type[Policy]], should_warn: bool
):
    graph_schema = GraphSchema(
        {
            f"{idx}": SchemaNode({}, policy_type, "", "", {})
            for idx, policy_type in enumerate(policy_types)
        }
    )
    importer = DummyImporter()
    validator = DefaultV1RecipeValidator(graph_schema=graph_schema)
    monkeypatch.setattr(
        validator,
        "_raise_if_a_rule_policy_is_incompatible_with_domain",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        validator,
        "_warn_if_rule_based_data_is_unused_or_missing",
        lambda *args, **kwargs: None,
    )

    if should_warn:
        with pytest.warns(
            UserWarning,
            match=(f"'{RulePolicy.__name__}' is not " "included in the model's "),
        ) as records:
            validator.validate(importer)
    else:
        with pytest.warns(None) as records:
            validator.validate(importer)
        assert len(records) == 0


class CustomTempRulePolicy(RulePolicy):
    pass


@pytest.mark.parametrize(
    "policy_types, should_raise",
    [
        ([TEDPolicy], True),
        ([RulePolicy], False),
        ([MemoizationPolicy, RulePolicy], False),
        ([CustomTempRulePolicy], False),
    ],
)
def test_core_raise_if_domain_contains_form_names_but_no_rule_policy_given(
    monkeypatch: MonkeyPatch, policy_types: List[Type[Policy]], should_raise: bool
):
    domain_with_form = Domain.from_dict(
        {KEY_FORMS: {"some-form": {"required_slots": []}}}
    )
    importer = DummyImporter(domain=domain_with_form)
    graph_schema = GraphSchema(
        {
            "policy": SchemaNode({}, policy_type, "", "", {})
            for policy_type in policy_types
        }
    )
    validator = DefaultV1RecipeValidator(graph_schema)
    monkeypatch.setattr(validator, "_validate_nlu", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        validator, "_warn_if_no_rule_policy_is_contained", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        validator,
        "_warn_if_rule_based_data_is_unused_or_missing",
        lambda *args, **kwargs: None,
    )
    if should_raise:
        with pytest.raises(
            InvalidDomain,
            match="You have defined a form action, but have not added the",
        ):
            validator.validate(importer)
    else:
        validator.validate(importer)


def test_core_raise_if_a_rule_policy_is_incompatible_with_domain(
    monkeypatch: MonkeyPatch,
):

    domain = Domain.empty()

    num_instances = 2
    nodes = {}
    configs_for_rule_policies = []
    for feature_type in POLICY_CLASSSES:
        for idx in range(num_instances):
            unique_name = f"{feature_type.__name__}-{idx}"
            unique_config = {unique_name: None}
            nodes[unique_name] = SchemaNode({}, feature_type, "", "", unique_config)
        if feature_type == RulePolicy:
            configs_for_rule_policies.append(unique_config)

    mock = Mock()
    monkeypatch.setattr(RulePolicy, "raise_if_incompatible_with_domain", mock)

    validator = DefaultV1RecipeValidator(graph_schema=GraphSchema(nodes))
    monkeypatch.setattr(
        validator,
        "_warn_if_rule_based_data_is_unused_or_missing",
        lambda *args, **kwargs: None,
    )
    importer = DummyImporter()
    validator.validate(importer)

    # Note: this works because we validate nodes in insertion order
    mock.all_args_list == [
        {"config": config, "domain": domain} for config in configs_for_rule_policies
    ]


@pytest.mark.parametrize(
    "policy_types, num_duplicates, priority",
    [
        (POLICY_CLASSSES, 0, 0),
        (POLICY_CLASSSES, 1, 1),
        (list(POLICY_CLASSSES) * 2, 2, 3),
    ],
)
def test_core_warn_if_policy_priorities_are_not_unique(
    monkeypatch: MonkeyPatch,
    policy_types: Set[Type[Policy]],
    num_duplicates: bool,
    priority: int,
):

    assert (
        len(policy_types) >= priority + num_duplicates
    ), f"This tests needs at least {priority+num_duplicates} many types."

    # start with a schema where node i has priority i
    nodes = {
        f"{idx}": SchemaNode("", policy_type, "", "", {"priority": idx})
        for idx, policy_type in enumerate(policy_types)
    }

    # give nodes p+1, .., p+num_duplicates-1 priority "priority"
    for idx in range(num_duplicates):
        nodes[f"{priority+idx+1}"].config["priority"] = priority

    validator = DefaultV1RecipeValidator(graph_schema=GraphSchema(nodes))
    monkeypatch.setattr(
        validator,
        "_warn_if_rule_based_data_is_unused_or_missing",
        lambda *args, **kwargs: None,
    )

    importer = DummyImporter()

    if num_duplicates > 0:
        duplicates = [
            node.uses
            for idx_str, node in nodes.items()
            if priority <= int(idx_str) <= priority + num_duplicates
        ]
        expected_message = (
            f"Found policies {_types_to_str(duplicates)} with same priority {priority} "
        )
        expected_message = re.escape(expected_message)
        with pytest.warns(UserWarning, match=expected_message):
            validator.validate(importer)
    else:
        with pytest.warns(None) as records:
            validator.validate(importer)
        assert len(records) == 0


def test_core_raise_if_policy_has_no_priority():
    class PolicyWithoutPriority(Policy, GraphComponent):
        def __init__(
            self,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
        ) -> None:
            super().__init__(config, model_storage, resource, execution_context)

    nodes = {"policy": SchemaNode("", PolicyWithoutPriority, "", "", {})}
    graph_schema = GraphSchema(nodes)
    importer = DummyImporter()
    validator = DefaultV1RecipeValidator(graph_schema)
    with pytest.raises(
        InvalidConfigException, match="Every policy must have a priority value"
    ):
        validator.validate(importer)


@pytest.mark.parametrize("policy_type_consuming_rule_data", [RulePolicy])
def test_core_warn_if_rule_data_missing(policy_type_consuming_rule_data: Type[Policy]):

    importer = TrainingDataImporter.load_from_dict(
        domain_path="data/test_e2ebot/domain.yml",
        training_data_paths=[
            "data/test_e2ebot/data/nlu.yml",
            "data/test_e2ebot/data/stories.yml",
        ],
    )

    graph_schema = GraphSchema(
        {"policy": SchemaNode({}, policy_type_consuming_rule_data, "", "", {})}
    )
    validator = DefaultV1RecipeValidator(graph_schema)

    with pytest.warns(
        UserWarning,
        match=(
            "Found a rule-based policy in your configuration "
            "but no rule-based training data."
        ),
    ):
        validator.validate(importer)


@pytest.mark.parametrize(
    "policy_type_not_consuming_rule_data", [TEDPolicy, MemoizationPolicy]
)
def test_core_warn_if_rule_data_unused(
    policy_type_not_consuming_rule_data: Type[Policy],
):

    importer = TrainingDataImporter.load_from_dict(
        domain_path="data/test_moodbot/domain.yml",
        training_data_paths=[
            "data/test_moodbot/data/nlu.yml",
            "data/test_moodbot/data/rules.yml",
        ],
    )

    graph_schema = GraphSchema(
        {"policy": SchemaNode({}, policy_type_not_consuming_rule_data, "", "", {})}
    )
    validator = DefaultV1RecipeValidator(graph_schema)

    with pytest.warns(
        UserWarning,
        match=(
            "Found rule-based training data but no policy "
            "supporting rule-based data."
        ),
    ):
        validator.validate(importer)


def test_nlu_training_data_validation():
    importer = DummyImporter(
        training_data=TrainingData([Message({TEXT: "some text", INTENT: ""})])
    )
    nlu_validator = DefaultV1RecipeValidator(GraphSchema({}))

    with pytest.warns(UserWarning, match="Found empty intent"):
        nlu_validator.validate(importer)


def test_no_warnings_with_default_project(tmp_path: Path):
    rasa.utils.common.copy_directory(Path("rasa/cli/initial_project"), tmp_path)

    importer = TrainingDataImporter.load_from_config(
        config_path=str(tmp_path / "config.yml"),
        domain_path=str(tmp_path / "domain.yml"),
        training_data_paths=[str(tmp_path / "data")],
    )

    config, _missing_keys, _configured_keys = DefaultV1Recipe.auto_configure(
        importer.get_config_file_for_auto_config(),
        importer.get_config(),
        TrainingType.END_TO_END,
    )
    graph_config = DefaultV1Recipe().graph_config_for_recipe(
        config, cli_parameters={}, training_type=TrainingType.END_TO_END
    )
    validator = DefaultV1RecipeValidator(graph_config.train_schema)

    with warnings.catch_warnings() as records:
        validator.validate(importer)

    if records is not None:
        records = filter_expected_warnings(records)
        assert len(records) == 0


def test_importer_with_invalid_model_config(tmp_path: Path):
    invalid = {"version": "2.0", "policies": ["name"]}
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(invalid, config_file)

    with pytest.raises(YamlValidationException):
        importer = TrainingDataImporter.load_from_config(str(config_file))
        DefaultV1Recipe.auto_configure(
            importer.get_config_file_for_auto_config(),
            importer.get_config(),
            TrainingType.END_TO_END,
        )

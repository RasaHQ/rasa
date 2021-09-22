from typing import Dict, List, Optional, Set, Text, Any, Tuple, Type
import re

import pytest
from _pytest.monkeypatch import MonkeyPatch
from unittest.mock import Mock

from rasa.engine.graph import GraphComponent, GraphSchema, SchemaNode
from rasa.graph_components.validators.config_validator import (
    FEATURIZER_CLASSES,
    POLICY_CLASSSES,
    ConfigValidator,
    TRAINABLE_EXTRACTORS,
)
from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS
from rasa.nlu.classifiers.diet_classifier import DIETClassifierGraphComponent
from rasa.nlu.extractors.regex_entity_extractor import (
    RegexEntityExtractorGraphComponent,
)
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractorGraphComponent
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizerGraphComponent,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import (
    RegexFeaturizerGraphComponent,
)
from rasa.nlu.selectors.response_selector import ResponseSelectorGraphComponent
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizerGraphComponent
from rasa.core.policies.memoization import MemoizationPolicyGraphComponent
from rasa.core.policies.rule_policy import RulePolicyGraphComponent
from rasa.core.policies.ted_policy import TEDPolicyGraphComponent
from rasa.core.policies.policy import PolicyGraphComponent
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.core.domain import KEY_FORMS, Domain, InvalidDomain
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT_RESPONSE_KEY,
)
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.importers.importer import TrainingDataImporter


class DummyImporter(TrainingDataImporter):
    def __init__(
        self,
        training_data: Optional[TrainingData] = None,
        config: Optional[Dict[Text, Any]] = None,
        domain: Optional[Domain] = None,
    ):
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


def _test_warn_if_some_training_data_is_not_used(
    training_data: TrainingData, component_type: Type, match: Text
):
    dummy_importer = DummyImporter(training_data=training_data)
    graph_schema_without_critical_component = GraphSchema(
        {"tokenizer": SchemaNode({}, WhitespaceTokenizerGraphComponent, "", "", {})}
    )
    validator = ConfigValidator(graph_schema_without_critical_component)
    with pytest.warns(UserWarning, match=match):
        validator.validate(dummy_importer)

    graph_schema_with_critical_component = GraphSchema(
        {
            **graph_schema_without_critical_component.nodes,
            "critical_component": SchemaNode({}, component_type, "", "", {}),
        }
    )
    validator = ConfigValidator(graph_schema_with_critical_component)
    with pytest.warns(None) as records:
        validator.validate(dummy_importer)
        assert len(records) == 0


@pytest.mark.parametrize(
    "message_data, component_type, warning",
    [
        (message_data, component_type, warning)
        for message_data, component_types, warning in [
            (
                {INTENT_RESPONSE_KEY: "dummy"},
                [ResponseSelectorGraphComponent],
                "with examples for training a response selector",
            ),
            (
                {ENTITIES: [{ENTITY_ATTRIBUTE_TYPE: "dummy"}]},
                sorted(
                    TRAINABLE_EXTRACTORS,
                    key=lambda component_type: component_type.__name__,
                ),
                "consisting of entity examples",
            ),
            (
                {
                    ENTITIES: [
                        {
                            ENTITY_ATTRIBUTE_TYPE: "dummy",
                            ENTITY_ATTRIBUTE_ROLE: "dummy-role",
                        }
                    ]
                },
                [DIETClassifierGraphComponent, CRFEntityExtractorGraphComponent],
                "with entities that have roles/groups",
            ),
        ]
        for component_type in component_types
    ],
)
def test_nlu_warn_if_training_examples_are_unused(
    component_type: Type[GraphComponent], message_data: Dict[Text, Any], warning: Text,
):
    training_data = TrainingData(training_examples=[Message(message_data)])
    _test_warn_if_some_training_data_is_not_used(
        training_data=training_data,
        component_type=component_type,
        match=f"You have defined training data {warning}, but your NLU pipeline",
    )


@pytest.mark.parametrize(
    "component_type",
    [RegexFeaturizerGraphComponent, RegexEntityExtractorGraphComponent],
)
def test_nlu_warn_if_regex_features_are_not_used(component_type: Type[GraphComponent]):
    training_data = TrainingData(
        training_examples=[Message({})],
        regex_features=[{"name": "dummy", "pattern": "dummy"}],
    )
    _test_warn_if_some_training_data_is_not_used(
        training_data=training_data,
        component_type=component_type,
        match="You have defined training data with regexes, but your NLU",
    )


@pytest.mark.parametrize(
    "component_type",
    [RegexFeaturizerGraphComponent, RegexEntityExtractorGraphComponent],
)
def test_nlu_warn_if_lookup_table_is_not_used(component_type: Type[GraphComponent]):
    training_data = TrainingData(
        training_examples=[Message({})],
        lookup_tables=[{"elements": "this-is-no-file-and-that-does-not-matter"}],
    )
    assert training_data.lookup_tables is not None
    _test_warn_if_some_training_data_is_not_used(
        training_data=training_data,
        component_type=component_type,
        match=(
            "You have defined training data consisting of lookup tables, "
            "but your NLU"
        ),
    )


def test_nlu_warn_if_lookup_table_and_crf_extractor_pattern_feature_mismatch():
    training_data = TrainingData(
        training_examples=[Message({})],
        lookup_tables=[{"elements": "this-is-no-file-and-that-does-not-matter"}],
    )
    assert training_data.lookup_tables is not None
    importer = DummyImporter(training_data=training_data)

    match = (
        f"You have defined training data consisting of lookup tables, "
        f"but your NLU pipeline's "
        f"'{CRFEntityExtractorGraphComponent.__name__}' does not include the "
        f"'pattern' feature"
    )

    # without 'pattern'
    crf_schema_node = SchemaNode(
        {}, CRFEntityExtractorGraphComponent, "", "", {"features": [["pos"]]},
    )
    graph_schema = GraphSchema(
        {
            "tokenizer": SchemaNode({}, WhitespaceTokenizerGraphComponent, "", "", {}),
            "featurizer": SchemaNode({}, RegexFeaturizerGraphComponent, "", "", {}),
            "crf": crf_schema_node,
        }
    )
    validator = ConfigValidator(graph_schema)
    with pytest.warns(UserWarning, match=match):
        validator.validate(importer)

    # with 'pattern'
    crf_schema_node.config = {"features": [["suffix1", "pattern"], ["pos"]]}
    validator = ConfigValidator(graph_schema)
    with pytest.warns(None) as records:
        validator.validate(importer)
        assert len(records) == 0


def test_nlu_raise_if_more_than_one_tokenizer():
    graph_schema = GraphSchema(
        {
            "a": SchemaNode({}, WhitespaceTokenizerGraphComponent, "", "", {}),
            "b": SchemaNode({}, WhitespaceTokenizerGraphComponent, "", "", {}),
        }
    )
    importer = DummyImporter()
    validator = ConfigValidator(graph_schema)
    with pytest.raises(InvalidConfigException, match=".* more than one tokenizer"):
        validator.validate(importer)


@pytest.mark.parametrize(
    "component_types,should_warn",
    [
        (
            [
                WhitespaceTokenizerGraphComponent,
                LexicalSyntacticFeaturizerGraphComponent,
                CRFEntityExtractorGraphComponent,
                DIETClassifierGraphComponent,
            ],
            True,
        ),
        (
            [
                WhitespaceTokenizerGraphComponent,
                LexicalSyntacticFeaturizerGraphComponent,
                DIETClassifierGraphComponent,
            ],
            False,
        ),
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
    nlu_validator = ConfigValidator(graph_schema)
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
                WhitespaceTokenizerGraphComponent,
                LexicalSyntacticFeaturizerGraphComponent,
                RegexEntityExtractorGraphComponent,
                DIETClassifierGraphComponent,
            ],
            "data/test/overlapping_regex_entities.yml",
            True,
        ),
        (
            [
                WhitespaceTokenizerGraphComponent,
                LexicalSyntacticFeaturizerGraphComponent,
                RegexEntityExtractorGraphComponent,
            ],
            "data/test/overlapping_regex_entities.yml",
            False,
        ),
        (
            [
                WhitespaceTokenizerGraphComponent,
                LexicalSyntacticFeaturizerGraphComponent,
                DIETClassifierGraphComponent,
            ],
            "data/test/overlapping_regex_entities.yml",
            False,
        ),
        (
            [
                WhitespaceTokenizerGraphComponent,
                LexicalSyntacticFeaturizerGraphComponent,
                RegexEntityExtractorGraphComponent,
                DIETClassifierGraphComponent,
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
    importer = TrainingDataImporter.load_from_dict(training_data_paths=[data_path],)
    # there are no domain files for the above examples, so:
    monkeypatch.setattr(Domain, "check_missing_responses", lambda *args, **kwargs: None)

    graph_schema = GraphSchema(
        {
            f"{idx}": SchemaNode({}, component_type, "", "", {})
            for idx, component_type in enumerate(component_types)
        }
    )
    validator = ConfigValidator(graph_schema)
    monkeypatch.setattr(
        validator, "_warn_if_some_training_data_is_unused", lambda *args, **kwargs: None
    )

    if should_warn:
        with pytest.warns(
            UserWarning,
            match=(
                f"You have an overlap between the "
                f"{RegexEntityExtractorGraphComponent.__name__} and the statistical"
            ),
        ):
            validator.validate(importer)
    else:
        with pytest.warns(None) as records:
            validator.validate(importer)
        assert len(records) == 0


@pytest.mark.parametrize(
    "component_types_and_configs, should_raise",
    [
        (
            [
                (
                    LexicalSyntacticFeaturizerGraphComponent,
                    {FEATURIZER_CLASS_ALIAS: "different-class-same-name"},
                ),
                (
                    RegexFeaturizerGraphComponent,
                    {FEATURIZER_CLASS_ALIAS: "different-class-same-name"},
                ),
            ],
            True,
        ),
        (
            [
                (
                    RegexFeaturizerGraphComponent,
                    {FEATURIZER_CLASS_ALIAS: "same-class-other-name"},
                ),
                (
                    RegexFeaturizerGraphComponent,
                    {FEATURIZER_CLASS_ALIAS: "same-class-different-name"},
                ),
            ],
            False,
        ),
        (
            [(RegexFeaturizerGraphComponent, {}), (RegexFeaturizerGraphComponent, {})],
            False,
        ),
    ],
)
def test_nlu_raise_if_featurizers_are_not_compatible(
    component_types_and_configs: List[Tuple[Type[GraphComponent], Dict[Text, Any]]],
    should_raise: bool,
):
    graph_schema = GraphSchema(
        {
            f"{idx}": SchemaNode({}, component_type, "", "", config)
            for idx, (component_type, config) in enumerate(component_types_and_configs)
        }
    )
    importer = DummyImporter()
    validator = ConfigValidator(graph_schema)
    if should_raise:
        with pytest.raises(InvalidConfigException):
            validator.validate(importer)
    else:
        validator.validate(importer)


@pytest.mark.parametrize(
    "policy_type",
    [
        TEDPolicyGraphComponent,
        RulePolicyGraphComponent,
        MemoizationPolicyGraphComponent,
    ],
)
def test_core_warn_if_data_but_no_policy(
    monkeypatch: MonkeyPatch, policy_type: Optional[Type[PolicyGraphComponent]]
):

    importer = TrainingDataImporter.load_from_dict(
        domain_path="data/test_e2ebot/domain.yml",
        training_data_paths=[
            "data/test_e2ebot/data/nlu.yml",
            "data/test_e2ebot/data/stories.yml",
        ],
    )

    nodes = {"tokenizer": SchemaNode({}, WhitespaceTokenizerGraphComponent, "", "", {})}
    if policy_type is not None:
        nodes["some-policy"] = SchemaNode({}, policy_type, "", "", {})
    graph_schema = GraphSchema(nodes)

    validator = ConfigValidator(graph_schema)
    monkeypatch.setattr(validator, "_validate_nlu", lambda _: None)
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
        with pytest.warns(None) as records:
            validator.validate(importer)
        assert len(records) == 0


@pytest.mark.parametrize(
    "policy_types, should_warn",
    [
        ([TEDPolicyGraphComponent], True),
        ([RulePolicyGraphComponent], False),
        ([MemoizationPolicyGraphComponent, RulePolicyGraphComponent], False),
    ],
)
def test_core_warn_if_no_rule_policy(
    monkeypatch: MonkeyPatch,
    policy_types: List[Type[PolicyGraphComponent]],
    should_warn: bool,
):
    graph_schema = GraphSchema(
        {
            f"{idx}": SchemaNode({}, policy_type, "", "", {})
            for idx, policy_type in enumerate(policy_types)
        }
    )
    importer = DummyImporter()
    validator = ConfigValidator(graph_schema=graph_schema)
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
            match=(
                f"'{RulePolicyGraphComponent.__name__}' is not "
                "included in the model's "
            ),
        ) as records:
            validator.validate(importer)
    else:
        with pytest.warns(None) as records:
            validator.validate(importer)
        assert len(records) == 0


@pytest.mark.parametrize(
    "policy_types, should_raise",
    [
        ([TEDPolicyGraphComponent], True),
        ([RulePolicyGraphComponent], False),
        ([MemoizationPolicyGraphComponent, RulePolicyGraphComponent], False),
    ],
)
def test_core_raise_if_domain_contains_form_names_but_no_rule_policy_given(
    monkeypatch: MonkeyPatch,
    policy_types: List[Type[PolicyGraphComponent]],
    should_raise: bool,
):
    domain_with_form = Domain.from_dict({KEY_FORMS: {"some-form": {}}})
    importer = DummyImporter(domain=domain_with_form)
    graph_schema = GraphSchema(
        {
            "policy": SchemaNode({}, policy_type, "", "", {})
            for policy_type in policy_types
        }
    )
    validator = ConfigValidator(graph_schema)
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
        if feature_type == RulePolicyGraphComponent:
            configs_for_rule_policies.append(unique_config)

    mock = Mock()
    monkeypatch.setattr(
        RulePolicyGraphComponent, "raise_if_incompatible_with_domain", mock
    )

    validator = ConfigValidator(graph_schema=GraphSchema(nodes))
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
    policy_types: Set[Type[PolicyGraphComponent]],
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

    validator = ConfigValidator(graph_schema=GraphSchema(nodes))
    monkeypatch.setattr(
        validator,
        "_warn_if_rule_based_data_is_unused_or_missing",
        lambda *args, **kwargs: None,
    )

    importer = DummyImporter()

    if num_duplicates > 0:
        duplicates = [
            node.uses.__name__
            for idx_str, node in nodes.items()
            if priority <= int(idx_str) <= priority + num_duplicates
        ]
        expected_message = f"Found policies {duplicates} with same priority {priority} "
        expected_message = re.escape(expected_message)
        with pytest.warns(UserWarning, match=expected_message):
            validator.validate(importer)
    else:
        with pytest.warns(None) as records:
            validator.validate(importer)
        assert len(records) == 0


@pytest.mark.parametrize("policy_type_consuming_rule_data", [RulePolicyGraphComponent])
def test_core_warn_if_rule_data_missing(
    policy_type_consuming_rule_data: Type[PolicyGraphComponent],
):

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
    validator = ConfigValidator(graph_schema)

    with pytest.warns(
        UserWarning,
        match=(
            "Found a rule-based policy in your pipeline "
            "but no rule-based training data."
        ),
    ):
        validator.validate(importer)


@pytest.mark.parametrize(
    "policy_type_not_consuming_rule_data",
    [TEDPolicyGraphComponent, MemoizationPolicyGraphComponent,],
)
def test_core_warn_if_rule_data_unused(
    policy_type_not_consuming_rule_data: Type[PolicyGraphComponent],
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
    validator = ConfigValidator(graph_schema)

    with pytest.warns(
        UserWarning,
        match=(
            "Found rule-based training data but no policy "
            "supporting rule-based data."
        ),
    ):
        validator.validate(importer)

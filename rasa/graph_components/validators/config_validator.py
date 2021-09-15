from __future__ import annotations
from collections import defaultdict
import itertools
from rasa.nlu.featurizers.featurizer import Featurizer2
from typing import List, Dict, Set, Text, Any

from rasa.engine.graph import ExecutionContext, GraphComponent, GraphSchema, SchemaNode
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.core.policies.rule_policy import RulePolicy, RulePolicyGraphComponent
from rasa.nlu.tokenizers.tokenizer import TokenizerGraphComponent
from rasa.core.policies.policy import PolicyGraphComponent, SupportedData
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DOCS_URL_COMPONENTS,
    DOCS_URL_DEFAULT_ACTIONS,
    DOCS_URL_POLICIES,
    DOCS_URL_RULES,
)
from rasa.shared.core.domain import Domain, InvalidDomain
from rasa.shared.core.constants import (
    ACTION_BACK_NAME,
    ACTION_RESTART_NAME,
    USER_INTENT_BACK,
    USER_INTENT_RESTART,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.constants import TRAINABLE_EXTRACTORS
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.shared.utils.io


def names_of_component_types_used_in_schema(graph_schema: GraphSchema) -> Set[Text]:
    """Collects the names of the types used in the schema nodes.

    Args:
        graph_schema: a graph schema
    Returns:
        names of the types used in the schema nodes in the given `graph_schema`
    """
    node_names = set(node.uses.__name__ for node in graph_schema.nodes.values())
    # FIXME: remove this workaround when all 3.0 components have been renamed
    node_names = set(
        name.replace("GraphComponent", "").replace("Component", "")
        for name in node_names
    )
    return node_names


class NLUValidator(GraphComponent):
    """Checks whether the configuration and the NLU training data match."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> NLUValidator:
        """Creates a new `NLUValidator` (see parent class for full docstring)."""
        return cls(execution_context)

    def __init__(self, execution_context: ExecutionContext) -> None:
        """Instantiates a new `NLUValidator`.

        Args:
           execution_context: Information about the current graph run.
        """
        self._graph_schema = execution_context._graph_schema  # TODO: prune?
        self._component_type_names = names_of_component_types_used_in_schema(
            self._graph_schema
        )

    def validate(self, training_data: TrainingData) -> None:
        """Validates whether the configuration matches the training data.

        Args:
           training_data: the NLU training data
        """
        # TODO: where do we check whether this is consistent with the domain?
        self._warn_if_some_training_data_is_unused(training_data=training_data)
        tokenizer_schema_node = self._raise_if_more_than_one_tokenizer()
        self._warn_of_competing_extractors()
        self._warn_of_competition_with_regex_extractor(training_data=training_data)
        self._raise_or_warn_if_featurizers_are_not_compatible(
            tokenizer_schema_node=tokenizer_schema_node
        )

    # all NLU components

    def _warn_if_some_training_data_is_unused(
        self, training_data: TrainingData
    ) -> None:
        """Validates that all training data will be consumed by some component.

        For example, if you specify response examples in your training data, but there
        is no `ResponseSelector` component in your configuration, then this method
        issues a warning.

        NOTE: rasa/nlu/components/validate_required_components_from_data
        TODO: this also works the other way round, we can warn/raise if a
            component won't be trained instead of creating a placeholder model

        Args:
            training_data: training data
        """

        if (
            training_data.response_examples
            and "ResponseSelector" not in self._names_of_used_components
        ):
            rasa.shared.utils.io.raise_warning(
                "You have defined training data with examples for training a response "
                "selector, but your NLU pipeline does not include a response selector "
                "component. To train a model on your response selector data, add a "
                "'ResponseSelector' to your pipeline."
            )

        if training_data.entity_examples and self._names_of_used_components.disjoint(
            TRAINABLE_EXTRACTORS
        ):
            rasa.shared.utils.io.raise_warning(
                "You have defined training data consisting of entity examples, but "
                "your NLU pipeline does not include an entity extractor trained on "
                "your training data. To extract non-pretrained entities, add one of "
                f"{TRAINABLE_EXTRACTORS} to your pipeline."
            )

        if training_data.entity_examples and self._names_of_used_components.disjoint(
            {"DIETClassifier", "CRFEntityExtractor"}
        ):
            if training_data.entity_roles_groups_used():
                rasa.shared.utils.io.raise_warning(
                    "You have defined training data with entities that "
                    "have roles/groups, but your NLU pipeline does not "
                    "include a 'DIETClassifier' or a 'CRFEntityExtractor'. "
                    "To train entities that have roles/groups, "
                    "add either 'DIETClassifier' or 'CRFEntityExtractor' to your "
                    "pipeline."
                )

        if training_data.regex_features and self._names_of_used_components.disjoint(
            ["RegexFeaturizer", "RegexEntityExtractor"],
        ):
            rasa.shared.utils.io.raise_warning(
                "You have defined training data with regexes, but "
                "your NLU pipeline does not include a 'RegexFeaturizer' or a "
                "'RegexEntityExtractor'. To use regexes, include either a "
                "'RegexFeaturizer' or a 'RegexEntityExtractor' in your pipeline."
            )

        if training_data.lookup_tables and self._names_of_used_components.disjoint(
            ["RegexFeaturizer", "RegexEntityExtractor"],
        ):
            rasa.shared.utils.io.raise_warning(
                "You have defined training data consisting of lookup tables, but "
                "your NLU pipeline does not include a 'RegexFeaturizer' or a "
                "'RegexEntityExtractor'. To use lookup tables, include either a "
                "'RegexFeaturizer' or a 'RegexEntityExtractor' in your pipeline."
            )

        if training_data.lookup_tables:
            if self._names_of_used_components.disjoint(
                ["CRFEntityExtractor", "DIETClassifier"],
            ):
                rasa.shared.utils.io.raise_warning(
                    "You have defined training data consisting of lookup tables, but "
                    "your NLU pipeline does not include any components that use these "
                    "features. To make use of lookup tables, "
                    "add a 'DIETClassifier' or a "
                    "'CRFEntityExtractor' with the 'pattern' feature to your pipeline."
                )
            elif "CRFEntityExtractor" not in self._names_of_used_components:
                crf_schema_nodes = [
                    c
                    for c in self._graph_schema
                    if c.uses.__name__ == "CRFEntityExtractor"
                ]
                # check to see if any of the possible CRFEntityExtractors will
                # featurize `pattern`
                has_pattern_feature = False
                for crf in crf_schema_nodes:
                    crf_features = crf.config.get("features")
                    # iterate through [[before],[word],[after]] features
                    has_pattern_feature = "pattern" in itertools.chain(*crf_features)

                if not has_pattern_feature:
                    rasa.shared.utils.io.raise_warning(
                        "You have defined training data consisting of "
                        "lookup tables, but your NLU pipeline's "
                        "'CRFEntityExtractor' does not include the "
                        "'pattern' feature. To featurize lookup tables, "
                        "add the 'pattern' feature to the 'CRFEntityExtractor' "
                        "in your pipeline."
                    )

        if (
            training_data.entity_synonyms
            and "EntitySynonymMapper" not in self._names_of_used_components
        ):
            rasa.shared.utils.io.raise_warning(
                "You have defined synonyms in your training data, but "
                "your NLU pipeline does not include an 'EntitySynonymMapper'. "
                "To map synonyms, add an 'EntitySynonymMapper' to your pipeline."
            )

    def _raise_if_more_than_one_tokenizer(self) -> SchemaNode:
        """Validates that only one tokenizer is present in the pipeline.

        NOTE: rasa/nlu/components/validate_only_one_tokenizer_is_used

        Returns:
            the schema node describing the tokenizer

        Raises:
            `InvalidConfigException` in case there is more than one tokenizer
        """
        tokenizer_schema_nodes = [
            schema_node
            for schema_node in self._graph_schema.nodes.items()
            if isinstance(schema_node.uses, TokenizerGraphComponent)
        ]

        # TODO: why is 0 tokenizers ok? shouldn't this be ==1 ? Would also match the
        # old name better...
        if len(tokenizer_schema_nodes) > 1:
            names = [schema_node.node_name for schema_node in tokenizer_schema_nodes]
            raise InvalidConfigException(
                f"The pipeline configuration contains more than one tokenizer, "
                f"which is not possible at this time. You can only use one tokenizer. "
                f"The pipeline contains the following tokenizers: "
                f"{names}. "  # TODO: do we want to print node_name or id?
            )

        return tokenizer_schema_nodes[0]

    def _warn_of_competing_extractors(self) -> None:
        """Warns the user when using competing extractors.

        Competing extractors are e.g. `CRFEntityExtractor` and `DIETClassifier`.
        Both of these look for the same entities based on the same training data
        leading to ambiguity in the results.

        Args:
           node_names: names of all components
        """
        extractors_in_pipeline = self._component_type_names.intersection(
            TRAINABLE_EXTRACTORS.intersection
        )
        if len(extractors_in_pipeline) > 1:
            rasa.shared.utils.io.raise_warning(
                f"You have defined multiple entity extractors that do the same job "
                f"in your pipeline: "
                f"{', '.join(extractors_in_pipeline)}. "
                f"This can lead to the same entity getting "
                f"extracted multiple times. Please read the documentation section "
                f"on entity extractors to make sure you understand the implications: "
                f"{DOCS_URL_COMPONENTS}#entity-extractors"
            )

    def _warn_of_competition_with_regex_extractor(
        self, training_data: TrainingData
    ) -> None:
        """Warns when regex entity extractor is competing with a general one.

        This might be the case when the following conditions are all met:
        * You are using a general entity extractor and the `RegexEntityExtractor`
        * AND you have regex patterns for entity type A
        * AND you have annotated text examples for entity type A

        Args:
            training_data: the training data
        """
        present_general_extractors = self._component_type_names.intersection(
            TRAINABLE_EXTRACTORS
        )
        has_general_extractors = len(present_general_extractors) > 0
        has_regex_extractor = "RegexEntityExtractor" in self._component_type_names

        regex_entity_types = {rf["name"] for rf in training_data.regex_features}
        overlap_between_types = training_data.entities.intersection(regex_entity_types)
        has_overlap = len(overlap_between_types) > 0

        if has_general_extractors and has_regex_extractor and has_overlap:
            rasa.shared.utils.io.raise_warning(
                f"You have an overlap between the RegexEntityExtractor and the "
                f"statistical entity extractors "
                f"{', '.join(present_general_extractors)} "
                f"in your pipeline. Specifically both types of extractors will "
                f"attempt to extract entities of the types "
                f"{', '.join(overlap_between_types)}. This can lead to multiple "
                f"extraction of entities. Please read RegexEntityExtractor's "
                f"documentation section to make sure you understand the "
                f"implications: {DOCS_URL_COMPONENTS}#regexentityextractor"
            )

    def _raise_or_warn_if_featurizers_are_not_compatible(
        self, tokenizer_schema_node: SchemaNode
    ) -> None:
        """Raises or warns if there are problems regarding the featurizers.

        We require featurizers to...
        1. have unique names because otherwise we cannot distinguish the features
           produces by them
        2. be compatible with the used tokenizer (e.g. `SpacyFeaturizer` won't do
           anything if it is not used with the `SpacyTokenizer`)
        """
        featurizers: List[SchemaNode] = [
            node
            for node in self._graph_schema.nodes.values()
            if isinstance(node.uses, Featurizer2)
        ]

        Featurizer2.validate_configs_compatible(
            [schema_node.config for schema_node in featurizers]
        )

        for schema_node in featurizers:
            Featurizer2.validate_compatibility_with_tokenizer(
                config=schema_node.config, tokenizer_type=tokenizer_schema_node.uses
            )


class CoreValidator(GraphComponent):
    """Checks whether the configuration and the Core training data match."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new `CoreValidator` (see parent class for full docstring)."""
        return cls(execution_context)

    def __init__(self, execution_context: ExecutionContext) -> None:
        """Instantiates a new `CoreValidator`.

        Args:
           execution_context: Information about the current graph run.
        """
        self._graph_schema = execution_context._graph_schema  # TODO: prune?
        self._policy_schema_nodes: List[SchemaNode] = [
            node
            for node in self._graph_schema.nodes.values()
            if isinstance(node.uses, PolicyGraphComponent)
        ]

    def validate(
        self, training_trackers: List[DialogueStateTracker], domain: Domain
    ) -> None:
        """Validates whether the configuration matches the training data.

        Args:
           training_trackers: trackers containing the Core training data
           domain: the domain
        """
        if not self._self._policy_schema_nodes:
            return
        self._warn_if_no_rule_policy_is_contained()
        self._raise_if_rule_policies_are_incompatible_with_domain(domain)
        self._warn_if_rule_based_data_is_unused_or_missing(
            training_trackers=training_trackers
        )

    def _warn_if_no_rule_policy_is_contained(self) -> None:
        """Warns if there is no rule policy among the given policies.

        NOTE: core/policies/_ensemble/check_for_important_policies
        """
        if not any(
            isinstance(node.uses, RulePolicyGraphComponent)
            for node in self._policy_schema_nodes
        ):
            rasa.shared.utils.io.raise_warning(
                f"'{RulePolicy.__name__}' is not included in the model's "
                f"policy configuration. Default intents such as "
                f"'{USER_INTENT_RESTART}' and '{USER_INTENT_BACK}' will "
                f"not trigger actions '{ACTION_RESTART_NAME}' and "
                f"'{ACTION_BACK_NAME}'.",
                docs=DOCS_URL_DEFAULT_ACTIONS,
            )

    def _warn_if_rule_based_data_is_unused_or_missing(
        self, training_trackers: List[DialogueStateTracker]
    ) -> None:
        """Warns if rule-data is unused or missing.

        NOTE: _ensemble/_emit_rule_policy_warning (called in train)

        Args:
            training_trackers: trackers to be used for training
        """
        consuming_rule_data = any(
            policy_node.uses.supported_data()
            in [SupportedData.RULE_DATA, SupportedData.ML_AND_RULE_DATA]
            for policy_node in self._policy_schema_nodes
        )
        contains_rule_tracker = any(
            tracker.is_rule_tracker for tracker in training_trackers
        )

        if consuming_rule_data and not contains_rule_tracker:
            rasa.shared.utils.io.raise_warning(
                f"Found a rule-based policy in your pipeline but "
                f"no rule-based training data. Please add rule-based "
                f"stories to your training data or "
                f"remove the rule-based policy (`{RulePolicy.__name__}`) from your "
                f"your pipeline.",
                docs=DOCS_URL_RULES,
            )
        elif not consuming_rule_data and contains_rule_tracker:
            rasa.shared.utils.io.raise_warning(
                f"Found rule-based training data but no policy supporting rule-based "
                f"data. Please add `{RulePolicy.__name__}` or another rule-supporting "
                f"policy to the `policies` section in `{DEFAULT_CONFIG_PATH}`.",
                docs=DOCS_URL_RULES,
            )

    def _warn_if_priorities_are_not_unique(self) -> None:
        """Warns if the priorities of the policies are not unique.

        NOTE: _ensemble/check_priorities
        """
        priority_dict = defaultdict(list)
        for schema_node in self._policy_schema_nodes:
            priority = schema_node.config["priority"]
            name = schema_node.node_name
            priority_dict[priority].append(name)

        for k, v in priority_dict.items():
            if len(v) > 1:
                rasa.shared.utils.io.raise_warning(
                    f"Found policies {v} with same priority {k} "
                    f"in PolicyEnsemble. When personalizing "
                    f"priorities, be sure to give all policies "
                    f"different priorities.",
                    docs=DOCS_URL_POLICIES,
                )

    def _raise_if_rule_policies_are_incompatible_with_domain(
        self, domain: Domain,
    ) -> None:
        """Validates the rule policies against the domain.

        NOTE: _ensemble/check_domain_ensemble_compatibility (called from agent.py)

        Raises:
            `InvalidConfigException` if domain and rule policies do not match
        """
        rule_policies: List[SchemaNode] = [
            schema_node
            for schema_node in self._graph_schema.nodes.items()
            if schema_node.uses == RulePolicyGraphComponent
        ]

        if domain.form_names and not rule_policies:
            raise InvalidDomain(
                "You have defined a form action, but have not added the "
                f"'{RulePolicyGraphComponent.__name__}' to your policy ensemble. "
                f"Either remove all forms from your domain or add the "
                f"'{RulePolicyGraphComponent.__name__}' to your policy configuration."
            )

        if rule_policies:
            first_rule_policy: SchemaNode = rule_policies[0]
            try:
                RulePolicyGraphComponent._validate_against_domain(
                    config=first_rule_policy.config, domain=domain
                )
            except InvalidDomain as e:
                raise InvalidConfigException(
                    f"The given domain does not match the configuration "
                    f"of {first_rule_policy.node_name}."
                    # TODO: do we want to print node_name or id?
                ) from e

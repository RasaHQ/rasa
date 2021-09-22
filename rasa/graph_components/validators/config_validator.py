from __future__ import annotations
from collections import defaultdict
import itertools
from typing import Iterable, List, Dict, Text, Any, Set, Type

from rasa.engine.graph import ExecutionContext, GraphComponent, GraphSchema, SchemaNode
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.nlu.featurizers.featurizer import Featurizer2
from rasa.nlu.extractors.mitie_entity_extractor import (
    MitieEntityExtractorGraphComponent,
)
from rasa.nlu.extractors.regex_entity_extractor import (
    RegexEntityExtractorGraphComponent,
)
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractorGraphComponent
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapperComponent
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import (
    RegexFeaturizerGraphComponent,
)
from rasa.nlu.classifiers.diet_classifier import DIETClassifierGraphComponent
from rasa.nlu.selectors.response_selector import ResponseSelectorGraphComponent
from rasa.nlu.tokenizers.tokenizer import TokenizerGraphComponent
from rasa.core.policies.rule_policy import RulePolicyGraphComponent
from rasa.core.policies.policy import PolicyGraphComponent, SupportedData
from rasa.core.policies.memoization import MemoizationPolicyGraphComponent
from rasa.core.policies.ted_policy import TEDPolicyGraphComponent
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizerGraphComponent,
)
from rasa.shared.core.training_data.structures import RuleStep, StoryGraph
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
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.shared.utils.io


# FIXME: check right paths, not just existence on any path (e.g. end2end featurization
# might be separate from NLU featurization and only one of them might have a
# tokenizer)

TRAINABLE_EXTRACTORS = {
    MitieEntityExtractorGraphComponent,
    CRFEntityExtractorGraphComponent,
    DIETClassifierGraphComponent,
}

# TODO: do we have a registries for these?
FEATURIZER_CLASSES = {
    LexicalSyntacticFeaturizerGraphComponent,
    RegexFeaturizerGraphComponent,
    # TODO: add rest / replace
}
POLICY_CLASSSES = {  # Note: this is used in tests only, should belong elsewhere anyway
    TEDPolicyGraphComponent,
    MemoizationPolicyGraphComponent,
    RulePolicyGraphComponent,
}


def types_to_str(types: Iterable[Type]) -> Text:
    """Returns a text containing the names of all given types.

    Args:
        types: some types
    Returns:
        text containing all type names
    """
    return ", ".join([type.__name__ for type in types])


class ConfigValidator(GraphComponent):
    """Validates the current graph schema against the training data and domain."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> ConfigValidator:
        """Creates a new `ConfigValidator` (see parent class for full docstring)."""
        return cls(execution_context.graph_schema)

    def __init__(self, graph_schema: GraphSchema) -> None:
        """Instantiates a new `ConfigValidator`.

        Args:
           graph_schema: a graph schema
        """
        self._graph_schema = graph_schema
        self._component_types = set(node.uses for node in graph_schema.nodes.values())
        self._policy_schema_nodes: List[SchemaNode] = [
            node
            for node in self._graph_schema.nodes.values()
            if issubclass(node.uses, PolicyGraphComponent)
        ]

    def validate(self, importer: TrainingDataImporter) -> None:
        """Validates the current graph schema against the training data and domain.

        Args:
            importer: the training data importer which can also load the domain
        """
        nlu_data = importer.get_nlu_data()
        self._validate_nlu(nlu_data)

        story_graph = importer.get_stories()
        domain = importer.get_domain()
        self._validate_core(story_graph, domain)

    def _validate_nlu(self, training_data: TrainingData) -> None:
        """Validates whether the configuration matches the training data.

        Args:
           training_data: training_data
        """
        # TODO: raise if training data is empty?
        # TODO: where do we check whether NLU training data is consistent with
        # the domain?
        self._raise_if_more_than_one_tokenizer()
        self._raise_if_featurizers_are_not_compatible()
        self._warn_of_competing_extractors()
        self._warn_of_competition_with_regex_extractor(training_data=training_data)
        self._warn_if_some_training_data_is_unused(training_data=training_data)

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
            and ResponseSelectorGraphComponent not in self._component_types
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data with examples for training a response "
                f"selector, but your NLU pipeline does not include a response selector "
                f"component. To train a model on your response selector data, add a "
                f"'{ResponseSelectorGraphComponent.__name__}' to your pipeline."
            )

        if training_data.entity_examples and self._component_types.isdisjoint(
            TRAINABLE_EXTRACTORS
        ):
            rasa.shared.utils.io.raise_warning(
                "You have defined training data consisting of entity examples, but "
                "your NLU pipeline does not include an entity extractor trained on "
                "your training data. To extract non-pretrained entities, add one of "
                f"{types_to_str(TRAINABLE_EXTRACTORS)} to your pipeline."
            )

        if training_data.entity_examples and self._component_types.isdisjoint(
            {DIETClassifierGraphComponent, CRFEntityExtractorGraphComponent}
        ):
            if training_data.entity_roles_groups_used():
                rasa.shared.utils.io.raise_warning(
                    f"You have defined training data with entities that "
                    f"have roles/groups, but your NLU pipeline does not "
                    f"include a '{DIETClassifierGraphComponent.__name__}' "
                    f"or a '{CRFEntityExtractorGraphComponent.__name__}'. "
                    f"To train entities that have roles/groups, "
                    f"add either '{DIETClassifierGraphComponent.__name__}' "
                    f"or '{CRFEntityExtractorGraphComponent.__name__}' to your "
                    f"pipeline."
                )

        if training_data.regex_features and self._component_types.isdisjoint(
            [RegexFeaturizerGraphComponent, RegexEntityExtractorGraphComponent],
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data with regexes, but "
                f"your NLU pipeline does not include a 'RegexFeaturizer' or a "
                f"'RegexEntityExtractor'. To use regexes, include either a "
                f"'{RegexFeaturizerGraphComponent.__name__}' or a "
                f"'{RegexEntityExtractorGraphComponent.__name__}' in your pipeline."
            )

        # NOTE: bugs
        # 1. removed the check that complains if neither "CRFEntityExtractor",
        #   nor "DIETClassifier" are contained in case a lookup table is given...
        #   ... because
        #   - CRFEntityExtractor: only uses lookup tables if a RegexFeaturizer is
        #       present *and* the patterns feature is defined (see also 2.)
        #   - DIET: does not use the lookup table...
        # 2. previously we warned if there is a lookup table and the
        #    CRF entity extractor has no "patterns" defined
        #    a) this is optional but not mandatory, so we *could* remove that warning
        #       (it is still below - see if "CRFEntityExtractor...")
        #    b) we definitely should warn if "patterns" is defined but there
        #       is no regex featurizer
        #       (this is a new warning added under the if "CRFEntityExtractor...")

        if training_data.lookup_tables and self._component_types.isdisjoint(
            [RegexFeaturizerGraphComponent, RegexEntityExtractorGraphComponent],
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data consisting of lookup tables, but "
                f"your NLU pipeline does not include a "
                f"'{RegexFeaturizerGraphComponent.__name__}' or a "
                f"'{RegexEntityExtractorGraphComponent.__name__}'. "
                f"To use lookup tables, include either a "
                f"'{RegexFeaturizerGraphComponent.__name__}' "
                f"or a '{RegexEntityExtractorGraphComponent.__name__}' "
                f"in your pipeline."
            )

        if CRFEntityExtractorGraphComponent in self._component_types:

            crf_schema_nodes = [
                schema_node
                for schema_node in self._graph_schema.nodes.values()
                if schema_node.uses == CRFEntityExtractorGraphComponent
            ]
            # check to see if any of the possible CRFEntityExtractors will
            # featurize `pattern`
            has_pattern_feature = False

            for crf in crf_schema_nodes:
                crf_features = crf.config.get("features", [])
                # iterate through [[before],[word],[after]] features
                has_pattern_feature = "pattern" in itertools.chain(*crf_features)

            if training_data.lookup_tables and not has_pattern_feature:
                rasa.shared.utils.io.raise_warning(
                    f"You have defined training data consisting of "
                    f"lookup tables, but your NLU pipeline's "
                    f"'{CRFEntityExtractorGraphComponent.__name__}' "
                    f"does not include the "
                    f"'pattern' feature. To featurize lookup tables, "
                    f"add the 'pattern' feature to the "
                    f"'{CRFEntityExtractorGraphComponent.__name__}' "
                    "in your pipeline."
                )

            if not training_data.lookup_tables and has_pattern_feature:
                rasa.shared.utils.io.raise_warning(
                    "You have specified the 'pattern' feature for your "
                    f"'{CRFEntityExtractorGraphComponent.__name__}' "
                    f"but your training data does not"
                    f"contain a lookup table. "
                    f"To use the 'pattern' feature, "
                    f"add a lookup tale to your training data and a "
                    f"'{RegexFeaturizerGraphComponent.__name__}' "
                    f"to your pipeline."
                )

        if (
            training_data.entity_synonyms
            and EntitySynonymMapperComponent not in self._component_types
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined synonyms in your training data, but "
                f"your NLU pipeline does not include an "
                f"'{EntitySynonymMapperComponent.__name__}'. "
                f"To map synonyms, add an "
                f"'{EntitySynonymMapperComponent.__name__}' to your pipeline."
            )

    def _raise_if_more_than_one_tokenizer(self) -> None:
        """Validates that only one tokenizer is present in the pipeline.

        NOTE: rasa/nlu/components/validate_only_one_tokenizer_is_used

        Raises:
            `InvalidConfigException` in case there is more than one tokenizer
        """
        tokenizer_schema_nodes = [
            schema_node
            for schema_node in self._graph_schema.nodes.values()
            if issubclass(schema_node.uses, TokenizerGraphComponent)
        ]

        # TODO: is 0 tokenizers ok? shouldn't this be ==1 ? Would also match the
        # old name better...
        if len(tokenizer_schema_nodes) > 1:
            names = [
                schema_node.uses.__name__ for schema_node in tokenizer_schema_nodes
            ]
            raise InvalidConfigException(
                f"The pipeline configuration contains more than one tokenizer, "
                f"which is not possible at this time. You can only use one tokenizer. "
                f"The pipeline contains the following tokenizers: "
                f"{names}. "
            )

    def _warn_of_competing_extractors(self) -> None:
        """Warns the user when using competing extractors.

        Competing extractors are e.g. `CRFEntityExtractor` and `DIETClassifier`.
        Both of these look for the same entities based on the same training data
        leading to ambiguity in the results.

        Args:
           node_names: names of all components
        """
        extractors_in_pipeline: Set[
            Type[GraphComponent]
        ] = self._component_types.intersection(TRAINABLE_EXTRACTORS)
        if len(extractors_in_pipeline) > 1:
            rasa.shared.utils.io.raise_warning(
                f"You have defined multiple entity extractors that do the same job "
                f"in your pipeline: "
                f"{types_to_str(extractors_in_pipeline)}. "
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
        present_general_extractors = self._component_types.intersection(
            TRAINABLE_EXTRACTORS
        )
        has_general_extractors = len(present_general_extractors) > 0
        has_regex_extractor = (
            RegexEntityExtractorGraphComponent in self._component_types
        )

        regex_entity_types = {rf["name"] for rf in training_data.regex_features}
        overlap_between_types = training_data.entities.intersection(regex_entity_types)
        has_overlap = len(overlap_between_types) > 0

        if has_general_extractors and has_regex_extractor and has_overlap:
            rasa.shared.utils.io.raise_warning(
                f"You have an overlap between the "
                f"{RegexEntityExtractorGraphComponent.__name__} and the "
                f"statistical entity extractors "
                f"{types_to_str(present_general_extractors)} "
                f"in your pipeline. Specifically both types of extractors will "
                f"attempt to extract entities of the types "
                f"{', '.join(overlap_between_types)}. "
                f"This can lead to multiple "
                f"extraction of entities. Please read "
                f"{RegexEntityExtractorGraphComponent.__name__}'s "
                f"documentation section to make sure you understand the "
                f"implications: {DOCS_URL_COMPONENTS}#regexentityextractor"
            )

    def _raise_if_featurizers_are_not_compatible(self,) -> None:
        """Raises or warns if there are problems regarding the featurizers.

        Raises:
            `InvalidConfigException`
        """
        featurizers: List[SchemaNode] = [
            node
            for node in self._graph_schema.nodes.values()
            if issubclass(node.uses, Featurizer2)
        ]
        Featurizer2.raise_if_featurizer_configs_are_not_compatible(
            [schema_node.config for schema_node in featurizers]
        )

    def _validate_core(self, story_graph: StoryGraph, domain: Domain) -> None:
        """Validates whether the configuration matches the training data.

        Args:
           story_graph: a story graph (core training data)
           domain: the domain
        """
        if not self._policy_schema_nodes and story_graph.story_steps:
            rasa.shared.utils.io.raise_warning(
                "Found data for training policies but no policy was configured.",
                docs=DOCS_URL_POLICIES,
            )
        if not self._policy_schema_nodes:
            return
        self._warn_if_no_rule_policy_is_contained()
        self._raise_if_domain_contains_form_names_but_no_rule_policy_given(domain)
        self._raise_if_a_rule_policy_is_incompatible_with_domain(domain)
        self._warn_if_priorities_are_not_unique()
        self._warn_if_rule_based_data_is_unused_or_missing(story_graph=story_graph)

    def _warn_if_no_rule_policy_is_contained(self) -> None:
        """Warns if there is no rule policy among the given policies.

        NOTE: core/policies/_ensemble/check_for_important_policies
        """
        if not any(
            node.uses == RulePolicyGraphComponent for node in self._policy_schema_nodes
        ):
            rasa.shared.utils.io.raise_warning(
                f"'{RulePolicyGraphComponent.__name__}' is not included in the model's "
                f"policy configuration. Default intents such as "
                f"'{USER_INTENT_RESTART}' and '{USER_INTENT_BACK}' will "
                f"not trigger actions '{ACTION_RESTART_NAME}' and "
                f"'{ACTION_BACK_NAME}'.",
                docs=DOCS_URL_DEFAULT_ACTIONS,
            )

    def _raise_if_domain_contains_form_names_but_no_rule_policy_given(
        self, domain: Domain,
    ) -> None:
        """Validates that there exists a rule policy if forms are defined.

        NOTE: _ensemble/check_domain_ensemble_compatibility (called from agent.py)

        Raises:
            `InvalidConfigException` if domain and rule policies do not match
        """
        contains_rule_policy = any(
            schema_node
            for schema_node in self._graph_schema.nodes.values()
            if schema_node.uses == RulePolicyGraphComponent
        )

        if domain.form_names and not contains_rule_policy:
            raise InvalidDomain(
                "You have defined a form action, but have not added the "
                f"'{RulePolicyGraphComponent.__name__}' to your policy ensemble. "
                f"Either remove all forms from your domain or add the "
                f"'{RulePolicyGraphComponent.__name__}' to your policy configuration."
            )

    def _raise_if_a_rule_policy_is_incompatible_with_domain(
        self, domain: Domain,
    ) -> None:
        """Validates the rule policies against the domain.

        NOTE: _ensemble/check_domain_ensemble_compatibility (called from agent.py)

        Raises:
            `InvalidDomain` if domain and rule policies do not match
        """
        for schema_node in self._graph_schema.nodes.values():
            if schema_node.uses == RulePolicyGraphComponent:
                RulePolicyGraphComponent.raise_if_incompatible_with_domain(
                    config=schema_node.config, domain=domain
                )

    def _warn_if_priorities_are_not_unique(self) -> None:
        """Warns if the priorities of the policies are not unique.

        NOTE: _ensemble/check_priorities
        """
        priority_dict = defaultdict(list)
        for schema_node in self._policy_schema_nodes:
            if "priority" in schema_node.config:
                priority = schema_node.config["priority"]
                name = schema_node.uses.__name__
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

    def _warn_if_rule_based_data_is_unused_or_missing(
        self, story_graph: StoryGraph
    ) -> None:
        """Warns if rule-data is unused or missing.

        NOTE: _ensemble/_emit_rule_policy_warning (called in train)

        Args:
            story_graph: a story graph (core training data)
        """
        consuming_rule_data = any(
            policy_node.uses.supported_data()
            in [SupportedData.RULE_DATA, SupportedData.ML_AND_RULE_DATA]
            for policy_node in self._policy_schema_nodes
        )

        # Reminder: We generate rule trackers via:
        #   rasa/shared/core/generator/...
        #   .../TrainingDataGenerator/_generate_rule_trackers/
        contains_rule_tracker = any(
            isinstance(step, RuleStep) for step in story_graph.ordered_steps()
        )

        if consuming_rule_data and not contains_rule_tracker:
            rasa.shared.utils.io.raise_warning(
                f"Found a rule-based policy in your pipeline but "
                f"no rule-based training data. Please add rule-based "
                f"stories to your training data or "
                f"remove the rule-based policy "
                f"(`{RulePolicyGraphComponent.__name__}`) from your "
                f"your pipeline.",
                docs=DOCS_URL_RULES,
            )
        elif not consuming_rule_data and contains_rule_tracker:
            rasa.shared.utils.io.raise_warning(
                f"Found rule-based training data but no policy supporting rule-based "
                f"data. Please add `{RulePolicyGraphComponent.__name__}` "
                f"or another rule-supporting "
                f"policy to the `policies` section in `{DEFAULT_CONFIG_PATH}`.",
                docs=DOCS_URL_RULES,
            )

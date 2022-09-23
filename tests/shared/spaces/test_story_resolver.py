# TODO: test basic stories: intents, actions,
# TODO: test basic rules: intents, actions
# TODO: test entities in stories
# TODO: test rule / story names
# TODO: test slot set events
# TODO: test form actions and active loops
# TODO: test story checkpoints
# TODO: test story or switches
# TODO: test rule conditions
from pydot import frozendict

from rasa.shared.spaces.domain_resolver import DomainResolver
from rasa.shared.spaces.story_resolver import StoryResolver


def test_basic_rule_resolving():
    rules_path = "data/test_spaces/money/rules.yml"
    domain_path = "data/test_spaces/money/domain.yml"
    space_name = "money"
    expected_rules_path = "data/test_spaces/money/rules_prefixed_money.yml"
    _, domain_info = DomainResolver.load_and_resolve(domain_path, space_name)
    resolved_stories = StoryResolver.load_and_resolve(rules_path, space_name,
                                                      domain_info)
    expected_stories = StoryResolver.load(expected_rules_path)
    assert frozendict(resolved_stories) == frozendict(expected_stories)


def test_more_complex_rule_resolving():
    rules_path = "data/test_spaces/transfer_money/rules.yml"
    domain_path = "data/test_spaces/transfer_money/domain.yml"
    space_name = "transfer_money"
    expected_rules_path = "data/test_spaces/transfer_money/rules_namespaced.yml"
    _, domain_info = DomainResolver.load_and_resolve(domain_path, space_name)
    resolved_stories = StoryResolver.load_and_resolve(rules_path, space_name,
                                                      domain_info)
    expected_stories = StoryResolver.load(expected_rules_path)
    assert frozendict(resolved_stories) == frozendict(expected_stories)
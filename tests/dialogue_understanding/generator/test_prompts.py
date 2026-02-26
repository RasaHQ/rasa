from pathlib import Path

import pytest

PROMPT_TEMPLATES_DIR = Path("rasa/dialogue_understanding/generator/prompt_templates")
AGENT_DIALOGUE_UNDERSTANDING_PROMPT_TEMPLATES = sorted(
    PROMPT_TEMPLATES_DIR.glob("agent*.jinja2")
)


@pytest.mark.parametrize(
    "prompt_template_path",
    AGENT_DIALOGUE_UNDERSTANDING_PROMPT_TEMPLATES,
    ids=lambda path: path.name,
)
def test_verify_agent_prompt_templates_contain_agent_related_parts(
    prompt_template_path: Path,
) -> None:
    """Ensure every prompt for sub-agents contains necessary parts"""
    with prompt_template_path.open("r") as f:
        prompt_template = f.read()

    # sub agents in "Available Flows and Slots" section
    assert '"sub-agents":[{% for agent in flow.agent_info %}' in prompt_template

    # commands
    assert "`continue agent`" in prompt_template
    assert "`restart agent`" in prompt_template

    # active agent info
    assert '{% if active_agent %},"active_agent":' in prompt_template

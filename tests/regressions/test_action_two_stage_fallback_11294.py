import textwrap
from pathlib import Path
from typing import Callable

import rasa.core.agent
from rasa.core.channels import UserMessage
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.events import ActionExecuted, ActiveLoop, BotUttered, UserUttered


async def test_action_two_stage_fallback_does_not_return_key_error(
    tmp_path: Path,
    trained_async: Callable,
):
    config = tmp_path / "config.yml"

    config.write_text(
        textwrap.dedent(
            """
            recipe: default.v1
            assistant_id: placeholder_default

            language: en

            pipeline:
               - name: WhitespaceTokenizer
               - name: RegexFeaturizer
               - name: LexicalSyntacticFeaturizer
               - name: CountVectorsFeaturizer
               - name: DIETClassifier
                 epochs: 20
                 constrain_similarities: true
               - name: FallbackClassifier
                 threshold: 0.7
                 ambiguity_threshold: 0.1

            policies:
               - name: RulePolicy
                 enable_fallback_prediction: True
               - name: AugmentedMemoizationPolicy
               - name: TEDPolicy
                 epochs: 20
                 constrain_similarities: true
            """
        )
    )

    nlu_file = tmp_path / "nlu.yml"
    nlu_file.write_text(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            nlu:
                - intent: greet
                  examples: |
                    - hey
                    - hello
                    - hi

                - intent: affirm
                  examples: |
                    - yes
                    - y
                    - indeed
                    - of course
                    - that sounds good
                    - correct

                - intent: deny
                  examples: |
                    - no
                    - n
                    - never
                    - I don't think so
                    - don't like that
                    - no way
                    - not really

                - intent: mood_great
                  examples: |
                    - amazing
                    - perfect
                    - wonderful

                - intent: goodbye
                  examples: |
                    - cu
                    - goodbye
                    - good night
                    - have a nice day
                    - see you around
                    - bye bye
            """
        )
    )

    utter_default_text = "I'm sorry, I can't help you. Bypass to agent"
    domain = tmp_path / "domain.yml"
    domain.write_text(
        textwrap.dedent(
            f"""
                version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                intents:
                  - greet
                  - affirm
                  - deny
                  - mood_great

                slots:
                    custom_slot:
                        type: text
                        influence_conversation: true
                        mappings:
                        - type: custom

                responses:
                  utter_greet:
                  - text: "Hey! How are you?"

                  utter_ask_rephrase:
                  - text: "I'm sorry, I didn't understand that. Could you rephrase?"

                  utter_default:
                  - text: {utter_default_text}
                """
        )
    )
    rules_file = tmp_path / "rules.yml"
    rules_file.write_text(
        f"""
                version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                rules:
                - rule: test
                  steps:
                  - intent: nlu_fallback
                  - action: action_two_stage_fallback
                  - active_loop: action_two_stage_fallback
               """
    )
    model = await trained_async(
        domain=domain,
        config=config,
        training_files=[nlu_file, rules_file],
    )
    agent = await rasa.core.agent.load_agent(model)
    agent.load_model(model)

    tracker = await agent.tracker_store.get_or_create_tracker(sender_id="test_11294")
    tracker.update_with_events(
        [
            UserUttered("Hi"),
            ActionExecuted("utter_greet"),
            BotUttered("Hey! How are you?"),
            ActionExecuted("action_listen"),
            UserUttered("wunderbar"),
            ActiveLoop("action_two_stage_fallback"),
        ],
        domain,
    )
    await agent.tracker_store.save(tracker)

    message = UserMessage(
        parse_data={
            "entities": [],
            "intent": {"confidence": 0.94, "name": "deny"},
            "message_id": None,
            "metadata": {},
            "text": "/out_of_scope",
        },
        sender_id="test_11294",
    )
    responses = await agent.handle_message(message)

    assert any(bot_msg["text"] == utter_default_text for bot_msg in responses)

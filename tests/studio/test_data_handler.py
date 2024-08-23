from typing import Dict, List
from unittest.mock import MagicMock

import pytest
from rasa.shared.core.flows import Flow
from rasa.shared.core.flows.flow_step_links import (
    ElseFlowStepLink,
    FlowStepLinks,
    IfFlowStepLink,
    StaticFlowStepLink,
)
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.steps import (
    ActionFlowStep,
    CollectInformationFlowStep,
    LinkFlowStep,
)
from rasa.studio import data_handler

from rasa.studio.config import StudioConfig
from rasa.studio.data_handler import (
    StudioDataHandler,
)


@pytest.mark.parametrize(
    "domain1, domain2, domain_diff",
    [
        (
            {
                "version": "3.1",
                "intents": ["greet", "goodbye"],
                "entities": ["first_name", "last_name"],
            },
            {
                "version": "3.1",
                "intents": ["greet", "goodbye", "affirm"],
                "entities": ["first_name", "last_name", "age"],
            },
            {"intents": ["affirm"], "entities": ["age"]},
        ),
        (
            {
                "version": "3.1",
                "intents": ["greet", "goodbye"],
                "entities": ["first_name", "last_name"],
            },
            {
                "version": "3.1",
                "intents": ["greet", "goodbye"],
                "entities": ["first_name", "last_name"],
            },
            {},
        ),
        (
            {
                "version": "3.1",
                "intents": ["greet", "goodbye"],
                "entities": ["first_name", "last_name"],
            },
            {
                "version": "3.1",
                "intents": [
                    "welcome",
                    "exit",
                    "affirm",
                ],
                "entities": [
                    "last_name",
                    "age",
                    {"city": {"roles": ["departure", "destination"]}},
                ],
            },
            {
                "intents": ["welcome", "exit", "affirm"],
                "entities": ["age", {"city": {"roles": ["departure", "destination"]}}],
            },
        ),
        (
            {
                "version": "3.1",
                "actions": ["action_reset_unk_slots", "validate_order_tracking_form"],
                "responses": {
                    "utter_greet": [
                        {"text": "Hey! How are you?"},
                        {
                            "text": "Hey, {name}. Welcome back! How can I help you?",
                            "condition": [
                                {"type": "slot", "name": "logged_in", "value": True}
                            ],
                        },
                    ],
                    "utter_did_that_help": [{"text": "Did that help you?"}],
                },
                "slots": {
                    "first_name": {
                        "type": "text",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "first_name",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                        ],
                    },
                    "age": {
                        "type": "float",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "age",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                        ],
                    },
                },
            },
            {
                "version": "3.1",
                "actions": [
                    "action_reset_unk_slots",
                    "validate_order_tracking_form",
                    "action_restart_order_tracking_form",
                ],
                "responses": {
                    "utter_greet": [
                        {"text": "Hey! How are you?"},
                        {
                            "text": "Hey, {name}. Welcome back! How can I help you?",
                            "condition": [
                                {"type": "slot", "name": "logged_in", "value": True}
                            ],
                        },
                    ],
                    "utter_cheer_up": [
                        {
                            "text": "Here is something to cheer you up:",
                            "image": "https://i.imgur.com/nGF1K8f.jpg",
                        }
                    ],
                    "utter_did_that_help": [{"text": "Did that help you?"}],
                },
                "slots": {
                    "stargazers_count": {
                        "type": "text",
                        "mappings": [{"type": "from_text"}],
                    },
                    "first_name": {
                        "type": "text",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "first_name",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                        ],
                    },
                    "age": {
                        "type": "float",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "age",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                        ],
                    },
                },
            },
            {
                "actions": ["action_restart_order_tracking_form"],
                "responses": {
                    "utter_cheer_up": [
                        {
                            "text": "Here is something to cheer you up:",
                            "image": "https://i.imgur.com/nGF1K8f.jpg",
                        }
                    ],
                },
                "slots": {
                    "stargazers_count": {
                        "type": "text",
                        "mappings": [{"type": "from_text"}],
                    },
                },
            },
        ),
        (
            {
                "version": "3.1",
                "actions": ["action_reset_unk_slots", "validate_order_tracking_form"],
                "responses": {
                    "utter_greet": [
                        {"text": "Hey! How are you?"},
                        {
                            "text": "Hey, {name}. Welcome back! How can I help you?",
                            "condition": [
                                {"type": "slot", "name": "logged_in", "value": True}
                            ],
                        },
                    ],
                    "utter_did_that_help": [{"text": "Did that help you?"}],
                },
                "slots": {
                    "first_name": {
                        "type": "text",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "first_name",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                        ],
                    },
                    "age": {
                        "type": "float",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "age",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                        ],
                    },
                },
            },
            {
                "version": "3.1",
                "actions": [
                    "action_reset_unk_slots",
                    "validate_order_tracking_form",
                ],
                "responses": {
                    "utter_greet": [
                        {"text": "Hey! How are you?"},
                        {
                            "text": "Hey, {name}. Welcome back! How can I help you?",
                            "condition": [
                                {"type": "slot", "name": "logged_in", "value": True}
                            ],
                        },
                    ],
                    "utter_did_that_help": [{"text": "Did that help you?"}],
                },
                "slots": {
                    "first_name": {
                        "type": "text",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "first_name",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                        ],
                    },
                    "age": {
                        "type": "float",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "age",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                        ],
                    },
                },
            },
            {},
        ),
        (
            {
                "version": "3.1",
                "actions": ["action_unk_slots", "validate_order_tracking"],
                "responses": {
                    "utter_greet": [
                        {"text": "How are you?"},
                        {
                            "text": "Hey, {name}. Welcome back?",
                            "condition": [
                                {"type": "slot", "name": "logged_in", "value": True}
                            ],
                        },
                    ],
                    "utter_did_that_help": [{"text": "Did that help?"}],
                },
                "slots": {
                    "first_name": {
                        "type": "text",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "first_name",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                        ],
                    },
                    "age": {
                        "type": "float",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "age",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                        ],
                    },
                },
            },
            {
                "version": "3.1",
                "actions": [
                    "action_reset_unk_slots",
                    "validate_order_tracking_form",
                ],
                "responses": {
                    "utter_greet": [
                        {"text": "Hey! How are you?"},
                        {
                            "text": "Hey, {name}. Welcome back! How can I help you?",
                            "condition": [
                                {"type": "slot", "name": "logged_in", "value": True}
                            ],
                        },
                    ],
                    "utter_did_that_help": [{"text": "Did that help you?"}],
                },
                "slots": {
                    "first_name": {
                        "type": "text",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "first_name",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                        ],
                    },
                    "age": {
                        "type": "float",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "age",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                        ],
                    },
                },
            },
            {
                "actions": [
                    "action_reset_unk_slots",
                    "validate_order_tracking_form",
                ],
                "responses": {
                    "utter_greet": [
                        {"text": "Hey! How are you?"},
                        {
                            "text": "Hey, {name}. Welcome back! How can I help you?",
                            "condition": [
                                {"type": "slot", "name": "logged_in", "value": True}
                            ],
                        },
                    ],
                    "utter_did_that_help": [{"text": "Did that help you?"}],
                },
                "slots": {
                    "first_name": {
                        "type": "text",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "first_name",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "first_name",
                                    }
                                ],
                            },
                        ],
                    },
                    "age": {
                        "type": "float",
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": "age",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                            {
                                "type": "from_text",
                                "intent": "inform",
                                "conditions": [
                                    {
                                        "active_loop": "personal_details_form",
                                        "requested_slot": "age",
                                    }
                                ],
                            },
                        ],
                    },
                },
            },
        ),
    ],
)
def test_diff_generator_domain(domain1: Dict, domain2: Dict, domain_diff: Dict) -> None:
    actual_diff = data_handler.combine_domains(domain2, domain1)
    assert actual_diff == domain_diff


@pytest.mark.parametrize(
    "nlu1, nlu2, nlu_diff",
    [
        (
            {
                "version": "3.1",
                "nlu": [
                    {
                        "intent": "greet",
                        "examples": (
                            "- hey\n- hello\n"
                            "- hi\n- hello there\n"
                            "- good morning\n"
                            "- good evening\n"
                            "- moin\n- hey there\n"
                            "- let's go\n- hey dude\n"
                            "- goodmorning\n"
                            "- goodevening\n"
                            "- good afternoon\n"
                        ),
                    },
                    {
                        "intent": "goodbye",
                        "examples": (
                            "- cu\n- good by\n"
                            "- cee you later\n"
                            "- good night\n"
                            "- bye\n- goodbye\n"
                            "- have a nice day\n"
                            "- see you around\n"
                            "- bye bye\n"
                            "- see you later\n"
                        ),
                    },
                    {
                        "intent": "are_u_a_boot",
                        "examples": (
                            "- are you a bot?\n"
                            "- you a robot?\n"
                            "- are you a machine?\n"
                            "- is this live chat?\n"
                        ),
                    },
                    {
                        "intent": "how_many_stars",
                        "examples": (
                            "- How many stars do you have?\n"
                            "- github stars?\n- how many stargazers?\n"
                        ),
                    },
                    {
                        "intent": "inform",
                        "examples": (
                            "- my name is [Uros](first_name)\n"
                            "- I'm [John](first_name)\n"
                            "- Hi, my first name is [Luis](first_name)\n"
                            "- Milica\n- Karin\n- Steven\n"
                            "- I'm [18](age)\n"
                            "- I am [32](age) years old\n"
                            "- 9(age)\n"
                        ),
                    },
                ],
            },
            {
                "version": "3.1",
                "nlu": [
                    {
                        "intent": "greet",
                        "examples": (
                            "- hey\n- hello\n"
                            "- hi\n- hello there\n"
                            "- good morning\n"
                            "- good evening\n"
                            "- moin\n- hey there\n"
                            "- let's go\n- hey dude\n"
                            "- goodmorning\n"
                            "- goodevening\n"
                            "- good afternoon\n"
                        ),
                    },
                    {
                        "intent": "goodbye",
                        "examples": (
                            "- cu\n- good by\n"
                            "- cee you later\n"
                            "- good night\n"
                            "- bye\n- goodbye\n"
                            "- have a nice day\n"
                            "- see you around\n"
                            "- bye bye\n"
                            "- see you later\n"
                        ),
                    },
                    {
                        "intent": "are_u_a_boot",
                        "examples": (
                            "- are you a bot?\n"
                            "- you a robot?\n"
                            "- are you a machine?\n"
                            "- is this live chat?\n"
                        ),
                    },
                    {
                        "intent": "feel_sad",
                        "examples": (
                            "- sad.\n" "- not well\n" "- so so\n- I feel bad.\n"
                        ),
                    },
                    {
                        "intent": "feel_great",
                        "examples": (
                            "- I feel good.\n- great.\n" "- good.\n- im fine.\n"
                        ),
                    },
                    {
                        "intent": "how_many_stars",
                        "examples": (
                            "- How many stars do you have?\n"
                            "- github stars?\n- how many stargazers?\n"
                        ),
                    },
                    {
                        "intent": "inform",
                        "examples": (
                            "- my name is [Uros](first_name)\n"
                            "- I'm [John](first_name)\n"
                            "- Hi, my first name is [Luis](first_name)\n"
                            "- Milica\n- Karin\n- Steven\n"
                            "- I'm [18](age)\n"
                            "- I am [32](age) years old\n"
                            "- 9(age)\n"
                        ),
                    },
                ],
            },
            {
                "nlu": [
                    {
                        "intent": "feel_sad",
                        "examples": (
                            "- sad.\n" "- not well\n" "- so so\n- I feel bad.\n"
                        ),
                    },
                    {
                        "intent": "feel_great",
                        "examples": (
                            "- I feel good.\n- great.\n" "- good.\n- im fine.\n"
                        ),
                    },
                ]
            },
        ),
        (
            {
                "version": "3.1",
                "nlu": [
                    {
                        "intent": "greet",
                        "examples": (
                            "- hey\n- hello\n"
                            "- hi\n- hello there\n"
                            "- good morning\n"
                            "- good evening\n"
                            "- moin\n- hey there\n"
                            "- let's go\n- hey dude\n"
                            "- goodmorning\n"
                            "- good afternoon\n"
                        ),
                    },
                    {
                        "intent": "goodbye",
                        "examples": (
                            "- cu\n- good by\n"
                            "- cee you later\n"
                            "- good night\n"
                            "- bye\n- goodbye\n"
                            "- have a nice day\n"
                            "- see you around\n"
                            "- see you later\n"
                        ),
                    },
                    {
                        "intent": "are_u_a_boot",
                        "examples": (
                            "- are you a bot?\n"
                            "- you a robot?\n"
                            "- is this live chat?\n"
                        ),
                    },
                    {
                        "intent": "how_many_stars",
                        "examples": (
                            "- How many stars do you have?\n" "- github stars?\n"
                        ),
                    },
                    {
                        "intent": "inform",
                        "examples": (
                            "- my name is [Uros](first_name)\n"
                            "- I'm [John](first_name)\n"
                            "- Hi, my first name is [Luis](first_name)\n"
                            "- Milica\n- Karin\n- Steven\n"
                            "- I'm [18](age)\n"
                            "- 9(age)\n"
                        ),
                    },
                ],
            },
            {
                "version": "3.1",
                "nlu": [
                    {
                        "intent": "greet",
                        "examples": (
                            "- hey\n- hello\n"
                            "- hi\n- hello there\n"
                            "- good morning\n"
                            "- good evening\n"
                            "- moin\n- hey there\n"
                            "- let's go\n- hey dude\n"
                            "- goodmorning\n"
                            "- goodevening\n"
                            "- good afternoon\n"
                        ),
                    },
                    {
                        "intent": "goodbye",
                        "examples": (
                            "- cu\n- good by\n"
                            "- cee you later\n"
                            "- good night\n"
                            "- bye\n- goodbye\n"
                            "- have a nice day\n"
                            "- see you around\n"
                            "- bye bye\n"
                            "- see you later\n"
                        ),
                    },
                    {
                        "intent": "are_u_a_boot",
                        "examples": (
                            "- are you a bot?\n"
                            "- you a robot?\n"
                            "- are you a machine?\n"
                            "- is this live chat?\n"
                        ),
                    },
                    {
                        "intent": "feel_sad",
                        "examples": (
                            "- sad.\n" "- not well\n" "- so so\n- I feel bad.\n"
                        ),
                    },
                    {
                        "intent": "feel_great",
                        "examples": (
                            "- I feel good.\n- great.\n" "- good.\n- im fine.\n"
                        ),
                    },
                    {
                        "intent": "how_many_stars",
                        "examples": (
                            "- How many stars do you have?\n"
                            "- github stars?\n- how many stargazers?\n"
                        ),
                    },
                    {
                        "intent": "inform",
                        "examples": (
                            "- my name is [Uros](first_name)\n"
                            "- I'm [John](first_name)\n"
                            "- Hi, my first name is [Luis](first_name)\n"
                            "- Milica\n- Karin\n- Steven\n"
                            "- I'm [18](age)\n"
                            "- I am [32](age) years old\n"
                            "- 9(age)\n"
                        ),
                    },
                ],
            },
            {
                "nlu": [
                    {
                        "intent": "greet",
                        "examples": "- goodevening",
                    },
                    {
                        "intent": "goodbye",
                        "examples": "- bye bye",
                    },
                    {
                        "intent": "are_u_a_boot",
                        "examples": "- are you a machine?",
                    },
                    {
                        "intent": "feel_sad",
                        "examples": "- sad.\n- not well\n- so so\n- I feel bad.\n",
                    },
                    {
                        "intent": "feel_great",
                        "examples": "- I feel good.\n- great.\n- good.\n- im fine.\n",
                    },
                    {
                        "intent": "how_many_stars",
                        "examples": "- how many stargazers?",
                    },
                    {
                        "intent": "inform",
                        "examples": "- I am [32](age) years old",
                    },
                ]
            },
        ),
        (
            {
                "nlu": [
                    {
                        "intent": "greet",
                        "examples": "- goodevening\n- good afternoon\n",
                    },
                    {
                        "intent": "goodbye",
                        "examples": "- bye bye\n- see you later\n",
                    },
                ]
            },
            {
                "nlu": [
                    {
                        "intent": "greet",
                        "examples": "- goodevening\n- good afternoon\n",
                    },
                    {
                        "intent": "goodbye",
                        "examples": "- bye bye\n- see you later\n",
                    },
                ]
            },
            {"nlu": []},
        ),
        (
            {
                "nlu": [
                    {
                        "intent": "greet",
                        "examples": "- goodevening\n- good afternoon\n",
                    },
                    {
                        "intent": "goodbye",
                        "examples": "- bye bye\n- see you later\n",
                    },
                    {"synonym": "1", "examples": "- first\n"},
                    {"synonym": "free-wifi", "examples": "- wifi\n"},
                    {"synonym": "2", "examples": "- second one\n"},
                ]
            },
            {
                "nlu": [
                    {
                        "intent": "greet",
                        "examples": "- goodevening\n- good afternoon\n",
                    },
                    {
                        "intent": "goodbye",
                        "examples": "- bye bye\n- see you later\n",
                    },
                    {"synonym": "1", "examples": "- first\n"},
                    {"synonym": "free-wifi", "examples": "- wifi\n"},
                    {"synonym": "price-range", "examples": "- price range\n"},
                    {"synonym": "2", "examples": "- second\n- second one\n"},
                ]
            },
            {
                "nlu": [
                    {"synonym": "price-range", "examples": "- price range\n"},
                    {"synonym": "2", "examples": "- second"},
                ]
            },
        ),
    ],
)
def test_diff_generator_nlu(
    nlu1: Dict[str, List], nlu2: Dict[str, List], nlu_diff: Dict[str, List]
) -> None:
    actual_diff = data_handler.create_new_nlu_from_diff(nlu2, nlu1)
    assert actual_diff == nlu_diff


def test_diff_generator_nlu_empty_studio_data() -> None:
    original_nlu = {
        "nlu": [
            {
                "intent": "greet",
                "examples": "- hey",
            }
        ]
    }
    studio_nlu = {}
    actual_diff = data_handler.create_new_nlu_from_diff(studio_nlu, original_nlu)
    assert actual_diff == {"nlu": []}


def test_diff_generator_nlu_empty_original_data() -> None:
    original_nlu = {}
    studio_nlu = {
        "nlu": [
            {
                "intent": "greet",
                "examples": "- hey",
            }
        ]
    }
    actual_diff = data_handler.create_new_nlu_from_diff(studio_nlu, original_nlu)
    assert actual_diff == {"nlu": [{"examples": "- hey", "intent": "greet"}]}


def test_diff_generator_nlu_empty_everything() -> None:
    original_nlu = {}
    studio_nlu = {}
    actual_diff = data_handler.create_new_nlu_from_diff(studio_nlu, original_nlu)
    assert actual_diff == {"nlu": []}


@pytest.mark.parametrize(
    "flows1, flows2, flows_diff",
    [
        (
            [
                Flow(
                    id="check_balance",
                    custom_name="check your balance",
                    description="check the user's account balance.",
                    step_sequence=FlowStepSequence(
                        child_steps=[
                            ActionFlowStep(
                                custom_id="0_check_balance",
                                idx=0,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[
                                        StaticFlowStepLink(
                                            target_step_id="1_utter_current_balance"
                                        )
                                    ]
                                ),
                                action="check_balance",
                            ),
                            ActionFlowStep(
                                custom_id="1_utter_current_balance",
                                idx=1,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[StaticFlowStepLink(target_step_id="END")]
                                ),
                                action="utter_current_balance",
                            ),
                        ]
                    ),
                ),
                Flow(
                    id="replace_card",
                    custom_name="replace_card",
                    description="The user needs to replace their card.",
                    step_sequence=FlowStepSequence(
                        child_steps=[
                            CollectInformationFlowStep(
                                custom_id="0_collect_confirm_correct_card",
                                idx=0,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[
                                        IfFlowStepLink(
                                            target_reference=FlowStepSequence(
                                                child_steps=[
                                                    LinkFlowStep(
                                                        custom_id="1_link_replace_eligible_card",
                                                        idx=1,
                                                        description=None,
                                                        metadata={},
                                                        next=FlowStepLinks(links=[]),
                                                        link="replace_eligible_card",
                                                    )
                                                ]
                                            ),
                                            condition="confirm_correct_card",
                                        ),
                                        ElseFlowStepLink(
                                            target_reference=FlowStepSequence(
                                                child_steps=[
                                                    ActionFlowStep(
                                                        custom_id="2_utter_relevant_card_not_linked",
                                                        idx=2,
                                                        description=None,
                                                        metadata={},
                                                        next=FlowStepLinks(
                                                            links=[
                                                                StaticFlowStepLink(
                                                                    target_step_id="END"
                                                                )
                                                            ]
                                                        ),
                                                        action="utter_relevant_card_not_linked",
                                                    )
                                                ]
                                            )
                                        ),
                                    ]
                                ),
                                collect="confirm_correct_card",
                                utter="utter_ask_confirm_correct_card",
                                collect_action="action_ask_confirm_correct_card",
                                rejections=[],
                                ask_before_filling=True,
                                reset_after_flow_ends=True,
                            )
                        ]
                    ),
                ),
            ],
            [
                Flow(
                    id="check_balance",
                    custom_name="check your balance",
                    description="check the user's account balance.",
                    step_sequence=FlowStepSequence(
                        child_steps=[
                            ActionFlowStep(
                                custom_id="0_check_balance",
                                idx=0,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[
                                        StaticFlowStepLink(
                                            target_step_id="1_utter_current_balance"
                                        )
                                    ]
                                ),
                                action="check_balance",
                            ),
                            ActionFlowStep(
                                custom_id="1_utter_current_balance",
                                idx=1,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[StaticFlowStepLink(target_step_id="END")]
                                ),
                                action="utter_current_balance",
                            ),
                        ]
                    ),
                ),
                Flow(
                    id="replace_card",
                    custom_name="replace_card",
                    description="The user needs to replace their card.",
                    step_sequence=FlowStepSequence(
                        child_steps=[
                            CollectInformationFlowStep(
                                custom_id="0_collect_confirm_correct_card",
                                idx=0,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[
                                        IfFlowStepLink(
                                            target_reference=FlowStepSequence(
                                                child_steps=[
                                                    LinkFlowStep(
                                                        custom_id="1_link_replace_eligible_card",
                                                        idx=1,
                                                        description=None,
                                                        metadata={},
                                                        next=FlowStepLinks(links=[]),
                                                        link="replace_eligible_card",
                                                    )
                                                ]
                                            ),
                                            condition="confirm_correct_card",
                                        ),
                                        ElseFlowStepLink(
                                            target_reference=FlowStepSequence(
                                                child_steps=[
                                                    ActionFlowStep(
                                                        custom_id="2_utter_relevant_card_not_linked",
                                                        idx=2,
                                                        description=None,
                                                        metadata={},
                                                        next=FlowStepLinks(
                                                            links=[
                                                                StaticFlowStepLink(
                                                                    target_step_id="END"
                                                                )
                                                            ]
                                                        ),
                                                        action="utter_relevant_card_not_linked",
                                                    )
                                                ]
                                            )
                                        ),
                                    ]
                                ),
                                collect="confirm_correct_card",
                                utter="utter_ask_confirm_correct_card",
                                collect_action="action_ask_confirm_correct_card",
                                rejections=[],
                                ask_before_filling=True,
                                reset_after_flow_ends=True,
                            )
                        ]
                    ),
                ),
            ],
            [],
        ),
        (
            [
                Flow(
                    id="replace_card",
                    custom_name="replace_card",
                    description="The user needs to replace their card.",
                    step_sequence=FlowStepSequence(
                        child_steps=[
                            CollectInformationFlowStep(
                                custom_id="0_collect_confirm_correct_card",
                                idx=0,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[
                                        IfFlowStepLink(
                                            target_reference=FlowStepSequence(
                                                child_steps=[
                                                    LinkFlowStep(
                                                        custom_id="1_link_replace_eligible_card",
                                                        idx=1,
                                                        description=None,
                                                        metadata={},
                                                        next=FlowStepLinks(links=[]),
                                                        link="replace_eligible_card",
                                                    )
                                                ]
                                            ),
                                            condition="confirm_correct_card",
                                        ),
                                        ElseFlowStepLink(
                                            target_reference=FlowStepSequence(
                                                child_steps=[
                                                    ActionFlowStep(
                                                        custom_id="2_utter_relevant_card_not_linked",
                                                        idx=2,
                                                        description=None,
                                                        metadata={},
                                                        next=FlowStepLinks(
                                                            links=[
                                                                StaticFlowStepLink(
                                                                    target_step_id="END"
                                                                )
                                                            ]
                                                        ),
                                                        action="utter_relevant_card_not_linked",
                                                    )
                                                ]
                                            )
                                        ),
                                    ]
                                ),
                                collect="confirm_correct_card",
                                utter="utter_ask_confirm_correct_card",
                                collect_action="action_ask_confirm_correct_card",
                                rejections=[],
                                ask_before_filling=True,
                                reset_after_flow_ends=True,
                            )
                        ]
                    ),
                ),
            ],
            [
                Flow(
                    id="check_balance",
                    custom_name="check your balance",
                    description="check the user's account balance.",
                    step_sequence=FlowStepSequence(
                        child_steps=[
                            ActionFlowStep(
                                custom_id="0_check_balance",
                                idx=0,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[
                                        StaticFlowStepLink(
                                            target_step_id="1_utter_current_balance"
                                        )
                                    ]
                                ),
                                action="check_balance",
                            ),
                            ActionFlowStep(
                                custom_id="1_utter_current_balance",
                                idx=1,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[StaticFlowStepLink(target_step_id="END")]
                                ),
                                action="utter_current_balance",
                            ),
                        ]
                    ),
                ),
                Flow(
                    id="replace_card",
                    custom_name="replace_card",
                    description="The user needs to replace their card.",
                    step_sequence=FlowStepSequence(
                        child_steps=[
                            CollectInformationFlowStep(
                                custom_id="0_collect_confirm_correct_card",
                                idx=0,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[
                                        IfFlowStepLink(
                                            target_reference=FlowStepSequence(
                                                child_steps=[
                                                    LinkFlowStep(
                                                        custom_id="1_link_replace_eligible_card",
                                                        idx=1,
                                                        description=None,
                                                        metadata={},
                                                        next=FlowStepLinks(links=[]),
                                                        link="replace_eligible_card",
                                                    )
                                                ]
                                            ),
                                            condition="confirm_correct_card",
                                        ),
                                        ElseFlowStepLink(
                                            target_reference=FlowStepSequence(
                                                child_steps=[
                                                    ActionFlowStep(
                                                        custom_id="2_utter_relevant_card_not_linked",
                                                        idx=2,
                                                        description=None,
                                                        metadata={},
                                                        next=FlowStepLinks(
                                                            links=[
                                                                StaticFlowStepLink(
                                                                    target_step_id="END"
                                                                )
                                                            ]
                                                        ),
                                                        action="utter_relevant_card_not_linked",
                                                    )
                                                ]
                                            )
                                        ),
                                    ]
                                ),
                                collect="confirm_correct_card",
                                utter="utter_ask_confirm_correct_card",
                                collect_action="action_ask_confirm_correct_card",
                                rejections=[],
                                ask_before_filling=True,
                                reset_after_flow_ends=True,
                            )
                        ]
                    ),
                ),
            ],
            [
                Flow(
                    id="check_balance",
                    custom_name="check your balance",
                    description="check the user's account balance.",
                    step_sequence=FlowStepSequence(
                        child_steps=[
                            ActionFlowStep(
                                custom_id="0_check_balance",
                                idx=0,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[
                                        StaticFlowStepLink(
                                            target_step_id="1_utter_current_balance"
                                        )
                                    ]
                                ),
                                action="check_balance",
                            ),
                            ActionFlowStep(
                                custom_id="1_utter_current_balance",
                                idx=1,
                                description=None,
                                metadata={},
                                next=FlowStepLinks(
                                    links=[StaticFlowStepLink(target_step_id="END")]
                                ),
                                action="utter_current_balance",
                            ),
                        ]
                    ),
                )
            ],
        ),
    ],
)
def test_diff_generator_flows(
    flows1: List[Flow], flows2: List[Flow], flows_diff: List[Flow]
) -> None:
    actual_diff = data_handler.create_new_flows_from_diff(flows2, flows1)
    assert actual_diff == flows_diff


@pytest.fixture
def handler() -> StudioDataHandler:
    return StudioDataHandler(
        StudioConfig(
            authentication_server_url="http://studio.amazonaws.com",
            studio_url="http://studio.amazonaws.com",
            realm_name="rasa-test",
            client_id="rasa-cli",
        ),
        "test",
    )


def test_build_request_simple(handler: StudioDataHandler) -> None:
    request = handler._build_request()
    assert request == {
        "query": (
            "query ExportAsEncodedYaml($input: ExportAsEncodedYamlInput!)"
            " { exportAsEncodedYaml(input: $input) "
            "{ ... on ExportModernAsEncodedYamlOutput "
            "{ nlu flows domain endpoints config }"
            " ... on ExportClassicAsEncodedYamlOutput "
            "{ nlu domain }}}"
        ),
        "variables": {"input": {"assistantName": "test"}},
    }


def test_build_request(handler: StudioDataHandler) -> None:
    request = handler._build_request(["inform"], ["city"])
    assert request == {
        "query": (
            "query ExportAsEncodedYaml($input: ExportAsEncodedYamlInput!)"
            " { exportAsEncodedYaml(input: $input) "
            "{ ... on ExportModernAsEncodedYamlOutput "
            "{ nlu flows domain endpoints config }"
            " ... on ExportClassicAsEncodedYamlOutput "
            "{ nlu domain }}}"
        ),
        "variables": {
            "input": {
                "assistantName": "test",
                "objects": [
                    {"name": ["inform"], "type": "Intent"},
                    {"name": ["city"], "type": "Entity"},
                ],
            }
        },
    }


@pytest.mark.parametrize(
    "data",
    [
        {
            "data": {
                "exportAsEncodedYaml": {
                    "nlu": "test",
                    "domain": "test",
                    "domainIntents": "test",
                    "domainEntities": "test",
                }
            }
        },
        {
            "data": {
                "exportAsEncodedYaml": {
                    "flows": "test",
                    "domain": "test",
                    "domainActions": "test",
                    "domainResponses": "test",
                    "domainSlots": "test",
                }
            }
        },
    ],
)
def test_validate_valid(handler: StudioDataHandler, data: Dict) -> None:
    assert handler._validate_response(data)


@pytest.mark.parametrize(
    "data, error",
    [
        ({"errors": [{"message": "Test error message"}]}, "Test error message"),
        ({"data": {"exportAsEncodedYaml": None}}, "{'exportAsEncodedYaml': None}"),
    ],
)
def test_validate_invalid(
    handler: StudioDataHandler, data: Dict, error: str, caplog: pytest.LogCaptureFixture
) -> None:
    assert not handler._validate_response(data)
    assert error in caplog.text


def test_refresh_token(
    handler: StudioDataHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = MagicMock()
    token.can_refresh.return_value = True
    auth_mock = MagicMock()
    reader_mock = MagicMock()
    monkeypatch.setattr("rasa.studio.data_handler.StudioAuth", auth_mock)
    monkeypatch.setattr("rasa.studio.data_handler.KeycloakTokenReader", reader_mock)
    handler.refresh_token(token)
    assert auth_mock.called
    assert auth_mock.return_value.refresh_token.called
    assert reader_mock.called


def test_extract_data_nlu(handler: StudioDataHandler) -> None:
    """Test that the nlu data (base64) is extracted correctly from the response."""
    handler._extract_data(
        {
            "data": {
                "exportAsEncodedYaml": {
                    "domain": (
                        "dmVyc2lvbjogJzMuMScKaW50ZW50czoKLSBhcmVfdV9hX2Jvb3QKLS"
                        "BmZWVsX2dyZWF0Ci0gZmVlbF9zYWQKLSBnb29kYnllCi0gZ3JlZXQK"
                        "LSBob3dfbWFueV9zdGFycwotIGluZm9ybQotIHN0YXJ0X2Zvcm0KZW"
                        "50aXRpZXM6Ci0gZmlyc3RfbmFtZQotIGFnZQ=="
                    ),
                    "nlu": (
                        "dmVyc2lvbjogJzMuMScKbmx1OgotIGludGVudDogZ3JlZXQKICBleG"
                        "FtcGxlczogfAogICAgLSBoZXkKICAgIC0gaGVsbG8KICAgIC0gaGkK"
                        "ICAgIC0gaGVsbG8gdGhlcmUKICAgIC0gZ29vZCBtb3JuaW5nCiAgIC"
                        "AtIGdvb2QgZXZlbmluZwogICAgLSBtb2luCiAgICAtIGhleSB0aGVy"
                        "ZQogICAgLSBsZXQncyBnbwogICAgLSBoZXkgZHVkZQogICAgLSBnb2"
                        "9kbW9ybmluZwogICAgLSBnb29kZXZlbmluZwogICAgLSBnb29kIGFm"
                        "dGVybm9vbgotIGludGVudDogZ29vZGJ5ZQogIGV4YW1wbGVzOiB8Ci"
                        "AgICAtIGN1CiAgICAtIGdvb2QgYnkKICAgIC0gY2VlIHlvdSBsYXRl"
                        "cgogICAgLSBnb29kIG5pZ2h0CiAgICAtIGJ5ZQogICAgLSBnb29kYn"
                        "llCiAgICAtIGhhdmUgYSBuaWNlIGRheQogICAgLSBzZWUgeW91IGFy"
                        "b3VuZAogICAgLSBieWUgYnllCiAgICAtIHNlZSB5b3UgbGF0ZXIKLS"
                        "BpbnRlbnQ6IGFyZV91X2FfYm9vdAogIGV4YW1wbGVzOiB8CiAgICAt"
                        "IGFyZSB5b3UgYSBib3Q/CiAgICAtIHlvdSBhIHJvYm90PwogICAgLS"
                        "BhcmUgeW91IGEgbWFjaGluZT8KICAgIC0gaXMgdGhpcyBsaXZlIGN"
                        "oYXQ/"
                    ),
                }
            }
        }
    )

    assert (
        handler.nlu
        == """version: '3.1'
nlu:
- intent: greet
  examples: |
    - hey
    - hello
    - hi
    - hello there
    - good morning
    - good evening
    - moin
    - hey there
    - let's go
    - hey dude
    - goodmorning
    - goodevening
    - good afternoon
- intent: goodbye
  examples: |
    - cu
    - good by
    - cee you later
    - good night
    - bye
    - goodbye
    - have a nice day
    - see you around
    - bye bye
    - see you later
- intent: are_u_a_boot
  examples: |
    - are you a bot?
    - you a robot?
    - are you a machine?
    - is this live chat?"""
    )
    assert (
        handler.domain
        == """version: '3.1'
intents:
- are_u_a_boot
- feel_great
- feel_sad
- goodbye
- greet
- how_many_stars
- inform
- start_form
entities:
- first_name
- age"""
    )
    assert handler.has_nlu()


def test_extract_data_flows(handler: StudioDataHandler) -> None:
    """Test that the nlu data (base64) is extracted correctly from the response."""
    handler._extract_data(
        {
            "data": {
                "exportAsEncodedYaml": {
                    "flows": (
                        "Zmxvd3M6CiAgcmVwbGFjZV9lbGlnaWJsZV9jYXJkOgogICAgZGVzY3Jpc"
                        "HRpb246IE5ldmVyIHByZWRpY3QgU3RhcnRGbG93IGZvciB0aGlzIGZsb3"
                        "csIHVzZXJzIGFyZSBub3QgYWJsZSB0byB0cmlnZ2VyLgogICAgbmFtZTo"
                        "gcmVwbGFjZSBlbGlnaWJsZSBjYXJkCiAgICBzdGVwczoKICAgICAgLSBj"
                        "b2xsZWN0OiByZXBsYWNlbWVudF9yZWFzb24KICAgICAgICBuZXh0OgogI"
                        "CAgICAgICAgLSBpZjogcmVwbGFjZW1lbnRfcmVhc29uID09ICJsb3N0Ig"
                        "ogICAgICAgICAgICB0aGVuOgogICAgICAgICAgICAgIC0gY29sbGVjdDo"
                        "gd2FzX2NhcmRfdXNlZF9mcmF1ZHVsZW50bHkKICAgICAgICAgICAgICAg"
                        "IGFza19iZWZvcmVfZmlsbGluZzogdHJ1ZQogICAgICAgICAgICAgICAgb"
                        "mV4dDoKICAgICAgICAgICAgICAgICAgLSBpZjogd2FzX2NhcmRfdXNlZF"
                        "9mcmF1ZHVsZW50bHkKICAgICAgICAgICAgICAgICAgICB0aGVuOgogICA"
                        "gICAgICAgICAgICAgICAgICAgLSBhY3Rpb246IHV0dGVyX3JlcG9ydF9m"
                        "cmF1ZAogICAgICAgICAgICAgICAgICAgICAgICBuZXh0OiBFTkQKICAgI"
                        "CAgICAgICAgICAgICAgLSBlbHNlOiBzdGFydF9yZXBsYWNlbWVudAogIC"
                        "AgICAgICAgLSBpZjogInJlcGxhY2VtZW50X3JlYXNvbiA9PSAnZGFtYWd"
                        "lZCciCiAgICAgICAgICAgIHRoZW46IHN0YXJ0X3JlcGxhY2VtZW50CiAg"
                        "ICAgICAgICAtIGVsc2U6CiAgICAgICAgICAgIC0gYWN0aW9uOiB1dHRlc"
                        "l91bmtub3duX3JlcGxhY2VtZW50X3JlYXNvbl9oYW5kb3ZlcgogICAgIC"
                        "AgICAgICAgIG5leHQ6IEVORAogICAgICAtIGlkOiBzdGFydF9yZXBsYWN"
                        "lbWVudAogICAgICAgIGFjdGlvbjogdXR0ZXJfd2lsbF9jYW5jZWxfYW5k"
                        "X3NlbmRfbmV3CiAgICAgIC0gYWN0aW9uOiB1dHRlcl9uZXdfY2FyZF9oY"
                        "XNfYmVlbl9vcmRlcmVk"
                    ),
                    "domain": (
                        "dmVyc2lvbjogJzMuMScKcmVzcG9uc2VzOgogIHV0dGVyX2Zvcm1fY29tc"
                        "GxldGVkOgogIC0gdGV4dDogJ0ZpcnN0IE5hbWU6IHtmaXJzdF9uYW1lfS"
                        "wgQWdlOiB7YWdlfScKICB1dHRlcl9hc2tfYWdlOgogIC0gdGV4dDogaG9"
                        "3IG9sZCBhcmUgdT8KICB1dHRlcl9hc2tfZmlyc3RfbmFtZToKICAtIHRl"
                        "eHQ6IFdoYXQgaXMgdXIgbmFtZT8KICB1dHRlcl9oaV9maXJzdF9uYW1lO"
                        "gogIC0gdGV4dDogaGkge2ZpcnN0X25hbWV9CnNsb3RzOgogIHN0YXJnYX"
                        "plcnNfY291bnQ6CiAgICB0eXBlOiB0ZXh0CiAgICBtYXBwaW5nczoKICA"
                        "gIC0gdHlwZTogZnJvbV90ZXh0CiAgZmlyc3RfbmFtZToKICAgIHR5cGU6"
                        "IHRleHQKICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBmcm9tX2VudGl0e"
                        "QogICAgICBlbnRpdHk6IGZpcnN0X25hbWUKICAgICAgaW50ZW50OiBpbm"
                        "Zvcm0KICAgICAgY29uZGl0aW9uczoKICAgICAgLSBhY3RpdmVfbG9vcDo"
                        "gcGVyc29uYWxfZGV0YWlsc19mb3JtCiAgICAgICAgcmVxdWVzdGVkX3Ns"
                        "b3Q6IGZpcnN0X25hbWUKICAgIC0gdHlwZTogZnJvbV90ZXh0CiAgICAgI"
                        "GludGVudDogaW5mb3JtCiAgICAgIGNvbmRpdGlvbnM6CiAgICAgIC0gYW"
                        "N0aXZlX2xvb3A6IHBlcnNvbmFsX2RldGFpbHNfZm9ybQogICAgICAgIHJ"
                        "lcXVlc3RlZF9zbG90OiBmaXJzdF9uYW1lCiAgYWdlOgogICAgdHlwZTog"
                        "ZmxvYXQKICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBmcm9tX2VudGl0e"
                        "QogICAgICBlbnRpdHk6IGFnZQogICAgICBpbnRlbnQ6IGluZm9ybQogIC"
                        "AgICBjb25kaXRpb25zOgogICAgICAtIGFjdGl2ZV9sb29wOiBwZXJzb25"
                        "hbF9kZXRhaWxzX2Zvcm0KICAgICAgICByZXF1ZXN0ZWRfc2xvdDogYWdl"
                        "CiAgICAtIHR5cGU6IGZyb21fdGV4dAogICAgICBpbnRlbnQ6IGluZm9yb"
                        "QogICAgICBjb25kaXRpb25zOgogICAgICAtIGFjdGl2ZV9sb29wOiBwZX"
                        "Jzb25hbF9kZXRhaWxzX2Zvcm0KICAgICAgICByZXF1ZXN0ZWRfc2xvdDo"
                        "gYWdlCmFjdGlvbnM6Ci0gYWN0aW9uX2V4YW1wbGUKLSBhY3Rpb25fcmFz"
                        "YV9zdGFyZ2F6ZXJzX2NvdW50"
                    ),
                }
            }
        }
    )

    assert (
        handler.flows
        == """flows:
  replace_eligible_card:
    description: Never predict StartFlow for this flow, users are not able to trigger.
    name: replace eligible card
    steps:
      - collect: replacement_reason
        next:
          - if: replacement_reason == "lost"
            then:
              - collect: was_card_used_fraudulently
                ask_before_filling: true
                next:
                  - if: was_card_used_fraudulently
                    then:
                      - action: utter_report_fraud
                        next: END
                  - else: start_replacement
          - if: "replacement_reason == 'damaged'"
            then: start_replacement
          - else:
            - action: utter_unknown_replacement_reason_handover
              next: END
      - id: start_replacement
        action: utter_will_cancel_and_send_new
      - action: utter_new_card_has_been_ordered"""
    )
    assert (
        handler.domain
        == """version: '3.1'
responses:
  utter_form_completed:
  - text: 'First Name: {first_name}, Age: {age}'
  utter_ask_age:
  - text: how old are u?
  utter_ask_first_name:
  - text: What is ur name?
  utter_hi_first_name:
  - text: hi {first_name}
slots:
  stargazers_count:
    type: text
    mappings:
    - type: from_text
  first_name:
    type: text
    mappings:
    - type: from_entity
      entity: first_name
      intent: inform
      conditions:
      - active_loop: personal_details_form
        requested_slot: first_name
    - type: from_text
      intent: inform
      conditions:
      - active_loop: personal_details_form
        requested_slot: first_name
  age:
    type: float
    mappings:
    - type: from_entity
      entity: age
      intent: inform
      conditions:
      - active_loop: personal_details_form
        requested_slot: age
    - type: from_text
      intent: inform
      conditions:
      - active_loop: personal_details_form
        requested_slot: age
actions:
- action_example
- action_rasa_stargazers_count"""
    )
    assert handler.has_flows()

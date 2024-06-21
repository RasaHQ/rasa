import json
from abc import ABC
from typing import List, Optional, Dict, Any, Text, Union
from unittest.mock import MagicMock

import grpc
import pytest
import structlog
from google.protobuf.json_format import Parse

from pytest import MonkeyPatch
from rasa_sdk.grpc_errors import ResourceNotFound, ResourceNotFoundType
from rasa_sdk.grpc_py import action_webhook_pb2

from rasa.core.actions.action_exceptions import DomainNotFound
from rasa.core.actions.constants import (
    SSL_CLIENT_CERT_FIELD,
    SSL_CLIENT_KEY_FIELD,
)
from rasa.core.actions.custom_action_executor import CustomActionRequestWriter
from rasa.core.actions.grpc_custom_action_executor import GRPCCustomActionExecutor
from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import (
    FloatSlot,
    BooleanSlot,
    CategoricalSlot,
    ListSlot,
    AnySlot,
    Slot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig
from tests.utilities import filter_logs


def create_domain_yaml() -> str:
    return """
version: "3.1"

intents:
  - greet
  - withdraw
  - check_balance
  - goodbye

slots:
  account_type:
    type: categorical
    values:
      - primary
      - secondary
    mappings:
      - type: custom
  can_withdraw:
    type: bool
    mappings:
      - type: custom
  balance:
    type: float
    mappings:
      - type: custom

responses:
  utter_withdraw:
    - text: "You are not allowed to withdraw any amounts. Please check permission."
      condition:
        - type: slot
          name: can_withdraw
          value: False
    - text: "Withdrawal has been approved."
      condition:
        - type: slot
          name: can_withdraw
          value: True
        - type: slot
          name: account_type
          value: primary
    - text: "Withdrawal was sent for approval to primary account holder."
      condition:
        - type: slot
          name: account_type
          value: secondary
  utter_check_balance:
    - text: "As a primary account holder, \
    you can now set-up your access on mobile app too."
      condition:
        - type: slot
          name: account_type
          value: primary
      channel: os
    - text: "Welcome to your app account overview."
      condition:
        - type: slot
          name: account_type
          value: primary
      channel: app
    """


def action_endpoint() -> EndpointConfig:
    """Create an action endpoint."""
    return EndpointConfig(url="http://localhost:5055", enable_selective_domain=False)


# Note: this is a snapshot of the events from a real conversation
def create_events_without_tuple() -> List[Dict[Text, Any]]:
    """Create a list of events from a real conversation."""
    return [
        {
            "event": "action",
            "timestamp": 1718964815.060891,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_session_start",
            "policy": None,
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "session_started",
            "timestamp": 1718964815.061358,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
        },
        {
            "event": "action",
            "timestamp": 1718964815.061561,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_listen",
            "policy": None,
            "confidence": None,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "user",
            "timestamp": 1718964816.098185,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "text": "Add contact",
            "parse_data": {
                "intent": {
                    "name": "add_contact",
                    "confidence": 0.8393586103804717,
                },
                "entities": [],
                "text": "Add contact",
                "message_id": "42d39043f2014b77ae779f048b08488e",
                "metadata": {},
                "commands": [
                    {"flow": "add_contact", "command": "start flow"},
                    {
                        "name": "route_session_to_calm",
                        "value": True,
                        "extractor": "LLM",
                        "command": "set slot",
                    },
                ],
                "text_tokens": [[0, 3], [4, 11]],
                "intent_ranking": [
                    {"name": "add_contact", "confidence": 0.8393586103804717},
                    {"name": "goodbye", "confidence": 0.05454142097988517},
                    {
                        "name": "remove_contact",
                        "confidence": 0.029512668719663977,
                    },
                    {"name": "stop", "confidence": 0.02608296239420464},
                    {"name": "greet", "confidence": 0.016733967031172997},
                    {"name": "deny", "confidence": 0.011564870141654937},
                    {"name": "ask_help", "confidence": 0.004680628135799852},
                    {"name": "affirm", "confidence": 0.003915224409168441},
                    {"name": "add_card", "confidence": 0.0024325929924862368},
                    {
                        "name": "correct_pizza_type",
                        "confidence": 0.0022302123570833634,
                    },
                ],
                "flows_from_semantic_search": [
                    ["list_contacts", 0.8434605002403259],
                    ["add_contact", 0.8285885453224182],
                    ["add_card", 0.8191986083984375],
                    ["remove_contact", 0.7622846364974976],
                    ["check_balance", 0.7404183745384216],
                    ["transaction_search", 0.7292126417160034],
                    ["order_pizza", 0.7165393829345703],
                    ["job_vacancies", 0.7154850959777832],
                    ["fill_pizza_order", 0.7014303207397461],
                    ["correct_order", 0.6807493567466736],
                    ["transfer_money", 0.6750742793083191],
                    ["replace_card", 0.6680017709732056],
                    ["setup_recurrent_payment", 0.6676250696182251],
                    ["verify_account", 0.6650057435035706],
                    ["check_portfolio", 0.6394549012184143],
                ],
                "flows_in_prompt": [
                    "transaction_search",
                    "transfer_money",
                    "check_balance",
                    "correct_order",
                    "add_contact",
                    "list_contacts",
                    "order_pizza",
                    "add_card",
                    "remove_contact",
                    "job_vacancies",
                    "replace_card",
                    "verify_account",
                    "fill_pizza_order",
                    "setup_recurrent_payment",
                    "check_portfolio",
                ],
            },
            "input_channel": "inspector",
            "message_id": "42d39043f2014b77ae779f048b08488e",
        },
        {
            "event": "slot",
            "timestamp": 1718964816.119426,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "flow_hashes",
            "value": {
                "add_card": "711ba960eb1175a1c925770aa431f0e8",
                "add_contact": "41ebfdb28bafbec4ae4a368b9ddc78c8",
                "authenticate_user": "a583571018323398c801af2eaf840010",
                "check_balance": "3f94456d6a0c9e6ab05dd15ee4d0c544",
                "check_portfolio": "de8890a3227c41a4c380f221a90df828",
                "list_contacts": "ab5fe5e344022299f96be76b067d2587",
                "order_pizza": "160fe8c75f6a0554b6005d7e3343f65d",
                "fill_pizza_order": "eaa3801d1f72d14f435b41629520a959",
                "auth_user": "828d22b43f3a85e4edf8b32b21a9749f",
                "use_membership_points": "2258d087ccf9522034c86b3d6bde5026",
                "correct_order": "c39b9804b0eedba659b1f7fedf61abc1",
                "job_vacancies": "46fb99222a76a20ce50d91077dc4505f",
                "pattern_correction": "c478e4db6ea1ddbea0d6062213a0b627",
                "pattern_chitchat": "34dd56a016bc01e672a7321716a5b5e3",
                "pattern_search": "898e27a29fcc35c00fcb65d1a5921677",
                "pattern_cancel_flow": "4d04451d613094f3ced64738c707e07d",
                "pattern_completed": "d362b3766210c7b191b8af441c9f89dc",
                "register_to_vote_in_california": "f6f79550613b1083cdc5ff5e3509ca45",
                "remove_contact": "959808bd21fe5a6b25e7edc26ad63a3b",
                "replace_card": "371f481ad37067c2043c6b05845e8d26",
                "replace_eligible_card": "675ab76aa42c441ed6bde1f2f5bc03a6",
                "setup_recurrent_payment": "ec9c1db6719b76fb7e69e8dbf03cdea9",
                "transaction_search": "454de372b0323bf52f87f374cec50168",
                "transfer_money": "05dc106e195d3731963f8bf392fc7dfa",
                "verify_account": "da6896734655c2c9ca7962dfd347753a",
                "pattern_cannot_handle": "b5daf3b86f5653ae753091e6d77c99eb",
                "pattern_clarification": "224a1fd1170c32b2d0be56ab7efb9534",
                "pattern_code_change": "29890a48b8343949e027b4c705a70659",
                "pattern_collect_information": "d9dd96fdad51563e8af1b18231724717",
                "pattern_continue_interrupted": "251484804d2001725b49eb811d0ca193",
                "pattern_human_handoff": "0f45c504f8d9955215ee2b1854e06d2a",
                "pattern_internal_error": "c1173139609c67add46bc5ef45c43368",
                "pattern_restart": "f618c5be91977add377fc51a059e8b49",
                "pattern_session_start": "9a3aa0a08e249877149ebcf722091660",
                "pattern_skip_question": "66b09b8938edd43d95f10b0dccc75c40",
            },
        },
        {
            "event": "slot",
            "timestamp": 1718964816.1194298,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "route_session_to_calm",
            "value": True,
        },
        {
            "event": "stack",
            "timestamp": 1718964816.1194339,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "add", "path": "/0", \
                "value": {"frame_id": "OQIATOZE", "flow_id": "add_contact", \
                    "step_id": "START", "frame_type": "regular", "type": "flow"}}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964816.170923,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/0/step_id",\
             "value": "0_collect_add_contact_handle"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964816.171048,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "add", "path": "/1", \
               "value": {"frame_id": "OA7WJAJM", "flow_id": \
                "pattern_collect_information", "step_id": "START", \
                "collect": "add_contact_handle", \
                "utter": "utter_ask_add_contact_handle", \
                "collect_action": "action_ask_add_contact_handle", \
                "rejections": [], "type": "pattern_collect_information"}}]',
        },
        {
            "event": "flow_started",
            "timestamp": 1718964816.1711822,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "add_contact",
        },
        {
            "event": "stack",
            "timestamp": 1718964816.1711848,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", "value": "start"}]',
        },
        {
            "event": "flow_started",
            "timestamp": 1718964816.1713731,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_collect_information",
        },
        {
            "event": "user_featurization",
            "timestamp": 1718964816.171375,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "use_text_for_featurization": False,
        },
        {
            "event": "action",
            "timestamp": 1718964816.17139,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_run_slot_rejections",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "stack",
            "timestamp": 1718964816.2111812,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
              "value": "1_validate_{{context.collect}}"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964816.211367,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
            "value": "ask_collect"}]',
        },
        {
            "event": "action",
            "timestamp": 1718964816.2115479,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "utter_ask_add_contact_handle",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "bot",
            "timestamp": 1718964816.211714,
            "metadata": {
                "utter_action": "utter_ask_add_contact_handle",
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "text": "What's the handle of the user you want to add?",
            "data": {
                "elements": None,
                "quick_replies": None,
                "buttons": None,
                "attachment": None,
                "image": None,
                "custom": None,
            },
        },
        {
            "event": "stack",
            "timestamp": 1718964816.245588,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
              "value": "3_{{context.collect_action}}"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964816.245774,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
               "value": "4_action_listen"}]',
        },
        {
            "event": "action",
            "timestamp": 1718964816.246057,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_listen",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "user",
            "timestamp": 1718964820.462629,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "text": "Handle is John",
            "parse_data": {
                "intent": {"name": "inform", "confidence": 0.31622857984272357},
                "entities": [],
                "text": "Handle is John",
                "message_id": "e62dff9943e94e20997100bdea937ca9",
                "metadata": {},
                "commands": [
                    {
                        "name": "add_contact_handle",
                        "value": "John",
                        "extractor": "LLM",
                        "command": "set slot",
                    },
                    {
                        "name": "route_session_to_calm",
                        "value": True,
                        "extractor": "LLM",
                        "command": "set slot",
                    },
                ],
                "text_tokens": [[0, 6], [7, 9], [10, 14]],
                "intent_ranking": [
                    {"name": "inform", "confidence": 0.31622857984272357},
                    {"name": "greet", "confidence": 0.24710447198357743},
                    {"name": "ask_help", "confidence": 0.2088005503903839},
                    {"name": "goodbye", "confidence": 0.056003064787953825},
                    {"name": "deny", "confidence": 0.04022587775178235},
                    {"name": "stop", "confidence": 0.026536392848407767},
                    {"name": "affirm", "confidence": 0.021056469242458468},
                    {
                        "name": "hotel_search",
                        "confidence": 0.017175447752648284,
                    },
                    {
                        "name": "list_restaurants",
                        "confidence": 0.0077276788167619905,
                    },
                    {
                        "name": "ask_availability",
                        "confidence": 0.006542979503157815,
                    },
                ],
                "flows_from_semantic_search": [
                    ["list_contacts", 0.7548446655273438],
                    ["check_balance", 0.7474148869514465],
                    ["add_card", 0.7430962324142456],
                    ["job_vacancies", 0.7250053882598877],
                    ["remove_contact", 0.7116490602493286],
                    ["add_contact", 0.7054607272148132],
                    ["transaction_search", 0.7040430307388306],
                    ["order_pizza", 0.671093761920929],
                    ["fill_pizza_order", 0.6538543105125427],
                    ["transfer_money", 0.6505253314971924],
                    ["correct_order", 0.6487415432929993],
                    ["replace_card", 0.6446723937988281],
                    ["setup_recurrent_payment", 0.6306261420249939],
                    ["verify_account", 0.6289434432983398],
                    ["check_portfolio", 0.618802547454834],
                ],
                "flows_in_prompt": [
                    "transaction_search",
                    "transfer_money",
                    "check_balance",
                    "correct_order",
                    "add_contact",
                    "list_contacts",
                    "order_pizza",
                    "add_card",
                    "remove_contact",
                    "job_vacancies",
                    "replace_card",
                    "verify_account",
                    "fill_pizza_order",
                    "setup_recurrent_payment",
                    "check_portfolio",
                ],
            },
            "input_channel": "inspector",
            "message_id": "e62dff9943e94e20997100bdea937ca9",
        },
        {
            "event": "slot",
            "timestamp": 1718964820.490399,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "add_contact_handle",
            "value": "John",
        },
        {
            "event": "stack",
            "timestamp": 1718964820.5268621,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", "value": "start"}]',
        },
        {
            "event": "flow_started",
            "timestamp": 1718964820.527052,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_collect_information",
        },
        {
            "event": "user_featurization",
            "timestamp": 1718964820.527054,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "use_text_for_featurization": False,
        },
        {
            "event": "action",
            "timestamp": 1718964820.527065,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_run_slot_rejections",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "stack",
            "timestamp": 1718964820.56412,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
              "value": "1_validate_{{context.collect}}"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964820.564302,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", "value": "END"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964820.5644732,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "remove", "path": "/1"}]',
        },
        {
            "event": "flow_completed",
            "timestamp": 1718964820.564627,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_collect_information",
            "step_id": "1_validate_{{context.collect}}",
        },
        {
            "event": "stack",
            "timestamp": 1718964820.564629,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/0/step_id", \
              "value": "1_collect_add_contact_name"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964820.564738,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "add", "path": "/1", \
              "value": {"frame_id": "5JPYBPBP", \
                "flow_id": "pattern_collect_information", \
                "step_id": "START", \
                "collect": "add_contact_name", \
                "utter": "utter_ask_add_contact_name", \
                "collect_action": "action_ask_add_contact_name", \
                "rejections": [], "type": "pattern_collect_information"}}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964820.564852,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", "value": "start"}]',
        },
        {
            "event": "flow_started",
            "timestamp": 1718964820.5650198,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_collect_information",
        },
        {
            "event": "action",
            "timestamp": 1718964820.5650299,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_run_slot_rejections",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "stack",
            "timestamp": 1718964820.601402,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
               "value": "1_validate_{{context.collect}}"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964820.6015801,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
              "value": "ask_collect"}]',
        },
        {
            "event": "action",
            "timestamp": 1718964820.601758,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "utter_ask_add_contact_name",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "bot",
            "timestamp": 1718964820.601942,
            "metadata": {
                "utter_action": "utter_ask_add_contact_name",
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "text": "What's the name of the user you want to add?",
            "data": {
                "elements": None,
                "quick_replies": None,
                "buttons": None,
                "attachment": None,
                "image": None,
                "custom": None,
            },
        },
        {
            "event": "stack",
            "timestamp": 1718964820.638693,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
                "value": "3_{{context.collect_action}}"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964820.638905,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
               "value": "4_action_listen"}]',
        },
        {
            "event": "action",
            "timestamp": 1718964820.6390898,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_listen",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "user",
            "timestamp": 1718964833.548838,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "text": "Name is Uros",
            "parse_data": {
                "intent": {"name": "inform", "confidence": 0.5595198353416092},
                "entities": [],
                "text": "Name is Uros",
                "message_id": "affac3c448eb4dfeaacf19962c3cf1c6",
                "metadata": {},
                "commands": [
                    {
                        "name": "add_contact_name",
                        "value": "Uros",
                        "extractor": "LLM",
                        "command": "set slot",
                    },
                    {
                        "name": "route_session_to_calm",
                        "value": True,
                        "extractor": "LLM",
                        "command": "set slot",
                    },
                ],
                "text_tokens": [[0, 4], [5, 7], [8, 12]],
                "intent_ranking": [
                    {"name": "inform", "confidence": 0.5595198353416092},
                    {"name": "ask_help", "confidence": 0.07485666244373927},
                    {"name": "goodbye", "confidence": 0.06674381418198033},
                    {"name": "deny", "confidence": 0.06512897211706174},
                    {"name": "affirm", "confidence": 0.06104158044696839},
                    {"name": "greet", "confidence": 0.0410453666668231},
                    {
                        "name": "list_restaurants",
                        "confidence": 0.02241958481514177,
                    },
                    {"name": "stop", "confidence": 0.017850497952076536},
                    {"name": "hotel_search", "confidence": 0.01041746630298052},
                    {
                        "name": "replace_card",
                        "confidence": 0.010351664589848293,
                    },
                ],
                "flows_from_semantic_search": [
                    ["list_contacts", 0.7493193745613098],
                    ["add_card", 0.7339915633201599],
                    ["check_balance", 0.7316513061523438],
                    ["job_vacancies", 0.7071036696434021],
                    ["transaction_search", 0.6897743940353394],
                    ["order_pizza", 0.6720184087753296],
                    ["remove_contact", 0.6572635173797607],
                    ["add_contact", 0.6564961671829224],
                    ["transfer_money", 0.6542760729789734],
                    ["correct_order", 0.6447532176971436],
                    ["setup_recurrent_payment", 0.6442460417747498],
                    ["fill_pizza_order", 0.6435969471931458],
                    ["replace_card", 0.6376330852508545],
                    ["check_portfolio", 0.637299656867981],
                    ["verify_account", 0.6303889751434326],
                ],
                "flows_in_prompt": [
                    "transaction_search",
                    "transfer_money",
                    "check_balance",
                    "correct_order",
                    "add_contact",
                    "list_contacts",
                    "order_pizza",
                    "add_card",
                    "remove_contact",
                    "job_vacancies",
                    "replace_card",
                    "verify_account",
                    "fill_pizza_order",
                    "setup_recurrent_payment",
                    "check_portfolio",
                ],
            },
            "input_channel": "inspector",
            "message_id": "affac3c448eb4dfeaacf19962c3cf1c6",
        },
        {
            "event": "slot",
            "timestamp": 1718964833.5864031,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "add_contact_name",
            "value": "Uros",
        },
        {
            "event": "stack",
            "timestamp": 1718964833.622764,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", "value": "start"}]',
        },
        {
            "event": "flow_started",
            "timestamp": 1718964833.6229482,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_collect_information",
        },
        {
            "event": "user_featurization",
            "timestamp": 1718964833.6229498,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "use_text_for_featurization": False,
        },
        {
            "event": "action",
            "timestamp": 1718964833.62296,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_run_slot_rejections",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "stack",
            "timestamp": 1718964833.6633458,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
              "value": "1_validate_{{context.collect}}"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964833.663525,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", "value": "END"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964833.6636941,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "remove", "path": "/1"}]',
        },
        {
            "event": "flow_completed",
            "timestamp": 1718964833.663847,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_collect_information",
            "step_id": "1_validate_{{context.collect}}",
        },
        {
            "event": "stack",
            "timestamp": 1718964833.6638489,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/0/step_id", \
              "value": "2_collect_add_contact_confirmation"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964833.6639512,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "add", "path": "/1", \
              "value": {"frame_id": "XFD4VOQC", \
                "flow_id": "pattern_collect_information", \
                "step_id": "START", \
                "collect": "add_contact_confirmation", \
                "utter": "utter_ask_add_contact_confirmation", \
                "collect_action": "action_ask_add_contact_confirmation", \
                "rejections": [], "type": "pattern_collect_information"}}]',
        },
        {
            "event": "slot",
            "timestamp": 1718964833.664062,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "add_contact_confirmation",
            "value": None,
        },
        {
            "event": "stack",
            "timestamp": 1718964833.664066,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", "value": "start"}]',
        },
        {
            "event": "flow_started",
            "timestamp": 1718964833.664232,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_collect_information",
        },
        {
            "event": "action",
            "timestamp": 1718964833.6642408,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_run_slot_rejections",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "stack",
            "timestamp": 1718964833.701149,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
              "value": "1_validate_{{context.collect}}"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964833.701329,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
                "value": "ask_collect"}]',
        },
        {
            "event": "action",
            "timestamp": 1718964833.701508,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "utter_ask_add_contact_confirmation",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "bot",
            "timestamp": 1718964833.7016902,
            "metadata": {
                "utter_action": "utter_ask_add_contact_confirmation",
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "text": "Do you want to add Uros(John) to your contacts?",
            "data": {
                "elements": None,
                "quick_replies": None,
                "buttons": [
                    {"payload": "yes", "title": "Yes"},
                    {"payload": "no", "title": "No, cancel"},
                ],
                "attachment": None,
                "image": None,
                "custom": None,
            },
        },
        {
            "event": "stack",
            "timestamp": 1718964833.742558,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
              "value": "3_{{context.collect_action}}"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964833.742759,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
            "value": "4_action_listen"}]',
        },
        {
            "event": "action",
            "timestamp": 1718964833.742943,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_listen",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
    ]


def create_events_with_tuple() -> List[Dict[Text, Any]]:
    return [
        {
            "event": "user",
            "timestamp": 1718964835.954178,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "text": "yes",
            "parse_data": {
                "intent": {"name": "affirm", "confidence": 0.9143817369918581},
                "entities": [],
                "text": "yes",
                "message_id": "82f2f706079b424e8d5f2c7c9bb15c56",
                "metadata": {},
                "commands": [
                    {
                        "name": "add_contact_confirmation",
                        "value": "True",
                        "extractor": "LLM",
                        "command": "set slot",
                    },
                    {
                        "name": "route_session_to_calm",
                        "value": True,
                        "extractor": "LLM",
                        "command": "set slot",
                    },
                ],
                "text_tokens": [(0, 3)],
                "intent_ranking": [
                    {"name": "affirm", "confidence": 0.9143817369918581},
                    {"name": "goodbye", "confidence": 0.026915233965718207},
                    {"name": "greet", "confidence": 0.025586624364794926},
                    {"name": "ask_help", "confidence": 0.009618884391661083},
                    {"name": "inform", "confidence": 0.007110499664667112},
                    {"name": "deny", "confidence": 0.00553821784984332},
                    {"name": "stop", "confidence": 0.004175694352318176},
                    {
                        "name": "inform_num_pizza",
                        "confidence": 0.0012075836571628319,
                    },
                    {
                        "name": "list_restaurants",
                        "confidence": 0.0008876769015624743,
                    },
                    {
                        "name": "hotel_search",
                        "confidence": 0.0005172187982708012,
                    },
                ],
                "flows_from_semantic_search": [
                    ["list_contacts", 0.7562662363052368],
                    ["add_card", 0.7465101480484009],
                    ["check_balance", 0.7431129813194275],
                    ["job_vacancies", 0.7041592001914978],
                    ["transaction_search", 0.7029867768287659],
                    ["order_pizza", 0.6690036654472351],
                    ["correct_order", 0.6670126914978027],
                    ["fill_pizza_order", 0.6658919453620911],
                    ["add_contact", 0.6644800901412964],
                    ["transfer_money", 0.6598181128501892],
                    ["setup_recurrent_payment", 0.6578066945075989],
                    ["remove_contact", 0.6576454043388367],
                    ["replace_card", 0.6563082933425903],
                    ["verify_account", 0.6510206460952759],
                    ["check_portfolio", 0.6289162039756775],
                ],
                "flows_in_prompt": [
                    "transaction_search",
                    "transfer_money",
                    "check_balance",
                    "correct_order",
                    "add_contact",
                    "list_contacts",
                    "order_pizza",
                    "add_card",
                    "remove_contact",
                    "job_vacancies",
                    "replace_card",
                    "verify_account",
                    "fill_pizza_order",
                    "setup_recurrent_payment",
                    "check_portfolio",
                ],
            },
            "input_channel": "inspector",
            "message_id": "82f2f706079b424e8d5f2c7c9bb15c56",
        },
        {
            "event": "slot",
            "timestamp": 1718964835.996863,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "add_contact_confirmation",
            "value": True,
        },
        {
            "event": "stack",
            "timestamp": 1718964836.0416481,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", "value": "start"}]',
        },
        {
            "event": "flow_started",
            "timestamp": 1718964836.041831,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_collect_information",
        },
        {
            "event": "user_featurization",
            "timestamp": 1718964836.041833,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "use_text_for_featurization": False,
        },
        {
            "event": "action",
            "timestamp": 1718964836.041845,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_run_slot_rejections",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "stack",
            "timestamp": 1718964840.478105,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", \
                  "value": "1_validate_{{context.collect}}"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964840.478573,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/1/step_id", "value": "END"}]',
        },
        {
            "event": "stack",
            "timestamp": 1718964840.478862,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "remove", "path": "/1"}]',
        },
        {
            "event": "flow_completed",
            "timestamp": 1718964840.479121,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_collect_information",
            "step_id": "1_validate_{{context.collect}}",
        },
        {
            "event": "stack",
            "timestamp": 1718964840.479126,
            "metadata": {
                "model_id": "14637f7d3d15487b9de4d21663c8797b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/0/step_id", \
                  "value": "add_contact"}]',
        },
    ]


def create_float_slot() -> FloatSlot:
    """Return float slot."""
    return FloatSlot(
        name="float_slot",
        mappings=[
            {
                "a": 1.0,
            }
        ],
        initial_value=2.0,
        value_reset_delay=3,
        max_value=1.0,
        min_value=0.0,
        influence_conversation=True,
        is_builtin=False,
        shared_for_coexistence=False,
    )


def create_boolean_slot() -> BooleanSlot:
    """Return boolean slot."""
    return BooleanSlot(
        name="test2",
        mappings=[{}],
        initial_value=False,
        influence_conversation=False,
    )


def create_categorical_slot() -> CategoricalSlot:
    """Return categorical slot."""
    return CategoricalSlot(
        name="test3",
        mappings=[{}],
        initial_value="test",
        influence_conversation=False,
    )


def create_list_slot() -> ListSlot:
    """Return list slot."""
    return ListSlot("test", mappings=[{}], influence_conversation=False)


def create_any_slot() -> AnySlot:
    """Return any slot."""
    return AnySlot(
        name="account", mappings=[{}], initial_value=None, influence_conversation=False
    )


def create_slots() -> List[Slot]:
    """Return list of various slots."""
    return [
        create_float_slot(),
        create_categorical_slot(),
        create_boolean_slot(),
        create_list_slot(),
        create_any_slot(),
    ]


# Note: this is a representation of the tracker which resembles the real world scenario
# where the tracker has multiple events and slots.
# Its main purpose is to test the grpc request creation.
def create_tracker_with_tuple() -> DialogueStateTracker:
    """Return dialogue state tracker constructed from events and slots."""
    return DialogueStateTracker.from_dict(
        sender_id="test_sender_id",
        events_as_dict=create_events_without_tuple() + create_events_with_tuple(),
        slots=create_slots(),
        max_event_history=5,
    )


# Note: this is a representation of the tracker which resembles the real world scenario
# where the tracker has multiple events and slots.
# Its main purpose is to test the grpc request creation.
@pytest.fixture
def tracker_with_tuple() -> DialogueStateTracker:
    """Return dialogue state tracker constructed from events and slots."""
    return create_tracker_with_tuple()


# Note: this is a representation of the tracker which resembles the real world scenario
# where the tracker has multiple events and slots.
# Its main purpose is to test the grpc request creation.
def create_tracker_without_tuple() -> DialogueStateTracker:
    """Return dialogue state tracker constructed from events and slots."""
    return DialogueStateTracker.from_dict(
        sender_id="test_sender_id",
        events_as_dict=create_events_without_tuple(),
        slots=create_slots(),
        max_event_history=5,
    )


# Note: this is a representation of the tracker which resembles the real world scenario
# where the tracker has multiple events and slots.
# Its main purpose is to test the grpc request creation.
@pytest.fixture
def tracker_without_tuple() -> DialogueStateTracker:
    """Return dialogue state tracker constructed from events and slots."""
    return create_tracker_without_tuple()


def create_domain() -> Domain:
    """Return domain from YAML string."""
    return Domain.from_yaml(create_domain_yaml())


@pytest.fixture
def domain() -> Domain:
    """Return domain from YAML string."""
    return create_domain()


@pytest.fixture
def grpc_url() -> str:
    """Return grpc url."""
    return "grpc://localhost:5055"


def action_name() -> str:
    """Return action name."""
    return "action_test"


def create_action_request_writer() -> CustomActionRequestWriter:
    """Return request writer."""
    return CustomActionRequestWriter(action_name(), action_endpoint())


@pytest.fixture
def grpc_custom_action_executor() -> GRPCCustomActionExecutor:
    """Return grpc custom action executor."""
    return GRPCCustomActionExecutor(action_name(), action_endpoint())


@pytest.fixture
def grpc_insecure_channel_result() -> MagicMock:
    """Return grpc insecure channel result."""
    return MagicMock()


@pytest.fixture
def grpc_insecure_channel(
    monkeypatch: MonkeyPatch, grpc_insecure_channel_result: MagicMock
) -> MagicMock:
    """Mock grpc.insecure channel."""
    _grpc_insecure_channel = MagicMock()
    _grpc_insecure_channel.return_value = grpc_insecure_channel_result
    monkeypatch.setattr(
        "grpc.insecure_channel",
        _grpc_insecure_channel,
    )
    return _grpc_insecure_channel


@pytest.fixture
def grpc_secure_channel(monkeypatch: MonkeyPatch) -> MagicMock:
    """Mock grpc.secure channel."""
    _grpc_secure_channel = MagicMock()
    monkeypatch.setattr(
        "grpc.secure_channel",
        _grpc_secure_channel,
    )
    return _grpc_secure_channel


@pytest.fixture
def grpc_client() -> MagicMock:
    """Return grpc client."""
    return MagicMock()


@pytest.fixture
def grpc_action_servicer_stub(
    monkeypatch: MonkeyPatch, grpc_client: MagicMock
) -> MagicMock:
    """Return grpc action servicer stub."""
    _grpc_action_servicer_stub = MagicMock()
    _grpc_action_servicer_stub.return_value = grpc_client
    monkeypatch.setattr(
        "rasa.core.actions.grpc_custom_action_executor.action_webhook_pb2_grpc.ActionServiceStub",
        _grpc_action_servicer_stub,
    )
    return _grpc_action_servicer_stub


@pytest.fixture
def mock_ssl_credentials(monkeypatch: MonkeyPatch) -> MagicMock:
    """Mock ssl credentials."""
    return MagicMock()


@pytest.fixture
def grpc_ssl_channel_credentials(
    monkeypatch: MonkeyPatch,
    mock_ssl_credentials: MagicMock,
) -> MagicMock:
    _grpc_ssl_channel_credentials = MagicMock()
    monkeypatch.setattr(
        "grpc.ssl_channel_credentials",
        _grpc_ssl_channel_credentials,
    )
    _grpc_ssl_channel_credentials.return_value = mock_ssl_credentials
    return _grpc_ssl_channel_credentials


def create_grpc_payload(
    tracker: DialogueStateTracker,
) -> action_webhook_pb2.WebhookRequest:
    """Create grpc payload with given tracker."""
    request_writer = create_action_request_writer()

    json_payload = json.dumps(
        request_writer.create(tracker=tracker, domain=create_domain())
    )
    return Parse(
        text=json_payload,
        message=action_webhook_pb2.WebhookRequest(),
        ignore_unknown_fields=True,
    )


# We use this as general fixture for grpc payload
# because we just need the grpc payload
@pytest.fixture
def grpc_payload() -> action_webhook_pb2.WebhookRequest:
    """Fixture for grpc payload for tracker without tuple."""
    return create_grpc_payload(tracker=create_tracker_without_tuple())


@pytest.fixture
def mock_file_as_bytes(monkeypatch: MonkeyPatch) -> MagicMock:
    _file_as_bytes = MagicMock()
    monkeypatch.setattr(
        "rasa.core.actions.grpc_custom_action_executor.file_as_bytes",
        _file_as_bytes,
    )
    return _file_as_bytes


def test_create_grpc_insecure_channel(
    grpc_insecure_channel_result: MagicMock,
    grpc_insecure_channel: MagicMock,
    grpc_custom_action_executor: GRPCCustomActionExecutor,
) -> None:
    result = grpc_custom_action_executor._create_channel()

    assert result == grpc_insecure_channel_result
    grpc_insecure_channel.assert_called_once_with(
        target=grpc_custom_action_executor.request_url,
        compression=grpc.Compression.Gzip,
    )


@pytest.mark.parametrize(
    "endpoint_config, "
    "expected_root_certificate,"
    "expected_private_key, "
    "expected_certificate_chain",
    [
        (
            EndpointConfig.from_dict(
                {
                    "url": "grpc://localhost:5055",
                    "cafile": "cafile",
                    SSL_CLIENT_CERT_FIELD: "client_cert",
                    SSL_CLIENT_KEY_FIELD: "client_key",
                }
            ),
            b"cafile content",
            b"keyfile content",
            b"cert file content",
        ),
        (
            EndpointConfig.from_dict(
                {
                    "url": "grpc://localhost:5055",
                    "cafile": "cafile",
                    SSL_CLIENT_KEY_FIELD: "client_key",
                }
            ),
            b"cafile content",
            None,
            None,
        ),
        (
            EndpointConfig.from_dict(
                {
                    "url": "grpc://localhost:5055",
                    "cafile": "cafile",
                    SSL_CLIENT_CERT_FIELD: "client_cert",
                }
            ),
            b"cafile content",
            None,
            None,
        ),
        (
            EndpointConfig.from_dict(
                {
                    "url": "grpc://localhost:5055",
                    "cafile": "cafile",
                }
            ),
            b"cafile content",
            None,
            None,
        ),
    ],
)
def test_create_grpc_secure_channel_with_ca_cert(
    endpoint_config: EndpointConfig,
    expected_root_certificate: bytes,
    expected_private_key: Optional[bytes],
    expected_certificate_chain: Optional[bytes],
    grpc_secure_channel: MagicMock,
    mock_file_as_bytes: MagicMock,
    mock_ssl_credentials: MagicMock,
    grpc_ssl_channel_credentials: MagicMock,
) -> None:
    """Test create grpc secure channel with ca cert"""
    # Given
    mock_file_as_bytes.side_effect = [
        expected_root_certificate,
        expected_certificate_chain,
        expected_private_key,
    ]

    secure_channel = MagicMock()
    grpc_secure_channel.return_value = secure_channel

    grpc_custom_action_executor = GRPCCustomActionExecutor(
        "action_test", endpoint_config
    )

    # When
    result = grpc_custom_action_executor._create_channel()

    # Then
    assert result == secure_channel
    grpc_ssl_channel_credentials.assert_called_once_with(
        root_certificates=expected_root_certificate,
        private_key=expected_private_key,
        certificate_chain=expected_certificate_chain,
    )
    grpc_secure_channel.assert_called_once_with(
        target=grpc_custom_action_executor.request_url,
        credentials=mock_ssl_credentials,
        compression=grpc.Compression.Gzip,
    )


@pytest.mark.parametrize(
    "endpoint_config, expected_log_event, expected_log_message",
    [
        (
            EndpointConfig.from_dict(
                {
                    "url": "grpc://localhost:5055",
                    "cafile": "cafile",
                    SSL_CLIENT_KEY_FIELD: "client_key",
                }
            ),
            f"rasa.core.actions.grpc_custom_action_executor.{SSL_CLIENT_CERT_FIELD}_missing",
            f"Client key file '{SSL_CLIENT_KEY_FIELD}' is provided but "
            f"client certificate file '{SSL_CLIENT_CERT_FIELD}' "
            f"is not provided in the endpoint configuration. "
            f"Both fields are required for client TLS authentication."
            f"Continuing without client TLS authentication.",
        ),
        (
            EndpointConfig.from_dict(
                {
                    "url": "grpc://localhost:5055",
                    "cafile": "cafile",
                    SSL_CLIENT_CERT_FIELD: "client_cert",
                }
            ),
            f"rasa.core.actions.grpc_custom_action_executor.{SSL_CLIENT_KEY_FIELD}_missing",
            f"Client certificate file '{SSL_CLIENT_CERT_FIELD}' "
            f" is provided but client key file '{SSL_CLIENT_KEY_FIELD}'"
            f" is not provided in the endpoint configuration. "
            f"Both fields are required for client TLS authentication."
            f"Continuing without client TLS authentication.",
        ),
    ],
)
def test_test_grpc_custom_action_executor_logs_error_on_wrong_cert_configuration(
    endpoint_config: EndpointConfig,
    expected_log_event: str,
    expected_log_message: str,
    mock_file_as_bytes: MagicMock,
) -> None:
    """Test logging error on wrong cert configuration"""
    # Given
    mock_file_as_bytes.return_value = b"cafile content"

    with structlog.testing.capture_logs() as caplog:
        # When
        grpc_custom_action_executor = GRPCCustomActionExecutor(
            "action_test", endpoint_config
        )
        # Then

        assert grpc_custom_action_executor.client_key is None
        assert grpc_custom_action_executor.client_cert is None
        logs = filter_logs(
            caplog,
            event=expected_log_event,
            log_level="error",
            log_message_parts=[expected_log_message],
        )
        assert len(logs) == 1


@pytest.mark.parametrize(
    "dialogue_tracker, expected_warning_logs",
    [(create_tracker_with_tuple(), 1), (create_tracker_without_tuple(), 0)],
)
def test_grpc_custom_action_executor_create_payload(
    dialogue_tracker: DialogueStateTracker,
    expected_warning_logs: int,
    domain: Domain,
    grpc_custom_action_executor: GRPCCustomActionExecutor,
) -> None:
    with structlog.testing.capture_logs() as caplog:
        result_grpc_payload = grpc_custom_action_executor._create_payload(
            tracker=dialogue_tracker, domain=domain
        )

        expected_grpc_payload = create_grpc_payload(tracker=dialogue_tracker)

        assert result_grpc_payload == expected_grpc_payload
        logs = filter_logs(
            caplog,
            event=(
                "rasa.core.actions.grpc_custom_action_executor."
                "create_grpc_payload_from_dict_failed"
            ),
            log_level="warning",
            log_message_parts=[
                (
                    "Failed to create gRPC payload from Python dictionary. "
                    "Falling back to create payload from JSON intermediary."
                )
            ],
        )
        assert len(logs) == expected_warning_logs


class StubRpcError(grpc.RpcError):
    pass


class StubGrpcCallException(grpc.RpcError, grpc.Call, ABC):
    def __init__(self, code: grpc.StatusCode, details: str):
        self._code = code
        self._details = details

    def initial_metadata(self):
        return []

    def trailing_metadata(self):
        return []

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    def __str__(self) -> str:
        return f"{self.code}: {self.details}"


@pytest.fixture
def mock_grpc_call_exception(monkeypatch: MonkeyPatch) -> MagicMock:
    """Mock grpc call exception"""
    _grpc_call = MagicMock(spec=StubGrpcCallException)
    return _grpc_call


@pytest.mark.parametrize(
    "exception, expected_exception_message",
    [
        (
            StubRpcError(),
            f"Failed to execute custom action '{action_name()}'. "
            f"Unknown error occurred while calling the "
            "action server over gRPC protocol.",
        ),
        (
            StubGrpcCallException(grpc.StatusCode.UNKNOWN, "Unknown error"),
            f"Failed to execute custom action '{action_name()}'. "
            "Error: Unknown error",
        ),
    ],
)
@pytest.mark.usefixtures("grpc_insecure_channel", "grpc_action_servicer_stub")
def test_grpc_webhook_request_rpc_error(
    exception: Union[StubRpcError, StubGrpcCallException],
    expected_exception_message: str,
    grpc_client: MagicMock,
    grpc_custom_action_executor: GRPCCustomActionExecutor,
    grpc_payload: action_webhook_pb2.WebhookRequest,
) -> None:
    grpc_client.Webhook.side_effect = exception

    with pytest.raises(RasaException) as raised_exception:
        grpc_custom_action_executor._request(grpc_payload)
        assert expected_exception_message in str(raised_exception.value)


@pytest.mark.usefixtures("grpc_insecure_channel", "grpc_action_servicer_stub")
def test_grpc_webhook_request_domain_not_found_rpc_call_error(
    grpc_client: MagicMock,
    grpc_custom_action_executor: GRPCCustomActionExecutor,
    grpc_payload: action_webhook_pb2.WebhookRequest,
) -> None:
    resource_not_found = ResourceNotFound(
        action_name="action_name",
        message="message",
        resource_type=ResourceNotFoundType.DOMAIN,
    )
    details = resource_not_found.model_dump_json()

    grpc_client.Webhook.side_effect = StubGrpcCallException(
        grpc.StatusCode.NOT_FOUND, details
    )

    with pytest.raises(DomainNotFound):
        with structlog.testing.capture_logs() as caplog:
            grpc_custom_action_executor._request(grpc_payload)

            logs = filter_logs(
                caplog,
                event="rasa.core.actions.grpc_custom_action_executor.domain_not_found",
                log_level="error",
                log_message_parts=[
                    f"Failed to execute custom action "
                    f"'{grpc_custom_action_executor.action_endpoint}'. "
                    f"Could not find domain. {resource_not_found.message}"
                ],
            )
            assert len(logs) == 1


@pytest.mark.usefixtures("grpc_insecure_channel", "grpc_action_servicer_stub")
def test_grpc_webhook_request_action_not_found_rpc_call_error(
    grpc_client: MagicMock,
    grpc_custom_action_executor: GRPCCustomActionExecutor,
    grpc_payload: action_webhook_pb2.WebhookRequest,
) -> None:
    resource_not_found = ResourceNotFound(
        action_name="action_name",
        message="message",
        resource_type=ResourceNotFoundType.ACTION,
    )
    details = resource_not_found.model_dump_json()

    grpc_client.Webhook.side_effect = StubGrpcCallException(
        grpc.StatusCode.NOT_FOUND, details
    )

    with pytest.raises(RasaException) as exception:
        grpc_custom_action_executor._request(grpc_payload)
        assert (
            f"Failed to execute custom action "
            f"'{grpc_custom_action_executor.action_name}'. "
            f"Error: {details}" in str(exception.value)
        )


@pytest.mark.usefixtures("grpc_insecure_channel", "grpc_action_servicer_stub")
async def test_grpc_custom_action_executor_run(
    grpc_client: MagicMock,
    grpc_custom_action_executor: GRPCCustomActionExecutor,
    tracker_without_tuple: DialogueStateTracker,
    domain: Domain,
    grpc_payload: action_webhook_pb2.WebhookRequest,
) -> None:
    await grpc_custom_action_executor.run(tracker=tracker_without_tuple, domain=domain)

    grpc_client.Webhook.assert_called_once_with(grpc_payload)

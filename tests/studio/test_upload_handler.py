import argparse
import base64
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Set, Text, Union
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from rasa.shared.exceptions import RasaException

import rasa.studio.upload
from rasa.studio.config import StudioConfig
from rasa.studio.results_logger import ResultsLogger

CALM_DOMAIN_YAML = dedent(
    """\
    version: '3.1'
    actions:
    - action_add_contact
    - action_check_balance
    - action_execute_transfer
    - action_remove_contact
    - action_search_hotel
    - action_transaction_search
    - action_check_transfer_funds
    responses:
      utter_add_contact_cancelled:
      - text: Okay, I am cancelling this adding of a contact.
      utter_add_contact_error:
      - text: Something went wrong, please try again.
      utter_ca_income_insufficient:
      - text: Unfortunately, we cannot increase your transfer limits under these circumstances.
      utter_cant_advice_on_health:
      - text: I'm sorry, I can't give you advice on your health.
      utter_contact_added:
      - text: Contact added successfully.
      utter_contact_already_exists:
      - text: There's already a contact with that handle in your list.
      utter_contact_not_in_list:
      - text: That contact is not in your list.
      utter_current_balance:
      - text: You still have {current_balance} in your account.
      utter_hotel_inform_rating:
      - text: The {hotel_name} has an average rating of {hotel_average_rating}
      utter_remove_contact_cancelled:
      - text: Okay, I am cancelling this removal of a contact.
      utter_remove_contact_error:
      - text: Something went wrong, please try again.
      utter_remove_contact_success:
      - text: Removed {remove_contact_handle}({remove_contact_name}) from your contacts.
      utter_transactions:
      - text: 'Your current transactions are:  {transactions_list}'
      utter_verify_account_cancelled:
      - text: Cancelling account verification...
        metadata:
          rephrase: true
      utter_verify_account_success:
      - text: Your account was successfully verified
      utter_ask_add_contact_confirmation:
      - text: Do you want to add {add_contact_name}({add_contact_handle}) to your contacts?
      utter_ask_add_contact_handle:
      - text: What's the handle of the user you want to add?
      utter_ask_add_contact_name:
      - text: What's the name of the user you want to add?
        metadata:
          rephrase: true
      utter_ask_based_in_california:
      - text: Are you based in California?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
      utter_ask_remove_contact_confirmation:
      - text: Should I remove {remove_contact_handle} from your contact list?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
      utter_ask_remove_contact_handle:
      - text: What's the handle of the user you want to remove?
      utter_ask_verify_account_confirmation:
      - text: Your email address is {verify_account_email} and you are not based in California, correct?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
      utter_ask_verify_account_confirmation_california:
      - text: Your email address is {verify_account_email} and you are based in California with a yearly income exceeding 100,000$, correct?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
      utter_ask_verify_account_email:
      - text: What's your email address?
      utter_ask_verify_account_sufficient_california_income:
      - text: Does your yearly income exceed 100,000 USD?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
    slots:
      add_contact_confirmation:
        type: bool
        mappings:
        - type: custom
      add_contact_handle:
        type: text
        mappings:
        - type: custom
      add_contact_name:
        type: text
        mappings:
        - type: custom
      based_in_california:
        type: bool
        mappings:
        - type: custom
      current_balance:
        type: float
        mappings:
        - type: custom
      hotel_average_rating:
        type: float
        mappings:
        - type: custom
      hotel_name:
        type: text
        mappings:
        - type: custom
      remove_contact_confirmation:
        type: bool
        mappings:
        - type: custom
      remove_contact_handle:
        type: text
        mappings:
        - type: custom
      remove_contact_name:
        type: text
        mappings:
        - type: custom
      return_value:
        type: text
        mappings:
        - type: custom
      transactions_list:
        type: text
        mappings:
        - type: custom
      set_slots_test_text:
        type: text
        mappings:
        - type: custom
      set_slots_test_categorical:
        type: categorical
        mappings:
        - type: custom
        values:
        - value_1
        - value_2
      verify_account_confirmation:
        type: bool
        mappings:
        - type: custom
      verify_account_confirmation_california:
        type: bool
        mappings:
        - type: custom
      verify_account_email:
        type: text
        mappings:
        - type: custom
      verify_account_sufficient_california_income:
        type: bool
        mappings:
        - type: custom
    intents:
    - health_advice
    entities:
    - name
    - age
    session_config:
      session_expiration_time: 60
      carry_over_slots_to_new_session: true
    """  # noqa: E501
)


CALM_FLOWS_YAML = dedent(
    """\
    flows:
      health_advice:
        steps:
        - id: 0_utter_cant_advice_on_health
          next: END
          action: utter_cant_advice_on_health
        name: health advice
        description: user asks for health advice
        nlu_trigger:
        - intent:
            name: health_advice
            confidence_threshold: 0.8
      add_contact:
        steps:
        - id: 0_collect_add_contact_handle
          next: 1_collect_add_contact_name
          description: a user handle starting with @
          collect: add_contact_handle
          utter: utter_ask_add_contact_handle
          ask_before_filling: false
          reset_after_flow_ends: true
          rejections: []
        - id: 1_collect_add_contact_name
          next: 2_collect_add_contact_confirmation
          description: a name of a person
          collect: add_contact_name
          utter: utter_ask_add_contact_name
          ask_before_filling: false
          reset_after_flow_ends: true
          rejections: []
        - id: 2_collect_add_contact_confirmation
          next:
          - if: not slots.add_contact_confirmation
            then:
            - id: 3_utter_add_contact_cancelled
              next: END
              action: utter_add_contact_cancelled
          - else: action_add_contact
          description: a confirmation to add contact
          collect: add_contact_confirmation
          utter: utter_ask_add_contact_confirmation
          ask_before_filling: false
          reset_after_flow_ends: true
          rejections: []
        - id: action_add_contact
          next:
          - if: slots.return_value is 'already_exists'
            then:
            - id: 5_utter_contact_already_exists
              next: END
              action: utter_contact_already_exists
          - if: slots.return_value is 'success'
            then:
            - id: 6_utter_contact_added
              next: END
              action: utter_contact_added
          - else:
            - id: 7_utter_add_contact_error
              next: END
              action: utter_add_contact_error
          action: action_add_contact
        name: add_contact
        description: add a contact to your contact list
      check_balance:
        steps:
        - id: 0_action_check_balance
          next: 1_utter_current_balance
          action: action_check_balance
        - id: 1_utter_current_balance
          next: END
          action: utter_current_balance
        name: check_balance
        description: check the user's account balance.
      hotel_search:
        steps:
        - id: 0_action_search_hotel
          next: 1_utter_hotel_inform_rating
          action: action_search_hotel
        - id: 1_utter_hotel_inform_rating
          next: END
          action: utter_hotel_inform_rating
        name: hotel_search
        description: search for hotels
      remove_contact:
        steps:
        - id: 0_collect_remove_contact_handle
          next: 1_collect_remove_contact_confirmation
          description: a contact handle starting with @
          collect: remove_contact_handle
          utter: utter_ask_remove_contact_handle
          ask_before_filling: false
          reset_after_flow_ends: true
          rejections: []
        - id: 1_collect_remove_contact_confirmation
          next:
          - if: not slots.remove_contact_confirmation
            then:
            - id: 2_utter_remove_contact_cancelled
              next: END
              action: utter_remove_contact_cancelled
          - else: action_remove_contact
          collect: remove_contact_confirmation
          utter: utter_ask_remove_contact_confirmation
          ask_before_filling: true
          reset_after_flow_ends: true
          rejections: []
        - id: action_remove_contact
          next:
          - if: slots.return_value is 'not_found'
            then:
            - id: 4_utter_contact_not_in_list
              next: END
              action: utter_contact_not_in_list
          - if: slots.return_value is 'success'
            then:
            - id: 5_utter_remove_contact_success
              next: END
              action: utter_remove_contact_success
          - else:
            - id: 6_utter_remove_contact_error
              next: END
              action: utter_remove_contact_error
          action: action_remove_contact
        name: remove_contact
        description: remove a contact from your contact list
      transaction_search:
        steps:
        - id: 0_action_transaction_search
          next: 1_utter_transactions
          action: action_transaction_search
        - id: 1_utter_transactions
          next: END
          action: utter_transactions
        name: transaction_search
        description: lists the last transactions of the user account
      transfer_money:
        steps:
        - id: 0_collect_transfer_money_recipient
          next: 1_collect_transfer_money_amount_of_money
          description: Asks user for the recipient's name.
          collect: transfer_money_recipient
          utter: utter_ask_transfer_money_recipient
          ask_before_filling: false
          reset_after_flow_ends: true
          rejections: []
        - id: 1_collect_transfer_money_amount_of_money
          next: 2_action_check_transfer_funds
          description: Asks user for the amount to transfer.
          collect: transfer_money_amount_of_money
          utter: utter_ask_transfer_money_amount_of_money
          ask_before_filling: false
          reset_after_flow_ends: true
          rejections: []
        - id: 2_action_check_transfer_funds
          next:
          - if: not slots.transfer_money_has_sufficient_funds
            then:
            - id: 3_utter_transfer_money_insufficient_funds
              next: 4_set_slots
              action: utter_transfer_money_insufficient_funds
            - id: 4_set_slots
              next: END
              set_slots:
              - transfer_money_amount_of_money: null
              - transfer_money_has_sufficient_funds: null
              - set_slots_test_text: This is a test!
              - set_slots_test_categorical: value_1
          - else: collect_transfer_money_final_confirmation
          action: action_check_transfer_funds
        - id: collect_transfer_money_final_confirmation
          next:
          - if: not slots.transfer_money_final_confirmation
            then:
            - id: 6_utter_transfer_cancelled
              next: END
              action: utter_transfer_cancelled
          - else: action_execute_transfer
          description: Asks user for final confirmation to transfer money.
          collect: transfer_money_final_confirmation
          utter: utter_ask_transfer_money_final_confirmation
          ask_before_filling: true
          reset_after_flow_ends: true
          rejections: []
        - id: action_execute_transfer
          next:
          - if: slots.transfer_money_transfer_successful
            then:
            - id: 8_utter_transfer_complete
              next: END
              action: utter_transfer_complete
          - else:
            - id: 9_utter_transfer_failed
              next: END
              action: utter_transfer_failed
          action: action_execute_transfer
        name: transfer_money
        description: This flow let's users send money to friends and family.
      verify_account:
        steps:
        - id: 0_collect_verify_account_email
          next: 1_collect_based_in_california
          description: Asks user for their email address.
          collect: verify_account_email
          utter: utter_ask_verify_account_email
          ask_before_filling: true
          reset_after_flow_ends: true
          rejections: []
        - id: 1_collect_based_in_california
          next:
          - if: slots.based_in_california
            then:
            - id: 2_collect_verify_account_sufficient_california_income
              next:
              - if: not slots.verify_account_sufficient_california_income
                then:
                - id: 3_utter_ca_income_insufficient
                  next: END
                  action: utter_ca_income_insufficient
              - else: collect_verify_account_confirmation_california
              description: Asks user if they have sufficient income in California.
              collect: verify_account_sufficient_california_income
              utter: utter_ask_verify_account_sufficient_california_income
              ask_before_filling: true
              reset_after_flow_ends: true
              rejections: []
            - id: collect_verify_account_confirmation_california
              next:
              - if: slots.verify_account_confirmation_california
                then:
                - id: 5_utter_verify_account_success
                  next: END
                  action: utter_verify_account_success
              - else:
                - id: 6_utter_verify_account_cancelled
                  next: END
                  action: utter_verify_account_cancelled
              description: Asks user for final confirmation to verify their account in California.
              collect: verify_account_confirmation_california
              utter: utter_ask_verify_account_confirmation_california
              ask_before_filling: true
              reset_after_flow_ends: true
              rejections: []
          - else: collect_verify_account_confirmation
          description: Asks user if they are based in California.
          collect: based_in_california
          utter: utter_ask_based_in_california
          ask_before_filling: true
          reset_after_flow_ends: true
          rejections: []
        - id: collect_verify_account_confirmation
          next:
          - if: slots.verify_account_confirmation
            then:
            - id: 8_utter_verify_account_success
              next: END
              action: utter_verify_account_success
          - else:
            - id: 9_utter_verify_account_cancelled
              next: END
              action: utter_verify_account_cancelled
          description: Asks user for final confirmation to verify their account.
          collect: verify_account_confirmation
          utter: utter_ask_verify_account_confirmation
          ask_before_filling: true
          reset_after_flow_ends: true
          rejections: []
        name: verify_account
        description: Verify an account for higher transfer limits
    """  # noqa: E501
)

CALM_NLU_YAML = dedent(
    """\
    version: "3.1"
    nlu:
    - intent: health_advice
      examples: |
        - I need some medical advice.
        - Can you help me with some health issues?
        - I need medical support.
        - I'm experiencing some symptoms and I need guidance on what to do.
        - Can you provide me with health recommendations?
        - I'm struggling with some health concerns. Can you offer advice?
        - Can you suggest ways to improve my overall well-being?
        - I'm looking for tips on managing stress and anxiety. Any advice?
        - I have a specific health question. Can you offer me some insights?
        - I need suggestions on maintaining a healthy diet and exercise routine.
        - Is there anyone knowledgeable about natural remedies who can give me advice?
        - Can you provide me with information on preventing common illnesses?
        - I'm interested in learning about alternative therapies. Can you share your expertise?
        - Can you recommend a good doctor? I'm not feeling well.
    responses:
      utter_transfer_money_insufficient_funds:
      - text: You don't have so much money on your account!
      utter_transfer_failed:
      - text: something went wrong transferring the money.
      utter_out_of_scope:
      - text: Sorry, I'm not sure how to respond to that. Type "help" for assistance.
      utter_ask_transfer_money_amount_of_money:
      - text: How much money do you want to transfer?
      utter_ask_transfer_money_recipient:
      - text: Who do you want to transfer money to?
      utter_transfer_complete:
      - text: Successfully transferred {transfer_money_amount_of_money} to {transfer_money_recipient}.
      utter_transfer_cancelled:
      - text: Transfer cancelled.
      utter_ask_transfer_money_final_confirmation:
      - buttons:
        - payload: yes
          title: Yes
        - payload: no
          title: No, cancel the transaction
        text: Would you like to transfer {transfer_money_amount_of_money} to {transfer_money_recipient}?
      utter_add_contact_cancelled:
      - text: Okay, I am cancelling this adding of a contact.
      utter_add_contact_error:
      - text: Something went wrong, please try again.
      utter_ca_income_insufficient:
      - text: Unfortunately, we cannot increase your transfer limits under these circumstances.
      utter_cant_advice_on_health:
      - text: I'm sorry, I can't give you advice on your health.
      utter_contact_added:
      - text: Contact added successfully.
      utter_contact_already_exists:
      - text: There's already a contact with that handle in your list.
      utter_contact_not_in_list:
      - text: That contact is not in your list.
      utter_current_balance:
      - text: You still have {current_balance} in your account.
      utter_hotel_inform_rating:
      - text: The {hotel_name} has an average rating of {hotel_average_rating}
      utter_remove_contact_cancelled:
      - text: Okay, I am cancelling this removal of a contact.
      utter_remove_contact_error:
      - text: Something went wrong, please try again.
      utter_remove_contact_success:
      - text: Removed {remove_contact_handle}({remove_contact_name}) from your contacts.
      utter_transactions:
      - text: 'Your current transactions are:  {transactions_list}'
      utter_verify_account_cancelled:
      - text: Cancelling account verification...
        metadata:
          rephrase: true
      utter_verify_account_success:
      - text: Your account was successfully verified
      utter_ask_add_contact_confirmation:
      - text: Do you want to add {add_contact_name}({add_contact_handle}) to your contacts?
      utter_ask_add_contact_handle:
      - text: What's the handle of the user you want to add?
      utter_ask_add_contact_name:
      - text: What's the name of the user you want to add?
        metadata:
          rephrase: true
      utter_ask_based_in_california:
      - text: Are you based in California?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
      utter_ask_remove_contact_confirmation:
      - text: Should I remove {remove_contact_handle} from your contact list?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
      utter_ask_remove_contact_handle:
      - text: What's the handle of the user you want to remove?
      utter_ask_verify_account_confirmation:
      - text: Your email address is {verify_account_email} and you are not based in California, correct?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
      utter_ask_verify_account_confirmation_california:
      - text: Your email address is {verify_account_email} and you are based in California with a yearly income exceeding 100,000$, correct?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
      utter_ask_verify_account_email:
      - text: What's your email address?
      utter_ask_verify_account_sufficient_california_income:
      - text: Does your yearly income exceed 100,000 USD?
        buttons:
        - title: Yes
          payload: Yes
        - title: No
          payload: No
    """  # noqa: E501
)


def encode_yaml(yaml):
    return base64.b64encode(yaml.encode("utf-8")).decode("utf-8")


@pytest.mark.parametrize(
    "args, endpoint, expected",
    [
        (
            argparse.Namespace(
                assistant_name=["test"],
                domain="data/upload/domain.yml",
                data=["data/upload/data/nlu.yml"],
                entities=["name"],
                intents=["greet", "inform"],
            ),
            "http://studio.amazonaws.com/api/graphql",
            {
                "query": (
                    "mutation ImportFromEncodedYaml"
                    "($input: ImportFromEncodedYamlInput!)"
                    "{\n  importFromEncodedYaml(input: $input)\n}"
                ),
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": (
                            "dmVyc2lvbjogJzMuMScKaW50ZW50czoKLSBncmVldAotIGluZm9ybQplbn"
                            "RpdGllczoKLSBuYW1lOgogICAgcm9sZXM6CiAgICAtIGZpcnN0X25hbWUK"
                            "ICAgIC0gbGFzdF9uYW1lCi0gYWdlCg=="
                        ),
                        "nlu": (
                            "dmVyc2lvbjogIjMuMSIKbmx1OgotIGludGVudDogZ3JlZXQKICBleGFtcGxlc"
                            "zogfAogICAgLSBoZXkKICAgIC0gaGVsbG8KICAgIC0gaGkKICAgIC0gaGVsbG8"
                            "gdGhlcmUKICAgIC0gZ29vZCBtb3JuaW5nCiAgICAtIGdvb2QgZXZlbmluZwogI"
                            "CAgLSBtb2luCiAgICAtIGhleSB0aGVyZQogICAgLSBsZXQncyBnbwogICAgLSB"
                            "oZXkgZHVkZQogICAgLSBnb29kbW9ybmluZwogICAgLSBnb29kZXZlbmluZwogI"
                            "CAgLSBnb29kIGFmdGVybm9vbgotIGludGVudDogaW5mb3JtCiAgZXhhbXBsZXM"
                            "6IHwKICAgIC0gbXkgbmFtZSBpcyBbVXJvc117ImVudGl0eSI6ICJuYW1lIiwgI"
                            "nJvbGUiOiAiZmlyc3RfbmFtZSJ9CiAgICAtIEknbSBbSm9obl17ImVudGl0eSI"
                            "6ICJuYW1lIiwgInJvbGUiOiAiZmlyc3RfbmFtZSJ9CiAgICAtIEhpLCBteSBma"
                            "XJzdCBuYW1lIGlzIFtMdWlzXXsiZW50aXR5IjogIm5hbWUiLCAicm9sZSI6ICJ"
                            "maXJzdF9uYW1lIn0KICAgIC0gTWlsaWNhCiAgICAtIEthcmluCiAgICAtIFN0Z"
                            "XZlbgogICAgLSBJJ20gWzE4XShhZ2UpCiAgICAtIEkgYW0gWzMyXShhZ2UpIHl"
                            "lYXJzIG9sZAogICAgLSA5Cg=="
                        ),
                    }
                },
            },
        ),
        (
            argparse.Namespace(
                assistant_name=["test"],
                calm=True,
                domain="data/upload/calm/domain/domain.yml",
                data=["data/upload/calm/"],
                config="data/upload/calm/config.yml",
                flows="data/upload/flows.yml",
            ),
            "http://studio.amazonaws.com/api/graphql",
            {
                "query": (
                    "mutation UploadModernAssistant"
                    "($input: UploadModernAssistantInput!)"
                    "{\n  uploadModernAssistant(input: $input)\n}"
                ),
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": (encode_yaml(CALM_DOMAIN_YAML)),
                        "flows": (encode_yaml(CALM_FLOWS_YAML)),
                        "nlu": (encode_yaml(CALM_NLU_YAML)),
                        "config": (
                            "cmVjaXBlOiBkZWZhdWx0LnYxCmxhbmd1YWdlOiBlbgpwaXBlbGluZToKLSBuYW1lOiBTaW5nbGVTdGVwTExNQ29"
                            "tbWFuZEdlbmVyYXRvcgogIGxsbToKICAgIG1vZGVsX25hbWU6IGdwdC00CnBvbGljaWVzOgotIG5hbWU"
                            "6IHJhc2EuY29yZS5wb2xpY2llcy5mbG93X3BvbGljeS5GbG93UG9saWN5Cg=="
                        ),
                    }
                },
            },
        ),
    ],
)
def test_handle_upload(
    monkeypatch: MonkeyPatch,
    args: argparse.Namespace,
    endpoint: str,
    expected: Dict[str, Any],
) -> None:
    mock = MagicMock()
    mock_token = MagicMock()
    mock_config = MagicMock()
    mock_config.read_config.return_value = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url=endpoint,
        realm_name="rasa-test",
        client_id="rasa-cli",
    )
    monkeypatch.setattr(rasa.studio.upload, "requests", mock)
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", mock_token)
    monkeypatch.setattr(
        rasa.studio.upload,
        "StudioConfig",
        mock_config,
    )

    rasa.studio.upload.handle_upload(args)

    assert mock.post.called
    assert mock.post.call_args[0][0] == endpoint
    assert mock.post.call_args[1]["json"] == expected


@pytest.mark.parametrize(
    "is_calm_bot, mock_fn_name",
    [
        (True, "upload_calm_assistant"),
        (False, "upload_nlu_assistant"),
    ],
)
def test_handle_upload_no_domain_path_specified(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    is_calm_bot: bool,
    mock_fn_name: str,
) -> None:
    """Test the handle_upload function when no domain path is specified in the CLI."""
    # setup test
    assistant_name = "test"
    endpoint = "http://studio.amazonaws.com/api/graphql"
    args = argparse.Namespace(
        assistant_name=[assistant_name],
        # this is the default value when running the cmd without specifying -d flag
        domain="domain.yml",
        calm=is_calm_bot,
    )

    domain_dir = tmp_path / "domain"
    domain_dir.mkdir(parents=True, exist_ok=True)
    domain_path = domain_dir / "domain.yml"
    domain_path.write_text("test domain")

    domain_paths = [str(domain_dir), str(tmp_path / "domain.yml")]
    # we need to monkeypatch the DEFAULT_DOMAIN_PATHS to be able to use temporary paths
    monkeypatch.setattr(rasa.studio.upload, "DEFAULT_DOMAIN_PATHS", domain_paths)

    mock_config = MagicMock()
    mock_config.read_config.return_value = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url=endpoint,
        realm_name="rasa-test",
        client_id="rasa-cli",
    )
    monkeypatch.setattr(
        rasa.studio.upload,
        "StudioConfig",
        mock_config,
    )

    mock = MagicMock()
    monkeypatch.setattr(rasa.studio.upload, mock_fn_name, mock)

    rasa.studio.upload.handle_upload(args)

    expected_args = argparse.Namespace(
        assistant_name=[assistant_name],
        calm=is_calm_bot,
        domain=str(domain_dir),
    )

    mock.assert_called_once_with(expected_args, assistant_name, endpoint)


@pytest.mark.parametrize(
    "assistant_name, nlu_examples_yaml, domain_yaml",
    [
        (
            "test",
            dedent(
                """\
                version: '3.1'
                intents:
                - greet
                - inform
                entities:
                - name:
                    roles:
                    - first_name
                    - last_name
                - age"""
            ),
            dedent(
                """\
                version: "3.1"
                nlu:
                - intent: greet
                examples: |
                    - hey
                    - hello
                    - hi
                    - hello there
                    - good morning
                    - good evening
                    - hey there
                    - let's go
                    - hey dude
                    - good afternoon
                - intent: inform
                examples: |
                    - I'm [John]{"entity": "name", "role": "first_name"}
                    - My first name is [Luis]{"entity": "name", "role": "first_name"}
                    - Karin
                    - Steven
                    - I'm [18](age)
                    - I am [32](age) years old"""
            ),
        )
    ],
)
def test_build_request(
    assistant_name: str, nlu_examples_yaml: str, domain_yaml: str
) -> None:
    domain_base64 = base64.b64encode(domain_yaml.encode("utf-8")).decode("utf-8")

    nlu_examples_base64 = base64.b64encode(nlu_examples_yaml.encode("utf-8")).decode(
        "utf-8"
    )

    graphQL_req = rasa.studio.upload.build_request(
        assistant_name, nlu_examples_yaml, domain_yaml
    )

    assert graphQL_req["variables"]["input"]["domain"] == domain_base64
    assert graphQL_req["variables"]["input"]["nlu"] == nlu_examples_base64
    assert graphQL_req["variables"]["input"]["assistantName"] == assistant_name


@pytest.mark.parametrize("assistant_name", ["test"])
def test_build_import_request(assistant_name: str) -> None:
    """Test the build_import_request function.

    :param assistant_name: The name of the assistant
    :return: None
    """

    base64_flows = encode_yaml(CALM_FLOWS_YAML)
    base64_domain = encode_yaml(CALM_DOMAIN_YAML)
    base64_config = encode_yaml("")
    base64_nlu = encode_yaml(CALM_NLU_YAML)

    graphql_req = rasa.studio.upload.build_import_request(
        assistant_name, CALM_FLOWS_YAML, CALM_DOMAIN_YAML, base64_config, CALM_NLU_YAML
    )

    assert graphql_req["variables"]["input"]["domain"] == base64_domain
    assert graphql_req["variables"]["input"]["flows"] == base64_flows
    assert graphql_req["variables"]["input"]["assistantName"] == assistant_name
    assert graphql_req["variables"]["input"]["nlu"] == base64_nlu


def test_build_import_request_no_nlu() -> None:
    """Test the build_import_request function when there is no NLU content to upload.

    :return: None
    """
    assistant_name = "test"
    empty_string = ""

    base64_flows = encode_yaml(CALM_FLOWS_YAML)
    base64_domain = encode_yaml(CALM_DOMAIN_YAML)
    base64_config = encode_yaml(empty_string)

    graphql_req = rasa.studio.upload.build_import_request(
        assistant_name,
        flows_yaml=CALM_FLOWS_YAML,
        domain_yaml=CALM_DOMAIN_YAML,
        config_yaml=empty_string,
    )

    assert graphql_req["variables"]["input"]["domain"] == base64_domain
    assert graphql_req["variables"]["input"]["flows"] == base64_flows
    assert graphql_req["variables"]["input"]["assistantName"] == assistant_name
    assert graphql_req["variables"]["input"]["config"] == base64_config
    assert graphql_req["variables"]["input"]["nlu"] == empty_string


@pytest.mark.parametrize(
    "graphQL_req, endpoint, return_value, expected_response, expected_status",
    [
        (
            {
                "query": """\
                    mutation ImportFromEncodedYaml($input: ImportFromEncodedYamlInput!)\
                        {\n  importFromEncodedYaml(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": "dmFyY2FzZSxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0WJhc2U=",
                        "nlu": "hcyBhIGxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0YWJhc2U=",
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {"data": {"importFromEncodedYaml": ""}},
                "status_code": 200,
            },
            "Upload successful",
            True,
        ),
        (
            {
                "query": """\
                    mutation ImportFromEncodedYaml($input: ImportFromEncodedYamlInput!)\
                        {\n  importFromEncodedYaml(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": "dmFyY2FzZSxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0WJhc2U=",
                        "nlu": "hcyBhIGxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0YWJhc2U=",
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {
                    "data": {"importFromEncodedYaml": ""},
                    "errors": [{"message": "Upload failed with status code 405"}],
                },
                "status_code": 405,
            },
            (
                "Upload failed with the following errors: "
                "Upload failed with status code 405"
            ),
            False,
        ),
        (
            {
                "query": """\
                    mutation ImportFromEncodedYaml($input: ImportFromEncodedYamlInput!)\
                        {\n  importFromEncodedYaml(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": "dmFyY2FzZSxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0WJhc2U=",
                        "nlu": "hcyBhIGxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0YWJhc2U=",
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {
                    "data": {"importFromEncodedYaml": None},
                    "errors": [{"message": "Any error message"}],
                },
                "status_code": 500,
            },
            "Upload failed with the following errors: Any error message",
            False,
        ),
        (
            {
                "query": """\
          mutation UploadModernAssistant($input: UploadModernAssistantInput!)\
              {\n  uploadModernAssistant(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": encode_yaml(CALM_DOMAIN_YAML),
                        "flows": encode_yaml(CALM_FLOWS_YAML),
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {"data": {"uploadModernAssistant": ""}},
                "status_code": 200,
            },
            "Upload successful",
            True,
        ),
        (
            {
                "query": """\
          mutation UploadModernAssistant($input: UploadModernAssistantInput!)\
              {\n  uploadModernAssistant(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": encode_yaml(CALM_DOMAIN_YAML),
                        "flows": encode_yaml(CALM_FLOWS_YAML),
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {
                    "data": {"importFromEncodedYaml": None},
                    "errors": [
                        {"message": "Upload failed with status code 405"},
                        {"message": "Another message"},
                    ],
                },
                "status_code": 405,
            },
            (
                "Upload failed with the following errors: "
                "Upload failed with status code 405; "
                "Another message"
            ),
            False,
        ),
        (
            {
                "query": """\
          mutation UploadModernAssistant($input: UploadModernAssistantInput!)\
              {\n  uploadModernAssistant(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": encode_yaml(CALM_DOMAIN_YAML),
                        "flows": encode_yaml(CALM_FLOWS_YAML),
                        "config": "",
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {
                    "data": {"importFromEncodedYaml": None},
                    "errors": [{"message": "Upload failed with status code 500"}],
                },
                "status_code": 500,
            },
            (
                "Upload failed with the following errors: "
                "Upload failed with status code 500"
            ),
            False,
        ),
    ],
)
def test_make_request(
    monkeypatch: MonkeyPatch,
    graphQL_req: Dict[str, Any],
    return_value: Dict[str, Any],
    expected_response: str,
    expected_status: bool,
    endpoint: str,
    caplog,
) -> None:
    # Mock the requests.post method
    return_mock = MagicMock()
    return_mock.status_code = return_value["status_code"]
    return_mock.json.return_value = return_value["json"]
    mock_post = MagicMock(return_value=return_mock)
    monkeypatch.setattr(rasa.studio.upload.requests, "post", mock_post)

    # Mock the KeycloakTokenReader
    mock_token = MagicMock()
    mock_token.get_token.return_value.token_type = "Bearer"
    mock_token.get_token.return_value.access_token = "mock_token"
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", lambda: mock_token)

    # Create an instance of ResultsLogger
    results_logger = ResultsLogger()

    # Apply the decorator to a test function
    @results_logger.wrap
    def test_make_request_func(inner_endpoint: str, graphQL_req: Dict[str, Any]):
        return rasa.studio.upload.make_request(inner_endpoint, graphQL_req)

    # Call the decorated function
    ret_val, status = test_make_request_func(endpoint, graphQL_req)

    # Assert that the mock was called with the correct arguments
    assert mock_post.called
    assert mock_post.call_args[0][0] == endpoint
    assert mock_post.call_args[1]["json"] == graphQL_req
    assert mock_post.call_args[1]["headers"] == {
        "Authorization": "Bearer mock_token",
        "Content-Type": "application/json",
    }

    # Assert that the KeycloakTokenReader was called
    assert mock_token.get_token.called

    # Check the return value and status
    assert ret_val == expected_response
    assert status == expected_status


@pytest.mark.parametrize(
    "domain_from_files, intents, entities, expected_domain",
    [
        (
            {
                "version": "3.1",
                "intents": [
                    "greet",
                    "inform",
                    "goodbye",
                    "deny",
                ],
                "entities": [
                    {"name": {"roles": ["first_name", "last_name"]}},
                    "age",
                    "destination",
                    "origin",
                ],
            },
            ["greet", "inform"],
            ["name"],
            {
                "version": "3.1",
                "intents": [
                    "greet",
                    "inform",
                ],
                "entities": [{"name": {"roles": ["first_name", "last_name"]}}],
            },
        ),
    ],
)
def test_filter_domain(
    domain_from_files: Dict[str, Any],
    intents: List[str],
    entities: List[Union[str, Dict[Any, Any]]],
    expected_domain: Dict[str, Any],
) -> None:
    filtered_domain = rasa.studio.upload._filter_domain(
        domain_from_files=domain_from_files, intents=intents, entities=entities
    )
    assert filtered_domain == expected_domain


@pytest.mark.parametrize(
    "intents, entities, found_intents, found_entities",
    [
        (
            ["greet", "inform"],
            ["name"],
            ["greet", "goodbye", "deny"],
            ["name", "destination", "origin"],
        ),
    ],
)
def test_check_for_missing_primitives(
    intents: List[str],
    entities: List[str],
    found_intents: List[str],
    found_entities: List[str],
) -> None:
    with pytest.raises(RasaException) as excinfo:
        rasa.studio.upload._check_for_missing_primitives(
            intents, entities, found_intents, found_entities
        )
        assert "The following intents were not found in the domain: inform" in str(
            excinfo.value
        )
        assert "The following entities were not found in the domain: age" in str(
            excinfo.value
        )


@pytest.mark.parametrize(
    "args, intents_from_files, entities_from_files, "
    "expected_intents, expected_entities",
    [
        (
            argparse.Namespace(
                intents={"greet", "inform"},
                entities={"name"},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["greet", "inform"],
            ["name"],
        ),
        (
            argparse.Namespace(
                intents=None,
                entities={"name"},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["goodbye", "greet", "deny"],
            ["name"],
        ),
        (
            argparse.Namespace(
                intents={},
                entities={"name"},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["goodbye", "greet", "deny"],
            ["name"],
        ),
        (
            argparse.Namespace(
                intents={"greet", "inform"},
                entities=None,
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["greet", "inform"],
            ["destination", "name", "origin"],
        ),
        (
            argparse.Namespace(
                intents={"greet", "inform"},
                entities={},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["greet", "inform"],
            ["destination", "name", "origin"],
        ),
    ],
)
def test_get_selected_entities_and_intents(
    args: argparse.Namespace,
    intents_from_files: Set[Text],
    entities_from_files: List[Text],
    expected_intents: List[Text],
    expected_entities: List[Text],
) -> None:
    entities, intents = rasa.studio.upload._get_selected_entities_and_intents(
        args=args,
        intents_from_files=intents_from_files,
        entities_from_files=entities_from_files,
    )

    assert intents.sort() == expected_intents.sort()
    assert entities.sort() == expected_entities.sort()

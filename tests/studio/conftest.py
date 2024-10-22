import base64
from textwrap import dedent
from unittest.mock import MagicMock

import pytest

from rasa.studio.auth import StudioAuth
from rasa.studio.config import StudioConfig


@pytest.fixture
def mock_keycloak_instance_object() -> MagicMock:
    keycloak_open_id_instance = MagicMock()
    keycloak_open_id_instance.server_url = "http://localhost:8080"
    keycloak_open_id_instance.client_id = "client_id"
    keycloak_open_id_instance.realm_name = "realm_name"
    keycloak_open_id_instance.token = MagicMock()
    return keycloak_open_id_instance


@pytest.fixture
def mock_keycloak_instance(
    mock_keycloak_open_id: MagicMock, mock_keycloak_instance_object: MagicMock
) -> MagicMock:
    mock_keycloak_open_id.return_value = mock_keycloak_instance_object
    return mock_keycloak_instance_object


@pytest.fixture
def studio_auth(mock_keycloak_instance: MagicMock) -> StudioAuth:
    config = StudioConfig(
        authentication_server_url="http://localhost:8080",
        studio_url="http://localhost:8080/graphql",
        client_id="client_id",
        realm_name="realm_name",
    )
    auth = StudioAuth(config)
    return auth


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
          metadata:
            line_numbers: 11-11
          action: utter_cant_advice_on_health
        name: health advice
        description: user asks for health advice
        nlu_trigger:
        - intent:
            name: health_advice
            confidence_threshold: 0.8
        file_path: data/upload/calm/data/flows.yml
      add_contact:
        steps:
        - id: 0_collect_add_contact_handle
          next: 1_collect_add_contact_name
          description: a user handle starting with @
          metadata:
            line_numbers: 16-17
          collect: add_contact_handle
          utter: utter_ask_add_contact_handle
          ask_before_filling: false
          reset_after_flow_ends: true
          rejections: []
        - id: 1_collect_add_contact_name
          next: 2_collect_add_contact_confirmation
          description: a name of a person
          metadata:
            line_numbers: 18-19
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
              metadata:
                line_numbers: 25-26
              action: utter_add_contact_cancelled
          - else: action_add_contact
          description: a confirmation to add contact
          metadata:
            line_numbers: 20-27
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
              metadata:
                line_numbers: 33-34
              action: utter_contact_already_exists
          - if: slots.return_value is 'success'
            then:
            - id: 6_utter_contact_added
              next: END
              metadata:
                line_numbers: 37-38
              action: utter_contact_added
          - else:
            - id: 7_utter_add_contact_error
              next: END
              metadata:
                line_numbers: 40-41
              action: utter_add_contact_error
          metadata:
            line_numbers: 28-41
          action: action_add_contact
        name: add_contact
        description: add a contact to your contact list
        file_path: data/upload/calm/data/flows.yml
      check_balance:
        steps:
        - id: 0_action_check_balance
          next: 1_utter_current_balance
          metadata:
            line_numbers: 46-46
          action: action_check_balance
        - id: 1_utter_current_balance
          next: END
          metadata:
            line_numbers: 47-47
          action: utter_current_balance
        name: check_balance
        description: check the user's account balance.
        file_path: data/upload/calm/data/flows.yml
      hotel_search:
        steps:
        - id: 0_action_search_hotel
          next: 1_utter_hotel_inform_rating
          metadata:
            line_numbers: 52-52
          action: action_search_hotel
        - id: 1_utter_hotel_inform_rating
          next: END
          metadata:
            line_numbers: 53-53
          action: utter_hotel_inform_rating
        name: hotel_search
        description: search for hotels
        file_path: data/upload/calm/data/flows.yml
      remove_contact:
        steps:
        - id: 0_collect_remove_contact_handle
          next: 1_collect_remove_contact_confirmation
          description: a contact handle starting with @
          metadata:
            line_numbers: 58-59
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
              metadata:
                line_numbers: 65-66
              action: utter_remove_contact_cancelled
          - else: action_remove_contact
          metadata:
            line_numbers: 60-67
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
              metadata:
                line_numbers: 73-74
              action: utter_contact_not_in_list
          - if: slots.return_value is 'success'
            then:
            - id: 5_utter_remove_contact_success
              next: END
              metadata:
                line_numbers: 77-78
              action: utter_remove_contact_success
          - else:
            - id: 6_utter_remove_contact_error
              next: END
              metadata:
                line_numbers: 80-81
              action: utter_remove_contact_error
          metadata:
            line_numbers: 68-81
          action: action_remove_contact
        name: remove_contact
        description: remove a contact from your contact list
        file_path: data/upload/calm/data/flows.yml
      transaction_search:
        steps:
        - id: 0_action_transaction_search
          next: 1_utter_transactions
          metadata:
            line_numbers: 86-86
          action: action_transaction_search
        - id: 1_utter_transactions
          next: END
          metadata:
            line_numbers: 87-87
          action: utter_transactions
        name: transaction_search
        description: lists the last transactions of the user account
        file_path: data/upload/calm/data/flows.yml
      transfer_money:
        steps:
        - id: 0_collect_transfer_money_recipient
          next: 1_collect_transfer_money_amount_of_money
          description: Asks user for the recipient's name.
          metadata:
            line_numbers: 92-93
          collect: transfer_money_recipient
          utter: utter_ask_transfer_money_recipient
          ask_before_filling: false
          reset_after_flow_ends: true
          rejections: []
        - id: 1_collect_transfer_money_amount_of_money
          next: 2_action_check_transfer_funds
          description: Asks user for the amount to transfer.
          metadata:
            line_numbers: 94-95
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
              metadata:
                line_numbers: 100-100
              action: utter_transfer_money_insufficient_funds
            - id: 4_set_slots
              next: END
              metadata:
                line_numbers: 101-106
              set_slots:
              - transfer_money_amount_of_money: null
              - transfer_money_has_sufficient_funds: null
              - set_slots_test_text: This is a test!
              - set_slots_test_categorical: value_1
          - else: collect_transfer_money_final_confirmation
          metadata:
            line_numbers: 96-107
          action: action_check_transfer_funds
        - id: collect_transfer_money_final_confirmation
          next:
          - if: not slots.transfer_money_final_confirmation
            then:
            - id: 6_utter_transfer_cancelled
              next: END
              metadata:
                line_numbers: 115-116
              action: utter_transfer_cancelled
          - else: action_execute_transfer
          description: Asks user for final confirmation to transfer money.
          metadata:
            line_numbers: 108-117
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
              metadata:
                line_numbers: 123-124
              action: utter_transfer_complete
          - else:
            - id: 9_utter_transfer_failed
              next: END
              metadata:
                line_numbers: 126-127
              action: utter_transfer_failed
          metadata:
            line_numbers: 118-127
          action: action_execute_transfer
        name: transfer_money
        description: This flow let's users send money to friends and family.
        file_path: data/upload/calm/data/flows.yml
      verify_account:
        steps:
        - id: 0_collect_verify_account_email
          next: 1_collect_based_in_california
          description: Asks user for their email address.
          metadata:
            line_numbers: 132-134
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
                  metadata:
                    line_numbers: 147-148
                  action: utter_ca_income_insufficient
              - else: collect_verify_account_confirmation_california
              description: Asks user if they have sufficient income in California.
              metadata:
                line_numbers: 141-149
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
                  metadata:
                    line_numbers: 157-158
                  action: utter_verify_account_success
              - else:
                - id: 6_utter_verify_account_cancelled
                  next: END
                  metadata:
                    line_numbers: 160-161
                  action: utter_verify_account_cancelled
              description: Asks user for final confirmation to verify their account in California.
              metadata:
                line_numbers: 150-161
              collect: verify_account_confirmation_california
              utter: utter_ask_verify_account_confirmation_california
              ask_before_filling: true
              reset_after_flow_ends: true
              rejections: []
          - else: collect_verify_account_confirmation
          description: Asks user if they are based in California.
          metadata:
            line_numbers: 135-162
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
              metadata:
                line_numbers: 170-171
              action: utter_verify_account_success
          - else:
            - id: 9_utter_verify_account_cancelled
              next: END
              metadata:
                line_numbers: 173-174
              action: utter_verify_account_cancelled
          description: Asks user for final confirmation to verify their account.
          metadata:
            line_numbers: 163-174
          collect: verify_account_confirmation
          utter: utter_ask_verify_account_confirmation
          ask_before_filling: true
          reset_after_flow_ends: true
          rejections: []
        name: verify_account
        description: Verify an account for higher transfer limits
        file_path: data/upload/calm/data/flows.yml
    """  # noqa: E501
)

CALM_CUSTOMIZED_PATTERNS_YAML = dedent(
    """\
    flows:
      pattern_completed:
        steps:
        - id: action_listen
          next: END
          metadata:
            line_numbers: 4-7
          action: action_listen
        name: pattern_completed
        description: This is a pattern
        file_path: data/upload/customized_default_flows.yml
    """
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

CALM_CONFIG_YAML = dedent(
    """\
    recipe: default.v1
    language: en
    pipeline:
    - name: LLMCommandGenerator
      llm:
        model_name: gpt-4
        request_timeout: 7
        max_tokens: 256

    policies:
    - name: FlowPolicy
    - name: IntentlessPolicy
    """
)

CALM_ENDPOINTS_YAML = "nlg: \ntype: rephrase\n"


def encode_yaml(yaml):
    return base64.b64encode(yaml.encode("utf-8")).decode("utf-8")


def mock_questionary_text(question, default=""):
    return MagicMock(ask=lambda: "test")

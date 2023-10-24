import pytest

from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.exceptions import SchemaValidationError


def test_schema_validation_fails_on_empty_steps() -> None:
    with pytest.raises(SchemaValidationError):
        flows_from_str(
            """
        flows:
          empty_flow:
            description: "A flow without steps"
            steps: []
        """
        )


def test_schema_validation_fails_on_empty_steps_for_branch() -> None:
    valid_flows = """
        flows:
          empty_branch_flow:
            description: "A flow with an empty branch"
            steps:
              - action: utter_greet
                next:
                  - if: "status == logged_in"
                    then: 
                      - action: utter_already_logged_in
                        next: "END"
                  - else:
                    - action: "utter_need_to_log_in"
                      next: "END"
        """

    flows_from_str(valid_flows)

    invalid_flows = """
        flows:
          empty_branch_flow:
            description: "A flow with an empty branch"
            steps:
              - action: utter_greet
                next:
                  - if: "status == logged_in"
                    then: []
                  - else:
                    - action: "utter_need_to_log_in"
                      next: "END"
        """

    with pytest.raises(SchemaValidationError):
        flows_from_str(invalid_flows)

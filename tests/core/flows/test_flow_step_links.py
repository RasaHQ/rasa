from rasa.shared.core.flows.yaml_flows_io import flows_from_str


def test_depth_in_tree():
    flows = flows_from_str(
        """
            flows:
              flow_1:
                steps:
                  - id: step_1
                    collect: slot
                    next:
                      - if: condition
                        then:
                          - collect: slot
                            next:
                              - if: not condition
                                then:
                                  - action: utter
                                    next: END
                              - else: step_2
                      - else: step_2
                  - id: step_2
                    collect: slot
                    next:
                      - if: condition
                        then:
                          - action: utter
                            next: END
                      - else:
                          - action: utter
                            next: step_3
                  - id: step_3
                    action: some_action
                    next: step_4
                  - id: step_4
                    action: some_action
                    next:
                    - if: condition
                      then:
                        - action: some_action
                          next:
                          - if: condition
                            then:
                              - action: utter
                                next: step_4_2
                          - else:
                              - action: utter
                                next: step_4_1
                        - id: step_4_1
                          action: some_action
                          next: END
                        - id: step_4_2
                          action: some_action
                          next:
                          - if: condition
                            then:
                              - collect: slot
                                next:
                                  - if: not condition
                                    then:
                                      - action: utter
                                        next: END
                                  - else: step_4_1
                          - else: END
                    - else:
                      - collect: slot
                        next: END
    """
    )

    steps = flows.underlying_flows[0].steps

    assert steps[0].custom_id == "step_1"
    assert steps[0].next.depth_in_tree() == 2
    assert steps[3].custom_id == "step_2"
    assert steps[3].next.depth_in_tree() == 1
    assert steps[6].custom_id == "step_3"
    assert steps[6].next.depth_in_tree() == 0
    assert steps[7].custom_id == "step_4"
    assert steps[7].next.depth_in_tree() == 3

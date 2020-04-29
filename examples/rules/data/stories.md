<!-- each story will be perceived as independent rule -->


## activate loop q_form
<!-- required slots for q_form should be listed somewhere else -->
* activate_q_form  <!-- like request_restaurant -->
    - action_activate_q_form  <!-- default action with an utterance like action_restart -->
    - form{"name": "q_form"} <!-- problem with predicting this event, because, it has to be used as condition -->
    - action_loop_q_form  <!-- can be anything -->

## ask loop q_form
<!-- RulePolicy should substitute requested_slot with actual slot value -->
    - form{"name": "q_form"}  <!-- condition that form is active-->
    - slot{"requested_slot": "some_slot"}  <!-- some condition -->
* inform{"some_slot":"bla"} <!-- can be ANY -->
    - validate_loop
    - action_loop_q_form <!-- can be internal core action, can be anything -->

## explain loop q_form
    - form{"name": "q_form"} <!-- condition that form is active-->
    - slot{"requested_slot": "some_slot"}  <!-- some condition -->
* explain                          <!-- can be anything -->
    - utter_explain_some_slot
    - action_loop_q_form

## stop loop q_form
    - form{"name": "q_form"} <!-- condition that form is active-->
* stopp
    - action_stop_q_form

## finish loop q_form
    - form{"name": "q_form"} <!-- condition that form is active-->
    - slot{"requested_slot": null} <!-- some condition to finish -->
    - action_stop_q_form
    - form{"name": null}


## questions
* ask_possibilities
    - utter_list_possibilities


## switch faq
* switch_faq
    - action_switch_faq


## FAQ simple
    - slot{"detailed_faq": false}
* faq
    - utter_faq

## FAQ detailed
    - slot{"detailed_faq": true}
* faq
    - utter_faq
    - utter_ask_did_help


## FAQ helped - continue
    - utter_faq
    - utter_ask_did_help
* affirm
    - utter_continue


## FAQ not helped
    - utter_faq
    - utter_ask_did_help
* deny
    - utter_detailed_faq
    - utter_ask_did_help
 

## detailed FAQ not helped - continue
    - utter_detailed_faq
    - utter_ask_did_help
* deny
    - utter_ask_stop
* deny
    - utter_continue


## detailed FAQ not helped - stop
    - utter_detailed_faq
    - utter_ask_did_help
* deny
    - utter_ask_stop
* affirm
    - utter_stop



## Greet
<!-- add start indicator condition -->
* greet
    - utter_greet
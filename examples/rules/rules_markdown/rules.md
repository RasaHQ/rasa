<!-- each story starting with `>>` will be perceived as independent rule -->

>> Activate form 'q_form'
<!-- required slots for q_form are listed in the domain. -->
    - ... <!-- `...` indicates that this rule applies at any point within a conversation -->
* activate_q_form  <!-- like request_restaurant -->
    - loop_q_form  <!-- Activate and run form -->
    - form{"name": "loop_q_form"}


>> Example of an unhappy path for the 'q_form'
    - form{"name": "loop_q_form"} <!-- condition that form is active-->
    - slot{"requested_slot": "some_slot"}  <!-- some condition -->
    - ...
* explain                          <!-- can be anything -->
    - utter_explain_some_slot
    - loop_q_form
    - form{"name": "loop_q_form"}


>> submit form
    - form{"name": "loop_q_form"} <!-- condition that form is active-->
    - ...
    - loop_q_form <!-- condition that form is active -->
    - form{"name": null}
    - slot{"requested_slot": null}
    - utter_stop  <!-- can be any action -->


>> FAQ question
    - ...
* ask_possibilities
    - utter_list_possibilities


>> Another FAQ example
    - ...
* switch_faq
    - action_switch_faq


>> FAQ simple
    - slot{"detailed_faq": false}
    - ... <!-- indicator that there might be a story before hand -->
* faq
    - utter_faq
<!-- no ... means predict action_listen here -->


>> FAQ detailed
    - slot{"detailed_faq": true}
    - ...
* faq
    - utter_faq
    - ... <!-- don't predict action_listen by the rule -->


>> FAQ helped - continue
    - slot{"detailed_faq": true}
    - ...  <!-- putting actions before ... shouldn't be allowed -->
    - utter_faq
    - utter_ask_did_help  <!--problem: it will learn that after utter_faq goes utter_ask_did_help -->
* affirm
    - utter_continue


>> FAQ not helped
    - slot{"detailed_faq": true}
    - ...
    - utter_faq
    - utter_ask_did_help
* deny
    - utter_detailed_faq
    - ...  <!-- indicator that the story is continued, no action_listen -->
 

>> detailed FAQ not helped - continue
    - slot{"detailed_faq": true}
    - ...
    - utter_detailed_faq
    - utter_ask_did_help
* deny
    - utter_ask_stop
* deny
    - utter_continue


>> detailed FAQ not helped - stop
    - slot{"detailed_faq": true}
    - ...
    - utter_detailed_faq
    - utter_ask_did_help
* deny
    - utter_ask_stop
* affirm
    - utter_stop


>> Greet
<!-- lack of ... is story start indicator condition -->
* greet
    - utter_greet


>> Implementation of the TwoStageFallbackPolicy
    - ...
* nlu_fallback  <!-- like request_restaurant -->
    - two_stage_fallback  <!-- Activate and run form -->
    - form{"name": "two_stage_fallback"}

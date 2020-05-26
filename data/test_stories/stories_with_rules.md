>> rule 1
    - form{"name": "loop_q_form"}  <!-- condition that form is active-->
    - slot{"requested_slot": "some_slot"}  <!-- some condition -->
    - ...
* inform{"some_slot":"bla"} <!-- can be ANY -->
    - loop_q_form <!-- can be internal core action, can be anything -->

>> rule 2
    - form{"name": "loop_q_form"} <!-- condition that form is active-->
    - slot{"requested_slot": "some_slot"}  <!-- some condition -->
    - ...
* explain                          <!-- can be anything -->
    - utter_explain_some_slot
    - loop_q_form
    - form{"name": "loop_q_form"} <!-- condition that form is active-->
    
## ML story 1
* greet
    - utter_greet
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - form{"name": null}
    - utter_slots_values
* thankyou
    - utter_noworries

>> rule 3
    - form{"name": "loop_q_form"} <!-- condition that form is active-->
    - ...
    - loop_q_form <!-- condition that form is active -->
    - form{"name": null}
    - slot{"requested_slot": null}
    - action_stop_q_form

## ML story 2
* greet
    - utter_greet
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* chitchat
    - utter_chitchat
    - restaurant_form
    - form{"name": null}
    - utter_slots_values
* thankyou
    - utter_noworries

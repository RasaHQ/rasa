>> greeting
* greet
    - utter_greet

>> bot challenge
    - ...
* bot_challenge
    - utter_iamabot

>> thank you
    - ...
* thankyou
    - utter_noworries

>> chitchat
    - ...
* chitchat
    - utter_chitchat

>> activate restaurant form
    - ...
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}

>> submit restaurant form
    - form{"name": "restaurant_form"}
    - ...
    - restaurant_form
    - form{"name": null}
    - utter_submit
    - utter_slots_values

>> stop form
    - form{"name": "restaurant_form"}
    - ...
    - restaurant_form
* stop
    - utter_ask_continue

>> stop but continue form
    - form{"name": "restaurant_form"}
    - ...
    - utter_ask_continue
* affirm
    - restaurant_form

>> stop and really stop path
    - form{"name": "restaurant_form"}
    - ...
    - utter_ask_continue
* deny
    - action_deactivate_form
    - form{"name": null}

## OLD Generated Story 3490283781720101690 (example from interactive learning, "form: " will be excluded from training)
* greet
    - utter_greet
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - slot{"requested_slot": "cuisine"}
* chitchat
    - utter_chitchat  <!-- restaurant_form was predicted by FormPolicy and rejected, other policy predicted utter_chitchat -->
    - restaurant_form
    - slot{"requested_slot": "cuisine"}
* form: inform{"cuisine": "mexican"}
    - slot{"cuisine": "mexican"}
    - form: restaurant_form
    - slot{"cuisine": "mexican"}
    - slot{"requested_slot": "num_people"}
* form: inform{"number": "2"}
    - form: restaurant_form
    - slot{"num_people": "2"}
    - slot{"requested_slot": "outdoor_seating"}
* chitchat
    - utter_chitchat
    - restaurant_form
    - slot{"requested_slot": "outdoor_seating"}
* stop
    - utter_ask_continue
* affirm
    - restaurant_form  <!-- FormPolicy predicted FormValidation(False), other policy predicted restaurant_form -->
    - slot{"requested_slot": "outdoor_seating"}
* form: affirm
    - form: restaurant_form
    - slot{"outdoor_seating": true}
    - slot{"requested_slot": "preferences"}
* form: inform
    - form: restaurant_form
    - slot{"preferences": "/inform"}
    - slot{"requested_slot": "feedback"}
* form: inform{"feedback": "great"}
    - slot{"feedback": "great"}
    - form: restaurant_form
    - slot{"feedback": "great"}
    - form{"name": null}
    - slot{"requested_slot": null}
    - utter_submit
    - utter_slots_values
* thankyou
    - utter_noworries

## NEW Generated story interactive_story_1 (no "form: " prefix)
* greet
    - utter_greet
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - slot{"requested_slot": "cuisine"}
* chitchat
    - utter_chitchat
    - restaurant_form
    - slot{"requested_slot": "cuisine"}
* chitchat
    - utter_chitchat
    - restaurant_form
    - slot{"requested_slot": "cuisine"}
    - slot{"cuisine": "mexican"}
    - slot{"requested_slot": "feedback"}
    - slot{"feedback": "good"}
    - slot{"requested_slot": "num_people"}
    - slot{"num_people": "5"}
    - slot{"requested_slot": "outdoor_seating"}
    - slot{"outdoor_seating": false}
    - slot{"requested_slot": "preferences"}
    - slot{"preferences": "no additional preferences"}
    - slot{"requested_slot": null}
    - form{"name": null}
    - utter_submit
    - utter_slots_values
* thankyou
    - utter_noworries

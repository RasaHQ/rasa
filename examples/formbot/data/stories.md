## happy path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - form{"name": null}
    - utter_slots_values
* thank
    - utter_noworries

## unhappy path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* chitchat
    - utter_chitchat
    - restaurant_form
* chitchat
    - utter_chitchat
    - restaurant_form
* chitchat
    - utter_chitchat
    - restaurant_form
    - form{"name": null}
    - utter_slots_values
* thank
    - utter_noworries

## stop but continue path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* stop
    - utter_ask_continue
* affirm
    - restaurant_form
    - form{"name": null}
    - utter_slots_values
* thank
    - utter_noworries

## chitchat stop but continue path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* chitchat
    - utter_chitchat
    - restaurant_form
* stop
    - utter_ask_continue
* affirm
    - restaurant_form
    - form{"name": null}
    - utter_slots_values
* thank
    - utter_noworries

## stop but continue and chitchat path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* stop
    - utter_ask_continue
* affirm
    - restaurant_form
* chitchat
    - utter_chitchat
    - restaurant_form
    - form{"name": null}
    - utter_slots_values
* thank
    - utter_noworries

## chitchat stop but continue and chitchat path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* chitchat
    - utter_chitchat
    - restaurant_form
* stop
    - utter_ask_continue
* affirm
    - restaurant_form
* chitchat
    - utter_chitchat
    - restaurant_form
    - form{"name": null}
    - utter_slots_values
* thank
    - utter_noworries

## stop and really stop path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* stop
    - utter_ask_continue
* deny
    - action_restart
    - restart

## stop and really stop path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* chitchat
    - utter_chitchat
* stop
    - utter_ask_continue
* deny
    - action_restart
    - restart

## Generated Story 3490283781720101690 (example from interactive learning, "form: " will be excluded from training)
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - slot{"requested_slot": "cuisine"}
* chitchat
    - utter_chitchat
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
* form: affirm
    - form: restaurant_form
    - slot{"outdoor_seating": true}
    - slot{"requested_slot": "preferences"}
* form: rubbish
    - form: restaurant_form
    - slot{"preferences": "rubbish"}
    - form{"name": null}
    - slot{"requested_slot": null}
    - utter_slots_values
* thank
    - utter_noworries
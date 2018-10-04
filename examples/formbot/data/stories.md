## happy path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - form{"name": null}
* thank
    - utter_noworries

## chitchat once
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* chitchat
    - utter_chitchat
    - restaurant_form
    - form{"name": null}
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
* thank
    - utter_noworries

## unhappy path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* thank
    - utter_chitchat
    - restaurant_form
    - form{"name": null}
* thank
    - utter_noworries

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
* thank
    - utter_noworries
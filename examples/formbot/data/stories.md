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

## Generated Story -9155310465400161964 (example from interactive learning, "form: " will be excluded from training)
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - slot{"requested_slot": "cuisine"}
* chitchat
    - utter_chitchat
    - restaurant_form
    - form: slot{"requested_slot": "cuisine"}
* form: inform{"cuisine": "1"}
    - form: slot{"cuisine": "1"}
    - form: restaurant_form
    - form: slot{"cuisine": "1"}
    - form: slot{"requested_slot": "num_people"}
* form: inform{"num_people": "1"}
    - form: slot{"num_people": "1"}
    - form: restaurant_form
    - form: slot{"num_people": "1"}
    - form{"name": null}
    - slot{"requested_slot": null}
* thank
    - utter_noworries

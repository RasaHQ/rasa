## Happy path
* greet: /greet
    - utter_greet
* request_restaurant: /request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - form{"name": null}
    - utter_submit
    - utter_slots_values
* thankyou: /thankyou
    - utter_noworries

## Happy path with form prefix
* greet: /greet
    - utter_greet
* request_restaurant: /request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* form: /inform{"cuisine": "afghan"} <!-- intent "inform" is ignored inside the form -->
    - form: restaurant_form
    - form{"name": null}
    - utter_submit
    - utter_slots_values
* thankyou: /thankyou
    - utter_noworries
 
## unhappy path
* greet: /greet
    - utter_greet
* request_restaurant: /request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* chitchat: /chitchat
    - utter_chitchat
    - restaurant_form
    - form{"name": null}
    - utter_submit
    - utter_slots_values
* thankyou: /thankyou
    - utter_noworries

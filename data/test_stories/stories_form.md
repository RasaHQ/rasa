## simple_story_with_multiple_turns
* greet
    - utter_greet
* default
    - utter_default
* goodbye
    - utter_goodbye

## simple_story_with_form_happy
* greet
    - utter_greet
* start_form
    - some_form
    - form{"name": "some_form"}
* form: inform
    - form: some_form
    - form{"name": null}
* default
    - utter_default
* goodbye
    - utter_goodbye

## simple_story_with_form_unhappy
* greet
    - utter_greet
* start_form
    - some_form
    - form{"name": "some_form"}
* default
    - utter_default
    - some_form
    - form{"name": null}
* goodbye
    - utter_goodbye

## simple_story_with_form_stop_continue
* greet
    - utter_greet
* start_form
    - some_form
    - form{"name": "some_form"}
* stop
    - utter_ask_continue
* affirm
    - some_form
    - form{"name": null}
* goodbye
    - utter_goodbye

## simple_story_with_form_stop_inform
* greet
    - utter_greet
* start_form
    - some_form
    - form{"name": "some_form"}
* stop
    - utter_ask_continue
    - action_listen
* form: inform
    - some_form
    - form{"name": null}
* goodbye
    - utter_goodbye

## simple_story_with_form_stop_deactivate
* greet
    - utter_greet
* start_form
    - some_form
    - form{"name": "some_form"}
* stop
    - utter_ask_continue
* deny
    - action_deactivate_form
    - form{"name": null}
    - utter_goodbye

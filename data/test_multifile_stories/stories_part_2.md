## simple_story_with_multiple_turns
* affirm OR thank_you
    - utter_default
* goodbye
    - utter_goodbye
> check_goodbye

## why does the user want to leave?
> check_goodbye
* why
    - utter_default
> check_greet

## show_it_all
> check_greet
> check_hello                   <!-- allows multiple entry points -->
* next_intent
    - utter_greet              <!-- actions taken by the bot -->
> check_intermediate            <!-- allows intermediate checkpoints -->
* change_bank_details
    - utter_default            <!-- allows to end without checkpoints -->

## entities and slots
* greet{"cuisine": "italian"}
    - slot{"cuisine": "italian"}
    - utter_default
    - utter_default <!-- just so length is different from the checkpoint story -->

## simple_story_without_checkpoint
* simple
    - utter_default
    - utter_greet

## simple_story_with_only_start
> check_greet                   <!-- checkpoints at the start define entry points -->
* simple
    - utter_default

## simple_story_with_only_end
* hello
    - utter_greet
    - slot{"name": "peter"}
> check_greet                   <!-- checkpoint defining the end of this turn -->

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

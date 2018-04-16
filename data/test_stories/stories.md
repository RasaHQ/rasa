## simple_story_without_checkpoint
* default                       <!-- user utterance in _intent[entities] format -->
    - utter_default
    - utter_greet

## simple_story_with_only_start
> check_greet                   <!-- checkpoints at the start define entry points -->
* default
    - slot{"nice_person": ""}
    - utter_default

## simple_story_with_only_end
* greet
    - utter_greet
    - slot{"name": "peter"}
    - slot{"nice_person": ""}
> check_greet                   <!-- checkpoint defining the end of this turn -->

## simple_story_with_multiple_turns
* affirm OR thank_you
    - utter_default
* goodbye
    - utter_goodbye
> check_goodbye        

## why does the user want to leave?
> check_goodbye
* default
    - utter_default
> check_greet

## show_it_all
> check_greet
> check_hello                   <!-- allows multiple entry points -->

## intermediate checkpoint
* greet
    - utter_greet              <!-- actions taken by the bot -->
> check_intermediate            <!-- allows intermediate checkpoints -->
* change_bank_details
    - utter_default            <!-- allows to end without checkpoints -->

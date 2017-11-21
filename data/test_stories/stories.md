## simple_story_without_checkpoint
* _simple                       <!-- user utterance in _intent[entities] format -->
    - utter_default
    - utter_greet

## simple_story_with_only_start
> check_greet                   <!-- checkpoints at the start define entry points -->
* _simple
    - slot["nice_person"]
    - utter_default

## simple_story_with_only_end
* _hello
    - utter_greet
    - slot{"name": "peter"}
    - slot{"nice_person": ""}
> check_greet                   <!-- checkpoint defining the end of this turn -->

## simple_story_with_multiple_turns
* _affirm OR _thank_you
    - utter_default
* _goodbye
    - utter_goodbye
> check_goodbye        

## why does the user want to leave?
> check_goodbye
* _why
    - utter_default
> check_greet

## show_it_all
> check_greet
> check_hello                   <!-- allows multiple entry points -->

* _next_intent            
    - utter_greet              <!-- actions taken by the bot -->
    
> check_intermediate            <!-- allows intermediate checkpoints -->

* _change_bank_details
    - utter_default            <!-- allows to end without checkpoints -->

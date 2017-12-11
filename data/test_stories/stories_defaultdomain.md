## simple_story_with_only_start
> check_greet                   <!-- checkpoints at the start define entry points -->
* default
    - utter_default

## simple_story_with_only_end
* greet OR greet{"name": "Peter"}
    - utter_greet
> check_greet                   <!-- checkpoint defining the end of this turn -->

## simple_story_with_multiple_turns
* greet
    - utter_greet
* default
    - utter_default
* goodbye
    - utter_goodbye

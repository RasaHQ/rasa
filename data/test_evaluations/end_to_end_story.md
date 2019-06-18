## simple_story_with_only_start
> check_greet                   <!-- checkpoints at the start define entry points -->
* default:/default
    - utter_default

## simple_story_with_only_end
* greet:/greet
    - utter_greet
> check_greet                   <!-- checkpoint defining the end of this turn -->

## simple_story_with_multiple_turns
* greet:/greet
    - utter_greet
* default:/default
    - utter_default
* goodbye:/goodbye
    - utter_goodbye

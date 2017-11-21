## simple_story_with_only_start
> check_greet                   <!-- checkpoints at the start define entry points -->
* _default
    - utter_default

## simple_story_with_only_end
* _greet  OR _greet[name=Peter]
    - utter_greet
> check_greet                   <!-- checkpoint defining the end of this turn -->

## simple_story_with_multiple_turns
* _greet
    - utter_greet
* _default
    - utter_default
* _goodbye
    - utter_goodbye

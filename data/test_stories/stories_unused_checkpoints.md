## simple_story_with_end_checkpoint_1
* simple
    - utter_default
    - utter_greet
> check_end_1

## simple_story_with_end_checkpoint_2
* hello
    - utter_greet
    - slot{"name": "peter"}
> check_end_2

## simple_story_with_start
> check_start
* hello
    - utter_greet
    - slot{"name": "peter"}

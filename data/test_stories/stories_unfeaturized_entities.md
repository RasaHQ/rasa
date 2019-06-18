## simple_story
* greet{"name": "Peter", "unrelated_recognized_entity": "whatever", "other": "foo"}
    - utter_greet

## simple_story_with_multiple_turns
* greet
    - utter_greet
* default{"name": "Peter", "unrelated_recognized_entity": "whatever", "other": "foo"}
    - utter_default
* goodbye{"name": "Peter", "unrelated_recognized_entity": "whatever", "other": "foo"}
    - utter_goodbye
* thank{"name": "Peter", "unrelated_recognized_entity": "whatever", "other": "foo"}
    - utter_default
* ask{"name": "Peter", "unrelated_recognized_entity": "whatever", "other": "foo"}
    - utter_default

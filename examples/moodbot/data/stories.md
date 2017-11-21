## happy path               <!-- name of the story - just for debugging -->
* _greet              
  - utter_greet
* _mood_great               <!-- user utterance, in format _intent[entities] -->
  - utter_happy

## sad path 1               <!-- this is already the start of the next story -->
* _greet
  - utter_greet             <!-- action of the bot to execute -->
* _mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* _mood_affirm
  - utter_happy

## sad path 2
* _greet
  - utter_greet
* _mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* _mood_deny
  - utter_goodbye

## say goodbye
* _goodbye
  - utter_goodbye

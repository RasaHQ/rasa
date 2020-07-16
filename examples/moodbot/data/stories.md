## happy path               <!-- name of the story - just for debugging -->
* mood_great                <!-- user utterance, in the following format: * intent{"entity_name": value} -->
  - utter_happy             <!-- action of the bot to execute -->

## sad path 1               <!-- this is already the start of the next story -->            
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* affirm
  - utter_happy

## sad path 2
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* deny
  - utter_goodbye


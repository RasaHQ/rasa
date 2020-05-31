## happy path               <!-- name of the story - just for debugging -->
* greet: hi
  - utter_greet
* mood_great: amazing               <!-- user utterance, in the following format: * intent{"entity_name": value} -->
  - utter_happy

## sad path 1               <!-- this is already the start of the next story -->
* greet: hi
  - utter_greet             <!-- action of the bot to execute -->
* mood_unhappy: my day was horrible
  - utter_cheer_up
  - utter_did_that_help
* affirm: yes
  - utter_happy

## sad path 2
* greet: goodmorning
  - utter_greet
* mood_unhappy: I am sad
  - utter_cheer_up
  - utter_did_that_help
* deny: I don't think so
  - utter_goodbye

## say goodbye
* goodbye: bye
  - utter_goodbye

## bot challenge
* bot_challenge: are you human?
  - utter_iamabot

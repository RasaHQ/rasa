<!-- a
fancy
multiline
comment -->
## happy path
* greet
  - utter_greet
* mood_great
  - utter_happy

## sad path 1
* greet                 <!-- inline comment -->
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* affirm
  - utter_happy

<!-- just one line comment-->
## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* deny
  - utter_goodbye

<!--
one more
-->

## say goodbye
* goodbye
  - utter_goodbye
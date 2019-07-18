## simple_story_with_only_start
* greet: Hello
    - utter_ask_howcanhelp

## simple_story_with_multiple_turns
* greet: good morning
 - utter_ask_howcanhelp
* inform: im looking for a [moderately](price:moderate) priced restaurant in the [east](location) part of town
 - utter_on_it
 - utter_ask_cuisine
* inform: [french](cuisine) food
 - utter_ask_numpeople
 
 ## story_with_multiple_entities_correction_and_search
* greet: hello
 - utter_ask_howcanhelp
* inform: im looking for a [cheap](price:lo) restaurant which has [french](cuisine) food and is located in [bombay](location)
 - utter_on_it
 - utter_ask_numpeople
* inform: for [six](people:6) please
 - utter_ask_moreupdates
* inform: actually i need a [moderately](price:moderate) priced restaurant
 - utter_ask_moreupdates
* deny: no
 - utter_ack_dosearch
 - action_search_restaurants
 - action_suggest
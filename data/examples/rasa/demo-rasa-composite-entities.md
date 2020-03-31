<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

  - [intent:affirm](#intentaffirm)
  - [intent:goodbye](#intentgoodbye)
  - [intent:greet](#intentgreet)
  - [intent:chitchat/ask_name](#intentchitchatask_name)
  - [intent:chitchat/ask_weather](#intentchitchatask_weather)
  - [intent:restaurant_search](#intentrestaurant_search)
- [intent:order_pizza](#intentorder_pizza)
  - [synonym:chinese](#synonymchinese)
  - [synonym:vegetarian](#synonymvegetarian)
  - [regex:zipcode](#regexzipcode)
  - [regex:greet](#regexgreet)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## intent:affirm
- yes
- yep
- yeah
- indeed
- that's right
- ok
- great
- right, thank you
- correct
- great choice
- sounds really good

## intent:goodbye
- bye
- goodbye
- good bye
- stop
- end
- farewell
- Bye bye
- have a good one

## intent:greet
- hey
- howdy
- hey there
- hello
- hi
- good morning
- good evening
- dear sir

## intent:chitchat/ask_name
- What's your name?
- What can I call you?

## intent:chitchat/ask_weather
- How's the weather?
- Is it too hot outside?

## intent:restaurant_search
- i'm looking for a place to eat
- I want to grab lunch
- I am searching for a dinner spot
- i'm looking for a place in the [north](location) of town
- show me [italian]{"entity": "cuisine", "role": "european"} restaurants
- show me [chines]{"entity": "cuisine", "value": "chinese", "role": "asian"} restaurants in the [north](location)
- show me a [mexican]{"entity": "cuisine", "role": "latin america"} place in the [centre](location)
- i am looking for an [indian]{"entity": "cuisine", "role": "asian"} spot called olaolaolaolaolaola
- search for restaurants
- anywhere in the [west](location)
- anywhere near [18328](location)
- I am looking for [asian fusion](cuisine) food
- I am looking a restaurant in [29432](location)
- I am looking for [mexican indian fusion](cuisine)
- [central](location) [indian]{"entity": "cuisine", "role": "asian"} restaurant

# intent:order_pizza
- i want a [large]{"entity": "size", "group": "1"} pizza with [tomato]{"entity": "topping", "group": "1"} and a [small]{"entity": "size", "group": "2"} pizza with [bacon]{"entity": "topping", "group": "2"}
- one [large]{"entity": "size", "group": "1"} with [pepperoni]{"entity": "topping", "group": "1"} and a [medium]{"entity": "size", "group": "2"} with [mushrooms]{"entity": "topping", "group": "2"}
- I would like a [medium]{"entity": "size", "group": "1"} standard pizza and a [medium]{"entity": "size", "group": "2"} pizza with [extra cheese]{"entity": "topping", "group": "2"}
- [large]{"entity": "size", "group": "1"} with [onions]{"entity": "topping", "group": "1"} and [small]{"entity": "size", "group": "1"} with [olives]{"entity": "topping", "group": "1"}
- a pizza with [onions]{"entity": "topping", "group": "1"} in [medium]{"entity": "size", "group": "1"} and one with [mushrooms]{"entity": "topping", "group": "2"} in [small]{"entity": "size", "group": "2"} please

## synonym:chinese
+ Chines
* Chinese

## synonym:vegetarian
- vegg
- veggie

## regex:zipcode
- [0-9]{5}

## regex:greet
- hey[^\s]*
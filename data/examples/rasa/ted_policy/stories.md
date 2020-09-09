## SCENARIO CHECK
* greet
  - utter_SCENARIOCHECK

## faq
* faq
  - respond_faq
* thank
  - utter_express_positive-emo
* faq
  - respond_faq

## Story from conversation with b59c8f85-caea-4143-bae0-62421e93b664 on October 27th 2019
* greet
    - utter_SCENARIOCHECK
* SCENARIO{"context_scenario":"holiday","holiday_name":"thanksgiving"}
    - slot{"context_scenario":"holiday"}
    - slot{"holiday_name":"thanksgiving"}
    - action_start
    - slot{"link_1_url":"https://rasa.com/carbon/index.html?&rasaxhost=http://localhost:5002&conversationId=b59c8f85-caea-4143-bae0-62421e93b664&destination=https://offset.climateneutralnow.org/allprojects&label=link-1-clicked"}
    - slot{"link_2_url":"https://rasa.com/carbon/index.html?&rasaxhost=http://localhost:5002&conversationId=b59c8f85-caea-4143-bae0-62421e93b664&destination=https://offset.climateneutralnow.org/allprojects&label=link-2-clicked"}
    - action_disclaimer
    - utter_holiday-travel_offer_help
* why
    - slot{"context_scenario":"holiday"}
    - slot{"holiday_name":"thanksgiving"}
    - utter_explain_why_offset_travel
    - action_explain_typical_emissions
    - utter_ask_detailed_estimate
* affirm
    - airtravel_form
    - form{"name":"airtravel_form"}
    - slot{"requested_slot":"travel_flight_class"}
* affirm
    - airtravel_form
    - form{"name":"airtravel_form"}
    - slot{"requested_slot":"travel_flight_class"}
* why
    - utter_explain_economy_class
    - airtravel_form
    - slot{"requested_slot":"travel_flight_class"}
* affirm
    - airtravel_form
    - slot{"travel_flight_class":"economy"}
    - slot{"requested_slot":"travel_departure"}
* inform
    - airtravel_form
    - slot{"travel_departure":"Seattle"}
    - slot{"requested_slot":"travel_destination"}
* inform{"city":"Grand Rapids"}
    - slot{"city":"Grand Rapids"}
    - airtravel_form
    - slot{"travel_destination":"Grand Rapids"}
    - form{"name":null}
    - slot{"requested_slot":null}
* express_surprise
    - utter_explain_offset_calculation
* thank
    - utter_express_positive-emo
    - utter_farewell

## interactive_story_1
* greet
    - utter_SCENARIOCHECK
* SCENARIO{"context_scenario": "holiday", "holiday_name": "christmas"}
    - slot{"context_scenario": "holiday"}
    - slot{"holiday_name": "christmas"}
    - action_start
    - slot{"link_1_url": "https://rasa.com/carbon/index.html?&rasaxhost=https://carbon.rasa.com&conversationId=3bc0234bd09447aaaaaa9beafa9550f3&destination=https://offset.climateneutralnow.org/allprojects&label=link-1-clicked"}
    - slot{"link_2_url": "https://rasa.com/carbon/index.html?&rasaxhost=https://carbon.rasa.com&conversationId=3bc0234bd09447aaaaaa9beafa9550f3&destination=https://offset.climateneutralnow.org/allprojects&label=link-2-clicked"}
    - action_disclaimer
    - utter_holiday-travel_offer_help
* affirm
    - utter_explain_why_offset_travel
    - action_explain_typical_emissions
    - utter_ask_detailed_estimate
* affirm
    - airtravel_form
    - form{"name": "airtravel_form"}
    - slot{"requested_slot": "travel_flight_class"}
* form: affirm
    - form: airtravel_form
    - slot{"travel_flight_class": "economy"}
    - slot{"requested_slot": "travel_departure"}
* form: inform{"city": "Auckland"}
    - slot{"city": ["Auckland"]}
    - form: airtravel_form
    - slot{"travel_departure": "Auckland International Airport"}
    - slot{"iata_departure": "AKL"}
    - slot{"requested_slot": "travel_destination"}
* form: inform{"city": "Glasgow"}
    - slot{"city": ["Glasgow"]}
    - form: airtravel_form
    - slot{"travel_destination": "Wokal Field Glasgow International Airport"}
    - slot{"iata_destination": "GGW"}
    - form{"name": null}
    - slot{"requested_slot": null}
* thank
    - utter_express_positive-emo
    - utter_farewell

## filling travel plan before airtravel_form
* SCENARIO{"context_scenario": "holiday", "holiday_name": "thanksgiving"}
    - slot{"context_scenario": "holiday"}
    - slot{"holiday_name": "thanksgiving"}
    - action_start
    - slot{"link_1_url": "..."}
    - slot{"link_2_url": "..."}
    - action_disclaimer
    - utter_holiday-travel_offer_help
* inform{"city": ["berlin", "Madrid"]}
    - slot{"city": ["berlin", "Madrid"]}
    - airtravel_form
    - form{"name": "airtravel_form"}
    - slot{"travel_departure": "Berlin-Schönefeld Airport"}
    - slot{"travel_destination": "Adolfo Suárez Madrid–Barajas Airport"}
    - slot{"iata_departure": "SXF"}
    - slot{"iata_destination": "MAD"}
    - slot{"requested_slot": "travel_flight_class"}
* form: affirm
    - form: airtravel_form
    - slot{"travel_flight_class": "economy"}
    - form{"name": null}
    - slot{"requested_slot": null}

## stopover with via
* greet
    - utter_SCENARIOCHECK
* SCENARIO{"context_scenario": "holiday", "holiday_name": "christmas"}
    - slot{"context_scenario": "holiday"}
    - slot{"holiday_name": "christmas"}
    - action_start
    - slot{"link_1_url": "https://rasa.com/carbon/index.html?&rasaxhost=https://carbon.rasa.com&conversationId=3bc0234bd09447aaaaaa9beafa9550f3&destination=https://offset.climateneutralnow.org/allprojects&label=link-1-clicked"}
    - slot{"link_2_url": "https://rasa.com/carbon/index.html?&rasaxhost=https://carbon.rasa.com&conversationId=3bc0234bd09447aaaaaa9beafa9550f3&destination=https://offset.climateneutralnow.org/allprojects&label=link-2-clicked"}
    - action_disclaimer
    - utter_holiday-travel_offer_help
* affirm
    - utter_explain_why_offset_travel
    - action_explain_typical_emissions
    - utter_ask_detailed_estimate
* affirm
    - airtravel_form
    - form{"name": "airtravel_form"}
    - slot{"requested_slot": "travel_flight_class"}
* form: affirm
    - form: airtravel_form
    - slot{"travel_flight_class": "economy"}
    - slot{"requested_slot": "travel_departure"}
* form: inform{"city": ["Berlin", "Paris", "New York"]}
    - slot{"city": ["Berlin", "Paris", "New York"]}
    - form: airtravel_form
    - slot{"travel_departure": "Berlin-Schönefeld Airport"}
    - slot{"iata_departure": "SXF"}
    - slot{"travel_destination": "Charles de Gaulle International Airport"}
    - slot{"iata_destination": "CDG"}
    - slot{"travel_stopover": "John F Kennedy International Airport"}
    - slot{"iata_stopover": "JFK"}
    - form{"name": null}
    - slot{"requested_slot": null}
* thank
    - utter_express_positive-emo
    - utter_farewell

## stopover with second leg
* greet
    - utter_SCENARIOCHECK
* SCENARIO{"context_scenario": "holiday", "holiday_name": "christmas"}
    - slot{"context_scenario": "holiday"}
    - slot{"holiday_name": "christmas"}
    - action_start
    - slot{"link_1_url": "https://rasa.com/carbon/index.html?&rasaxhost=https://carbon.rasa.com&conversationId=3bc0234bd09447aaaaaa9beafa9550f3&destination=https://offset.climateneutralnow.org/allprojects&label=link-1-clicked"}
    - slot{"link_2_url": "https://rasa.com/carbon/index.html?&rasaxhost=https://carbon.rasa.com&conversationId=3bc0234bd09447aaaaaa9beafa9550f3&destination=https://offset.climateneutralnow.org/allprojects&label=link-2-clicked"}
    - action_disclaimer
    - utter_holiday-travel_offer_help
* affirm
    - utter_explain_why_offset_travel
    - action_explain_typical_emissions
    - utter_ask_detailed_estimate
* affirm
    - airtravel_form
    - form{"name": "airtravel_form"}
    - slot{"requested_slot": "travel_flight_class"}
* form: deny
    - form: airtravel_form
    - slot{"travel_flight_class": "business"}
    - slot{"requested_slot": "travel_departure"}
* form: inform{"city": "London"}
    - slot{"city": ["London"]}
    - form: airtravel_form
    - slot{"travel_departure": "London Luton Airport"}
    - slot{"iata_departure": "LTN"}
    - slot{"requested_slot": "travel_destination"}
* form: inform{"city": "Madrid"}
    - slot{"city": ["Madrid "]}
    - form: airtravel_form
    - slot{"travel_destination": "Adolfo Suárez Madrid–Barajas Airport"}
    - slot{"iata_destination": "MAD"}
    - slot{"previous_entered_flight": [["London Luton Airport", "LTN"], ["Adolfo Suárez Madrid–Barajas Airport", "MAD"]]}
    - form{"name": null}
    - slot{"requested_slot": null}
    - action_listen
* inform{"city": ["Madrid", "New York"]}
    - slot{"city": ["Madrid", "New York"]}
    - airtravel_form
    - form{"name": "airtravel_form"}
    - slot{"travel_departure": "London Luton Airport"}
    - slot{"travel_destination": "John F Kennedy International Airport"}
    - slot{"travel_stopover": "Adolfo Suárez Madrid–Barajas Airport"}
    - slot{"iata_departure": "LTN"}
    - slot{"iata_destination": "JFK"}
    - slot{"iata_stopover": "MAD"}
    - slot{"requested_slot": "travel_flight_class"}
* form: affirm
    - form: airtravel_form
    - slot{"travel_flight_class": "economy"}
    - form{"name": null}
    - slot{"requested_slot": null}
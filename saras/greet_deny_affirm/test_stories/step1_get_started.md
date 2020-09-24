## prompt for getting started
* get_started_step1
    - action_greet_user

## prompt for getting started + confirm
* get_started_step1
    - action_greet_user
<U> accept
    - utter_getstarted
    - utter_first_bot_with_rasa

## new to rasa at start, built bot before
* how_to_get_started{"user_type": "new"}
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_getstarted_new
    - utter_built_bot_before
<U> ook
    - utter_ask_migration

## new to rasa at start
* how_to_get_started{"user_type": "new"}
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_getstarted_new
    - utter_built_bot_before
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2
## new to rasa at start
* how_to_get_started{"user_type": "new"}
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_getstarted_new
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product":"rasa"}
    - utter_explain_nlu
    - utter_explain_core


## new to rasa + built a bot before
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yepp
    - utter_ask_migration
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> cool beans
    - utter_ask_migration
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + migrating from dialogflow
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yeah do that
    - utter_ask_migration
* switch{"current_api": "dialogflow"}
    - utter_switch_dialogflow
    - utter_anything_else

## new to rasa + not new to chatbots + migrating from luis
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yep i want that
    - utter_ask_migration
* switch{"current_api": "luis"}
    - utter_switch_luis
    - utter_anything_else

## new to rasa + not new to chatbots + migrating from something else
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yesssss
    - utter_ask_migration
* switch
    - action_store_unknown_product
    - utter_no_guide_for_switch
    - utter_anything_else

##  migrating from dialogflow
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> fair enough
    - utter_ask_migration
<U> fair enough
    - utter_ask_which_tool
* switch{"current_api": "dialogflow"}
    - utter_switch_dialogflow
    - utter_anything_else

## new to rasa + not new to chatbots + migrating from luis
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yep i want that
    - utter_ask_migration
<U> yes go ahead
    - utter_ask_which_tool
* switch{"current_api": "luis"}
    - utter_switch_luis
    - utter_anything_else

## new to rasa + not new to chatbots + migrating from something else
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> alright
    - utter_ask_migration
<U> cool beans
    - utter_ask_which_tool
* switch OR enter_data
    - action_store_unknown_product
    - utter_no_guide_for_switch
    - utter_anything_else


## just switch
* switch{"current_api":"tensorflow"}
    - slot{"current_api":"__other__"}
    - action_store_unknown_product
    - utter_no_guide_for_switch
    - utter_anything_else

## just switch
* switch
    - utter_ask_which_tool
* switch{"current_api":"tensorflow"}
    - slot{"current_api":"__other__"}
    - action_store_unknown_product
    - utter_no_guide_for_switch
    - utter_anything_else

## just switch
* switch
    - utter_ask_which_tool
* switch OR enter_data
    - action_store_unknown_product
    - utter_no_guide_for_switch
    - utter_anything_else


## new to rasa/bots, explain stack
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa/bots, explain core
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> no i won't
    - utter_direct_to_step2

## new to rasa/bots, explain core
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product":"rasa"}
    - utter_explain_nlu
    - utter_explain_core
## new to rasa/bots, explain x, then nlu+core
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
<U> agreed
    - utter_explain_nlu
    - utter_explain_core
    - utter_direct_to_step2


## new to rasa/bots, explain x, then nlu+core
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product":"rasa"}
    - utter_explain_nlu
    - utter_explain_core

## new to rasa/bots, explain x
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
<U> not yet
    - utter_direct_to_step2

## new to rasa/bots, explain nlu
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "nlu"}
    - utter_explain_nlu
    - utter_also_explain_core
<U> no i won't
    - utter_direct_to_step2

## new to rasa/bots, explain nlu then core
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "nlu"}
    - utter_explain_nlu
    - utter_also_explain_core
<U> alright
    - utter_explain_core
    - utter_direct_to_step2

## not new to rasa/bots, explain all
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> na
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## not new to rasa/bots, explain all
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* faq{"product":"rasa"}
    - utter_explain_core
    - utter_explain_nlu



## new to rasa/bots, explain all
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa/bots, explain all
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product":"rasa"}
    - utter_explain_nlu
    - utter_explain_core

## new to rasa/bots, don't explain
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* affirm OR how_to_get_started{"user_type": "new"} OR explain
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> na
    - utter_direct_to_step2

## not new to rasa + not interested in products
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no. u r idiot
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
<U> no. u r idiot
    - utter_thumbsup

## not new to rasa + core
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> na
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_anything_else

## not new to rasa + rasa
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu

## not new to rasa + rasa
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* faq{"product": "rasa"}
    - slot{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu

## not new to rasa + rasax
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no i won't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* faq{"product": "rasa"}
    - slot{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu

## not new to rasa + rasa
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no i won't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu

## not new to rasa + rasa
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* faq{"product": "rasa"}
    - slot{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu

## not new to rasa + rasax
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no. u r idiot
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* faq{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore

## skip to info on rasa x
* how_to_get_started{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore

## skip to info on rasa
* how_to_get_started{"product": "rasa"}
    - utter_explain_nlu
    - utter_explain_core

## skip to info on rasa nlu
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation

## not new to rasa + x
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no i won't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore

## new to rasa + x
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> hell yeah
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> ja thats great
    - utter_ask_migration
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product":"x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
<U> yes give me information
    - utter_explain_nlu
    - utter_explain_core
    - utter_direct_to_step2

## new to rasa + rasa oss
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yyeeeh
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> cool beans
    - utter_ask_migration
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "rasa"}
    - utter_explain_nlu
    - utter_explain_core

## new to rasa + rasa oss
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yyeeeh
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> ook
    - utter_ask_migration
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product": "rasa"}
    - slot{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu

## new to rasa + rasa oss
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> that is cool
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> ok great
    - utter_ask_migration
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
    
## skip to info on core
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_anything_else

## not new to rasa + nlu + nothing special
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
<U> na
    - utter_tutorialnlu
    - utter_anything_else

## not new to rasa + nlu + unknown topic
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> na
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info
    - action_store_unknown_nlu_part
    - utter_dont_know_nlu_part
    - utter_search_bar
    - utter_anything_else

## not new to rasa + nlu + intent + no recommendation
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no i won't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
<U> not yet
    - utter_thumbsup
    - utter_anything_else


## pipeline
* pipeline_recommendation
    - utter_what_language
* enter_data{"language":"english"}
    - slot{"language": "english"}
    - utter_pipeline_english
    - utter_anything_else

## pipeline
* pipeline_recommendation
    - utter_what_language
* enter_data{"language":"spanish"}
    - slot{"language": "__other__"}
    - action_store_bot_language
    - slot{"can_use_spacy":true}
    - utter_pipeline_nonenglish_spacy
    - utter_anything_else

## pipeline
* pipeline_recommendation
    - utter_what_language
* enter_data{"language":"russian"}
    - slot{"language": "__other__"}
    - action_store_bot_language
    - slot{"can_use_spacy":false}
    - utter_pipeline_nonenglish_nospacy
    - utter_anything_else

## pipeline
* pipeline_recommendation
    - utter_what_language
* enter_data
    - action_store_bot_language
    - slot{"can_use_spacy":true}
    - utter_pipeline_nonenglish_spacy
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, spacy
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no. u r idiot
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_what_language
* enter_data{"language":"spanish"}
    - slot{"language": "__other__"}
    - action_store_bot_language
    - slot{"can_use_spacy":true}
    - utter_pipeline_nonenglish_spacy
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, not spacy
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_what_language
* enter_data{"language": "russian"}
    - slot{"language": "__other__"}
    - action_store_bot_language
    - slot{"can_use_spacy":false}
    - utter_pipeline_nonenglish_nospacy
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, not spacy
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_what_language
* enter_data
    - action_store_bot_language
    - slot{"can_use_spacy":false}
    - utter_pipeline_nonenglish_nospacy
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, english
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_what_language
* enter_data{"language": "english"}
    - slot{"language": "english"}
    - utter_pipeline_english
    - utter_anything_else

## not new to rasa + nlu + intent + tool recommendation
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no i won't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
* nlu_generation_tool_recommendation
    - utter_nlu_tools

## not new to rasa + nlu + entity + no recommendation
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no i won't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "entity recognition"}
    - utter_nlu_entity_tutorial
    - utter_offer_recommendation
<U> na
    - utter_thumbsup
    - utter_anything_else

## not new to rasa + nlu + entity + pipeline spacy
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no. u r idiot
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "entity recognition"}
    - utter_nlu_entity_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_ask_entities
* enter_data{"entity": "name"}
    - action_store_entity_extractor
    - slot{"entity_extractor": "SpacyEntityExtractor"}
    - utter_spacy
    - utter_anything_else

## not new to rasa + nlu + entity + pipeline duckling
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "entity recognition"}
    - utter_nlu_entity_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_ask_entities
* enter_data{"entity": "date ranges"}
    - action_store_entity_extractor
    - slot{"entity_extractor": "DucklingHTTPExtractor"}
    - utter_duckling
    - utter_anything_else

## not new to rasa + nlu + entity + pipeline CRFEntityExtractor
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no. u r idiot
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "entity recognition"}
    - utter_nlu_entity_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_ask_entities
* enter_data{"entity": "some custom entity"}
    - action_store_entity_extractor
    - slot{"entity_extractor": "CRFEntityExtractor"}
    - utter_crf
    - utter_anything_else

## not new to rasa + nlu + entity + duckling info
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no. u r idiot
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* nlu_info{"nlu_part": "duckling"}
    - utter_duckling_info
    - utter_anything_else

## skip to info on nlu entities
* nlu_info{"nlu_part": "entity recognition"}
    - utter_nlu_entity_tutorial
    - utter_offer_recommendation

## skip to info on nlu intents
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation

## skip to info on nlu intents
* nlu_info{"nlu_part": "duckling"}
    - utter_duckling_info
    - utter_anything_else

## switch immediately to luis
* switch{"current_api":"luis"}
    - utter_switch_luis
    - utter_anything_else

## switch immediately to dialogflow
* switch{"current_api": "dialogflow"}
    - utter_switch_dialogflow
    - utter_anything_else

## faq
* faq
    - utter_faq
    - action_set_faq_slot

## source code
* source_code
    - utter_source_code
    - utter_anything_else

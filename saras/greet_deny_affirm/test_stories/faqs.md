## next step prompt
* next_step
    - action_next_step

## faqs
* faq
    - utter_faq
    - action_set_faq_slot

## faqs
* faq{"product": "x"}
    - slot{"product": "x"}
    - utter_faq
    - action_set_faq_slot

## faqs
* faq{"product": "rasa"}
    - slot{"product": "rasa"}
    - utter_faq
    - action_set_faq_slot

## faqs
* faq{"product": "nlu"}
    - slot{"product": "nlu"}
    - utter_faq
    - action_set_faq_slot

## faqs
* faq{"product": "core"}
    - slot{"product": "core"}
    - utter_faq
    - action_set_faq_slot

## faqs
* faq{"product": "all"}
    - slot{"product": "all"}
    - utter_faq
    - action_set_faq_slot


## more faqs
<U> helloooo
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* faq
    - utter_faq
    - action_set_faq_slot

## just newsletter
<U> hi hi
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> alright, cool
    - utter_thumbsup
    - utter_anything_else

## just newsletter, continue + confirm
<U> hlo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_continue_newsletter
<U> yes give me information
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> yyeeeh
    - utter_thumbsup
    - utter_anything_else

## just newsletter, don't continue + confirm
<U> hello sweatheart
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_continue_newsletter
<U> na
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> yea
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already) + confirm
<U> ey boss
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* signup_newsletter{"email": "maxmeier@firma.de"}
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> lets do it
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already)
<U> ey boss
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* signup_newsletter{"email": "maxmeier@firma.de"}
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just sales
<U> hi hi
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback

## just sales, continue
<U> greet
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_continue_sales
<U> yeah do that
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, don't continue
<U> hellllooooooo
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_continue_sales
<U> not yet
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## not new to rasa + not interested in products
<U> greet
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_first_bot_with_rasa
<U> na
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
<U> not yet
    - utter_thumbsup

## not new to rasa + not interested in products
<U> halloooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no. u r idiot
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_which_product
<U> no i won't
    - utter_thumbsup

## not new to rasa + nlu + nothing special
<U> hello world
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> na
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_for_nlu_specifics
<U> no i won't
    - utter_tutorialnlu
    - utter_anything_else

## not new to rasa + nlu + unknown topic
<U> hellllooooooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no. u r idiot
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_for_nlu_specifics
* nlu_info
    - action_store_unknown_nlu_part
    - utter_dont_know_nlu_part
    - utter_search_bar
    - utter_anything_else

## not new to rasa + nlu + intent + no recommendation
<U> hello world
    - action_greet_user
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
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_offer_recommendation
<U> no. u r idiot
    - utter_thumbsup
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> greet
    - action_greet_user
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
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_what_language
* enter_data{"language": "english"}
    - slot{"language": "english"}
    - utter_pipeline_english
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> ey boss
    - action_greet_user
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
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_what_language
* enter_data
    - action_store_bot_language
    - slot{"can_use_spacy":false}
    - utter_pipeline_nonenglish_nospacy
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> hi man
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> na
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
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_what_language
* enter_data{"language": "russian"}
    - slot{"language": "__other__"}
    - action_store_bot_language
    - slot{"can_use_spacy":false}
    - utter_pipeline_nonenglish_nospacy
    - utter_anything_else

## not new to rasa + nlu + entity + pipeline duckling
<U> hellooo
    - action_greet_user
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
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_ask_entities
* enter_data{"entity": "date ranges"}
    - action_store_entity_extractor
    - slot{"entity_extractor": "DucklingHTTPExtractor"}
    - utter_duckling
    - utter_anything_else

## not new to rasa + nlu + entity + pipeline duckling
<U> helloooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> around one millon euros
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
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_entities
* enter_data{"entity": "date ranges"}
    - action_store_entity_extractor
    - slot{"entity_extractor": "DucklingHTTPExtractor"}
    - utter_duckling
    - utter_anything_else

## FAQ - tell more about rasa x ee
* faq
    - utter_faq
    - action_set_faq_slot
* explain
    - action_explain_faq

## FAQ - tell more about languages
* faq
    - utter_faq
    - action_set_faq_slot
* explain
    - action_explain_faq

## FAQ - tell more about voice
* faq
    - utter_faq
    - action_set_faq_slot
* explain
    - action_explain_faq

## FAQ - tell more about slots
* faq
    - utter_faq
    - action_set_faq_slot
* explain
    - action_explain_faq

## FAQ - tell more about channels
* faq
    - utter_faq
    - action_set_faq_slot
* explain
    - action_explain_faq

## FAQ - Step 2 flow interrupted
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build

## FAQ - Step 2 flow interrupted
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_ready_to_build

## FAQ - Step 2 flow interrupted
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_ready_to_build
<U> yes it is
    - utter_run_rasa_init
    - utter_direct_to_step3

## FAQ - Step 2 flow interrupted
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_ready_to_build
<U> no. u r idiot
    - utter_ask_if_problem
* technical_question OR enter_data OR out_of_scope
    - action_store_problem_description
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## technical_question - docs_found - deny
* technical_question
    - action_docs_search
    - slot{"docs_found": true}
    - utter_ask_docs_help
<U> no i won't
    - action_tag_docs_search
    - action_forum_search

## technical_question - docs_found - affirm
* technical_question
    - action_docs_search
    - slot{"docs_found": true}
    - utter_ask_docs_help
<U> yep!
    - action_tag_docs_search
    - utter_great

## technical_question - no docs_found
* technical_question
    - action_docs_search
    - slot{"docs_found": false}
    - action_forum_search

## FAQ - Step 2 flow interrupted
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> no. u r idiot
    - utter_ask_if_problem
* faq
    - utter_faq
    - action_set_faq_slot
    - action_store_problem_description
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## FAQ - Step 2 flow interrupted
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> no. u r idiot
    - utter_ask_if_problem
<U> yep i want that
    - utter_ask_describe_problem
* faq
    - utter_faq
    - action_set_faq_slot
    - action_store_problem_description
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## FAQ - Step 2 flow interrupted
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> yes you can
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_ready_to_build

## Just don't want to continue
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_ready_to_build

## new to rasa + not new to chatbots + not migrating
<U> guten morgen
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yas
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> awesome!
    - utter_ask_migration
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2


## new to rasa + not new to chatbots + not migrating
<U> hi !
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> thats good
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i do
    - utter_ask_migration
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product": "rasa"}
    - slot{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu

## new to rasa + not new to chatbots + not migrating
<U> hi hi
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> hell yeah
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> ja
    - utter_ask_migration
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore

## new to rasa + not new to chatbots + not migrating
<U> ey boss
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_first_bot_with_rasa
<U> yep!
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> ook
    - utter_ask_migration
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> ey boss
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> agreed
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_built_bot_before
<U> yesss
    - utter_ask_migration
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> halloooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> hell yeah
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yes give me information
    - utter_ask_migration
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_migration
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> sup
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes that's great
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i get it
    - utter_ask_migration
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> hi hi
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i get it
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> how nice!
    - utter_ask_migration
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## new to rasa/bots, explain stack and try it out
<U> halloooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes it was okay
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* faq
    - utter_faq
    - action_set_faq_slot
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

## new to rasa/bots, explain stack and try it out
<U> hey bot!
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yep. :/
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa/bots, explain x and try it out
<U> hlo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> accept
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
<U> not yet
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## new to rasa/bots, explain core and direct to step2
<U> hey rasa
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes it was okay
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_also_explain_nlu
<U> not yet
    - utter_direct_to_step2

## new to rasa/bots, explain core and direct to step2
<U> hellllooooooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> hm, i'd like that
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> not yet
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## new to rasa/bots, explain core, then nlu and direct to step2
<U> halo sara
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> hell yes
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
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_also_explain_nlu
<U> that sounds fine
    - utter_explain_nlu
    - utter_direct_to_step2

## new to rasa/bots, explain x, then direct to step2
<U> hi !
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes give me information
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
* faq
    - utter_faq
    - action_set_faq_slot

## new to rasa/bots, explain core, then nlu and direct to step2
<U> hi there it's me
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> considering
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> yes go ahead
    - utter_explain_nlu
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## new to rasa/bots, explain nlu and direct to step2
<U> hey bot!
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yea
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "nlu"}
    - utter_explain_nlu
    - utter_also_explain_core
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_also_explain_core
<U> no i won't
    - utter_direct_to_step2

## new to rasa/bots, explain x and direct to step2
<U> hi !
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> oh awesome!
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
* faq
    - utter_faq
    - action_set_faq_slot
<U> not yet
    - utter_direct_to_step2

## new to rasa/bots, explain nlu and direct to step2
<U> guten morgen
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> fcourse
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
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## new to rasa/bots, don't explain and direct to step2
<U> greetings
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yyeeeh
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_explain_nlucorex
<U> no. u r idiot
    - utter_direct_to_step2

## new to rasa/bots, don't explain and direct to step2
<U> hello sweatheart
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> well yes
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> na
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## not new to rasa + not interested in products
<U> greet
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no i won't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
<U> not yet
    - utter_thumbsup

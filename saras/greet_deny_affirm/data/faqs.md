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
<U> good moring
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* faq
    - utter_faq
    - action_set_faq_slot

## just newsletter
<U> hi rasa
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
<U> yes baby
    - utter_thumbsup
    - utter_anything_else

## just newsletter, continue + confirm
<U> hellio
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_continue_newsletter
<U> kk
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> i will
    - utter_thumbsup
    - utter_anything_else

## just newsletter, don't continue + confirm
<U> halo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_continue_newsletter
<U> no way
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> ok let's start
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already) + confirm
<U> hola
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
<U> thats great
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already)
<U> helleo
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
<U> howdy
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
<U> hi folks
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_continue_sales
<U> it is ok
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, don't continue
<U> ola sara
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_continue_sales
<U> i don't want to give you my email
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## not new to rasa + not interested in products
<U> rasa hello
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_first_bot_with_rasa
<U> no i can't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
<U> no don"t want this!
    - utter_thumbsup

## not new to rasa + not interested in products
<U> heylo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no!!!!
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_which_product
<U> no i can't
    - utter_thumbsup

## not new to rasa + nlu + nothing special
<U> hi sara..
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i dont want to accept :p lol
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_for_nlu_specifics
<U> i don not like this
    - utter_tutorialnlu
    - utter_anything_else

## not new to rasa + nlu + unknown topic
<U> whats popping
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i do not need help installing
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
<U> heya
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i'm not going to give it to you
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
<U> i don't want either of those
    - utter_thumbsup
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> hell9o
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> sorry not right now
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
<U> hi mrs rasa
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> nein
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
<U> hi sara!
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no bots at all
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
<U> hey there
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> nope
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
<U> good mourning
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
<U> jezz
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
<U> not really
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
<U> nope
    - action_tag_docs_search
    - action_forum_search

## technical_question - docs_found - affirm
* technical_question
    - action_docs_search
    - slot{"docs_found": true}
    - utter_ask_docs_help
<U> go for it
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
<U> thanks but no thanks
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
<U> no bots at all
    - utter_ask_if_problem
<U> yeah'=
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
<U> done
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
<U> hi,
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> amayzing
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> go
    - utter_ask_migration
<U> nah i'm good
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2


## new to rasa + not new to chatbots + not migrating
<U> hhola
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> okie
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> sure!
    - utter_ask_migration
<U> no, i hate it
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product": "rasa"}
    - slot{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu

## new to rasa + not new to chatbots + not migrating
<U> hi again
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ok, i understood
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> fine
    - utter_ask_migration
<U> this sucks
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore

## new to rasa + not new to chatbots + not migrating
<U> hello sara
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_first_bot_with_rasa
<U> its okay
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yes accept please
    - utter_ask_migration
<U> no stop
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> hello it is me again
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes go for it
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_built_bot_before
<U> okay sure
    - utter_ask_migration
<U> i do not need help installing
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> hieee
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i am using it
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i guess so
    - utter_ask_migration
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_migration
<U> i don't want to say
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> hi pal!
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes please
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yes of course
    - utter_ask_migration
<U> no, not really.
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
<U> hi, bot
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> you got me, i accept, if you want me to
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i'm sure i will!
    - utter_ask_migration
<U> no sir
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
<U> hello is anybody there
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes pls
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_built_bot_before
<U> i'm not going to give it to you
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa/bots, explain stack and try it out
<U> helleo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yop
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> nehi
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
<U> hey, let's talk
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> k
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i decline
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
<U> nah, first time
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## new to rasa/bots, explain core and direct to step2
<U> hello everybody
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> nein
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
<U> no and no again
    - utter_direct_to_step2

## new to rasa/bots, explain core and direct to step2
<U> whats up?
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> jo
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> deny
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> nope
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## new to rasa/bots, explain core, then nlu and direct to step2
<U> hii
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes'
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> this sucks
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
<U> i changed my mind. i want to accept it
    - utter_explain_nlu
    - utter_direct_to_step2

## new to rasa/bots, explain x, then direct to step2
<U> heya
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i'm sure i will!
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i don't want either of those
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
<U> whats up?
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes great
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i'm afraid not
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> jezz
    - utter_explain_nlu
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## new to rasa/bots, explain nlu and direct to step2
<U> hieee
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yap
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> nein
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
<U> it sucks
    - utter_direct_to_step2

## new to rasa/bots, explain x and direct to step2
<U> well hello there ;)
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes great
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no thank s
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
<U> i dont want to tell
    - utter_direct_to_step2

## new to rasa/bots, explain nlu and direct to step2
<U> hei
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> si
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> nevermind
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "nlu"}
    - utter_explain_nlu
    - utter_also_explain_core
<U> i don't want to give it to you
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## new to rasa/bots, don't explain and direct to step2
<U> ola sara
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> going super well
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> absolutely not
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_explain_nlucorex
<U> no sorry
    - utter_direct_to_step2

## new to rasa/bots, don't explain and direct to step2
<U> heylo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yres
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i guess it means - no
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> i don't want to say
    - utter_direct_to_step2
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_direct_to_step2

## not new to rasa + not interested in products
<U> wasssup!
    - action_greet_user
* faq
    - utter_faq
    - action_set_faq_slot
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i don't agree
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
<U> noooooooooooooooooooooooooooooooooooooooo
    - utter_thumbsup

## chitchat
* chitchat
    - utter_chitchat

## deny chitchat
* chitchat
    - utter_chitchat
<U> no. u r idiot
    - utter_nohelp

## more chitchat
<U> hi there it's me
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat

## greet chitchat
<U> hello world
    - action_greet_user
* chitchat
    - utter_chitchat

## just newsletter + confirm
<U> hi !
    - action_greet_user
* chitchat
    - utter_chitchat
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> ja cool
    - utter_thumbsup
    - utter_anything_else

## just newsletter, continue, + confirm
<U> guten morgen
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_newsletter
<U> definitely yes without a doubt
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> oh awesome!
    - utter_thumbsup
    - utter_anything_else

## just newsletter, don't continue, + confirm
<U> hi !
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_newsletter
<U> not yet
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_ask_feedback
<U> yes give me information
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already) + confirm
<U> halloooo
    - action_greet_user
* chitchat
    - utter_chitchat
* signup_newsletter{"email": "maxmeier@firma.de"}
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> yesss
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already)
<U> hi man
    - action_greet_user
* chitchat
    - utter_chitchat
* signup_newsletter{"email": "maxmeier@firma.de"}
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter
<U> sup
    - action_greet_user
* chitchat
    - utter_chitchat
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter, continue
<U> greetings
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_newsletter
<U> yes i have!
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter, don't continue
<U> heelio
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_newsletter
<U> no i won't
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_ask_feedback

## just sales
<U> sup
    - action_greet_user
* chitchat
    - utter_chitchat
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback

## just sales, continue
<U> hey bot!
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_sales
<U> ok good
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, don't continue
<U> greetings
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_sales
<U> not yet
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## new to rasa + not new to chatbots + not migrating
<U> heelio
    - action_greet_user
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ook
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yes sirfr
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
* chitchat
    - utter_chitchat
    - utter_first_bot_with_rasa
<U> oh good !!
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> very much
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
<U> greet
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yess
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* chitchat
    - utter_chitchat
    - utter_built_bot_before
<U> ok fine
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
<U> hello sweatheart
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> alright
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* chitchat
    - utter_chitchat
    - utter_built_bot_before
<U> ok great
    - utter_ask_migration
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product": "rasa"}
    - slot{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu


## new to rasa + not new to chatbots + not migrating
<U> hello sweatheart
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ook
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> that is cool
    - utter_ask_migration
* chitchat
    - utter_chitchat
    - utter_ask_migration
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> hello sweatheart
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> more a less
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> hell yes
    - utter_ask_migration
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* chitchat
    - utter_chitchat
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> hlo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> well yes
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yes it was okay
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
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## new to rasa/bots, explain NLU and try it out
<U> helloooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> well yes
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* chitchat
    - utter_chitchat
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "nlu"}
    - utter_explain_nlu
    - utter_also_explain_core
<U> ook
    - utter_explain_core
    - utter_direct_to_step2

## new to rasa/bots, explain rasa x only
<U> hellllooooooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yess
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex

## new to rasa/bots, explain stack and try it out
<U> hello sweatheart
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> a little
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* chitchat
    - utter_chitchat
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa/bots, explain stack and try it out
<U> hi man
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes it is
    - action_set_onboarding
    - slot{"onboarding": true}
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
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## new to rasa/bots, explain core and try out stack
<U> greet
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> alright
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
* chitchat
    - utter_chitchat
    - utter_also_explain_nlu
<U> not yet
    - utter_direct_to_step2

## new to rasa/bots, explain core and try out stack
<U> hi there it's me
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> oh good !!
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
<U> no i won't
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## new to rasa/bots, explain core, then nlu and try out stack
<U> hi hi
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> that is cool
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
* chitchat
    - utter_chitchat
    - utter_also_explain_nlu
<U> yesssss
    - utter_explain_nlu
    - utter_direct_to_step2

## new to rasa/bots, explain core, then nlu and try out stack
<U> hi man
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> well yes
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
<U> yes it was okay
    - utter_explain_nlu
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## new to rasa/bots, explain nlu and try out stack
<U> hey rasa
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i agree
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "nlu"}
    - utter_explain_nlu
    - utter_also_explain_core
* chitchat
    - utter_chitchat
    - utter_also_explain_core
<U> no. u r idiot
    - utter_direct_to_step2

## new to rasa/bots, explain nlu and try out stack
<U> hellooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> alright
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
<U> na
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## new to rasa/bots, don't explain and try out stack
<U> hellooo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> sort of
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* chitchat
    - utter_chitchat
    - utter_ask_explain_nlucorex
<U> no i won't
    - utter_direct_to_step2

## new to rasa/bots, don't explain and try out stack
<U> hello, my name is [charles pfeffer](name)
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ja thats great
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> na
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## not new to rasa + not interested in products
<U> hi there it's me
    - action_greet_user
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no i won't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
<U> no. u r idiot
    - utter_thumbsup

## not new to rasa + not interested in products
<U> hey bot!
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* chitchat
    - utter_chitchat
    - utter_first_bot_with_rasa
<U> no i won't
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
<U> no i won't
    - utter_thumbsup

## not new to rasa + not interested in products
<U> sup
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no. u r idiot
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* chitchat
    - utter_chitchat
    - utter_ask_which_product
<U> not yet
    - utter_thumbsup

## not new to rasa + nlu + nothing special
<U> heelio
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
* chitchat
    - utter_chitchat
    - utter_ask_for_nlu_specifics
<U> no i won't
    - utter_tutorialnlu
    - utter_anything_else

## not new to rasa + rasa x + nothing special
<U> hello, my name is [charles pfeffer](name)
    - action_greet_user
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
* chitchat
    - utter_also_explain_nlucore
* affirm OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_direct_to_step2

## not new to rasa + rasa x + nothing special
<U> greet
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* chitchat
    - utter_chitchat
    - utter_first_bot_with_rasa
<U> boo
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
<U> no i won't
    - utter_direct_to_step2

## new to rasa + rasa x + nothing special
<U> hello sweatheart
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* chitchat
    - utter_chitchat
    - utter_first_bot_with_rasa
<U> thats good
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
<U> na
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## not new to rasa + nlu + unknown topic
<U> halo sara
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
* chitchat
    - utter_chitchat
    - utter_ask_for_nlu_specifics
* nlu_info
    - action_store_unknown_nlu_part
    - utter_dont_know_nlu_part
    - utter_search_bar
    - utter_anything_else

## not new to rasa + nlu + intent + no recommendation
<U> halo sara
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
* chitchat
    - utter_chitchat
    - utter_offer_recommendation
<U> simple bpt
    - utter_thumbsup
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> sup
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
* chitchat
    - utter_chitchat
    - utter_what_language
* enter_data{"language": "english"}
    - slot{"language": "english"}
    - utter_pipeline_english
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> halloooo
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
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_what_language
* chitchat
    - utter_chitchat
    - utter_what_language
* enter_data{"language": "russian"}
    - slot{"language": "__other__"}
    - action_store_bot_language
    - slot{"can_use_spacy":false}
    - utter_pipeline_nonenglish_nospacy
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> hello sweatheart
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
* nlu_info{"nlu_part": "intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_what_language
* chitchat
    - utter_chitchat
    - utter_what_language
* enter_data
    - action_store_bot_language
    - slot{"can_use_spacy":false}
    - utter_pipeline_nonenglish_nospacy
    - utter_anything_else

## not new to rasa + nlu + entity + pipeline duckling
<U> hi man
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
* nlu_info{"nlu_part": "entity recognition"}
    - utter_nlu_entity_tutorial
    - utter_offer_recommendation
* chitchat
    - utter_chitchat
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_ask_entities
* enter_data{"entity": "date ranges"}
    - action_store_entity_extractor
    - slot{"entity_extractor": "DucklingHTTPExtractor"}
    - utter_duckling
    - utter_anything_else

## not new to rasa + nlu + entity + pipeline duckling
<U> heelio
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
* nlu_info{"nlu_part": "entity recognition"}
    - utter_nlu_entity_tutorial
    - utter_offer_recommendation
* pipeline_recommendation OR affirm
    - utter_ask_entities
* chitchat
    - utter_chitchat
    - utter_ask_entities
* enter_data{"entity": "date ranges"}
    - action_store_entity_extractor
    - slot{"entity_extractor": "DucklingHTTPExtractor"}
    - utter_duckling
    - utter_anything_else

## how to get started without privacy policy
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* chitchat
    - utter_chitchat
    - utter_first_bot_with_rasa
<U> agreed
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yep. :/
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

## chitchat interrupting step 2 flow
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_rasa_init
<U> a little
    - utter_run_rasa_init

## chitchat interrupting and stopping step 2 flow
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_rasa_init
<U> not yet
    - utter_thumbsup
    - utter_anything_else

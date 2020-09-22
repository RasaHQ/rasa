## chitchat
* chitchat
    - utter_chitchat

## deny chitchat
* chitchat
    - utter_chitchat
<U> no :(
    - utter_nohelp

## more chitchat
<U> hallo sara
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat

## greet chitchat
<U> heyho
    - action_greet_user
* chitchat
    - utter_chitchat

## just newsletter + confirm
<U> hi mrs rasa
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
<U> i am!
    - utter_thumbsup
    - utter_anything_else

## just newsletter, continue, + confirm
<U> halo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_newsletter
<U> really
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> you asked me a yes or no question, which i answered with yes
    - utter_thumbsup
    - utter_anything_else

## just newsletter, don't continue, + confirm
<U> salut
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_newsletter
<U> never
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_ask_feedback
<U> sure!
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already) + confirm
<U> bonjour
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
<U> jezz
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already)
<U> hi there
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
<U> hell9o
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
<U> hello robot
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_newsletter
<U> not bad
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter, don't continue
<U> helo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_newsletter
<U> i'm not giving you my email address
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_ask_feedback

## just sales
<U> bonjour
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
<U> hi!
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_sales
<U> ya thats cool
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, don't continue
<U> hii
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_sales
<U> nah
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## new to rasa + not new to chatbots + not migrating
<U> hi mrs rasa
    - action_greet_user
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> thats fine
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> oh cool
    - utter_ask_migration
<U> nah
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> hey sara
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* chitchat
    - utter_chitchat
    - utter_first_bot_with_rasa
<U> k
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> yeeeeezzzzz
    - utter_ask_migration
<U> no way
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> hi again
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yessoo
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* chitchat
    - utter_chitchat
    - utter_built_bot_before
<U> ok sara
    - utter_ask_migration
<U> neither
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2
## new to rasa + not new to chatbots + not migrating
<U> hi.........................................................
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> okay cool
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* chitchat
    - utter_chitchat
    - utter_built_bot_before
<U> ok let's start
    - utter_ask_migration
<U> i don't want to say
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product": "rasa"}
    - slot{"product": "rasa"}
    - utter_explain_core
    - utter_explain_nlu


## new to rasa + not new to chatbots + not migrating
<U> hi there
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> amazing!
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i'd absolutely love that
    - utter_ask_migration
* chitchat
    - utter_chitchat
    - utter_ask_migration
<U> no don"t want this!
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* affirm OR how_to_get_started{"product":"all"} OR explain
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## new to rasa + not new to chatbots + not migrating
<U> hello robot
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ys
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> you asked me a yes or no question, which i answered with yes
    - utter_ask_migration
<U> i dont want to tell
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
<U> jojojo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> please
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i accept.
    - utter_ask_migration
<U> no i can't
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
<U> hey, sara!
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i'm using it
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
* chitchat
    - utter_chitchat
    - utter_built_bot_before
<U> no thank you
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "nlu"}
    - utter_explain_nlu
    - utter_also_explain_core
<U> yes that's what i want
    - utter_explain_core
    - utter_direct_to_step2

## new to rasa/bots, explain rasa x only
<U> hey there..
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yez
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no go
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex

## new to rasa/bots, explain stack and try it out
<U> heeey
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ye
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i'm not giving you my email address
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
<U> helo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> going super well
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> nevermind
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
<U> hi rasa
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ah ok
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> thanks but no thanks
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
* chitchat
    - utter_chitchat
    - utter_also_explain_nlu
<U> i'm not going to give it to you
    - utter_direct_to_step2

## new to rasa/bots, explain core and try out stack
<U> hey sara
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yap
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> i don't want to say
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> no stop
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## new to rasa/bots, explain core, then nlu and try out stack
<U> hiihihi
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> it is ok
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no you did it wrong
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
* chitchat
    - utter_chitchat
    - utter_also_explain_nlu
<U> ya i want
    - utter_explain_nlu
    - utter_direct_to_step2

## new to rasa/bots, explain core, then nlu and try out stack
<U> wasssup
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes.
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no bots at all
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> of course
    - utter_explain_nlu
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## new to rasa/bots, explain nlu and try out stack
<U> hi sara
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes,i am
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> nehi
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "nlu"}
    - utter_explain_nlu
    - utter_also_explain_core
* chitchat
    - utter_chitchat
    - utter_also_explain_core
<U> this sucks
    - utter_direct_to_step2

## new to rasa/bots, explain nlu and try out stack
<U> well hello there ;)
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> good.
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> this sucks
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "nlu"}
    - utter_explain_nlu
    - utter_also_explain_core
<U> i don't wanna tell the name of my company
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## new to rasa/bots, don't explain and try out stack
<U> whats popping
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yaah
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> not going well at all
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* chitchat
    - utter_chitchat
    - utter_ask_explain_nlucorex
<U> i'm not giving you my email address
    - utter_direct_to_step2

## new to rasa/bots, don't explain and try out stack
<U> i am [karen mease](name)
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yep, will do thank you
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> definitely not
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> no i dont want to
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## not new to rasa + not interested in products
<U> hi, bot
    - action_greet_user
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i don't agree
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
<U> nehi
    - utter_thumbsup

## not new to rasa + not interested in products
<U> hey there
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* chitchat
    - utter_chitchat
    - utter_first_bot_with_rasa
<U> i don't want to say
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
<U> nö
    - utter_thumbsup

## not new to rasa + not interested in products
<U> hey dude
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> neither
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* chitchat
    - utter_chitchat
    - utter_ask_which_product
<U> noooooooooooooooooooooooooooooooooooooooo
    - utter_thumbsup

## not new to rasa + nlu + nothing special
<U> ayyyy whaddup
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> n
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "nlu"}
    - utter_ask_for_nlu_specifics
* chitchat
    - utter_chitchat
    - utter_ask_for_nlu_specifics
<U> i don't want either of those
    - utter_tutorialnlu
    - utter_anything_else

## not new to rasa + rasa x + nothing special
<U> i am [karen mease](name)
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i do not need help installing
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
<U> hello!
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* chitchat
    - utter_chitchat
    - utter_first_bot_with_rasa
<U> na
    - action_set_onboarding
    - slot{"onboarding": false}
    - utter_ask_which_product
* how_to_get_started{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
<U> i don't want either of those
    - utter_direct_to_step2

## new to rasa + rasa x + nothing special
<U> good morning
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
* chitchat
    - utter_chitchat
    - utter_first_bot_with_rasa
<U> y
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> no, my frst time
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product": "x"}
    - slot{"product": "x"}
    - utter_explain_x
    - utter_also_explain_nlucore
<U> i'm not giving you my email address
    - utter_direct_to_step2
* chitchat
    - utter_chitchat
    - utter_direct_to_step2

## not new to rasa + nlu + unknown topic
<U> ssup?
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no thank you
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
<U> hi friends
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i dont want to
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
<U> na
    - utter_thumbsup
    - utter_anything_else

## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> hellio
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no, not really.
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
<U> jojojo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not really
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
<U> yo
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> nevermind
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
<U> heyho
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i don't want to
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
<U> ayyyy whaddup
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> it sux
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
<U> okay sure
    - action_set_onboarding
    - slot{"onboarding": true}
    - utter_built_bot_before
<U> ofcoure i do
    - utter_ask_migration
<U> no, not really.
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
<U> ok cool
    - utter_run_rasa_init

## chitchat interrupting and stopping step 2 flow
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
* chitchat
    - utter_chitchat
    - utter_ask_continue_rasa_init
<U> not going well at all
    - utter_thumbsup
    - utter_anything_else

## story number 1
<U> hello world
    - action_greet_user
* out_of_scope
    - utter_out_of_scope
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* thank
    - utter_noworries
    - utter_anything_else
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback

## story number 2
<U> ey boss
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* out_of_scope
    - utter_thumbsup
    - utter_anything_else

## story number 4
<U> hey bot
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* thank
    - utter_noworries
    - utter_anything_else
* chitchat
    - utter_chitchat
* out_of_scope
    - utter_out_of_scope
    - utter_possibilities

## story number 5
<U> hey bot!
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* thank
    - utter_noworries
    - utter_anything_else
* out_of_scope
    - utter_out_of_scope
    - utter_possibilities

## story number 6
<U> hellllooooooo
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat

## story number 7
<U> greet
    - action_greet_user
<U> heelio
    - action_greet_user
* chitchat
    - utter_chitchat
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback

## story number 8
<U> hi !
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities
* enter_data
    - utter_not_sure
    - utter_possibilities

## story number 9
<U> hi hi
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities
<U> no. u r idiot
    - utter_nohelp

## story number 11, continue
<U> guten morgen
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> not yet
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> yess
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## story number 11, don't continue
<U> hey bot!
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> not yet
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> 300 rupees
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## story number 12
<U> greetings
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* enter_data
    - utter_thumbsup
    - utter_anything_else

## story number 12, continue
<U> greetings
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> how nice!
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* enter_data
    - utter_thumbsup
    - utter_anything_else

## story number 12, don't continue
<U> greetings
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> not yet
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_ask_feedback
* enter_data
    - utter_thumbsup
    - utter_anything_else

## story number 14
<U> halo sara
    - action_greet_user
<U> halo sara
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat

## story number 15
<U> hlo
    - action_greet_user
* chitchat
    - utter_chitchat
* enter_data
    - utter_not_sure
    - utter_possibilities

## story number 17
<U> hey rasa
    - action_greet_user
<U> not yet
    - utter_nohelp
* out_of_scope
    - utter_out_of_scope
    - utter_possibilities
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat

## story number 18
<U> hello sweatheart
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
<U> no. u r idiot
    - utter_nohelp
* enter_data
    - utter_not_sure
    - utter_possibilities
<U> no. u r idiot
    - utter_nohelp
* out_of_scope
    - utter_out_of_scope
    - utter_possibilities
* enter_data{"number":5}
    - utter_not_sure
    - utter_possibilities
* enter_data
    - utter_not_sure
    - utter_possibilities

## Story from conversation with 00e7815f79e4413abb0dfb4b392f1099 on November 15th 2018
<U> hello sweatheart
    - action_greet_user
<U> hi hi
    - action_greet_user
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## story from linda
<U> hellllooooooo
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with dfbb633d10854f97b880a2496d632f0d on November 16th 2018
<U> hi man
    - action_greet_user
<U> hi hi
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with alan on November 16th 2018
* nlu_info{"nlu_part":"duckling"}
    - slot{"nlu_part":"duckling"}
    - utter_duckling_info
    - utter_anything_else
<U> ja cool
    - utter_what_help

## Story from conversation with alan on November 16th 2018 2
* nlu_info{"nlu_part":"intent classification"}
    - slot{"nlu_part":"intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
<U> cool beans
    - utter_what_language
* enter_data{"language":"spanish"}
    - slot{"language": "__other__"}
    - action_store_bot_language
    - slot{"can_use_spacy":true}
    - utter_pipeline_nonenglish_spacy
    - utter_anything_else
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with alan on November 16th 2018 2
* nlu_info{"nlu_part":"intent classification"}
    - slot{"nlu_part":"intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
<U> ok, i behave now
    - utter_what_language
* enter_data
    - action_store_bot_language
    - slot{"can_use_spacy":false}
    - utter_pipeline_nonenglish_nospacy
    - utter_anything_else

## Story from conversation with linda on November 15th 2018
<U> hlo
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with 477ddbe73e374eedb07104c5d9f42c31 on November 16th 2018
<U> sup
    - action_greet_user
<U> halo sara
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> hey rasa
    - action_greet_user

## Story from conversation with 4986d88ccb784dc19dc5a553a8e07890 on November 19th 2018
<U> ey boss
    - action_greet_user
<U> halloooo
    - action_greet_user
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yep. :/
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product":"core"}
    - slot{"product":"core"}
    - utter_explain_core
    - utter_also_explain_nlu
* how_to_get_started{"product":"core"}
    - slot{"product":"core"}
    - utter_direct_to_step2

## Story from conversation with 7830abb04e1c49809d89b0d420443928 on November 19th 2018
<U> hi there it's me
    - action_greet_user
* chitchat
    - utter_chitchat
* out_of_scope
    - utter_out_of_scope
* chitchat
    - utter_chitchat
* how_to_get_started{"product":"nlu"}
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with 29d264d8ce574a11bde572f0e79b73f3 on November 19th 2018
<U> hey bot!
    - action_greet_user
<U> hellooo
    - action_greet_user
* chitchat
    - utter_chitchat
<U> agreed
    - utter_thumbsup

## Story from conversation with 6fd65c93e374489f9c8d76697ab9c493 on November 19th 2018
<U> ey boss
    - action_greet_user
<U> halo sara
    - action_greet_user
* chitchat
    - utter_chitchat
<U> lets do it
    - utter_thumbsup

## Story from conversation with 35d1ecc91c364cbf8a6edf006e5d8c9a on November 19th 2018

<U> hellllooooooo
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
<U> no i won't
    - utter_thumbsup
    - utter_anything_else
<U> ja thats great
    - utter_thumbsup

## Story from conversation with 4c274f8d470e4b77adbfefe7cda7cad7 on October 27th 2018

<U> hey rasa
    - action_greet_user
<U> hello world
    - action_greet_user
<U> fair enough
    - utter_thumbsup
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with d041ba4b0a89479e9bb6a5007f2cdc87 on November 15th 2018
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
* enter_data
    - utter_thumbsup
    - utter_anything_else
* chitchat
    - utter_chitchat
* enter_data
    - utter_not_sure
    - utter_possibilities

## Story from conversation with dominik on November 19th 2018
<U> hello world
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> thats good
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> yes give me information
    - utter_ask_migration
* switch{"current_api":"luis"}
    - slot{"current_api":"luis"}
    - utter_switch_luis
    - utter_anything_else
* how_to_get_started{"product":"core"}
    - slot{"product":"core"}
    - utter_explain_core
    - utter_anything_else
* how_to_get_started{"product":"nlu"}
    - slot{"product":"nlu"}
    - utter_explain_nlu
    - utter_anything_else

## Story from conversation with dominik
* nlu_info{"nlu_part":"intent classification"}
    - slot{"nlu_part":"intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
<U> ye splease
    - utter_what_language
* enter_data{"language":"spanish"}
    - slot{"language": "__other__"}
    - action_store_bot_language
    - slot{"can_use_spacy":true}
    - utter_pipeline_nonenglish_spacy
    - utter_anything_else
* enter_data{"language":"french"}
    - slot{"language": "__other__"}
    - action_store_bot_language
    - slot{"can_use_spacy":true}
    - utter_pipeline_nonenglish_spacy
    - utter_anything_else

## Story from conversation with dominik on November 19th 2018
<U> hi hi
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> not yet
    - action_set_onboarding
    - slot{"onboarding":false}
    - utter_ask_which_product
* how_to_get_started{"product":"core"}
    - utter_explain_core
    - utter_anything_else

## Story from conversation with 201bb55841154f858f524a485f8816c3 on November 18th 2018

<U> hellooo
    - action_greet_user
* how_to_get_started{"product":"nlu"}
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with cd483ab3456d47dfb40bd1f51043fb54 on November 18th 2018

<U> hi !
    - action_greet_user
<U> hello, my name is [charles pfeffer](name)
    - action_greet_user
* how_to_get_started{"product":"core"}
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with cfa8bb9deaf0427498c662745431a282 on December 15th 2018
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
* chitchat
    - utter_chitchat
* enter_data
    - utter_not_sure
    - utter_possibilities

## Story from conversation with cdd14d763a664a5b95e998ce165bd86f on December 16th 2018
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
* enter_data
    - utter_not_sure
    - utter_possibilities
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> cool beans
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> no i won't
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> yess
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2
* enter_data
    - utter_not_sure
    - utter_possibilities

## Story from conversation with 67a8696eb5894b25a800b6cbd7a695bb on December 15th 2018
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* enter_data
    - utter_not_sure
    - utter_possibilities

## Story from conversation with 67a8696eb5894b25a800b6cbd7a695bb on December 15th 2018
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* enter_data
    - utter_not_sure
    - utter_possibilities
* faq
    - utter_faq
    - action_set_faq_slot
* enter_data
    - utter_not_sure
    - utter_possibilities

## Story from conversation with 67a8696eb5894b25a800b6cbd7a695bb on December 15th 2018
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* enter_data
    - utter_not_sure
    - utter_possibilities
* faq
    - utter_faq
    - action_set_faq_slot
* enter_data
    - utter_not_sure
    - utter_possibilities
* enter_data
    - utter_not_sure
    - utter_possibilities

## Story from conversation with 030829eb30ed4339985d7e71737f6c2d on January 1st 2019
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
<U> halo sara
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yes i have!
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> yep!
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2
* enter_data
    - utter_not_sure
    - utter_possibilities

## Story from conversation with e977c395a2404bef8ba7f0c3b7e65970 on January 8th 2019

* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
* enter_data
    - utter_not_sure
    - utter_possibilities
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> a little
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> yepp
    - utter_ask_migration
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product":"core"}
    - slot{"product":"core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> yea
    - utter_explain_nlu
    - utter_direct_to_step2

## Story from conversation with 602ddccbe2de4f46822e06e2781cf02d on January 8th 2019

* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
<U> greetings
    - action_greet_user
* out_of_scope
    - utter_out_of_scope
<U> thats good
    - utter_thumbsup
<U> ja
    - utter_thumbsup

## Story from conversation with 4cdb0b3c6f6c4b3694fcb311ae72743b on January 21st 2019

* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
    - slot{"step":"1"}
<U> oh awesome!
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ok great
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> no. u r idiot
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_ask_explain_nlucorex
* how_to_get_started{"product":"core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> no i won't
    - utter_direct_to_step2

## Story from conversation with 4b7ecc2cab6e42c5b1fedb8ab4056866 on December 29th 2018
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* out_of_scope
    - utter_out_of_scope
    - utter_possibilities
* out_of_scope
    - utter_out_of_scope
    - utter_possibilities

## Story from conversation with af5a6b3c39d04c6db2b682960e63f01c on January 21st 2019
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
    - slot{"step":"1"}
<U> greet
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* react_positive
    - utter_react_positive
    - utter_ask_continue_sales
<U> yep
    - utter_great
    - sales_form
    - form{"name": null}

## Story from conversation with af5a6b3c39d04c6db2b682960e63f01c on January 21st 2019
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
    - slot{"step":"1"}
<U> hey rasa
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* react_positive
    - utter_react_positive
    - utter_ask_continue_sales
<U> not yet
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## Story from conversation with 53d4ca53494d4469b7d94aca2f7b3fec on January 21st 2019
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
    - slot{"step":"1"}
<U> hi !
    - action_greet_user
* chitchat
    - utter_chitchat
* enter_data
    - utter_not_sure
    - utter_possibilities
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> that is cool
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* faq{"product":"nlu"}
    - slot{"product":"nlu"}
    - utter_faq
    - action_set_faq_slot
    - utter_ask_explain_nlucorex
* how_to_get_started{"product":"core"}
    - utter_explain_core
    - utter_also_explain_nlu

## Story from conversation with 4a4e903fc43940db9ccdb9153dfdadcb on January 21st 2019
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
    - slot{"step":"1"}
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build

## Story from conversation with 5f3a2ea92d184a9f96df7240e4f7e2d9 on January 21st 2019
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
    - slot{"step":"1"}
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> sort of
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> na
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> hm, i'd like that
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

## Story from conversation with ced8c1eb9a8d485f88a02d931b2879bd on January 16th 2019
* get_started_step4
    - action_greet_user
    - slot{"shown_privacy":true}
* faq
    - utter_faq
    - action_set_faq_slot
    - utter_anything_else
<U> fuck yeah!
    - utter_what_help
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> yeah sure
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> considering
    - utter_ask_migration
<U> not yet
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> yes you can
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

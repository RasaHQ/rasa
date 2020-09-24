## story number 1
<U> whats up
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
<U> heyho
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
<U> konichiwa
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
<U> good evening
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
<U> what up
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat

## story number 7
<U> whats up my bot
    - action_greet_user
<U> hi bot
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
<U> hello robot
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities
* enter_data
    - utter_not_sure
    - utter_possibilities

## story number 9
<U> hey there..
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities
<U> no i dont want to
    - utter_nohelp

## story number 11, continue
<U> hhola
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
<U> i don't think so
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> ok
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## story number 11, don't continue
<U> good morning
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
<U> stop it, i do not care!!!
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> na
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## story number 12
<U> hallo sara
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
<U> hieeeeeeeeeeeee
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> i will
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* enter_data
    - utter_thumbsup
    - utter_anything_else

## story number 12, don't continue
<U> hello friend
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> no
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_ask_feedback
* enter_data
    - utter_thumbsup
    - utter_anything_else

## story number 14
<U> hi pal!
    - action_greet_user
<U> hi rasa
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat

## story number 15
<U> howdy
    - action_greet_user
* chitchat
    - utter_chitchat
* enter_data
    - utter_not_sure
    - utter_possibilities

## story number 17
<U> hello rasa
    - action_greet_user
<U> i guess it means - no
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
<U> hieee
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
<U> no thank s
    - utter_nohelp
* enter_data
    - utter_not_sure
    - utter_possibilities
<U> non
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
<U> hi rasa
    - action_greet_user
<U> hello bot
    - action_greet_user
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## story from linda
<U> hi sara!
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with dfbb633d10854f97b880a2496d632f0d on November 16th 2018
<U> hai
    - action_greet_user
<U> hey
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with alan on November 16th 2018
* nlu_info{"nlu_part":"duckling"}
    - slot{"nlu_part":"duckling"}
    - utter_duckling_info
    - utter_anything_else
<U> go ahead
    - utter_what_help

## Story from conversation with alan on November 16th 2018 2
* nlu_info{"nlu_part":"intent classification"}
    - slot{"nlu_part":"intent classification"}
    - utter_nlu_intent_tutorial
    - utter_offer_recommendation
<U> okay!
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
<U> ok let's start
    - utter_what_language
* enter_data
    - action_store_bot_language
    - slot{"can_use_spacy":false}
    - utter_pipeline_nonenglish_nospacy
    - utter_anything_else

## Story from conversation with linda on November 15th 2018
<U> hallo
    - action_greet_user
* chitchat
    - utter_chitchat
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with 477ddbe73e374eedb07104c5d9f42c31 on November 16th 2018
<U> yoo
    - action_greet_user
<U> good morning
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> jop
    - action_greet_user

## Story from conversation with 4986d88ccb784dc19dc5a553a8e07890 on November 19th 2018
<U> hi,
    - action_greet_user
<U> hi friend
    - action_greet_user
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> i accept.
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> no thank s
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
<U> hiii
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
<U> hello there
    - action_greet_user
<U> hello hi
    - action_greet_user
* chitchat
    - utter_chitchat
<U> oh super
    - utter_thumbsup

## Story from conversation with 6fd65c93e374489f9c8d76697ab9c493 on November 19th 2018
<U> merhaba
    - action_greet_user
<U> whats up?
    - action_greet_user
* chitchat
    - utter_chitchat
<U> yez
    - utter_thumbsup

## Story from conversation with 35d1ecc91c364cbf8a6edf006e5d8c9a on November 19th 2018

<U> hello sweet boy
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
<U> no thank s
    - utter_thumbsup
    - utter_anything_else
<U> yes go for it
    - utter_thumbsup

## Story from conversation with 4c274f8d470e4b77adbfefe7cda7cad7 on October 27th 2018

<U> hey there boy
    - action_greet_user
<U> good evening
    - action_greet_user
<U> accepted
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
<U> hey hey
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> k
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> i will
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
<U> great
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
<U> hey ther
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> no stop
    - action_set_onboarding
    - slot{"onboarding":false}
    - utter_ask_which_product
* how_to_get_started{"product":"core"}
    - utter_explain_core
    - utter_anything_else

## Story from conversation with 201bb55841154f858f524a485f8816c3 on November 18th 2018

<U> jop
    - action_greet_user
* how_to_get_started{"product":"nlu"}
    - utter_getstarted
    - utter_first_bot_with_rasa

## Story from conversation with cd483ab3456d47dfb40bd1f51043fb54 on November 18th 2018

<U> hi
    - action_greet_user
<U> hi i'm [sandra hernandez](name)
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
<U> yaah
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> no sir
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> okay cool
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
<U> hi mister
    - action_greet_user
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> that would be great
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> never
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> ok friend
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
<U> yes, i accept
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> yesyesyes
    - utter_ask_migration
<U> no sir
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
* how_to_get_started{"product":"core"}
    - slot{"product":"core"}
    - utter_explain_core
    - utter_also_explain_nlu
<U> yep please
    - utter_explain_nlu
    - utter_direct_to_step2

## Story from conversation with 602ddccbe2de4f46822e06e2781cf02d on January 8th 2019

* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
<U> hi sara..
    - action_greet_user
* out_of_scope
    - utter_out_of_scope
<U> yesyestyes
    - utter_thumbsup
<U> yep, will do thank you
    - utter_thumbsup

## Story from conversation with 4cdb0b3c6f6c4b3694fcb311ae72743b on January 21st 2019

* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
    - slot{"step":"1"}
<U> okay
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ofcourse
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> no, thanks
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
<U> i don't want to give you my email
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
<U> heyo
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
<U> yes cool
    - utter_great
    - sales_form
    - form{"name": null}

## Story from conversation with af5a6b3c39d04c6db2b682960e63f01c on January 21st 2019
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
    - slot{"step":"1"}
<U> hi pal!
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
<U> no i don't accept
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## Story from conversation with 53d4ca53494d4469b7d94aca2f7b3fec on January 21st 2019
* get_started_step1
    - action_greet_user
    - slot{"shown_privacy":true}
    - slot{"step":"1"}
<U> hei
    - action_greet_user
* chitchat
    - utter_chitchat
* enter_data
    - utter_not_sure
    - utter_possibilities
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> oh, ok
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> not really
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
<U> sure thing
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> nah, i'm good
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> sure
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
<U> accepted
    - utter_what_help
* chitchat
    - utter_chitchat
* how_to_get_started
    - utter_getstarted
    - utter_first_bot_with_rasa
<U> ofcourse
    - action_set_onboarding
    - slot{"onboarding":true}
    - utter_built_bot_before
<U> ok i accept
    - utter_ask_migration
<U> none of them
    - utter_explain_rasa_components
    - utter_rasa_components_details
    - utter_ask_explain_nlucorex
<U> kk
    - utter_explain_nlu
    - utter_explain_core
    - utter_explain_x
    - utter_direct_to_step2

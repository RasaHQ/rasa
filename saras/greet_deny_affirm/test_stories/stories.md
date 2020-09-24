## thanks
* thank
    - utter_noworries
    - utter_anything_else

## bye
* bye
    - utter_bye

## greet
* greet OR enter_data{"name": "akela"}
    - action_greet_user

## sales
<U> hello sweatheart
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
* feedback{"feedback_value": "positive"}
    - slot{"feedback_value": "positive"}
    - action_tag_feedback
    - utter_great
    - utter_anything_else

## newsletter + feedback
<U> hi !
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* feedback{"feedback_value": "positive"}
    - slot{"feedback_value": "positive"}
    - action_tag_feedback
    - utter_great
    - utter_anything_else

## newsletter + affirm feedback
<U> hellllooooooo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> fcourse
    - utter_thumbsup
    - utter_anything_else

## newsletter + deny feedback
<U> hey bot!
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> na
    - utter_thumbsup
    - utter_anything_else

## newsletter, don't give email once
<U> greetings
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> no i won't
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> yes i have!
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## newsletter, continue, affirm, then contact sales
<U> heelio
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> i do not know yet
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> ja cool
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* feedback{"feedback_value": "positive"}
    - slot{"feedback_value": "positive"}
    - action_tag_feedback
    - utter_great
    - utter_anything_else
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
* feedback{"feedback_value": "positive"}
    - slot{"feedback_value": "positive"}
    - action_tag_feedback
    - utter_great
    - utter_anything_else

## newsletter, don't continue, then contact sales
<U> hellllooooooo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> no. u r idiot
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> no. u r idiot
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
* feedback{"feedback_value": "positive"}
    - slot{"feedback_value": "positive"}
    - action_tag_feedback
    - utter_great
    - utter_anything_else

## newsletter, don't continue
<U> hey rasa
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> not yet
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> na
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## just newsletter (with email already) + confirm
<U> greet
    - action_greet_user
* signup_newsletter{"email": "maxmeier@firma.de"}
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* feedback{"feedback_value": "negative"}
    - slot{"feedback_value": "negative"}
    - action_tag_feedback
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already)
<U> hi !
    - action_greet_user
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
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## newsletter then sales
<U> greetings
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
* feedback{"feedback_value": "positive"}
    - slot{"feedback_value": "positive"}
    - action_tag_feedback
    - utter_great
    - utter_anything_else

## sales then newsletter
<U> hellooo
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## newsletter, confirm, then sales
<U> halloooo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* feedback{"feedback_value": "negative"}
    - slot{"feedback_value": "negative"}
    - action_tag_feedback
    - utter_thumbsup
    - utter_anything_else
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
* feedback{"feedback_value": "positive"}
    - slot{"feedback_value": "positive"}
    - action_tag_feedback
    - utter_great
    - utter_anything_else

## newsletter + ask why email

<U> hi im [amanda anderson](name)
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* explain
    - utter_response_why_email
    - utter_ask_continue_newsletter
<U> alright
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## newsletter + ask why email

<U> hellllooooooo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* explain
    - utter_response_why_email
    - utter_ask_continue_newsletter
<U> no i won't
    - utter_thumbsup
    - utter_anything_else

## sales, then newsletter, then confirm
<U> hey bot
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
* feedback{"feedback_value": "negative"}
    - slot{"feedback_value": "negative"}
    - action_tag_feedback
    - utter_thumbsup
    - utter_anything_else

## neither options
<U> hi man
    - action_greet_user
<U> na
    - utter_nohelp

## neither --> newsletter
<U> sup
    - action_greet_user
<U> not yet
    - utter_nohelp
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## neither --> sales
<U> halloooo
    - action_greet_user
<U> na
    - utter_nohelp
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback
* feedback{"feedback_value": "positive"}
    - slot{"feedback_value": "positive"}
    - action_tag_feedback
    - utter_great
    - utter_anything_else

## chitchat --> email --> no email
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
<U> health care
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> not yet
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
<U> yas
    - utter_thumbsup
    - utter_ask_feedback
* feedback{"feedback_value": "negative"}
    - slot{"feedback_value": "negative"}
    - action_tag_feedback
    - utter_thumbsup
    - utter_anything_else

## anything else? - yes
    - utter_anything_else
<U> yesssss
    - utter_what_help

## anything else? - no
    - utter_anything_else
<U> 5000
    - utter_thumbsup

## anything else?
    - utter_anything_else
* enter_data
    - utter_not_sure
    - utter_possibilities

## positive reaction
* react_positive
    - utter_react_positive

## negative reaction
* react_negative
    - utter_react_negative

## why rasa
* why_rasa
    - utter_why_rasa
    - utter_ask_more
<U> lets do it
    - utter_why_rasa_research
    - utter_why_rasa_nlu
    - utter_why_rasa_dialogue
    - utter_why_rasa_os
    - utter_why_rasa_compliant

## why rasa
* why_rasa{"current_api":"tensorflow"}
    - slot{"current_api":"__other__"}
    - utter_why_rasa
    - utter_ask_more
<U> hell yeah
    - utter_why_rasa_research
    - utter_why_rasa_nlu
    - utter_why_rasa_dialogue
    - utter_why_rasa_os
    - utter_why_rasa_compliant

## why rasa
* why_rasa{"current_api":"luis"}
    - slot{"current_api":"luis"}
    - utter_why_rasa
    - utter_switch_luis
    - utter_ask_more_migrate
<U> i get it
    - utter_why_rasa_research
    - utter_why_rasa_nlu
    - utter_why_rasa_dialogue
    - utter_why_rasa_os
    - utter_why_rasa_compliant


## why rasa
* why_rasa{"current_api":"dialogflow"}
    - slot{"current_api":"dialogflow"}
    - utter_why_rasa
    - utter_switch_dialogflow
    - utter_ask_more_migrate
<U> ok fine
    - utter_why_rasa_research
    - utter_why_rasa_nlu
    - utter_why_rasa_dialogue
    - utter_why_rasa_os
    - utter_why_rasa_compliant

## why rasa
* why_rasa
    - utter_why_rasa
    - utter_ask_more
<U> no. u r idiot
    - utter_anything_else

## why rasa
* why_rasa{"current_api":"tensorflow"}
    - slot{"current_api":"__other__"}
    - utter_why_rasa
    - utter_ask_more
<U> no. u r idiot
    - utter_anything_else

## why rasa
* why_rasa{"current_api":"luis"}
    - slot{"current_api":"luis"}
    - utter_why_rasa
    - utter_switch_luis
    - utter_ask_more_migrate
<U> no. u r idiot
    - utter_anything_else

## why rasa
* why_rasa{"current_api":"dialogflow"}
    - slot{"current_api":"dialogflow"}
    - utter_why_rasa
    - utter_switch_dialogflow
    - utter_ask_more_migrate
<U> not yet
    - utter_anything_else

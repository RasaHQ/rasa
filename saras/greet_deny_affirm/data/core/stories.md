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
<U> wasssup
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
<U> i said, helllllloooooo!!!!
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
<U> salut
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> that ok
    - utter_thumbsup
    - utter_anything_else

## newsletter + deny feedback
<U> hallo sara
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> i guess it means - no
    - utter_thumbsup
    - utter_anything_else

## newsletter, don't give email once
<U> hello]
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> never
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> yes i agree
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## newsletter, continue, affirm, then contact sales
<U> hi sara..
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> na
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> yes of course
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
<U> hey ther
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> i don't want either of those
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> nö
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
<U> good mourning
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> nah i'm good
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> very bad
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## just newsletter (with email already) + confirm
<U> hey
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
<U> howdy
    - action_greet_user
* signup_newsletter{"email": "maxmeier@firma.de"}
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter
<U> hi can you speak ?
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## newsletter then sales
<U> hi there
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
<U> hello!
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
<U> hell9o
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

<U> hi i'm [sandra hernandez](name)
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* explain
    - utter_response_why_email
    - utter_ask_continue_newsletter
<U> sounds good!
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## newsletter + ask why email

<U> ssup?
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* explain
    - utter_response_why_email
    - utter_ask_continue_newsletter
<U> absolutely not
    - utter_thumbsup
    - utter_anything_else

## sales, then newsletter, then confirm
<U> hi sara!
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
<U> hi rasa
    - action_greet_user
<U> i don't want to
    - utter_nohelp

## neither --> newsletter
<U> helo
    - action_greet_user
<U> i don not like this
    - utter_nohelp
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## neither --> sales
<U> hallo sara
    - action_greet_user
<U> deny
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
<U> hellio
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
<U> na
    - utter_cantsignup
    - utter_ask_continue_newsletter
<U> decline
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
<U> yaps
    - utter_thumbsup
    - utter_ask_feedback
* feedback{"feedback_value": "negative"}
    - slot{"feedback_value": "negative"}
    - action_tag_feedback
    - utter_thumbsup
    - utter_anything_else

## anything else? - yes
    - utter_anything_else
<U> ya please
    - utter_what_help

## anything else? - no
    - utter_anything_else
<U> na
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
<U> yop
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
<U> ok
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
<U> yesyesyes
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
<U> oki doki
    - utter_why_rasa_research
    - utter_why_rasa_nlu
    - utter_why_rasa_dialogue
    - utter_why_rasa_os
    - utter_why_rasa_compliant

## why rasa
* why_rasa
    - utter_why_rasa
    - utter_ask_more
<U> nah, first time
    - utter_anything_else

## why rasa
* why_rasa{"current_api":"tensorflow"}
    - slot{"current_api":"__other__"}
    - utter_why_rasa
    - utter_ask_more
<U> i don't want to give you my email
    - utter_anything_else

## why rasa
* why_rasa{"current_api":"luis"}
    - slot{"current_api":"luis"}
    - utter_why_rasa
    - utter_switch_luis
    - utter_ask_more_migrate
<U> no, thanks
    - utter_anything_else

## why rasa
* why_rasa{"current_api":"dialogflow"}
    - slot{"current_api":"dialogflow"}
    - utter_why_rasa
    - utter_switch_dialogflow
    - utter_ask_more_migrate
<U> nevermind
    - utter_anything_else

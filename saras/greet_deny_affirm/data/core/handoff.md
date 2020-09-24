## chitchat
* human_handoff
    - utter_contact_email

## greet + handoff
<U> hallo sara
    - action_greet_user
* human_handoff
    - utter_contact_email

## just newsletter + handoff, continue
<U> hey sara
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* human_handoff
    - utter_contact_email
    - utter_ask_continue_newsletter
<U> yes,i am
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter + handoff, don't continue
<U> heya
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* human_handoff
    - utter_contact_email
    - utter_ask_continue_newsletter
<U> nein
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## just sales, continue
<U> hi mrs rasa
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* human_handoff
    - utter_contact_email
    - utter_ask_continue_sales
<U> oh cool
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, don't continue
<U> hi bot
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* human_handoff
    - utter_contact_email
    - utter_ask_continue_sales
<U> neither
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## just sales, explain, continue
<U> yoo
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* explain
    - action_explain_sales_form
    - utter_ask_continue_sales
<U> you got me, i accept, if you want me to
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, explain, stop
<U> what is up?
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* explain
    - action_explain_sales_form
    - utter_ask_continue_sales
<U> no i dont want to accept :p lol
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## chitchat
* human_handoff
    - utter_contact_email

## greet + handoff
<U> greetings
    - action_greet_user
* human_handoff
    - utter_contact_email

## just newsletter + handoff, continue
<U> hello sweatheart
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* human_handoff
    - utter_contact_email
    - utter_ask_continue_newsletter
<U> fcourse
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter + handoff, don't continue
<U> ey boss
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* human_handoff
    - utter_contact_email
    - utter_ask_continue_newsletter
<U> no i won't
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## just sales, continue
<U> halo sara
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* human_handoff
    - utter_contact_email
    - utter_ask_continue_sales
<U> accept
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, don't continue
<U> hi man
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* human_handoff
    - utter_contact_email
    - utter_ask_continue_sales
<U> no. u r idiot
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## just sales, explain, continue
<U> hey rasa
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* explain
    - action_explain_sales_form
    - utter_ask_continue_sales
<U> yess
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, explain, stop
<U> hey bot
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* explain
    - action_explain_sales_form
    - utter_ask_continue_sales
<U> not yet
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

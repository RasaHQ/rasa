## out of scope
* out_of_scope
    - utter_out_of_scope
    - utter_possibilities

## say enter data outside the flows
<U> bonjour
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities

## say confirm outside the flows 2
<U> helleo
    - action_greet_user
<U> great
    - utter_thumbsup

## say greet outside the flows
<U> hey dude
    - action_greet_user
* greet OR enter_data{"name": "akela"}
    - action_greet_user

## just newsletter + confirm
<U> hey there..
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
<U> yes'
    - utter_thumbsup
    - utter_anything_else

## just newsletter, continue + confirm
<U> bom dia
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> of course
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> lets do this
    - utter_thumbsup
    - utter_anything_else

## just newsletter, don't continue + confirm
<U> hi, bot
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> no don"t want this!
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_ask_feedback
<U> yeeees
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already) + confirm
<U> wasssup
    - action_greet_user
* out_of_scope
    - utter_out_of_scope
* signup_newsletter{"email": "maxmeier@firma.de"}
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> yes i have built a bot before
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already)
<U> well hello there ;)
    - action_greet_user
* out_of_scope
    - utter_out_of_scope
* signup_newsletter{"email": "maxmeier@firma.de"}
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter
<U> hi rasa
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

## just newsletter, continue
<U> jojojo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> yeah, why not
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter, don't continue
<U> bonjour
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> i don't want to give you my email
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_ask_feedback

## just sales
<U> hi mister
    - action_greet_user
* out_of_scope
    - utter_out_of_scope
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    - utter_ask_feedback

## just sales, continue
<U> halloo
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_sales
<U> yep if i have to
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, don't continue
<U> whats up
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_sales
<U> nop
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

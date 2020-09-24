## out of scope
* out_of_scope
    - utter_out_of_scope
    - utter_possibilities

## say enter data outside the flows
<U> hlo
    - action_greet_user
* enter_data
    - utter_not_sure
    - utter_possibilities

## say confirm outside the flows 2
<U> hlo
    - action_greet_user
<U> yes go ahead
    - utter_thumbsup

## say greet outside the flows
<U> hellllooooooo
    - action_greet_user
* greet OR enter_data{"name": "akela"}
    - action_greet_user

## just newsletter + confirm
<U> hi there it's me
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
<U> awesome
    - utter_thumbsup
    - utter_anything_else

## just newsletter, continue + confirm
<U> hi hi
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> yes give me information
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback
<U> more a less
    - utter_thumbsup
    - utter_anything_else

## just newsletter, don't continue + confirm
<U> hey bot
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> na
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}
    - utter_ask_feedback
<U> yes it is
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already) + confirm
<U> sup
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
<U> i have used it in the past
    - utter_thumbsup
    - utter_anything_else

## just newsletter (with email already)
<U> hi man
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
<U> hellllooooooo
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
<U> hey bot
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_newsletter
<U> yesh
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu
    - utter_ask_feedback

## just newsletter, don't continue
<U> helloooo
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

## just sales
<U> greet
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
<U> hellooo
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_sales
<U> yep!
    - utter_great
    - sales_form
    - form{"name": null}

## just sales, don't continue
<U> guten morgen
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope
    - utter_out_of_scope
    - utter_ask_continue_sales
<U> not yet
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

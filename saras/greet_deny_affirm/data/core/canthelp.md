## chitchat
* canthelp
    - utter_canthelp

## greet + canthelp
<U> heyho
    - action_greet_user
* canthelp
    - utter_canthelp

## greet + newsletter + canthelp + continue
<U> helo
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* canthelp
    - utter_canthelp
    - utter_ask_continue_newsletter
<U> yes, give me information, please
    - utter_great
    - subscribe_newsletter_form
    - form{"name": null}
    - utter_docu

## greet + newsletter + canthelp + don't continue
<U> hieee
    - action_greet_user
* signup_newsletter
    - utter_can_do
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* canthelp
    - utter_canthelp
    - utter_ask_continue_newsletter
<U> i dont want to accept :p lol
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## just sales + continue
<U> hai
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* canthelp
    - utter_canthelp
    - utter_ask_continue_sales
<U> i accept
    - utter_great
    - sales_form
    - form{"name": null}

## just sales + don't continue
<U> halloo
    - action_greet_user
* contact_sales
    - utter_moreinformation
    - sales_form
    - form{"name": "sales_form"}
* canthelp
    - utter_canthelp
    - utter_ask_continue_sales
<U> i decline
    - utter_thumbsup
    - action_deactivate_form
    - form{"name": null}

## Story from conversation with 3db7df5613a2487a809694e2f6b2b34e on November 25th 2019

<U> hello robot
    - action_greet_user
    - slot{"shown_privacy":true}
* nlu_generation_tool_recommendation{"product":"nlu"}
    - slot{"product":"nlu"}
    - utter_nlu_tools

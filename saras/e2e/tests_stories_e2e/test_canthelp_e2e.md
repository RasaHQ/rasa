## greet + canthelp
* greet: wasssup
    <B> Hey there, my name is Sara.
* canthelp: you cannot help me with what i want
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else.

## chitchat
* canthelp: stop go back
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else.

## greet + newsletter + canthelp + continue
* greet: wasssup!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: subscribe me to newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* canthelp: you can't help me with what i need
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else.
    <B> Do you still want to subscribe to the newsletter?
* affirm: yes
    <B> Cool, let's do it!
    - subscribe_newsletter_form
    - form{"name": null}  
    <B> How is this conversation going?

## greet + newsletter + canthelp + don't continue
* greet: hell9o
    <B> Hey there, my name is Sara.
* signup_newsletter: can i sign up to the newsletter too?
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* canthelp: that does not help
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else.
    <B> Do you still want to subscribe to the newsletter?
* deny: no i haven't decided yet if i want to sign up
    <B> Got it.
## just sales + don't continue
* greet: hii
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: i would like to contact your sales team please
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"} 
* canthelp: do you get anything?
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else. 
    <B> Do you still want to talk to sales?
* deny: absolutely not
    <B> Got it.
## just sales + continue
* greet: hi man
    <B> Hey there, my name is Sara.
* contact_sales: sales please
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}  
* canthelp: hm i don't think you can do what i want
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else. 
    <B> Do you still want to talk to sales?
* affirm: confirm
    - sales_form
    - form{"name": "null"} 
## Story from conversation with 3db7df5613a2487a809694e2f6b2b34e on November 25th 2019
* greet: good moring
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* nlu_generation_tool_recommendation{"product":"nlu"}: how to integrate rasa with customer data?
    <B> We recommend using Rasa X to easily create and classify NLU data from within a UI.
    - slot{"product":"nlu"}

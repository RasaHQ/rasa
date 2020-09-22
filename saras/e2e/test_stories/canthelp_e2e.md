## greet + canthelp
<U> heyho
    <B> Hey there, my name is Sara.
<U> i think you cant help me
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else.

## chitchat
<U> this conversation is not really helpful
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else.

## greet + newsletter + canthelp + continue
<U> jop
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> register me for the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> quit
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else.
    <B> Do you still want to subscribe to the newsletter?
<U> cool story bro
    <B> Cool, let's do it!
    - subscribe_newsletter_form
    - form{"name": null}  
    <B> How is this conversation going?

## greet + newsletter + canthelp + don't continue
<U> ey boss
    <B> Hey there, my name is Sara.
<U> lets try the newsletter signup
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> this is leading to nothing
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else.
    <B> Do you still want to subscribe to the newsletter?
<U> i decline
    <B> Got it.
## just sales + don't continue
<U> good mourning
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> contact
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"} 
<U> you can't help me with what i need
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else. 
    <B> Do you still want to talk to sales?
<U> nein
    <B> Got it.
## just sales + continue
<U> i said, helllllloooooo!!!!
    <B> Hey there, my name is Sara.
<U> contact
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}  
<U> this is leading to nothing
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else. 
    <B> Do you still want to talk to sales? Do you still want to talk to sales?
<U> okay sure
    - sales_form
    - form{"name": "null"} 
## Story from conversation with 3db7df5613a2487a809694e2f6b2b34e on November 25th 2019
<U> hi man
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how can i get a docker image
    <B> We recommend using Rasa X to easily create and classify NLU data from within a UI.
    - slot{"product":"nlu"}

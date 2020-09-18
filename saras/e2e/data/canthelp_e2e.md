## greet + canthelp
<U> Hi [rasa](product)
    <B> Hey there!
<U> not helpful
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at hi@rasa.com for anything else.

## chitchat
<U> thanks for nothing
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at hi@rasa.com for anything else.

## greet + newsletter + canthelp + continue
<U> hey bot
    <B> Hey there, my name is Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Could I sign up to newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> not helpful
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at hi@rasa.com for anything else. Do you still want to subscribe to the newsletter?
<U> Yes
    <B> Cool, let's do it!
    - subscribe_newsletter_form
    - form{"name": null}  
    <B> How is this conversation going?

## greet + newsletter + canthelp + don't continue
<U> Hi
    <B> Hey there!
<U> i would love to get the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> ok then you cant help me
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our forum, otherwise contact us at hi@rasa.com for anything else. Do you still want to subscribe to the newsletter?
<U> No
    <B> Got it.
## just sales + don't continue
<U> greet
    <B> Hi, I'm Sara! By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sales call
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"} 
<U> please stop
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else. Do you still want to talk to sales?
<U> no
    <B> Got it.
## just sales + continue
<U> hi
    <B> Hi, I'm Sara!
<U> sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}  
<U> you cant help
    <B> I'm sorry I couldn't help you. If you have general questions about Rasa, please head to our [forum](https://forum.rasa.com/), otherwise contact us at [hi@rasa.com](mailto:hi@rasa.com) for anything else. Do you still want to talk to sales? Do you still want to talk to sales?
<U> yes
    - sales_form
    - form{"name": "null"} 
## Story from conversation with 3db7df5613a2487a809694e2f6b2b34e on November 25th 2019
<U> greet
    <B> Hey there, my name is Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> do i need much data to train the rasa [nlu](product)
    <B> We recommend using Rasa X to easily create and classify [nlu](product) data from within a UI.
    - slot{"product":"nlu"}

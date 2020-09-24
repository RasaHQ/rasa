## out of scope
* out_of_scope: what about wheather
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter

## say enter data outside the flows
* greet: ola sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: the people speak [english](language)
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## say confirm outside the flows 2
* greet: heylo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* affirm: yes give me information
    <B> Got it.
## say greet outside the flows
* greet: heelio
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: hlo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## just newsletter + confirm
* greet: hhola
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope: I am User
    <B> Hmm, not sure what you mean... 
* signup_newsletter: register me for the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
* affirm: well yes
    <B> Got it. 
    <B> Is there anything else I can help you with?
## just newsletter, continue + confirm
* greet: hellooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: subscribe me to you newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope: SEL ME SOMETHING
    <B> Hmm, not sure what you mean... 
    <B> Do you still want to subscribe to the newsletter?
* affirm: yes sirfr
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
* affirm: ok sara
    <B> Got it. 
    <B> Is there anything else I can help you with?

## just newsletter, don't continue + confirm
* greet: hello everybody
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: sign up newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope: are you vegan
    <B> Hmm, not sure what you mean...
    <B> Do you still want to subscribe to the newsletter?
* deny: no, i hate it
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
* affirm: it is ok
    <B> Got it. 
    <B> How is this conversation going?

## just newsletter (with email already) + confirm
* greet: hello rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope: can you learn from our conversation?
    <B> Hmm, not sure what you mean...
* signup_newsletter: do the newsletter then
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?

## just newsletter (with email already)
* greet: hi there it's me
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope: do you have a phone number?
    <B> Hmm, not sure what you mean...
* signup_newsletter: sign up newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?

## just newsletter
* greet: hellooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope: do you like football
    <B> Hmm, not sure what you mean... 
* signup_newsletter: add me to the newsletter list
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
* affirm: yes, cool
    <B> Got it. 
    <B> Is there anything else I can help you with?

## just newsletter, continue
* greet: salut
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i just want to signup for our newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope: give me a girl friend
    <B> Hmm, not sure what you mean...
    <B> Do you still want to subscribe to the newsletter?
* affirm: yes i have built a bot before
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
* affirm: yess
    <B> Got it. 
    <B> Is there anything else I can help you with?

## just newsletter, don't continue
* greet: konichiwa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: newsletter, here is my email: marcus.miller@yahoo.com
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope: how to get rasa studio
    <B> Hmm, not sure what you mean...
    <B> Do you still want to subscribe to the newsletter?
* deny: i don't wanna tell the name of my company
    <B> Got it. 
    <B> How is this conversation going?
    - form{"name": null}

## just sales
* greet: halloooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope: i want to find out what you can build with rasa
    <B> Hmm, not sure what you mean...
* contact_sales: i want to buy the enterprise edition
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?

## just sales, continue
* greet: hey let's talk
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: call sales
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope: machine learning
    <B> Hmm, not sure what you mean...
    <B> Do you still want to talk to sales?
* affirm: yes i accept
    - sales_form
    - form{"name": null}
    <B> How is this conversation going?

## just sales, don't continue
* greet: good mourning
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: please can you book call for me
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope: please help with my ice cream it's dripping
    <B> Hmm, not sure what you mean... 
    <B> Do you still want to talk to sales?
* deny: i do not need help installing
    <B> Got it. 
    - sales_form
    - form{"name": null}
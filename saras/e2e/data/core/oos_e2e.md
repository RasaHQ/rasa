## out of scope
* out_of_scope
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter

## say enter data outside the flows
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> [Spanish](language)
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## say confirm outside the flows 2
<U> hi
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Great. 
    <B> Got it.
## say greet outside the flows
<U> greet
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> greetings
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## just newsletter + confirm
<U> hey
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean... 
<U> newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> Good!
    <B> Got it. 
    <B> Is there anything else I can help you with?
## just newsletter, continue + confirm
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you sign me up for the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    <B> Hmm, not sure what you mean... 
    <B> Do you still want to subscribe to the newsletter?
<U> Sure!
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> yeah
    <B> Got it. 
    <B> Is there anything else I can help you with?

## just newsletter, don't continue + confirm
<U> hey there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Sign me up for the newsletter.
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    <B> Hmm, not sure what you mean...
    <B> Do you still want to subscribe to the newsletter?
<U> No thanks
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> Good!
    <B> Got it. 
    <B> How is this conversation going?

## just newsletter (with email already) + confirm
<U> greet
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean...
<U> newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?

## just newsletter (with email already)
<U> hey
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean...
<U> i want the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?

## just newsletter
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean... 
<U> can you sign me up for the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> yeah
    <B> Got it. 
    <B> Is there anything else I can help you with?

## just newsletter, continue
<U> hi bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you sign me up for the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    <B> Hmm, not sure what you mean...
    <B> Do you still want to subscribe to the newsletter?
<U> Sure!
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> yes
    <B> Got it. 
    <B> Is there anything else I can help you with?

## just newsletter, don't continue
<U> hey bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you sign me up for the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    <B> Hmm, not sure what you mean...
    <B> Do you still want to subscribe to the newsletter?
<U> No!
    <B> Got it. 
    <B> How is this conversation going?
    - form{"name": null}

## just sales
<U> hello bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean...
<U> request call with sales team
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?

## just sales, continue
<U> hey bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to be in touch with sales
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope
    <B> Hmm, not sure what you mean...
    <B> Do you still want to talk to sales?
<U> yes
    - sales_form
    - form{"name": null}
    <B> How is this conversation going?

## just sales, don't continue
<U> greet
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sales call
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope
    <B> Hmm, not sure what you mean... 
    <B> Do you still want to talk to sales?
<U> No thanks
    <B> Got it. 
    - sales_form
    - form{"name": null}
## out of scope
* out_of_scope
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter

## say enter data outside the flows
<U> guten morgen
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> [english](language) is the only language but i want to add more
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## say confirm outside the flows 2
<U> hi man
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> yep if i have to
    <B> Got it.
## say greet outside the flows
<U> ey boss
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> halloooo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## just newsletter + confirm
<U> ola sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean... 
<U> subscribe to your newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> sort of
    <B> Got it. 
    <B> Is there anything else I can help you with?
## just newsletter, continue + confirm
<U> hi mrs rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i'll subscribe to the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    <B> Hmm, not sure what you mean... 
    <B> Do you still want to subscribe to the newsletter?
<U> yesh
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> that would be great
    <B> Got it. 
    <B> Is there anything else I can help you with?

## just newsletter, don't continue + confirm
<U> hell9o
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    <B> Hmm, not sure what you mean...
    <B> Do you still want to subscribe to the newsletter?
<U> no go
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> that is cool
    <B> Got it. 
    <B> How is this conversation going?

## just newsletter (with email already) + confirm
<U> helleo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean...
<U> can you add edward@paul.com to the newsletter list?
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?

## just newsletter (with email already)
<U> yoo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean...
<U> subscribe denise@gmail.com to the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?

## just newsletter
<U> ayyyy whaddup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean... 
<U> now i want to signup for the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> sort of
    <B> Got it. 
    <B> Is there anything else I can help you with?

## just newsletter, continue
<U> salut
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> oh actually i want to get the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    <B> Hmm, not sure what you mean...
    <B> Do you still want to subscribe to the newsletter?
<U> yeeeeezzzzz
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> i agree
    <B> Got it. 
    <B> Is there anything else I can help you with?

## just newsletter, don't continue
<U> rasa hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> newsletter registration
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* out_of_scope
    <B> Hmm, not sure what you mean...
    <B> Do you still want to subscribe to the newsletter?
<U> i do not need help installing
    <B> Got it. 
    <B> How is this conversation going?
    - form{"name": null}

## just sales
<U> hey, let's talk
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope
    <B> Hmm, not sure what you mean...
<U> can i have a call?
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?

## just sales, continue
<U> jojojo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can i have a call?
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope
    <B> Hmm, not sure what you mean...
    <B> Do you still want to talk to sales?
<U> that sounds fine
    - sales_form
    - form{"name": null}
    <B> How is this conversation going?

## just sales, don't continue
<U> good evening
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want one platform please
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
* out_of_scope
    <B> Hmm, not sure what you mean... 
    <B> Do you still want to talk to sales?
<U> na
    <B> Got it. 
    - sales_form
    - form{"name": null}
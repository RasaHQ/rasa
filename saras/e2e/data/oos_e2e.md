## out of scope
<U> out_of_scope
    <B> I can't help you with that, I'm sorry. You can ask me about:    - how to get started with Rasa     - the components of Rasa,    - booking a sales call    - subscribing to our newsletter
## say enter data outside the flows
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> [Spanish](language)
    <B> Hmm, not sure what you mean... You can ask me about:    - how to get started with Rasa     - the components of Rasa,    - booking a sales call    - subscribing to our newsletter
## say confirm outside the flows 2
<U> hi hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Great. 
    <B> Got it.
## say greet outside the flows
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> greetings
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## just newsletter + confirm
<U> hey
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> out of scope
    <B> Hmm, not sure what you mean... 
<U> newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> Good!
    <B> Got it. Is there anything else I can help you with?
## just newsletter, continue + confirm
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Could I sign up to the newsletter?
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> out of scope
    <B> Hmm, not sure what you mean... Do you still want to subscribe to the newsletter?
<U> Sure!
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> yeah
    <B> Got it. Is there anything else I can help you with?
## just newsletter, don't continue + confirm
<U> hey there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> signup to newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> cow
    <B> Hmm, not sure what you mean... Do you still want to subscribe to the newsletter?
<U> No thanks
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> Good!
    <B> Got it. How is this conversation going?
## just newsletter (with email already) + confirm
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> romeo and juliet
    <B> Hmm, not sure what you mean...
<U> newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## just newsletter (with email already)
<U> hey
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> the guardian
    <B> Hmm, not sure what you mean...
<U> subscribing to newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## just newsletter
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> out of scope question
    <B> Hmm, not sure what you mean... 
<U> Could I sign up to the newsletter?
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> yeah
    <B> Got it. Is there anything else I can help you with?
## just newsletter, continue
<U> hi bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Could I sign up to the newsletter?
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> Can you play the piano?
    <B> Hmm, not sure what you mean... Do you still want to subscribe to the newsletter?
<U> Sure!
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> yes
    <B> Got it. Is there anything else I can help you with?
## just newsletter, don't continue
<U> hey  bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Could I sign up to the newsletter?
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> Can you play the piano?
    <B> Hmm, not sure what you mean... Do you still want to subscribe to the newsletter?
<U> No!
    <B> Got it. How is this conversation going?
    - form{"name": null}
## just sales
<U> hellio bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I have an out of scope question
    <B> Hmm, not sure what you mean...
<U> Call sales team
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just sales, continue
<U> hey bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to set up a call with sales guys
    - sales_form
    - form{"name": "sales_form"}
<U> a paw of hamster
    <B> Hmm, not sure what you mean... Do you still want to talk to sales?
<U> yes sure
    - sales_form
    - form{"name": null}
    <B> How is this conversation going?
## just sales, don't continue
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sales call
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
<U> a pet
    <B> Hmm, not sure what you mean... Do you still want to talk to sales?
<U> No thanks
    <B> Got it. 
    - sales_form
    - form{"name": null}
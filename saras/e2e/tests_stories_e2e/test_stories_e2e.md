## thanks
* thank: cool thanks
    <B> No worries! 
    <B> Is there anything else I can help you with?
## bye
* bye: then bye
    <B> Goodbye!
## greet
* greet: salut
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## sales
* greet: hey ther
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: let me talk to your sales guys
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
* thank: cool, thanks
    - slot{"feedback_value": "positive"}
    <B> Got it. 
    <B> Is there anything else I can help you with?
## newsletter + feedback
* greet: good evening
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i want to signup
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* affirm: yep if i have to
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter + affirm feedback
* greet: hello sweet boy
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: newsletter please.
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* affirm: yess
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter + deny feedback
* greet: good evening
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: subscribe bruce_harryman@olsen.com to the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* react_negative: i don't care!!!!
    <B> Got it. 
    <B> Is there anything else I can help you with?
## newsletter, don't give email once
* greet: wasssup!
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i'd like to subscribe
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## newsletter, continue, affirm, then contact sales
* greet: hieee
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i want to signup for the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* deny: nehi
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
* affirm: i am using it
    <B> Got it.  
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
* affirm: yes, i'd love to
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
* contact_sales: i want to book a sales call
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
* affirm: okay!
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter, don't continue, then contact sales
* greet: wasssup!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: subscription newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* deny: no i dont want
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
* deny: no ma'am
    <B> Got it.
    - form{"name": null}
* contact_sales: i choose the call with sales
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
* affirm: fuck yeah!
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter, don't continue
* greet: hello it is me again
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: add me to the newsletter list
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* deny: no ma'am
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
* deny: nah, first time
    <B> Got it.
    - form{"name": null}
## just newsletter (with email already) + confirm
* greet: hei
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: subscribe my email evan@palmer.net to the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* react_negative: i dont like bots
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "negative"}
## just newsletter (with email already)
* greet: what is up?
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: subscribe to the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just newsletter
* greet: good evening
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i want to signup for the nl
    <B> Can do! 
    <B> What's your email address?
* enter_data: whatever it costs
    <B> We sent a confirmation email to [bdc@bdce.com](email). If you want some more awesome Rasa content, check out our [forum](https://forum.rasa.com/). How is this conversation going?
## newsletter then sales
* greet: konichiwa
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: sign me up for the newsletter - my email is carolyn_caskey@yahoo.com
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* contact_sales: sales sales sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
* affirm: that is cool
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## sales then newsletter
* greet: ola sara
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: a call
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
* signup_newsletter: subscribe to your newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## newsletter, confirm, then sales
* greet: hello sweatheart
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: yeaaah lets go for the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* deny: no i don't accept
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "negative"}
* contact_sales: lets talk to sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
* affirm: i want that
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter + ask why email
* greet: hola
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i'm craving the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* explain: how come?
    <B> I need your email address to sign you up to the newsletter. 
    <B> Do you still want to subscribe to the newsletter?
* affirm: yes pleae
    <B> Got it.
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
## newsletter + ask why email
* greet: ayyyy whaddup
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: what about signing up for the newsletter?
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* explain: what does that mean
    <B> I need your email address to sign you up to the newsletter. 
    <B> Do you still want to subscribe to the newsletter?
* deny: no ma'am
    <B> Got it. 
    <B> Is there anything else I can help you with?
## sales, then newsletter, then confirm
* greet: hi mrs rasa
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: i want to be connected to your sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
* signup_newsletter: first lets sign up for the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* deny: i'm afraid not
    - slot{"feedback_value": "negative"}
    <B> Got it. 
    <B> Is there anything else I can help you with?
## neither options
* greet: halo sara
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: i'm a developer
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
## neither --> newsletter
* greet: heeey
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* deny: no, thankyou
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
* signup_newsletter: i go for the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## neither --> sales
* greet: hallo sara
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: customer service
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
* contact_sales: booking sales ca;;
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
* affirm: yes pleae
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## chitchat --> email --> no email
* greet: ayyyy whaddup
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: are you ok?
    <B> I'm great! Thanks for asking.
* chitchat: how about your age
    <B> Old enough to be a bot!
* chitchat: who is the person that invented you?
    <B> Engineers at Rasa.
* chitchat: who is your creator?
    <B> I was born in the coolest city on Earth, in Berlin.
* signup_newsletter: please subscribe me to your newsletter
    <B> Can do! 
    <B> What's your email address?
* deny: no i won't
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
* deny: i don not like this
    <B> Got it. 
* affirm: si
    <B> Got it. 
    <B> How is this conversation going?
* deny: i don't want to say
    <B> Got it. 
    <B> Is there anything else I can help you with?
## anything else? yes
    <B> Is there anything else I can help you with?
* affirm: yes you can
    <B> Ok, what can I help you with?
## anything else? no
    <B> Is there anything else I can help you with?
* deny: i don't want to say
    <B> Got it.
## anything else?
    <B> Is there anything else I can help you with?
* enter_data: chief lemonade officer
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## positive reaction
* affirm: i agree
    <B> amazing!
## negative reaction
* react_negative: documentation of rasa is very bad
    <B> horrible!
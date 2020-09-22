## thanks
<U> cheers bro
    <B> No worries! 
    <B> Is there anything else I can help you with?
## bye
<U> bye bye bot
    <B> Goodbye!
## greet
<U> salut
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## sales
<U> hellllooooooo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> let me talk to your sales guys
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> thanks a bunch for everything
    - slot{"feedback_value": "positive"}
    <B> Got it. 
    <B> Is there anything else I can help you with?
## newsletter + feedback
<U> hey rasa
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i'll subscribe to the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> ok, i behave now
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter + affirm feedback
<U> hlo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> subcribe
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> y
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter + deny feedback
<U> good moring
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to subsribe to the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> i am sad
    <B> Got it. 
    <B> Is there anything else I can help you with?
## newsletter, don't give email once
<U> hi there it's me
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> newsletter it is
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## newsletter, continue, affirm, then contact sales
<U> whats up my bot
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to signup for the nl
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> i'm not giving you my email address
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> ok great
    <B> Got it.  
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> fine
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
<U> please can you book call for me
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> going super well
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter, don't continue, then contact sales
<U> hi there it's me
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> register me for the newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> no way
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> no you did it wrong
    <B> Got it.
    - form{"name": null}
<U> i want to be connected to your sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> sounds good!
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter, don't continue
<U> hallo sara
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> subsribing to our newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> nopes
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> no sorry
    <B> Got it.
    - form{"name": null}
## just newsletter (with email already) + confirm
<U> helloooo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> subscribe
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> i am angry over you
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "negative"}
## just newsletter (with email already)
<U> good mourning
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign me up for that newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just newsletter
<U> hii
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i'm craving the newsletter
    <B> Can do! 
    <B> What's your email address?
<U> conversational
    <B> We sent a confirmation email to [bdc@bdce.com](email). If you want some more awesome Rasa content, check out our [forum](https://forum.rasa.com/). How is this conversation going?
## newsletter then sales
<U> hi folks
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> lets try the newsletter registration
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> i want ti booking about booking sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> yes great
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## sales then newsletter
<U> hell9o
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i think i want to talk to your sales folks
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> join that newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## newsletter, confirm, then sales
<U> hello world
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i would like to subscribe to your newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> non
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "negative"}
<U> i want to talk to someone about your pricing system
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> yessoo
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter + ask why email
<U> hello it is me again
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> join that newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> explain that
    <B> I need your email address to sign you up to the newsletter. 
    <B> Do you still want to subscribe to the newsletter?
<U> thats great
    <B> Got it.
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
## newsletter + ask why email
<U> whats up my bot
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> subcribe
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> could you tell me more
    <B> I need your email address to sign you up to the newsletter. 
    <B> Do you still want to subscribe to the newsletter?
<U> not right now
    <B> Got it. 
    <B> Is there anything else I can help you with?
## sales, then newsletter, then confirm
<U> hey let's talk
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign me up, my email is elizabeth@yahoo.com
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> i want to signup
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> no sorry
    - slot{"feedback_value": "negative"}
    <B> Got it. 
    <B> Is there anything else I can help you with?
## neither options
<U> ey boss
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> a killer bot
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
## neither --> newsletter
<U> hi man
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> no sorry
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> subscribe me to newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## neither --> sales
<U> rasa hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> able to integrate with paypal, wordpress, facebook andd twilio
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> i choose the call
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> oh good !!
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## chitchat --> email --> no email
<U> hello it is me again
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> h r u ?
    <B> I'm great! Thanks for asking.
<U> how about your age
    <B> Old enough to be a bot!
<U> who was the person who put you together?
    <B> Engineers at Rasa.
<U> what is you mother name
    <B> I was born in the coolest city on Earth, in Berlin.
<U> i prefer to get the newsletter
    <B> Can do! 
    <B> What's your email address?
<U> na
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> no ma'am
    <B> Got it. 
<U> oki doki
    <B> Got it. 
    <B> How is this conversation going?
<U> sorry not right now
    <B> Got it. 
    <B> Is there anything else I can help you with?
## anything else? yes
    <B> Is there anything else I can help you with?
<U> hm, i'd like that
    <B> Ok, what can I help you with?
## anything else? no
    <B> Is there anything else I can help you with?
<U> na
    <B> Got it.
## anything else?
    <B> Is there anything else I can help you with?
<U> acme brands
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## positive reaction
<U> yes it is
    <B> amazing!
## negative reaction
<U> i dont like bots
    <B> horrible!
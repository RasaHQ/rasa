## thanks
<U> Thank you
    <B> No worries! 
    <B> Is there anything else I can help you with?
## bye
<U> bye
    <B> Goodbye!
## greet
<U> hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## sales
<U> hey bot
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to talk to sales
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> great thanks
    - slot{"feedback_value": "positive"}
    <B> Got it. 
    <B> Is there anything else I can help you with?
## newsletter + feedback
<U> greet
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> Good!
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter + affirm feedback
<U> hi
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> Good!
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter + deny feedback
<U> hi
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> this is a really frustrating experience
    <B> Got it. 
    <B> Is there anything else I can help you with?
## newsletter, don't give email once
<U> hi
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## newsletter, continue, affirm, then contact sales
<U> hi
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> no
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> yes
    <B> Got it.  
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> Perfect! 
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
<U> how to book a sales call?
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> good
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter, don't continue, then contact sales
<U> greet
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Subscribe newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> nope
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> not really
    <B> Got it.
    - form{"name": null}
<U> call with sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> good
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter, don't continue
<U> hey there
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Subscribe newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> nope
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> no
    <B> Got it.
    - form{"name": null}
## just newsletter (with email already) + confirm
<U> hi
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to sign up for the newsletter.
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> this is a really frustrating experience
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "negative"}
## just newsletter (with email already)
<U> hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just newsletter
<U> hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do! 
    <B> What's your email address?
<U> my email is Richard@Simmons.com
    <B> We sent a confirmation email to [bdc@bdce.com](email). If you want some more awesome Rasa content, check out our [forum](https://forum.rasa.com/). How is this conversation going?
## newsletter then sales
<U> hello bot
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> talk to sales team
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> great
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## sales then newsletter
<U> hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> contact call with sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> sign up for newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## newsletter, confirm, then sales
<U> hey there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> Very bad
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "negative"}
<U> book a sales call
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> Perfect!
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## newsletter + ask why email
<U> hey
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> i would like to know why you need that
    <B> I need your email address to sign you up to the newsletter. 
    <B> Do you still want to subscribe to the newsletter?
<U> yep
    <B> Got it.
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
## newsletter + ask why email
<U> hi
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up for newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> why do you need to know that
    <B> I need your email address to sign you up to the newsletter. 
    <B> Do you still want to subscribe to the newsletter?
<U> no
    <B> Got it. 
    <B> Is there anything else I can help you with?
## sales, then newsletter, then confirm
<U> hi Sara
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> talk to sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> sign up for newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> Very bad
    - slot{"feedback_value": "negative"}
    <B> Got it. 
    <B> Is there anything else I can help you with?
## neither options
<U> hey there
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> nothing
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
## neither --> newsletter
<U> hello there
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> neither
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> sign up for newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## neither --> sales
<U> hi Sara
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> nothing
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> call with sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> Great!
    <B> Got it. 
    <B> Is there anything else I can help you with?
    - slot{"feedback_value": "positive"}
## chitchat --> email --> no email
<U> hi there
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how are you?
    <B> I'm great! Thanks for asking.
<U> how old are you?
    <B> Old enough to be a bot!
<U> who made you?
    <B> Engineers at Rasa.
<U> where were you made
    <B> I was born in the coolest city on Earth, in Berlin.
<U> sign up for newsletter
    <B> Can do! 
    <B> What's your email address?
<U> No I can't
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> no thanks
    <B> Got it. 
<U> Good!
    <B> Got it. 
    <B> How is this conversation going?
<U> it is going pretty badly
    <B> Got it. 
    <B> Is there anything else I can help you with?
## anything else? yes
    <B> Is there anything else I can help you with?
<U> Yes
    <B> Ok, what can I help you with?
## anything else? no
    <B> Is there anything else I can help you with?
<U> No
    <B> Got it.
## anything else?
    <B> Is there anything else I can help you with?
<U> Got it!
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## positive reaction
<U> Great!
    <B> amazing!
## negative reaction
<U> this is a really frustrating experience
    <B> horrible!
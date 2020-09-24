## chitchat
* human_handoff: i want to talk to someone who is smarter than you
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com.
## greet + handoff
* greet: whats popping
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* human_handoff: i want to speak to human
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com.
## just newsletter + handoff, continue
* greet: helleo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: add me to the newsletter list
    <B> Sure thing!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* human_handoff: talking to a bot is stupid
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com. 
    <B> Do you still want to subscribe to the newsletter?
* affirm: yaps
    <B> That's great!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## just newsletter + handoff, don't continue
* greet: hey dude
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: sign up newsletter
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* human_handoff: can i talk to a human instead
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com. 
    <B> Do you still want to subscribe to the newsletter?
* deny: nevermind
    <B> Got it.
    - form{"name": null} 
## just sales, continue
* greet: hello rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: can i get a ssales call
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
* human_handoff: that's annoying i'd like to speak to someone real
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com. 
    <B> Do you still want to talk to sales?
* affirm: ok, sara
    <B> Great!
    - sales_form
    - form{"name": null}
## just sales, don't continue
* greet: hello everybody
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: i want to connect to your sales team
    <B> Sure, we can book a sales call! Let's get to know each other first .
* human_handoff: i dont wanna talk to a bot
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com.
    - sales_form
    - form{"name": "sales_form"}
* deny: i don't want to give it to you
    <B> Got it.
    - sales_form
    - form{"name": null}
## just sales, explain, continue
* greet: whats popping
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: connect to sales
    <B> Sure, we can book a sales call! Let's get to know each other first . 
* explain: why is that necessary
    <B> Your job function helps us in understanding how we can talk to you best. Do you still want to talk to sales?
* affirm: i guess so
    <B> Great! 
    <B> What's your job? 
## just sales, explain, stop
* greet: hello sweatheart
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: booking a sall
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
* explain: please elaborate
    <B> Your job function helps us in understanding how we can talk to you best. Do you still want to talk to sales?
* deny: absolutely not
    <B> Got it.
    - sales_form
    - form{"name": null}

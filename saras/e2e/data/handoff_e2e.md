## chitchat
<U> i want to talk to human
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com.
## greet + handoff
<U> hi
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> speak to a real person
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com.
## just newsletter + handoff, continue
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to sign up to newsletter
    <B> Sure thing!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> let me talk to a human
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com. 
    <B> Do you still want to subscribe to the newsletter?
<U> yes
    <B> That's great!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## just newsletter + handoff, don't continue
<U> hey there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up to newsletter
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> can i speak to a human
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com. 
    <B> Do you still want to subscribe to the newsletter?
<U> not at all
    <B> Got it.
    - form{"name": null} 
## just sales, continue
<U> hey
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can i contact sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
<U> can I talk to human
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com. 
    <B> Do you still want to talk to sales?
<U> Yes
    <B> Great!
    - sales_form
    - form{"name": null}
## just sales, don't continue
<U> hi bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> contact sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
<U> I need a real person
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com.
    - sales_form
    - form{"name": "sales_form"}
<U> No
    <B> Got it.
    - sales_form
    - form{"name": null}
## just sales, explain, continue
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can i talk to sales
    <B> Sure, we can book a sales call! Let's get to know each other first . 
<U> could you explain why you need that
    <B> Your job function helps us in understanding how we can talk to you best. Do you still want to talk to sales?
<U> yes
    <B> Great! 
    <B> What's your job? 
## just sales, explain, stop
<U> hi
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i would like to talk to sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
<U> could you explain why you need that
    <B> Your job function helps us in understanding how we can talk to you best. Do you still want to talk to sales?
<U> no thanks
    <B> Got it.
    - sales_form
    - form{"name": null}

## chitchat
<U> talk to human
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com.
## greet + handoff
<U> i said, helllllloooooo!!!!
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you put me in touch with a human?
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com.
## just newsletter + handoff, continue
<U> whats up my bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i go for the newsletter
    <B> Sure thing!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> please give me a human
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com. 
    <B> Do you still want to subscribe to the newsletter?
<U> how nice!
    <B> That's great!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## just newsletter + handoff, don't continue
<U> hi friends
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to signup for the newsletter
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> no, i want to talk to human
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com. 
    <B> Do you still want to subscribe to the newsletter?
<U> noooooooooooooooooooooooooooooooooooooooo
    <B> Got it.
    - form{"name": null} 
## just sales, continue
<U> helo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to buy the enterprise edition
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
<U> can you put me in touch with a human?
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com. 
    <B> Do you still want to talk to sales?
<U> ya thats cool
    <B> Great!
    - sales_form
    - form{"name": null}
## just sales, don't continue
<U> heyho
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to talk to your sales people
    <B> Sure, we can book a sales call! Let's get to know each other first .
<U> no, i want to talk to human
    <B> You can't speak to a human directly in this chat, but you can email us at hi@rasa.com.
    - sales_form
    - form{"name": "sales_form"}
<U> no i dont want to accept :p lol
    <B> Got it.
    - sales_form
    - form{"name": null}
## just sales, explain, continue
<U> merhaba
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> give me someone who can explain your business model
    <B> Sure, we can book a sales call! Let's get to know each other first . 
<U> explain that
    <B> Your job function helps us in understanding how we can talk to you best. Do you still want to talk to sales?
<U> y
    <B> Great! 
    <B> What's your job? 
## just sales, explain, stop
<U> konichiwa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> let' contact sales now
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
<U> explain that
    <B> Your job function helps us in understanding how we can talk to you best. Do you still want to talk to sales?
<U> i decline
    <B> Got it.
    - sales_form
    - form{"name": null}

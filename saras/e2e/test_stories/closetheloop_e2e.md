## story number 1
<U> good evening
    <B> Hey there, my name is Sara.
<U> expert of rasa
    <B> I can't help you with that, I'm sorry.
<U> oh actually i want to get the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> ok thanks!
    <B> You're welcome! 
    <B> Is there anything else I can help you with?
<U> i want one platform please
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## story number 2
<U> ola sara
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> yeah how about the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> where is mexico?
    <B> I can't help you with that, I'm sorry. 
    <B> Is there anything else I can help you with?
## story number 4
<U> hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> newsletter registration
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> thanks
    <B> You're welcome! 
    <B> Is there anything else I can help you with?
<U> who am i speaking with
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> where are your origins?
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 5
<U> i said, helllllloooooo!!!!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i just want to signup for our newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> thanks!
    <B> You're welcome! 
    <B> Is there anything else I can help you with?
<U> what i do after cd starter-pack-rasa-stack?
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 6
<U> hi friends
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> who are created you
    <B> The engineers at Rasa
<U> how old will you be on your next birthday?
    <B> Old enough to be a bot!
## story number 7
<U> rasa hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hallo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> where did you come from?
    <B> I was born in the coolest city on Earth, in Berlin.
<U> i need to talk to sales
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## story number 8
<U> helloooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to build a sales bot
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> 1000000
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 9
<U> hlo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> lead generation
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> absolutely not
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
## story number 11, continue
<U> hallo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> shall i know who am i?
    <B> I hope you are being yourself.
<U> so who are you ?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> okay hi how are you?
    <B> I'm great! Thanks for asking. 
<U> what is your root?
    <B> I was born in the coolest city on Earth, in Berlin.
<U> can you pelase subscribe me to the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> not going well at all
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> yay
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null} 
    <B> How is this conversation going?
## story number 11, don't continue
<U> what up
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> who could i be?
    <B> I hope you are being yourself.
<U> tell me something about you
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> are you happy
    <B> I'm great! Thanks for asking. 
<U> what is your birthplace?
    <B> I was born in the coolest city on Earth, in Berlin.
<U> subscribe to our newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> no bots at all
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> i'm not sure
    <B> Got it.
    - subscribe_newsletter_form
    - form{"name": null} 
## story number 12
<U> hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> newslettwr
    <B> Can do! 
    <B> What's your email address?
<U> around $500,000 per year
    <B> We sent a confirmation email to [bdc@bdce.com](email). If you want some more awesome Rasa content, check out our [forum](https://forum.rasa.com/). How is this conversation going?
<U> i want to build a sales bot
    <B> Got it. 
    <B> Is there anything else I can help you with?
## story number 12, continue
<U> ayyyy whaddup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> register me for the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> this sucks
    <B> I can't help you with that, I'm sorry.
    <B> Do you still want to subscribe to the newsletter?
<U> yes you can
    <B> perfect!
    - subscribe_newsletter_form
     - form{"name": null} 
    <B> How is this conversation going?
<U> oov
    <B> Got it. 
    <B> Is there anything else I can help you with?
## story number 12, don't continue
<U> hello is anybody there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get a subscription
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> i sell turtles
    <B> I can't help you with that, I'm sorry.
    <B> Do you still want to subscribe to the newsletter?
<U> no, thankyou
    - subscribe_newsletter_form
     - form{"name": null} 
    <B> How is this conversation going?
<U> hahah
    <B> Got it. 
    <B> Is there anything else I can help you with?
## story number 14
<U> hello sweatheart
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hola
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> is everything all right
    <B> I'm great! Thanks for asking. 
<U> how is the weather?
    <B> I can't help you with that, I'm sorry.
## story number 15
<U> what up
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what is your count of years being alive so far?
    <B> Old enough to be a bot!
<U> i have about 200 bucks in my savings account
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 17
<U> guten morgen
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i don't agree
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> you should learn to count
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> what's my identity?
    <B> I hope you are being yourself.
<U> ok who are you ?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> how's life treating you friend?
    <B> I'm great! Thanks for asking. 
<U> where's your home town?
    <B> I was born in the coolest city on Earth, in Berlin.
## story number 18
<U> i said, helllllloooooo!!!!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hey sara, how's it going
    <B> I'm great! Thanks for asking. 
<U> what city are you in?
    <B> I was born in the coolest city on Earth, in Berlin.
<U> i dont want to
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> 1000000
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> nah not for me
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> can you understand ?
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> 50,000 dollar
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> a turtle
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 00e7815f79e4413abb0dfb4b392f1099 on November 15th 2018
<U> halloooo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> merhaba
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> real bot then?
    <B> Yes, I'm a bot.
<U> how to make a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## story from linda
<U> hello sweatheart
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> 100000
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> sure, give me the basics
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with dfbb633d10854f97b880a2496d632f0d on November 16th 2018
<U> hey rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hi there it's me
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get strarted with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with alan on November 16th 2018
<U> where to train [intents](nlu_part) in rasa?
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
<U> okay..
    <B> Ok, what can I help you with?
## Story from conversation with alan on November 16th 2018 2
<U> how is [entity recognition](nlu_part) managed in rasa?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yes pls
    <B> Which (human) language is your assistant in?
<U> [spanish](language) is the only language but i want to add more
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
<U> how easy is it to use rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with alan on November 16th 2018 2
<U> do you use [duckling](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> more a less
    <B> Which (human) language is your assistant in?
<U> user can communicate with the bot in [dutch](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## Story from conversation with linda on November 15th 2018
<U> good moring
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are you artificial
    <B> Yes, I'm a bot.
<U> where are from
    <B> I was born in the coolest city on Earth, in Berlin.
<U> start rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with 477ddbe73e374eedb07104c5d9f42c31 on November 16th 2018
<U> hey, let's talk
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hey let's talk
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what are rasa's components?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> jojojo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## Story from conversation with 4986d88ccb784dc19dc5a553a8e07890 on November 19th 2018
<U> heeey
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sup
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are you sure that you're a bot?
    <B> Yes, I'm a bot.
<U> where to start the development of rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep thats cool
    <B> And have you built a contextual assistant or a bot before?
<U> no thank s
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what is rasa [nlu](product)?
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> i need more info for rasa [core](product)
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with 7830abb04e1c49809d89b0d420443928 on November 19th 2018
<U> hey ther
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are you okay
    <B> I'm great! Thanks for asking. 
<U> i ned a gp in 94301
    - respond_out_of_scope
<U> what is your birthplace?
    <B> I was born in the coolest city on Earth, in Berlin.
<U> more about [rasa open source](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with 29d264d8ce574a11bde572f0e79b73f3 on November 19th 2018
<U> rasa hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hi friends
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how you doing
    <B> I'm great! Thanks for asking. 
<U> yep, will do thank you
    <B> Got it.
## Story from conversation with 6fd65c93e374489f9c8d76697ab9c493 on November 19th 2018
<U> jojojo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> whats popping
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how've you been?
    <B> I'm great! Thanks for asking. 
<U> love you
    <B> Got it.
## Story from conversation with 35d1ecc91c364cbf8a6edf006e5d8c9a on November 19th 2018
<U> hiihihi
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> one that will get me promoted
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> next the sales contact
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> no i dont want
    <B> Got it. 
    <B> Is there anything else I can help you with?
<U> amazing!
    <B> Got it. 
## Story from conversation with 4c274f8d470e4b77adbfefe7cda7cad7 on October 27th 2018
<U> hey, let's talk
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> helloooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> that's great
    <B> Got it. 
<U> how to download?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with d041ba4b0a89479e9bb6a5007f2cdc87 on November 15th 2018
<U> are there different packages customers can book?
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> you are doin great
    <B> Got it. 
    <B> Is there anything else I can help you with?
<U> who are your parents
    <B> I was born in the coolest city on Earth, in Berlin.
<U> i don't have one
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with dominik on November 19th 2018
<U> merhaba
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> pip
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> i want to know how to start with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> fcourse
    <B> And have you built a contextual assistant or a bot before?
<U> jo
    <B> Are you migrating from another tool?
<U> i want to change from [dialogflow](current_api) to rasa
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
<U> how do i get rasa [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
<U> can you help me with the rasa [core](product) ?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Is there anything else I can help you with?
## Story from conversation with dominik
<U> is [duckling](nlu_part) part of rasa?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> done
    <B> Which (human) language is your assistant in?
<U> user can communicate with the bot in [portuguese](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
<U> until now it’s only in [german](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## Story from conversation with dominik on November 19th 2018
<U> hallo sara
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how can i connect to rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no, thankyou
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what about [nlu](product)?
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
## Story from conversation with 201bb55841154f858f524a485f8816c3 on November 18th 2018
<U> hiihihi
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to get started with rasa [nlu](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with cd483ab3456d47dfb40bd1f51043fb54 on November 18th 2018
<U> heeey
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> bom dia
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> [both](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with cfa8bb9deaf0427498c662745431a282 on December 15th 2018
<U> hi folks
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> where did you grow up?
    <B> I was born in the coolest city on Earth, in Berlin.
<U> i do not care how are you
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with cdd14d763a664a5b95e998ce165bd86f on December 16th 2018
<U> sup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sales assitant
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> can you shw me some information about intallation?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yesyestyes
    <B> And have you built a contextual assistant or a bot before?
<U> this sucks
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yess
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> picking my nose
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 67a8696eb5894b25a800b6cbd7a695bb on December 15th 2018
<U> hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> howareyou
    <B> I'm great! Thanks for asking. 
<U> are you really a bbot?
    <B> Yes, I'm a bot.
<U> i am a driver
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 67a8696eb5894b25a800b6cbd7a695bb on December 15th 2018
<U> whats popping
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how is your day going
    <B> I'm great! Thanks for asking. 
<U> what are you, a bot?
    <B> Yes, I'm a bot.
<U> 10000k
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> acme bank
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 030829eb30ed4339985d7e71737f6c2d on January 1st 2019
<U> can you guide me know to create knowledge base chatbot
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hi folks
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i get started with rasa myself?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> thats great
    <B> And have you built a contextual assistant or a bot before?
<U> i guess it means - no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yes, cool
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> the master of desaster
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with e977c395a2404bef8ba7f0c3b7e65970 on January 8th 2019
<U> how to build assistant?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i don't have one
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> where should i start from
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> that is cool
    <B> And have you built a contextual assistant or a bot before?
<U> accept
    <B> Are you migrating from another tool?
<U> i do not need help installing
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what's rasa [core](product)?
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU? 
<U> yep please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with 602ddccbe2de4f46822e06e2781cf02d on January 8th 2019
<U> how to get started with rassa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> ssup?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> will u kill me
    <B> I can't help you with that, I'm sorry.
<U> definitely yes without a doubt
    <B> Got it. 
<U> yes of course
    <B> Got it.
## Story from conversation with 4cdb0b3c6f6c4b3694fcb311ae72743b on January 21st 2019
<U> the components of rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> thats fine
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yeeeeezzzzz
    <B> And have you built a contextual assistant or a bot before?
<U> no i haven't decided yet if i want to sign up
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what to do if i want to build a voice bot using rasa?
    <B> You can connect Rasa to any channel you like, including voice channels. You can find more information [here](https://rasa.com/docs/core/connectors/). 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how do i get rasa [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU? 
<U> nehi
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with 4b7ecc2cab6e42c5b1fedb8ab4056866 on December 29th 2018
<U> good mourning
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are you really a bbot?
    <B> Yes, I'm a bot.
<U> who were you created by?
    <B> The engineers at Rasa
<U> what is your exact age?
    <B> Old enough to be a bot!
<U> i am an opioid addic
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter 
<U> turn off my stove
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter 
## Story from conversation with af5a6b3c39d04c6db2b682960e63f01c on January 21st 2019
<U> how can i connect to rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> halloooo
    <B> Hey there, my name is Sara.
<U> who is it that built you?
    <B> The engineers at Rasa
<U> tell me your age?
    <B> Old enough to be a bot!
<U> please schedule a sales call
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
<U> 1231
    <B> Do you still want to talk to sales?
<U> hell yes
    <B> great! 
    - sales_form
    - form{"name": null} 
    <B> How is this conversation going?
## Story from conversation with af5a6b3c39d04c6db2b682960e63f01c on January 21st 2019
<U> how to start with it
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hii
    <B> Hey there, my name is Sara.
<U> who is your employer?
    <B> The engineers at Rasa
<U> when do you celebrate your day of birth?
    <B> Old enough to be a bot!
<U> please connect me to sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
<U> customer service automation bot
    <B> Do you still want to talk to sales?
<U> nah, first time
    - sales_form
    - form{"name": null} 
    <B> Got it.
## Story from conversation with 4a4e903fc43940db9ccdb9153dfdadcb on January 21st 2019
<U> how to start
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i install rasa in windows
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
## Story from conversation with 5f3a2ea92d184a9f96df7240e4f7e2d9 on January 21st 2019
<U> how to start rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how toget strated?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> amayzing
    <B> And have you built a contextual assistant or a bot before?
<U> nah not for me
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> about [both](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with ced8c1eb9a8d485f88a02d931b2879bd on January 16th 2019
<U> can i build a faq robot with rasa?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> slots can be described as ?
    <B> Slots act as a key-value store which can be used to store information the user provided (e.g their home city) as well as information gathered about the outside world (e.g. the result of a database query). Learn more here. 
    <B> Is there anything else I can help you with?
<U> yes that's what i want
    <B> Ok, what can I help you with?
<U> who is the builder of you?
    <B> The engineers at Rasa
<U> get starte
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yesssss
    <B> And have you built a contextual assistant or a bot before?
<U> yes, give me information, please
    <B> Are you migrating from another tool?
<U> na
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yess
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
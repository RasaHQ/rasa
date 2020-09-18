## story number 1
<U> hey there
    <B> Hey there, my name is Sara.
<U> do you have emotions
    <B> I can't help you with that, I'm sorry.
<U> i would love to get the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> Thank you Sara 
    <B> You're welcome! 
    <B> Is there anything else I can help you with?
<U> I want to get in touch with your sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## story number 2
<U> hi
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign up to newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> what do you like to eat
    <B> I can't help you with that, I'm sorry. 
    <B> Is there anything else I can help you with?
## story number 4
<U> hi there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> great thanks
    <B> You're welcome! 
    <B> Is there anything else I can help you with?
<U> what is your name
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> where were you born
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 5
<U> good afternoon
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I'd like to sign up to newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> perfect thank you
    <B> You're welcome! 
    <B> Is there anything else I can help you with?
<U> out of scope question
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 6
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> who built you?
    <B> The engineers at Rasa
<U> how old are you
    <B> Old enough to be a bot!
## story number 7
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hiya
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> where are you from
    <B> I was born in the coolest city on Earth, in Berlin.
<U> talk to sales
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## story number 8
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to build a cool bot
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> I want to build a cool bot
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 9
<U> good morning
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to use pip
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> nope
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
## story number 11, continue
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> who am I
    <B> I hope you are being yourself.
<U> who are you
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> how are you
    <B> I'm great! Thanks for asking. 
<U> where are you from
    <B> I was born in the coolest city on Earth, in Berlin.
<U> I want to sign up to news letter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> no
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> of course
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null} 
    <B> How is this conversation going?
## story number 11, don't continue
<U> hi
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> who am I
    <B> I hope you are being yourself.
<U> who are you
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> how are you
    <B> I'm great! Thanks for asking. 
<U> where are you from
    <B> I was born in the coolest city on Earth, in Berlin.
<U> Subscribe me to the newsletter please!
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> deny
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
<U> no
    <B> Got it.
    - subscribe_newsletter_form
    - form{"name": null} 
## story number 12
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> gimme the newsletter
    <B> Can do! 
    <B> What's your email address?
<U> my email is [bdc@bdce.com](email)
    <B> We sent a confirmation email to [bdc@bdce.com](email). If you want some more awesome Rasa content, check out our [forum](https://forum.rasa.com/). How is this conversation going?
<U> got it
    <B> Got it. 
    <B> Is there anything else I can help you with?
## story number 12, continue
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> gimme the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> I don't have an email
    <B> I can't help you with that, I'm sorry.
    <B> Do you still want to subscribe to the newsletter?
<U> sure
    <B> perfect!
    - subscribe_newsletter_form
     - form{"name": null} 
    <B> How is this conversation going?
<U> got it
    <B> Got it. 
    <B> Is there anything else I can help you with?
## story number 12, don't continue
<U> hello bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to subscribe
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> don't have one
    <B> I can't help you with that, I'm sorry.
    <B> Do you still want to subscribe to the newsletter?
<U> deny
    - subscribe_newsletter_form
     - form{"name": null} 
    <B> How is this conversation going?
<U> i'm lovely
    <B> Got it. 
    <B> Is there anything else I can help you with?
## story number 14
<U> hey there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hey
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how are you feeling
    <B> I'm great! Thanks for asking. 
<U> what's the weather like
    <B> I can't help you with that, I'm sorry.
## story number 15
<U> greetings
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how old are you
    <B> Old enough to be a bot!
<U> I want to build a lot of different bots
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 17
<U> greet
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> nope
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> what do you like to have for lunch
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> who am I
    <B> I hope you are being yourself.
<U> who are you
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> how are you
    <B> I'm great! Thanks for asking. 
<U> where are you from
    <B> I was born in the coolest city on Earth, in Berlin.
## story number 18
<U> hi bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how is it going
    <B> I'm great! Thanks for asking. 
<U> where do you live
    <B> I was born in the coolest city on Earth, in Berlin.
<U> disagree
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> 5 euros
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> I dont want that
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
<U> Can you book a restaurant for me?
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> 5 euros
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> I do not know yet
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 00e7815f79e4413abb0dfb4b392f1099 on November 15th 2018
<U> hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are you a bot
    <B> Yes, I'm a bot.
<U> how do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## story from linda
<U> hi !
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> got it
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with dfbb633d10854f97b880a2496d632f0d on November 16th 2018
<U> hey bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> heya
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I would like to learn how to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with alan on November 16th 2018
<U> I want to learn more about [duckling](nlu_part)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
<U> yes please
    <B> Ok, what can I help you with?
## Story from conversation with alan on November 16th 2018 2
<U> [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yep
    <B> Which (human) language is your assistant in?
<U> my bot is in [spanish](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
<U> how can i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with alan on November 16th 2018 2
<U> [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yes please
    <B> Which (human) language is your assistant in?
<U>  I wrote it in [mandarin](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## Story from conversation with linda on November 15th 2018
<U> hiya
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are you a bot
    <B> Yes, I'm a bot.
<U> where were you created
    <B> I was born in the coolest city on Earth, in Berlin.
<U> I want to learn how to get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with 477ddbe73e374eedb07104c5d9f42c31 on November 16th 2018
<U> hi 
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## Story from conversation with 4986d88ccb784dc19dc5a553a8e07890 on November 19th 2018
<U> hi 
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hello
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are you a bot
    <B> Yes, I'm a bot.
<U> get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Rasa [core](product) please
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> more on [core](product)
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with 7830abb04e1c49809d89b0d420443928 on November 19th 2018
<U> hey there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how is it going
    <B> I'm great! Thanks for asking. 
<U> what's your favourite food
    - respond_out_of_scope
<U> where were you born
    <B> I was born in the coolest city on Earth, in Berlin.
<U> how do I start to use [nlu](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with 29d264d8ce574a11bde572f0e79b73f3 on November 19th 2018
<U> heillo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how are you
    <B> I'm great! Thanks for asking. 
<U> great!
    <B> Got it.
## Story from conversation with 6fd65c93e374489f9c8d76697ab9c493 on November 19th 2018
<U> hey bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hey
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how are you
    <B> I'm great! Thanks for asking. 
<U> gotcha
    <B> Got it.
## Story from conversation with 35d1ecc91c364cbf8a6edf006e5d8c9a on November 19th 2018
<U> heya
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> health bot
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> I would like to talk to sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> very bad
    <B> Got it. 
    <B> Is there anything else I can help you with?
<U> yup
    <B> Got it. 
## Story from conversation with 4c274f8d470e4b77adbfefe7cda7cad7 on October 27th 2018
<U> heya
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> heya
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> yeah
    <B> Got it. 
<U> how do I get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with d041ba4b0a89479e9bb6a5007f2cdc87 on November 15th 2018
<U> I want to contact sales
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
<U> it's going great
    <B> Got it. 
    <B> Is there anything else I can help you with?
<U> where were you made
    <B> I was born in the coolest city on Earth, in Berlin.
<U> None
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with dominik on November 19th 2018
<U> hi
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> 5$
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> how do I get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> affirm
    <B> And have you built a contextual assistant or a bot before?
<U> yeah
    <B> Are you migrating from another tool?
<U> yes I'm using Luis
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
<U> what is [core](product) for? 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
<U> what is [nlu](product) for?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Is there anything else I can help you with?
## Story from conversation with dominik
<U> I don't understand [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yep
    <B> Which (human) language is your assistant in?
<U> I've trained it in [spanish](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
<U> and if my bot is in [french](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## Story from conversation with dominik on November 19th 2018
<U> hey sara
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> I'm interested in [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
## Story from conversation with 201bb55841154f858f524a485f8816c3 on November 18th 2018
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do I get started with [nlu](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with cd483ab3456d47dfb40bd1f51043fb54 on November 18th 2018
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hello bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what can I use [core](product) for
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with cfa8bb9deaf0427498c662745431a282 on December 15th 2018
<U> hey
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> where are you from
    <B> I was born in the coolest city on Earth, in Berlin.
<U> interesting
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with cdd14d763a664a5b95e998ce165bd86f on December 16th 2018
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> 1 million
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> start with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yup
    <B> And have you built a contextual assistant or a bot before?
<U> no never
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yes
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> 1 million
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 67a8696eb5894b25a800b6cbd7a695bb on December 15th 2018
<U> good morning
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how are you
    <B> I'm great! Thanks for asking. 
<U> are you a bot
    <B> Yes, I'm a bot.
<U> My budget is oov
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 67a8696eb5894b25a800b6cbd7a695bb on December 15th 2018
<U> good morning
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how are you
    <B> I'm great! Thanks for asking. 
<U> are you a bot
    <B> Yes, I'm a bot.
<U> my email is [evgeniar@yahoo.com](email)
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> my email is [evgeniar@yahoo.com](email)
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 030829eb30ed4339985d7e71737f6c2d on January 1st 2019
<U> let's start
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hi
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yes please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> got it
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with e977c395a2404bef8ba7f0c3b7e65970 on January 8th 2019
<U> let's get started
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> got it
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
<U> how do i start with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
<U> yeah
    <B> Are you migrating from another tool?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> could you give more details about [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU? 
<U> yeah
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with 602ddccbe2de4f46822e06e2781cf02d on January 8th 2019
<U> let's get started
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hi
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what is your favourite food
    <B> I can't help you with that, I'm sorry.
<U> OK
    <B> Got it. 
<U> Perfect!
    <B> Got it.
## Story from conversation with 4cdb0b3c6f6c4b3694fcb311ae72743b on January 21st 2019
<U> I'd like to get started 
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> ok
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Can I build a rasa voice bot?
    <B> You can connect Rasa to any channel you like, including voice channels. You can find more information [here](https://rasa.com/docs/core/connectors/). 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Tell me more about [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU? 
<U> no thanks
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with 4b7ecc2cab6e42c5b1fedb8ab4056866 on December 29th 2018
<U> hey
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are you a bot
    <B> Yes, I'm a bot.
<U> who built you?
    <B> The engineers at Rasa
<U> how old are you
    <B> Old enough to be a bot!
<U> what do you eat for lunch
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter 
<U> what do you like to eat
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter 
## Story from conversation with af5a6b3c39d04c6db2b682960e63f01c on January 21st 2019
<U> let's get started
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hello
    <B> Hey there, my name is Sara.
<U> who built you?
    <B> The engineers at Rasa
<U> how old are you
    <B> Old enough to be a bot!
<U> I would like to setup a call with sales team
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
<U> a good one
    <B> Do you still want to talk to sales?
<U> yes
    <B> great! 
    - sales_form
    - form{"name": null} 
    <B> How is this conversation going?
## Story from conversation with af5a6b3c39d04c6db2b682960e63f01c on January 21st 2019
<U> let's get started
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hi
    <B> Hey there, my name is Sara.
<U> who built you?
    <B> The engineers at Rasa
<U> how old are you
    <B> Old enough to be a bot!
<U> can i chat with with sales team
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
<U> a good one
    <B> Do you still want to talk to sales?
<U> nope
    - sales_form
    - form{"name": null} 
    <B> Got it.
## Story from conversation with 4a4e903fc43940db9ccdb9153dfdadcb on January 21st 2019
<U> let's get started
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How to install [rasa](product)?
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
## Story from conversation with 5f3a2ea92d184a9f96df7240e4f7e2d9 on January 21st 2019
<U> let's get started
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yeah i'm
    <B> And have you built a contextual assistant or a bot before?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yes please, [both](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with ced8c1eb9a8d485f88a02d931b2879bd on January 16th 2019
<U> let's get started
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Help me understand what slots are.
    <B> Slots act as a key-value store which can be used to store information the user provided (e.g their home city) as well as information gathered about the outside world (e.g. the result of a database query). Learn more here. 
    <B> Is there anything else I can help you with?
<U> yes
    <B> Ok, what can I help you with?
<U> who built you?
    <B> The engineers at Rasa
<U> how do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep
    <B> And have you built a contextual assistant or a bot before?
<U> yes
    <B> Are you migrating from another tool?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yes please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
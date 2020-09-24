## story number 1
* greet: hello it is me again
    <B> Hey there, my name is Sara.
* out_of_scope: i need a gp in 94301
    <B> I can't help you with that, I'm sorry.
* signup_newsletter: subscribe bruce_harryman@olsen.com to the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
* thank: cool thanks
    <B> You're welcome! 
    <B> Is there anything else I can help you with?
* contact_sales: let me contact sales
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## story number 2
* greet: hii
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: subscribe
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
* out_of_scope: is that any of your business
    <B> I can't help you with that, I'm sorry. 
    <B> Is there anything else I can help you with?
## story number 4
* greet: ayyyy whaddup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: get the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
* thank: thanks for the help
    <B> You're welcome! 
    <B> Is there anything else I can help you with?
* chitchat: who are you and what do you want from me
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
* chitchat: where are you?
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 5
* greet: hieeeeeeeeeeeee
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i would like to subscribe to your newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null} 
    <B> How is this conversation going?
* thank: ok thanks sara
    <B> You're welcome! 
    <B> Is there anything else I can help you with?
* out_of_scope: gsaf
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 6
* greet: jojojo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: and who built u?
    <B> The engineers at Rasa
* chitchat: how old are u
    <B> Old enough to be a bot!
## story number 7
* greet: i said, helllllloooooo!!!!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: where did you come from?
    <B> I was born in the coolest city on Earth, in Berlin.
* contact_sales: let me talk to your sales people
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## story number 8
* greet: yoo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: a customer service bot
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* enter_data: driver
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 9
* greet: rasa hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: 75000-150000 euro
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* deny: i don't want either of those
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
## story number 11, continue
* greet: halo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: who could i be?
    <B> I hope you are being yourself.
* chitchat: what your name?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
* chitchat: how are you doing today my sweet friend
    <B> I'm great! Thanks for asking. 
* chitchat: where did you spend your youth?
    <B> I was born in the coolest city on Earth, in Berlin.
* signup_newsletter: can you add edward@paul.com to the newsletter list?
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* deny: i don not like this
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
* affirm: ya thats cool
    <B> Got it. 
    - subscribe_newsletter_form
    - form{"name": null} 
    <B> How is this conversation going?
## story number 11, don't continue
* greet: halloo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: do you know who i am?
    <B> I hope you are being yourself.
* chitchat: i want to know who you are
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
* chitchat: nah, i'm good - how are you doing?
    <B> I'm great! Thanks for asking. 
* chitchat: where did you grow up?
    <B> I was born in the coolest city on Earth, in Berlin.
* signup_newsletter: i need this dope newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* deny: no thank s
    <B> I can't sign you up if you don't provide your email Do you still want to subscribe to the newsletter?
* deny: i don't think so
    <B> Got it.
    - subscribe_newsletter_form
    - form{"name": null} 
## story number 12
* greet: hai
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i'd like to subscribe
    <B> Can do! 
    <B> What's your email address?
* enter_data: a wolf bot
    <B> We sent a confirmation email to [bdc@bdce.com](email). If you want some more awesome Rasa content, check out our [forum](https://forum.rasa.com/). How is this conversation going?
* enter_data: 10000 dollars
    <B> Got it. 
    <B> Is there anything else I can help you with?
## story number 12, continue
* greet: hell9o
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: subcribe
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* deny: none of them
    <B> I can't help you with that, I'm sorry.
    <B> Do you still want to subscribe to the newsletter?
* affirm: done
    <B> perfect!
    - subscribe_newsletter_form
     - form{"name": null} 
    <B> How is this conversation going?
* enter_data: there is no budget
    <B> Got it. 
    <B> Is there anything else I can help you with?
## story number 12, don't continue
* greet: halloo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i would love to subscribe to a newsletter!
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* enter_data: boo
    <B> I can't help you with that, I'm sorry.
    <B> Do you still want to subscribe to the newsletter?
* deny: no i haven't decided yet if i want to sign up
    - subscribe_newsletter_form
     - form{"name": null} 
    <B> How is this conversation going?
* react_positive: like u
    <B> Got it. 
    <B> Is there anything else I can help you with?
## story number 14
* greet: good moring
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: jop
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: whats goin on
    <B> I'm great! Thanks for asking. 
* chitchat: is it raining?
    <B> I can't help you with that, I'm sorry.
## story number 15
* greet: merhaba
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: how many years have you lived?
    <B> Old enough to be a bot!
* enter_data: like 60 quid
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## story number 17
* greet: hello it is me again
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* deny: i don't want to
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
* out_of_scope: which kind
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* chitchat: who could i be?
    <B> I hope you are being yourself.
* chitchat: whats ur name
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
* chitchat: how r u ?
    <B> I'm great! Thanks for asking. 
* chitchat: what state were you born in?
    <B> I was born in the coolest city on Earth, in Berlin.
## story number 18
* greet: hieeeeeeeeeeeee
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: how ?
    <B> I'm great! Thanks for asking. 
* chitchat: where are you located?
    <B> I was born in the coolest city on Earth, in Berlin.
* deny: no bots at all
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
* enter_data: 10000 dollars
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* deny: no go
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
* out_of_scope: drhdtjfjfyj
    <B> I can't help you with that, I'm sorry.
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* enter_data: we don't have one
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* enter_data: something to talk to my friends while i'm busy working
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 00e7815f79e4413abb0dfb4b392f1099 on November 15th 2018
* greet: hello world
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: hi man
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: are you real lol
    <B> Yes, I'm a bot.
* how_to_get_started: want to build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## story from linda
* greet: i said, helllllloooooo!!!!
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: in health care domain
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* how_to_get_started: tell me about the components
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with dfbb633d10854f97b880a2496d632f0d on November 16th 2018
* greet: hi there it's me
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: heelio
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how do i set up a chatbot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with alan on November 16th 2018
* nlu_info: how do you integrate [duckling](nlu_part)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
* affirm: ja cool
    <B> Ok, what can I help you with?
## Story from conversation with alan on November 16th 2018 2
* nlu_info: i don’t understand how you handle [entity recognition](nlu_part) at rasa
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: oh super
    <B> Which (human) language is your assistant in?
* enter_data: its an [spanish](language) bot
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
* how_to_get_started: how toget strated?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with alan on November 16th 2018 2
* nlu_info: an explanation of how [intent classification](nlu_part) work would help
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: k
    <B> Which (human) language is your assistant in?
* enter_data: it’s trained only in [chinese](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## Story from conversation with linda on November 15th 2018
* greet: hey let's talk
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: are you sure that you're a bot?
    <B> Yes, I'm a bot.
* chitchat: what's the name of the place you came from?
    <B> I was born in the coolest city on Earth, in Berlin.
* how_to_get_started: can i just test features without having to deal with your predefined conversation
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with 477ddbe73e374eedb07104c5d9f42c31 on November 16th 2018
* greet: hey dude
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: halloooo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: more information on components in rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* greet: heylo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## Story from conversation with 4986d88ccb784dc19dc5a553a8e07890 on November 19th 2018
* greet: hey rasa
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: hlo
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: are you sure that you're a bot?
    <B> Yes, I'm a bot.
* how_to_get_started: where to start the development of rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: confirm
    <B> And have you built a contextual assistant or a bot before?
* deny: it sux
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how can i start with rasa [core](product)?
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* how_to_get_started: how do i train rasa [core](product)
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with 7830abb04e1c49809d89b0d420443928 on November 19th 2018
* greet: sup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: hi sara, what are you up to?
    <B> I'm great! Thanks for asking. 
* out_of_scope: pizza bot
    - respond_out_of_scope
* chitchat: what location are you from?
    <B> I was born in the coolest city on Earth, in Berlin.
* how_to_get_started: just rasa [nlu](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with 29d264d8ce574a11bde572f0e79b73f3 on November 19th 2018
* greet: hi folks
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: halo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: how r u
    <B> I'm great! Thanks for asking. 
* affirm: uh-huh
    <B> Got it.
## Story from conversation with 6fd65c93e374489f9c8d76697ab9c493 on November 19th 2018
* greet: hello rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: helleo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: how are you today
    <B> I'm great! Thanks for asking. 
* react_positive: hèhè
    <B> Got it.
## Story from conversation with 35d1ecc91c364cbf8a6edf006e5d8c9a on November 19th 2018
* greet: hello it is me again
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: oov per year
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* contact_sales: hmmm sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
* deny: no i won't
    <B> Got it. 
    <B> Is there anything else I can help you with?
* affirm: yes, cool
    <B> Got it. 
## Story from conversation with 4c274f8d470e4b77adbfefe7cda7cad7 on October 27th 2018
* greet: good evening
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: hello sweatheart
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* affirm: that would be great
    <B> Got it. 
* how_to_get_started: can you shw me some information about intallation?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with d041ba4b0a89479e9bb6a5007f2cdc87 on November 15th 2018
* contact_sales: just gimme a call
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
* react_positive: you good
    <B> Got it. 
    <B> Is there anything else I can help you with?
* chitchat: i want to know the company which generated you
    <B> I was born in the coolest city on Earth, in Berlin.
* enter_data: a cool boy
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with dominik on November 19th 2018
* greet: hey let's talk
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: 1231
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* how_to_get_started: i wanted to build a bot my product customer support
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: i guess so
    <B> And have you built a contextual assistant or a bot before?
* affirm: sounds good!
    <B> Are you migrating from another tool?
* switch: i use [wit.ai](current_api)
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
* how_to_get_started: how to get started with rasa [core](product)?
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
* how_to_get_started: tell me about rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Is there anything else I can help you with?
## Story from conversation with dominik
* nlu_info: [intent classification](nlu_part) - what is that?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: top
    <B> Which (human) language is your assistant in?
* enter_data: it’s trained only in [mandarin](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
* enter_data: the people speak [french](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## Story from conversation with dominik on November 19th 2018
* greet: konichiwa
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: do you know how to set up a chatbot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i'm not going to give it to you
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: i wanna try rasa [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
## Story from conversation with 201bb55841154f858f524a485f8816c3 on November 18th 2018
* greet: heyho
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how do i get rasa [nlu](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with cd483ab3456d47dfb40bd1f51043fb54 on November 18th 2018
* greet: what up
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: hallo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: tell me how to get started with [core](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## Story from conversation with cfa8bb9deaf0427498c662745431a282 on December 15th 2018
* greet: hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: where did you originate?
    <B> I was born in the coolest city on Earth, in Berlin.
* out_of_scope: you lock sweety
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with cdd14d763a664a5b95e998ce165bd86f on December 16th 2018
* greet: hi man
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: im lonely
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* how_to_get_started: how easy is it to use rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes that's what i want
    <B> And have you built a contextual assistant or a bot before?
* deny: not going well at all
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: yes, cool
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* enter_data: 10000000
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 67a8696eb5894b25a800b6cbd7a695bb on December 15th 2018
* greet: helleo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: how r u ?
    <B> I'm great! Thanks for asking. 
* chitchat: are you robot
    <B> Yes, I'm a bot.
* enter_data: it's r_iuliucci@yahoo.com
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 67a8696eb5894b25a800b6cbd7a695bb on December 15th 2018
* greet: wasssup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: how r u>
    <B> I'm great! Thanks for asking. 
* chitchat: oh are you chatbot?
    <B> Yes, I'm a bot.
* enter_data: 20000k
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* enter_data: we don't have one
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with 030829eb30ed4339985d7e71737f6c2d on January 1st 2019
* how_to_get_started: how to build a bot?
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: hiihihi
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: what is componenbts
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: done
    <B> And have you built a contextual assistant or a bot before?
* deny: no. u r idiot
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: yes cool
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* enter_data: 100000
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
## Story from conversation with e977c395a2404bef8ba7f0c3b7e65970 on January 8th 2019
* how_to_get_started: what are the coponent
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* enter_data: my business mail is s_dibenedetto@simpson.net
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter
* how_to_get_started: how to make a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: jezz
    <B> And have you built a contextual assistant or a bot before?
* affirm: i want that
    <B> Are you migrating from another tool?
* deny: nah not for me
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: tell me about rasa [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU? 
* affirm: yep that's fine
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with 602ddccbe2de4f46822e06e2781cf02d on January 8th 2019
* how_to_get_started: how this rasa works
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: wasssup!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* out_of_scope: i can't deal with _your_ request
    <B> I can't help you with that, I'm sorry.
* affirm: ok i accept
    <B> Got it. 
* affirm: yep. :/
    <B> Got it.
## Story from conversation with 4cdb0b3c6f6c4b3694fcb311ae72743b on January 21st 2019
* how_to_get_started: ow to get started with rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* affirm: you got me, i accept, if you want me to
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: kk
    <B> And have you built a contextual assistant or a bot before?
* deny: no, thankyou
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq: can i use rasa to build a voice bot?
    <B> You can connect Rasa to any channel you like, including voice channels. You can find more information [here](https://rasa.com/docs/core/connectors/). 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: what's rasa [core](product)?
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU? 
* deny: nein
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with 4b7ecc2cab6e42c5b1fedb8ab4056866 on December 29th 2018
* greet: ayyyy whaddup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: are you really a bot
    <B> Yes, I'm a bot.
* chitchat: tell me more about your founders
    <B> The engineers at Rasa
* chitchat: can you tell me what number represents your age?
    <B> Old enough to be a bot!
* out_of_scope: i am asking you an out of scope question
    <B> Hmm, not sure what you mean... 
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter 
* out_of_scope: more
    <B> Hmm, not sure what you mean...
    <B> You can ask me about:     - how to get started with Rasa     - the components of Rasa,     - booking a sales call     - subscribing to our newsletter 
## Story from conversation with af5a6b3c39d04c6db2b682960e63f01c on January 21st 2019
* how_to_get_started: i want to know how can buld my own bot
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: jojojo
    <B> Hey there, my name is Sara.
* chitchat: tell me the name of your creator.
    <B> The engineers at Rasa
* chitchat: do you have friends the same age as you, if so, how old are they?
    <B> Old enough to be a bot!
* contact_sales: how can i contact the team ?
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
* enter_data: i sell turtles
    <B> Do you still want to talk to sales?
* affirm: ofcourse
    <B> great! 
    - sales_form
    - form{"name": null} 
    <B> How is this conversation going?
## Story from conversation with af5a6b3c39d04c6db2b682960e63f01c on January 21st 2019
* how_to_get_started: how do i build a bot?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* greet: whats up my bot
    <B> Hey there, my name is Sara.
* chitchat: who is behind all this?
    <B> The engineers at Rasa
* chitchat: do you have friends the same age as you, if so, how old are they?
    <B> Old enough to be a bot!
* contact_sales: sales sales sales
    <B> Sure, we can book a sales call! Let's get to know each other first .
    - sales_form
    - form{"name": "sales_form"}
* enter_data: customer service automation
    <B> Do you still want to talk to sales?
* deny: nah not for me
    - sales_form
    - form{"name": null} 
    <B> Got it.
## Story from conversation with 4a4e903fc43940db9ccdb9153dfdadcb on January 21st 2019
* how_to_get_started: sure, give me the basics
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* install_rasa: yes, i do need rasa stack.
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
## Story from conversation with 5f3a2ea92d184a9f96df7240e4f7e2d9 on January 21st 2019
* how_to_get_started: how to get started with rasa ?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: more information on components in rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes i accept
    <B> And have you built a contextual assistant or a bot before?
* deny: no you did it wrong
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: the [rasa open source](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## Story from conversation with ced8c1eb9a8d485f88a02d931b2879bd on January 16th 2019
* how_to_get_started: how do i use rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq: where can i find the definition of slots?
    <B> Slots act as a key-value store which can be used to store information the user provided (e.g their home city) as well as information gathered about the outside world (e.g. the result of a database query). Learn more here. 
    <B> Is there anything else I can help you with?
* affirm: okay
    <B> Ok, what can I help you with?
* chitchat: who was the one that built you?
    <B> The engineers at Rasa
* how_to_get_started: how this rasa works
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: i want that
    <B> And have you built a contextual assistant or a bot before?
* affirm: great lets do that
    <B> Are you migrating from another tool?
* deny: no i don't accept
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: yes it was okay
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
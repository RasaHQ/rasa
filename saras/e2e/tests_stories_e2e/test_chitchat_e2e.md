## chitchat
* chitchat: are you happy
    <B> I'm great! Thanks for asking.
## deny chitchat
* chitchat: you idiot
    <B> That's not very nice
* deny: non
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
## more chitchat
* greet: hi mister
    <B> Hey there, my name is Sara.
* chitchat: is everything okay
    <B> I'm great! Thanks for asking.
* chitchat: how old will you be on your next birthday?
    <B> Old enough to be a bot!
## greet chitchat
* greet: wasssup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: what's up
    <B> I'm great! Thanks for asking.
## just newsletter + confirm
* greet: hello world
    <B> Hey there, my name is Sara.
* chitchat: what's your name bot
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
* signup_newsletter: i need the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
* affirm: yes you can
    <B> Awesome! 
    <B> Is there anything else I can help you with?
## just newsletter, continue, + confirm
* greet: hola
    <B> Hey there, my name is Sara.
* signup_newsletter: i want to signup for your newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat: what are you up to?
    <B> I'm great! Thanks for asking. 
    <B> Do you still want to subscribe to the newsletter?
* affirm: well yes
    <B> Cool, let's do it! 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
* affirm: yay
    <B> Okay. 
    <B> Is there anything else I can help you with?
## just newsletter, don't continue, + confirm
* greet: hola
    <B> Hey there, my name is Sara.
* signup_newsletter: sign up
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat: what's new?
    <B> I'm great! Thanks for asking. 
    <B> Do you still want to subscribe to the newsletter?
* deny: i don not like this
    <B> Got it. 
    <B> How is this conversation going?
    - form{"name": null}
* affirm: ook
    <B> Okay.
    <B> Is there anything else I can help you with?
## just newsletter (with email already) + confirm
* greet: wasssup
    <B> Hey there, my name is Sara.
* chitchat: i want to know who you are
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
* signup_newsletter: newsletter please.
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* affirm: sure thing
    <B> Okay. 
    <B> Is there anything else I can help you with?
## just newsletter (with email already)
* greet: hello rasa
    <B> Hey there, my name is Sara.
* chitchat: do you know who i am?
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
* signup_newsletter: may i receive the newsletter from now on
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just newsletter
* greet: halo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: what can i do with this bot
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
* signup_newsletter: i go for the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just newsletter, continue
* greet: hey, let's talk
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: can you add edward@paul.com to the newsletter list?
    <B> You got it.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat: so what next?
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. 
    <B> Do you still want to subscribe to the newsletter?
* affirm: ok good
    <B> That's great!
    - subscribe_newsletter_form
    - form{"name": null} 
    <B> How is this conversation going?
## just newsletter, don't continue
* greet: jop
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: i want to signup for your newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* chitchat: okay hi how are you?
    <B> I'm great! Thanks for asking. 
    <B> Do you still want to subscribe to the newsletter?
* deny: no i dont want to accept :p lol
    <B> Got it.
    - form{"name": null}
    <B> How is this conversation going?
## just sales
* greet: hello sweet boy
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: are you a human ?
    <B> I am a bot
* contact_sales: how does your pricing work?
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
## just sales, continue
* greet: what is up?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: i want ti booking about booking sales
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
* chitchat: tell me something about you
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot. 
    <B> Do you still want to talk to sales?
* affirm: ok, sara
    - sales_form
    - form{"name": null}
## just sales, don't continue
* greet: halo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: i want to connect to your sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
* chitchat: what else can i do?
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
    <B> Do you still want to talk to sales?
* deny: stop it, i do not care!!!
    <B> Got it.
    - form{"name": null}
## new to rasa + not new to chatbots + not migrating
* greet: hello everybody
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: ok who are you ?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
* how_to_get_started: how toget strated?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
    - slot{"onboarding": true}
* affirm: accepted
    <B> And have you built a contextual assistant or a bot before?
* affirm: okay sure
    <B> Are you migrating from another tool?
* deny: no sorry
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: yep, will do thank you
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API.
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
* greet: heylo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how to build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* chitchat: how can i use you
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
    <B> Are you new to Rasa?
* affirm: i am using it
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: hm, i'd like that
    <B> Are you migrating from another tool?
* deny: n
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: [rasa open source](product) first
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
* greet: hai
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how to get started with rasa ?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes, cool
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
* chitchat: how are u?
    <B> I'm great! Thanks for asking. 
    <B> And have you built a contextual assistant or a bot before?
* affirm: thats good
    <B> Are you migrating from another tool?
* deny: still dont want to tell
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: yep thats cool
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
* greet: wasssup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: components of rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: really
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
* affirm: ye splease
    <B> Are you migrating from another tool?
* out_of_scope: i need money
    <B> I can't help you with that, I'm sorry. 
    <B> Are you migrating from another tool?
* deny: no stop
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: confirm
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
* greet: hi can you speak ?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: tell me how to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: a little
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: yep. :/
    <B> Are you migrating from another tool?
* deny: no sir
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* chitchat: what is your nme?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: confirm
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
* greet: hlo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i wanted to build a bot my product customer support
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: i have used it in the past
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: okay sure
    <B> Are you migrating from another tool?
* deny: i decline
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: y
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* chitchat: so what can you do
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain NLU and try it out
* greet: hello sweatheart
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: [core](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: okay cool
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
* chitchat: how's life treating you friend?
    <B> I'm great! Thanks for asking. 
    <B> And have you built a contextual assistant or a bot before?
* deny: i'm not going to give it to you
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: i would like to know more about rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
* affirm: i changed my mind. i want to accept it
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain NLU and try it out
* greet: bom dia
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: what are your features ?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes sirfr
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
* deny: i don't want to say
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
## new to rasa/bots, explain NLU and try it out
* greet: good evening
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i like to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: it is ok
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: i'm not sure
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* chitchat: what you can do for me?
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: yes that's great
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain stack and try it out
* greet: ssup?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i want to make intelligence chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: ok fine
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: i'm not giving you my email address
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: ofcourse
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* chitchat: which company created you?
    <B> Engineers at Rasa. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core and try out stack
* greet: hai
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: show me learning resources about rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: very much
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: i'm not giving you my email address
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: and rasa [nlu](product)?
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* out_of_scope: is rasa a studio?
    <B> I can't help you with that, I'm sorry.
    <B> Would you also like me to explain NLU?
* deny: no i don't accept
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core and try out stack
* greet: halloo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how to build a bot in rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yyeeeh
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: no i dont want
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how do i learn rasa [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* deny: not going well at all
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* chitchat: i'm looking for a spanish restaurant.
    <B> I can't help you with that, I'm sorry. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core, then nlu and try out stack
* greet: heyho
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how can i connect to rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yres
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: nevermind
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to get started with rasa [core](product)?
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* chitchat: i have a name, what is it?
    <B> I can't help you with that, I'm sorry.
    <B> Would you also like me to explain NLU?
* affirm: its okay
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core, then nlu and try out stack
* greet: ola sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i want to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: considering
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: it sux
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: tell me how to get started with [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* affirm: yes, give me information, please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* chitchat: beautiful day, isn't it?
    <B> I can't help you with that, I'm sorry. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu and try out stack
* greet: rasa hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: do you know how to set up a chatbot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: going super well
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: na
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: yes i wanna know more about rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
* chitchat: who could i be?
    <B> I can't help you with that, I'm sorry.
    <B> Would you also like me to explain Core?
* deny: i'm not giving you my email address
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu and try out stack
* greet: konichiwa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i like to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: very much
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: no don"t want this!
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: [rasa open source](product) first
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
* deny: nopes
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* chitchat: ok who are you ?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, don't explain and try out stack
* greet: helloooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how do you build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes i have built a bot before
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: i do not need help installing
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* chitchat: who r you
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: this sucks
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, don't explain and try out stack
* greet: helo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: can you help me build my bot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yeah do that
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: i'm afraid not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: nehi
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* chitchat: how r u ?
    <B> I'm great! Thanks for asking. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + not interested in products
* greet: what is up?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* chitchat: i want to know the company which generated you
    <B> Engineers at Rasa.
* how_to_get_started: tell me how to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no thank s
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: never mind
    <B> Got it.
## not new to rasa + not interested in products
* greet: guten morgen
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: get strarted with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* chitchat: hello. what's your name?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
    <B> Are you new to Rasa?
* deny: n
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* enter_data: slack
    <B> Got it.
## not new to rasa + not interested in products
* greet: hai
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how do i build a rasa chatbot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no i dont want to
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* chitchat: how is the weather?
    <B> I can't help you with that, I'm sorry.
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: no stop
    <B> Got it.
## not new to rasa + nlu + nothing special
* greet: wasssup!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how to build assistant?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no stop
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: i want to know about [rasa open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* deny: noooooooooo
    <B> Rasa Masterclass episodes 2-4 focus on NLU. Check out episode 2 [here](https://www.youtube.com/watch?v=k5UeywXA28k). 
    <B> Is there anything else I can help you with?
## not new to rasa + rasa x + nothing special
* greet: hello is anybody there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i am trying to build a bot using rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no i don't accept
    - slot{"onboarding": true}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: where can i find rasa document
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
* chitchat: what can you teache me?
    <B> Would you also like me to explain NLU and Core?
* explain: how come?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + rasa x + nothing special
* greet: hieee
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: information about rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no sir
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: what's rasa [nlu](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
* deny: no i won't
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + rasa x + nothing special
* greet: wasssup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how do i build a chatbot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* chitchat: how old were you on your last birthday?
    <B> Old enough to be a bot! 
    <B> Are you new to Rasa?
* affirm: ok cool
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: i don't want to give it to you
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how this rasa works
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
* deny: it sux
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* chitchat: who is this bot
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + nlu + unknown topic
* greet: hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how can i connect to rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i do not need help installing
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to get started with [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: how does [intent classification](nlu_part) work?
    <B> Hmm, seems like I haven't learned about that part of NLU yet! Try using the search bar above for now. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + no recommendation
* greet: ayyyy whaddup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: where do i start?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no thank s
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: just [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: i want to learn about [entity recognition](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* deny: none of them
    <B> Got it. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, spacy
* greet: ey boss
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: more info on components
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: sorry not right now
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: yes i wanna know more about rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: what is [duckling](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* pipeline_recommendation: what i a good pipeline to start with?
    <B> Which (human) language is your assistant in?
* enter_data: [portuguese](language) is the only language but i want to add more
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
* greet: rasa hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: get strarted with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i dont want to tell
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: rasa [core](product) quickstart
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: [intent classification](nlu_part) - what is that?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* pipeline_recommendation: spacy or tensorflow, what is better to start?
    <B> Which (human) language is your assistant in?
* enter_data: it’s in [portuguese](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
* greet: hola
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: tell me more about how to use rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no, thankyou
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how do i train rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: i checked the documentation on [intent classification](nlu_part) but i still don’t understand it
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: yep thats cool
    <B> Which (human) language is your assistant in?
* enter_data: so far it only speaks [chinese](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline duckling
* greet: halloooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i like to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i dont want to accept :p lol
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: what [rasa_nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: [intent](nlu_part) please
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
* chitchat: do you have friends the same age as you, if so, how old are they?
    <B> Old enough to be a bot! 
    <B> Would you like me to try to recommend a pipeline?
* affirm: thats good
    <B> What kind of entities do you want to extract?
* enter_data: [time](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline duckling
* greet: hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: what is componenbts
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i'm not sure
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: more about [rasa open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: sorry its [ner](nlu_part)
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
* affirm: fcourse
    <B> What kind of entities do you want to extract?
* chitchat: who is responsible for making you?
    <B> Engineers at Rasa. 
    <B> What kind of entities do you want to extract? 
* enter_data: i am interested in [ordinals](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## how to get started without privacy policy
* how_to_get_started: can give tell me about components of rosa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* out_of_scope: how much is 10 + 89
    <B> I can't help you with that, I'm sorry. 
    <B> Are you new to Rasa?
* affirm: yeeeeezzzzz
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: oh awesome!
    <B> Are you migrating from another tool?
* deny: nop
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: i would like to know more about rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## chitchat interrupting step 2 flow
* get_started_step2: go to step 2
    <B> I can show you how to start building your first AI assistant with Rasa. Are you ready?
* chitchat: what time is it
    <B> I can't help you with that, I'm sorry.
    <B> Do you want to go ahead with the steps to create a contextual assistant?
* affirm: yes i agree
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## chitchat interrupting and stopping step 2 flow
* get_started_step2: go to step 2
    <B> I can show you how to start building your first AI assistant with Rasa. Are you ready?
* greet: bom dia
    <B> I'm great! Thanks for asking. 
    <B> Do you want to go ahead with the steps to create a contextual assistant?
* deny: i don't agree
    <B> Got it. 
    <B> Is there anything else I can help you with?
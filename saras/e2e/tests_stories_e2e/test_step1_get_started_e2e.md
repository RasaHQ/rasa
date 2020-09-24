## prompt for getting started
* greet: wasssup!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## prompt for getting started + confirm
* greet: heylo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* affirm: hell yes
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## new to rasa at start, built bot before
* how_to_get_started: ok i am [new](user_type) to rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yeeeeezzzzz
    - slot{"onboarding": true}
    <B> Are you migrating from another tool?
## new to rasa at start
* how_to_get_started: i m [new](user_type)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't think so
    - slot{"onboarding": true}
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* explain: please elaborate
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + built a bot before
* how_to_get_started: ow to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: okay cool
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
* affirm: you got me, i accept, if you want me to
    <B> Are you migrating from another tool?
* deny: nehi
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* explain: why is that necessary
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
* how_to_get_started: start rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* how_to_get_started: ok i am [new](user_type) to rasa
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
* affirm: i will!
    <B> Are you migrating from another tool?
* deny: no don"t want this!
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* explain: give me more details
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + migrating from dialogflow
* how_to_get_started: how toget strated?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: i want that
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
* affirm: yes cool
    <B> Are you migrating from another tool?
* switch: [dialogflow](current_api)
    <B> We have a guide for migrating from DialogFlow [here](https://www.rasa.com/docs/rasa/migrate-from/google-dialogflow-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from luis
* how_to_get_started: i want to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yess
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
* affirm: fuck yeah!
    <B> Are you migrating from another tool?
* switch: [wit](current_api)
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from something else
* how_to_get_started: what is componenbts
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* how_to_get_started: tell me about the components
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: sounds good!
    <B> Are you migrating from another tool?
* technical_question: how about interactive learning
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
##  migrating from dialogflow
* how_to_get_started: what are the components?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: ok let's start
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: it is ok
    <B> Are you migrating from another tool?
* affirm: yesssss
    <B> Which tool are you migrating from?
* switch: i want to convert my [dialog flow](current_api) bot to rasa
    <B> We have a guide for migrating from DialogFlow [here](https://www.rasa.com/docs/rasa/migrate-from/google-dialogflow-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from luis
* how_to_get_started: can i build a faq robot with rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: oh awesome!
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: i agree
    <B> Are you migrating from another tool?
* affirm: yes you can
    <B> Which tool are you migrating from?
* switch: i am using [dialogflow](current_api) - how can i migrate
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from something else
* how_to_get_started: where can i learn to build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: i'm using it
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: i accept.
    <B> Are you migrating from another tool?
* affirm: ya i want
    <B> Which tool are you migrating from?
* switch: can i migrate to rasa from another tool?
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## just switch
* switch: i use [wit.ai](current_api)
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## just switch
* switch: switching
    <B> Which tool are you migrating from?
* switch: [dialogflow](current_api)
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## just switch
* switch: switching
    <B> Which tool are you migrating from?
* switch: switch to rasa from another platform
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## new to rasa/bots, explain stack
* how_to_get_started: i like to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: confirm
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: non
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: tell me how to get started with [core](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core
* how_to_get_started: how this rasa works
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: okay
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: i'm not giving you my email address
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: yes, i want to know more about [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* deny: i'm afraid not
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core
* how_to_get_started: i want to make intelligence chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: agreed
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: still dont want to tell
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: please teach me
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, explain x, then nlu+core
* how_to_get_started: components in rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: that would be great
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: no you did it wrong
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: want to build a chatbot
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, explain x
* how_to_get_started: whar are the components of rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: oui
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: no i dont want to accept :p lol
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* enter_data: [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
* deny: nah not for me
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu
* how_to_get_started: get strarted with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* how_to_get_started: [new](user_type)
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
* deny: no stop
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how do i train rasa [core](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
* deny: i'm not going to give it to you
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu then core
* how_to_get_started: rasa basics
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: i'd absolutely love that
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: i don't think so
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to learn rasa [core](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
* affirm: yaah
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa/bots, explain all
* how_to_get_started: how to build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't think so
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to get started with rasa [core](product)?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa/bots, explain all
* how_to_get_started: how to get sarted
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i dont want to accept :p lol
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: i want to learn about rasa [core](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, explain all
* how_to_get_started: how to start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: lets do this
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: no, my frst time
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* explain: why is that necessary
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain all
* how_to_get_started: i want to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes it was okay
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: no sorry
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: i want to learn about rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, don't explain
* how_to_get_started: how to build a bot in rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* how_to_get_started: i'm [new](user_type) to rasa
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* deny: no i dont want to
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: no way
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + not interested in products
* how_to_get_started: sure, give me the basics
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't want to give you my email
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: no i haven't decided yet if i want to sign up
    <B> Got it. 
## not new to rasa + core
* how_to_get_started: ow to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i do not need help installing
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: [nlu](product) and [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
## not new to rasa + rasa
* how_to_get_started: hot to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't want to
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to get starter?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + rasa
* how_to_get_started: how to learn rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i dont want to
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: the [rasa open source](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + rasa
* how_to_get_started: how do i build a rasa chatbot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't wanna tell the name of my company
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: can i build a faq robot with rasa?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + rasa
* how_to_get_started: start rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no don"t want this!
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to build a bot?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + [rasa](product)
* how_to_get_started: how about building chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't want either of those
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how this rasa works
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## skip to info on rasa x
* how_to_get_started: how can i start
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
## skip to info on rasa
* how_to_get_started: how t oget started
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## skip to info on rasa nlu
* how_to_get_started: get started with [rasa open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: it would be helpful to learn more about [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
## not new to rasa + x
* how_to_get_started: please teach me
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: nö
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq: what is [rasa x](product) ?
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API.
    <B> Would you also like me to explain NLU and Core?
## new to rasa + x
* how_to_get_started: more information on components in rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes give me information
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: yap
    <B> Are you migrating from another tool?
* deny: no, thankyou
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* enter_data: [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API.
    <B> Would you also like me to explain NLU and Core?
* affirm: i'm using it
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + rasa oss
* how_to_get_started: i want to know how can buld my own bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: oh super
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: very much
    <B> Are you migrating from another tool?
* deny: no and no again
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: what are rasa's components?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa + rasa oss
* how_to_get_started: where to start the development of rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yesh
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
* affirm: done
    <B> Are you migrating from another tool?
* deny: no, thankyou
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how do i build a rasa chatbot?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## skip to info on core
* how_to_get_started: get started with [rasa open source](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + nothing special
* how_to_get_started: i wanted to build a bot my product customer support
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: na
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: [nlu](product) and [core](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* deny: no i don't accept
    <B> Rasa Masterclass episodes 2-4 focus on NLU. Check out episode 2 [here](https://www.youtube.com/watch?v=k5UeywXA28k). 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + unknown topic
* how_to_get_started: how about building chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: nö
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: i need more info for rasa [core](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: how enttity extrcation works
    <B> Hmm, seems like I haven't learned about that part of NLU yet! Try using the search bar above for now. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + no recommendation
* how_to_get_started: where can i learn more about rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no i don't accept
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: more about [rasa open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: i still don’t get how [intent classification](nlu_part) works
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* deny: not right now
    <B> Got it. 
    <B> Is there anything else I can help you with?
## pipeline
* pipeline_recommendation: pipeline
    <B> Which (human) language is your assistant in?
* enter_data: it’s trained in [chinese](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?
## pipeline
* pipeline_recommendation: nlu pipeline
    <B> Which (human) language is your assistant in?
* enter_data: user can communicate with the bot in [spanish](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## pipeline
* pipeline_recommendation: what i a good pipeline to start with?
    <B> Which (human) language is your assistant in?
* enter_data: so far it only speaks [mandarin](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## pipeline
* pipeline_recommendation: which pipeline is better?
    <B> Which (human) language is your assistant in?
* enter_data: the language of the ai assistant is [french](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, spacy
* how_to_get_started: hot to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: not right now
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: about [both](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: i still don’t get how [entity recognition](nlu_part) works
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: yes, cool
    <B> Which (human) language is your assistant in?
* enter_data: i wrote it in [german](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
* how_to_get_started: i want to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: na
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to get started with rasa [core](product)?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: does rasa work with [duckling](nlu_part)?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: well yes
    <B> Which (human) language is your assistant in?
* enter_data: i’ve trained it in [portuguese](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
* how_to_get_started: from where i should start?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: n
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: rosa [open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: custom [ner](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: very much
    <B> Which (human) language is your assistant in?
* enter_data: [chinese](language) is the language of my bot
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, english
* how_to_get_started: how do i get started with rasa myself?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't think so
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to get started with rasa [core](product)?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: how is [entity recognition](nlu_part) managed in rasa?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* pipeline_recommendation: how to build a pipeline for the bot
    <B> Which (human) language is your assistant in?
* enter_data: it’s trained only in [portuguese](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + tool recommendation
* how_to_get_started: i want to make intelligence chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i guess it means - no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to get started with rasa [core](product)?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: how is [intent classification](nlu_part) managed in rasa?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* nlu_generation_tool_recommendation: how to work with [nlu](product) data
    <B> We recommend using Rasa X to easily create and classify NLU data from within a UI.
## not new to rasa + nlu + entity + no recommendation
* how_to_get_started: where to start the development of rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no i dont want to accept :p lol
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: tell me how to get started with [core](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: an explanation of how [intent classification](nlu_part) work would help
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
* deny: no don"t want this!
    <B> Got it. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline spacy
* how_to_get_started: lets start with the basics
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: absolutely not
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: rosa [open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: how do you integrate [duckling](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: alright, cool
    <B> What kind of entities do you want to extract?
* enter_data: get [dates](entity) from messages
    <B> SpaCy has excellent pre-trained named-entity recognizers for this -    check out this [interactive demo](https://demos.explosion.ai/displacy-ent/). I would suggest using the [SpacyEntityExtractor](https://rasa.com/docs/rasa/nlu/components/#spacyentityextractor) component in your pipeline. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline duckling
* how_to_get_started: i need help with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't wanna tell the name of my company
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: [open source rasa](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: [recognition](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: okay cool
    <B> What kind of entities do you want to extract?
* enter_data: [numbers](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline CRFEntityExtractor
* how_to_get_started: i wanted to build a bot my product customer support
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't want to give it to you
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: what's rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: what are [intents](nlu_part) ?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: i guess so
    <B> What kind of entities do you want to extract?
* enter_data: [how long](entity)
    <B> For custom entities like that, I would suggest using the [CRFEntityExtractor](https://rasa.com/docs/rasa/nlu/components/#crfentityextractor) component. Read more about how to annotate entities in your data [here](https://rasa.com/docs/nlu/dataformat/). 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + duckling info
* how_to_get_started: what is components
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no i dont want to accept :p lol
    - slot{"onboarding": true}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: rosa [open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: how is [entity recognition](nlu_part) managed in rasa?
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## skip to info on nlu entities
* nlu_info: i checked the documentation on [entity recognition](nlu_part) but i still don’t understand it
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
## skip to info on nlu intents
* nlu_info: what is [duckling](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
## skip to info on nlu intents
* nlu_info: is [duckling](nlu_part) part of rasa?
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## switch immediately to luis
* switch: [luis.ai](current_api)
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
## switch immediately to dialogflow
* switch: im migrating from [dialogflow](current_api)
    <B> We have a guide for migrating from DialogFlow [here](https://www.rasa.com/docs/rasa/migrate-from/google-dialogflow-to-rasa/). 
    <B> Is there anything else I can help you with?
## faq
* faq: What languages does rasa know?
    - respond_faq 
    - action_set_faq_slot

## source code
* source_code: demo bot source code
    <B> Like Rasa Open Source, my code is freely available on GitHub! You can find it here. 
    <B> Is there anything else I can help you with?
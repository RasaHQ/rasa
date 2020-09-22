## prompt for getting started
<U> what is up?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## prompt for getting started + confirm
<U> hola
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> amazing!
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## new to rasa at start, built bot before
<U> i am [new](user_type) but so how can i start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yesh
    - slot{"onboarding": true}
    <B> Are you migrating from another tool?
## new to rasa at start
<U> i am [new](user_type) but so how can i start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> sorry not right now
    - slot{"onboarding": true}
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what do you mean
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + built a bot before
<U> yeah please help me out
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> oh, ok
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> yes baby
    <B> Are you migrating from another tool?
<U> no. u r idiot
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> why is that necessary
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> hi, can you help in understanding nlu
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> [new](user_type)
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> alright
    <B> Are you migrating from another tool?
<U> i don not like this
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> could you tell me more
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + migrating from dialogflow
<U> can you tell me how to build a bot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i want that
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> ok great
    <B> Are you migrating from another tool?
<U> migration from [luis](current_api)
    <B> We have a guide for migrating from DialogFlow [here](https://www.rasa.com/docs/rasa/migrate-from/google-dialogflow-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from luis
<U> how to get started with rasa ?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes great
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> that is cool
    <B> Are you migrating from another tool?
<U> i currently use [dialog flow](current_api)
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from something else
<U> how to use
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> how do i sstart
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> done
    <B> Are you migrating from another tool?
<U> what is a synonym called?
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
##  migrating from dialogflow
<U> information about rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep thats cool
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i have used it in the past
    <B> Are you migrating from another tool?
<U> confirm
    <B> Which tool are you migrating from?
<U> [dialogflow](current_api) and implementation from scratch
    <B> We have a guide for migrating from DialogFlow [here](https://www.rasa.com/docs/rasa/migrate-from/google-dialogflow-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from luis
<U> information about rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yeah, why not
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> ok great
    <B> Are you migrating from another tool?
<U> top
    <B> Which tool are you migrating from?
<U> how can i migrate from [dialogflow](current_api)?
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from something else
<U> can i just test features without having to deal with your predefined conversation
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ya go for it
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i accept.
    <B> Are you migrating from another tool?
<U> yeah do that
    <B> Which tool are you migrating from?
<U> from which tools can i migrate to rasa?
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## just switch
<U> yes i want to switch from [luis](current_api) to rasa
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## just switch
<U> migration please
    <B> Which tool are you migrating from?
<U> how can i migrate from [dialogflow](current_api)?
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## just switch
<U> can i migrate to rasa from another tool?
    <B> Which tool are you migrating from?
<U> switching
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## new to rasa/bots, explain stack
<U> what is components
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep, will do thank you
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nö
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [nlu](product) and [core](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core
<U> what are your features ?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> it is ok
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i dont want to accept :p lol
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to train [rasa open source](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> i don't want to
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core
<U> i want to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes accept please
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nopes
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> more info on components of rasa pls
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, explain x, then nlu+core
<U> parts of rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ys
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i don't want to
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> help me build a bot
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, explain x
<U> how to build assistant with rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ya go for it
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i'm afraid not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
<U> nop
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu
<U> tell me how to start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i m [new](user_type)
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> no sir
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what about [nlu](product)?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
<U> i don't wanna tell the name of my company
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu then core
<U> what is a component in rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> thats good
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no i dont want to
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to train [rasa open source](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
<U> ok fine
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa/bots, explain all
<U> can i build a faq robot with rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nah not for me
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how do i get started with rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa/bots, explain all
<U> create chatbot steps
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nein
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how do i train rasa [core](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, explain all
<U> how do i download rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> absolutely
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i'm not sure
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> please elaborate
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain all
<U> can you help me build my bot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes, give me information, please
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no i haven't decided yet if i want to sign up
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what's rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, don't explain
<U> how to build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i m [new](user_type)
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> na
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> no i dont want to accept :p lol
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + not interested in products
<U> i wanna build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nop
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> no go
    <B> Got it. 
## not new to rasa + core
<U> i want to make intelligence chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nah, first time
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i wanna try rasa [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
## not new to rasa + rasa
<U> can give tell me about components of rosa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> absolutely not
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to build assistant with rasa?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + rasa
<U> how to download?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no sir
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i would like to know more about rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + rasa
<U> tell me about the components
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i don't wanna tell the name of my company
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> can you help me build my bot?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + rasa
<U> can i just test features without having to deal with your predefined conversation
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no. u r idiot
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> tell me how to start
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + [rasa](product)
<U> how to start rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nah, first time
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> can you shw me some information about intallation?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## skip to info on rasa x
<U> how to use
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
## skip to info on rasa
<U> how to setup rasa
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## skip to info on rasa nlu
<U> just [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> i still don’t get how [entity recognition](nlu_part) works
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
## not new to rasa + x
<U> i wanted to build a bot my product customer support
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no i dont want to accept :p lol
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what is [x](product) ?
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API.
    <B> Would you also like me to explain NLU and Core?
## new to rasa + x
<U> i want to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes i accept
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yesss
    <B> Are you migrating from another tool?
<U> not right now
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API.
    <B> Would you also like me to explain NLU and Core?
<U> agreed
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + rasa oss
<U> where do i find instructions
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i have used it in the past
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> ok good
    <B> Are you migrating from another tool?
<U> i decline
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to get started with rassa
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa + rasa oss
<U> more information on rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> okay..
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> sounds good!
    <B> Are you migrating from another tool?
<U> stop it, i do not care!!!
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> the components of rasa?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## skip to info on core
<U> what's rasa [core](product)?
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + nothing special
<U> sure, give me the basics
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i dont want to accept :p lol
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i need more info for rasa [core](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> i dont want to tell
    <B> Rasa Masterclass episodes 2-4 focus on NLU. Check out episode 2 [here](https://www.youtube.com/watch?v=k5UeywXA28k). 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + unknown topic
<U> how cna i get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> still dont want to tell
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> tell me about rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> how enttity extrcation works
    <B> Hmm, seems like I haven't learned about that part of NLU yet! Try using the search bar above for now. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + no recommendation
<U> how easy is it to use rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no i don't accept
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what is rasa [nlu](product)?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> custom [ner](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> n
    <B> Got it. 
    <B> Is there anything else I can help you with?
## pipeline
<U> nlu pipeline
    <B> Which (human) language is your assistant in?
<U> [chinese](language) is the only language but i want to add more
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?
## pipeline
<U> what is the right pipeline to choose?
    <B> Which (human) language is your assistant in?
<U> it speaks [french](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## pipeline
<U> what pipeline is better for what i want?
    <B> Which (human) language is your assistant in?
<U> it speaks [italian](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## pipeline
<U> which pipeline is better?
    <B> Which (human) language is your assistant in?
<U> the language is [french](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> more information on components in rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> absolutely not
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> about [both](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> i checked the documentation on [intent classification](nlu_part) but i still don’t understand it
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yres
    <B> Which (human) language is your assistant in?
<U> so far it only speaks [dutch](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> whar are the components of rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i'm afraid not
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how do i learn rasa [core](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intent classification](nlu_part) - what is that?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> considering
    <B> Which (human) language is your assistant in?
<U> all the training data was in [german](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> how to build assistant with rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nö
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i wanna try rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> how does [intent classification](nlu_part) work?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> absolutely
    <B> Which (human) language is your assistant in?
<U> [german](language) is the language of my bot
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, english
<U> how to learn rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no and no again
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yes i wanna know more about rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [entity recognition](nlu_part) - what is that?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> should i better start with the tensorflow pipeline or spacy?
    <B> Which (human) language is your assistant in?
<U> it’s available in [portuguese](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + tool recommendation
<U> do you know how to set up a chatbot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no thank s
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> can you help me with the rasa [core](product) ?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> is [duckling](nlu_part) part of rasa?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> i have to less [nlu](product) data
    <B> We recommend using Rasa X to easily create and classify NLU data from within a UI.
## not new to rasa + nlu + entity + no recommendation
<U> where to start?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no don"t want this!
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to get started with [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> sorry its [ner](nlu_part)
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
<U> no i dont want to accept :p lol
    <B> Got it. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline spacy
<U> how can i get stared
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no go
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to learn about rasa [core](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intent](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> ok good
    <B> What kind of entities do you want to extract?
<U> amounts of [money](entity)
    <B> SpaCy has excellent pre-trained named-entity recognizers for this -    check out this [interactive demo](https://demos.explosion.ai/displacy-ent/). I would suggest using the [SpacyEntityExtractor](https://rasa.com/docs/rasa/nlu/components/#spacyentityextractor) component in your pipeline. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline duckling
<U> how do i download rasa ?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no i haven't decided yet if i want to sign up
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> get started with [rasa open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> how do you integrate [duckling](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> amayzing
    <B> What kind of entities do you want to extract?
<U> i am interested in [ordinals](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline CRFEntityExtractor
<U> find out how to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> none of them
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> just rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> how is [intent classification](nlu_part) managed in rasa?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> absolutely
    <B> What kind of entities do you want to extract?
<U> how much [money](entity)
    <B> For custom entities like that, I would suggest using the [CRFEntityExtractor](https://rasa.com/docs/rasa/nlu/components/#crfentityextractor) component. Read more about how to annotate entities in your data [here](https://rasa.com/docs/nlu/dataformat/). 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + duckling info
<U> how do i get startd?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i don't agree
    - slot{"onboarding": true}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i wanna try rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intents](nlu_part)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## skip to info on nlu entities
<U> [intent recognition](nlu_part)
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
## skip to info on nlu intents
<U> can you explain me what [intents](nlu_part) are ?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
## skip to info on nlu intents
<U> [intent](nlu_part) please
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## switch immediately to luis
<U> can i migrate my [luis](current_api) bot to raza
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
## switch immediately to dialogflow
<U> do you have any tutorials how to migrate from [dialogflow](current_api)?
    <B> We have a guide for migrating from DialogFlow [here](https://www.rasa.com/docs/rasa/migrate-from/google-dialogflow-to-rasa/). 
    <B> Is there anything else I can help you with?
## faq
* faq
    - respond_faq 
    - action_set_faq_slot

## source code
<U> what is your source code
    <B> Like Rasa Open Source, my code is freely available on GitHub! You can find it here. 
    <B> Is there anything else I can help you with?
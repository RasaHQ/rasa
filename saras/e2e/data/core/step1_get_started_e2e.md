## prompt for getting started
<U> hi sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
## prompt for getting started + confirm
<U> hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> yes
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
## new to rasa at start, built bot before
<U> I am [new](user_type) to Rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> sure
    - slot{"onboarding": true}
    <B> Are you migrating from another tool?
## new to rasa at start
<U> I am [new](user_type)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": true}
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want more information
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + built a bot before
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yeah
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> yeah
    <B> Are you migrating from another tool?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> please explain
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> how to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> I'm [new](user_type) in Rasa, help me!
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> yes
    <B> Are you migrating from another tool?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> explain it to me
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + migrating from dialogflow
<U> how to get strated
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yeah
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> yep
    <B> Are you migrating from another tool?
<U> I use [dialogflow](current_api)
    <B> We have a guide for migrating from DialogFlow [here](https://www.rasa.com/docs/rasa/migrate-from/google-dialogflow-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from luis
<U> how do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> yeah
    <B> Are you migrating from another tool?
<U> I want to switch from [LUIS](current_api) to Rasa
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from something else
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> How can I get started with rasa?
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yep
    <B> Are you migrating from another tool?
<U> tensorflow
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
##  migrating from dialogflow
<U> how do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ya
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yes
    <B> Are you migrating from another tool?
<U> yepp
    <B> Which tool are you migrating from?
<U> migration from [dialogflow](current_api)
    <B> We have a guide for migrating from DialogFlow [here](https://www.rasa.com/docs/rasa/migrate-from/google-dialogflow-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from luis
<U> how to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yes
    <B> Are you migrating from another tool?
<U> yep
    <B> Which tool are you migrating from?
<U> im moving [luis](current_api)
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
## new to rasa + not new to chatbots + migrating from something else
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yeah
    <B> Are you migrating from another tool?
<U> Yep
    <B> Which tool are you migrating from?
<U> switch to rasa
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## just switch
<U> I want to switch from [dialogflow](current_api) to rasa
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## just switch
<U> switch to rasa
    <B> Which tool are you migrating from?
<U> I want to switch from [dialogflow](current_api) to rasa
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## just switch
<U> switch
    <B> Which tool are you migrating from?
<U> migrate to rasa
    <B> Sorry, but we don't have a migration guide for that tool yet. You can still follow the [tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/) to build a Rasa assistant based on your existing one, though! 
    <B> Is there anything else I can help you with?
## new to rasa/bots, explain stack
<U> how do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [both](product) please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core
<U> How can I get started with Rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> About [Core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> no
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ye
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to use rasa to build my chatbot
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, explain x, then nlu+core
<U> start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to know about the parts of rasa
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, explain x
<U> how do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [x](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
<U> no
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu
<U> let's start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> I am [new](user_type) to Rasa
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Rasa [NLU](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
<U> no thanks
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu then core
<U> how to get strated
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [nlu](product) part
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
<U> yeah
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa/bots, explain all
<U> How can I try out rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [Rasa open source](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa/bots, explain all
<U> How do I start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nah
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [rasa open source](product) to begin with
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, explain all
<U> How to get start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> please explain
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain all
<U> get started pls
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> tell me about [rasa open source](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa/bots, don't explain
<U> how to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> I'm [new](user_type)
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> not really
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + not interested in products
<U> Please tell me how I can start?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> no
    <B> Got it. 
## not new to rasa + core
<U> How get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Rasa [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
## not new to rasa + rasa
<U> How to get start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nah
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> help with rasa
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + rasa
<U> How do I start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [Rasa Open Source](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + rasa
<U> How do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> help with rasa
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + rasa
<U> How to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nah
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to start with rasa
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## not new to rasa + [rasa](product)
<U> What are the components of rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to use rasa to build my chatbot
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## skip to info on rasa x
<U> how to build bot with rasa x
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
## skip to info on rasa
<U> how to get started with rasa
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## skip to info on rasa nlu
<U> tell me more about [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
## not new to rasa + x
<U> how to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Rasa [x](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API.
    <B> Would you also like me to explain NLU and Core?
## new to rasa + x
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yeah
    <B> Are you migrating from another tool?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [x](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API.
    <B> Would you also like me to explain NLU and Core?
<U> yes please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + rasa oss
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yeah
    <B> Are you migrating from another tool?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> How do i get started with rasa?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## new to rasa + rasa oss
<U> I want to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yeah
    <B> Are you migrating from another tool?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to use rasa to build my chatbot
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
## skip to info on core
<U> i would like more explanation on RASA [Core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + nothing special
<U> How to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Can you tell me more about [NLU](product)?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> nope
    <B> Rasa Masterclass episodes 2-4 focus on NLU. Check out episode 2 [here](https://www.youtube.com/watch?v=k5UeywXA28k). 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + unknown topic
<U> How to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Rasa [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> what does nlu server do?
    <B> Hmm, seems like I haven't learned about that part of NLU yet! Try using the search bar above for now. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + no recommendation
<U> How to start using rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> deny
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [nlu](product) part
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intent classificaton](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> no
    <B> Got it. 
    <B> Is there anything else I can help you with?
## pipeline
<U> I don’t know which pipeline to use
    <B> Which (human) language is your assistant in?
<U> I wrote it in [english](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?
## pipeline
<U> pipeline recommendation
    <B> Which (human) language is your assistant in?
<U> I wrote it in [spanish](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## pipeline
<U> I don’t know which pipeline to use
    <B> Which (human) language is your assistant in?
<U> I wrote it in [spanish](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## pipeline
<U> what pipeline should i use?
    <B> Which (human) language is your assistant in?
<U> [Lithuanian](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> What are the components of rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yes please
    <B> Which (human) language is your assistant in?
<U> I wrote it in [spanish](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> How to get start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nah
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> can you explain to me how [intent classification](nlu_part) works?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yes please
    <B> Which (human) language is your assistant in?
<U> [dutch](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> How get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> I want to know more about [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> I don’t understand [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yeah
    <B> Which (human) language is your assistant in?
<U> I wrote it in [spanish](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, english
<U> Get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> NO
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Only [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intent classificaton](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> give me a recommendation
    <B> Which (human) language is your assistant in?
<U> I wrote it in [english](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + tool recommendation
<U> Help me get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> No.
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> More about [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intent classificaton](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> are there simpler ways to create [nlu](product) data
    <B> We recommend using Rasa X to easily create and classify NLU data from within a UI.
## not new to rasa + nlu + entity + no recommendation
<U> How do I get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> an explanation of how [entity recognition](nlu_part) work would help
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
<U> nah
    <B> Got it. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline spacy
<U> How do I get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> Tell me about [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> tell me about [entity recognition](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yes
    <B> What kind of entities do you want to extract?
<U> i want to extract [names](entity)
    <B> SpaCy has excellent pre-trained named-entity recognizers for this -    check out this [interactive demo](https://demos.explosion.ai/displacy-ent/). I would suggest using the [SpacyEntityExtractor](https://rasa.com/docs/rasa/nlu/components/#spacyentityextractor) component in your pipeline. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline duckling
<U> How do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [entity recognition](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yep
    <B> What kind of entities do you want to extract?
<U> [amounts](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline CRFEntityExtractor
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> No
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> More about [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> tell me about [entity recognition](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yes please
    <B> What kind of entities do you want to extract?
<U> [custom entity](entity)
    <B> For custom entities like that, I would suggest using the [CRFEntityExtractor](https://rasa.com/docs/rasa/nlu/components/#crfentityextractor) component. Read more about how to annotate entities in your data [here](https://rasa.com/docs/nlu/dataformat/). 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + duckling info
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": true}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> How do I get started with [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> I want to know if rasa works with [duckling](nlu_part)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## skip to info on nlu entities
<U> [entity recognition](nlu_part)
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
## skip to info on nlu intents
<U> I don’t understand how you handle [intent classification](nlu_part) at Rasa
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
## skip to info on nlu intents
<U> [duckling](nlu_part)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## switch immediately to luis
<U> How to migrate from [Luis](current_api)?
    <B> We have a guide for migrating from LUIS [here](https://www.rasa.com/docs/rasa/migrate-from/microsoft-luis-to-rasa/). 
    <B> Is there anything else I can help you with?
## switch immediately to dialogflow
<U> I use [DialogFlow](current_api)
    <B> We have a guide for migrating from DialogFlow [here](https://www.rasa.com/docs/rasa/migrate-from/google-dialogflow-to-rasa/). 
    <B> Is there anything else I can help you with?
## faq
* faq
    - respond_faq 
    - action_set_faq_slot

## source code
<U> can i see your code
    <B> Like Rasa Open Source, my code is freely available on GitHub! You can find it here. 
    <B> Is there anything else I can help you with?
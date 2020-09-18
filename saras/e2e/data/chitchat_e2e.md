## chitchat
<U> how are you
    <B> I'm great! Thanks for asking.
## deny chitchat
<U> hate
    <B> That's not very nice
<U> jk
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
## more chitchat
<U> hii sara
    <B> Hi!
<U> how are you?
    <B> I'm great! Thanks for asking.
<U> how old are you?
    <B> Old enough to be a bot!
## greet chitchat
<U> hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how are you
    <B> I'm great! Thanks for asking.
## just newsletter + confirm
<U> hi
    <B> Hey!
<U> What is your name 
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot
<U> subscribing
    <B> Can do.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
<U> positive
    <B> Awesome! Is there anything else I can help you with?
## just newsletter, continue, + confirm
<U> hi
    <B> Hi!
<U> newsletter
    <B> Can do.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> how do you do
    <B> I'm great! Thanks for asking. Do you still want to subscribe to the newsletter?
<U> yes
    <B> Cool, let's do it! 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> awesone
    <B> Okay. Is there anything else I can help you with?
## just newsletter, don't continue, + confirm
<U> hi
    <B> Hey there!
<U> subscribe to newsletter
    <B> Can do.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> how are you
    <B> I'm great! Thanks for asking. Do you still want to subscribe to the newsletter?
<U> no
    <B> Got it. How is this conversation going?
    - form{"name": null}
<U> great
    <B> Okay. Is there anything else I can help you with?
## just newsletter (with email already) + confirm
<U> hi
    <B> Hi!
<U> who are you
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot
<U> newsletter
    <B> Can do.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> awsone
    <B> Okay. Is there anything else I can help you with?
## just newsletter (with email already)
<U> hello
    <B> Hey!
<U> what do you have to show me?
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
<U> send me an e-mail
    <B> Can do.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just newsletter
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> show me menu
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
<U> newsletter
    <B> Can do.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just newsletter, continue
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> subscribing
    <B> You got it.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> help me
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. Do you still want to subscribe to the newsletter?
<U> on
    <B> That's great!
    - subscribe_newsletter_form
    - form{"name": null} 
    <B> How is this conversation going?
## just newsletter, don't continue
<U> hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i would like to join the newsletter
    <B> Can do.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> how are you
    <B> I'm great! Thanks for asking. Do you still want to subscribe to the newsletter?
<U> No
    <B> Got it.
    - form{"name": null}
    <B> How is this conversation going?
## just sales
<U> hey there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are you a real person
    <B> I am a bot
<U> I want to get in touch with your sales guys
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
## just sales, continue
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> booking sales call
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
<U> Whats your name?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot. Do you still want to talk to sales?
<U> yep
    - sales_form
    - form{"name": null}
## just sales, don't continue
<U> Hi Sara
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can i talk to a sales representative
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
<U> How can you help me?
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. Do you still want to talk to sales?
<U> nope
    <B> Got it.
    - form{"name": null}
## new to rasa + not new to chatbots + not migrating
<U> good morning
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> whats your name
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot
<U> How can I get started with [rasa](product)?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
    - slot{"onboarding": true}
<U> yeah
    <B> And have you built a contextual assistant or a bot before?
<U> yep
    <B> Are you migrating from another tool?
<U> deny
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> yes go ahead
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> Hi, bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to know how to get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> do yo do something  
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. Are you new to Rasa?
<U> yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yep
    <B> Are you migrating from another tool?
<U> no I'm not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> [both](product) please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> hey bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I need to know if I can use [rasa](product) to build an application?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes 
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> how are you
    <B> I'm great! Thanks for asking. And have you built a contextual assistant or a bot before?
<U> yes
    <B> Are you migrating from another tool?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> yes go for it
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> whats up
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How can I get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes 
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> yes
    <B> Are you migrating from another tool?
<U> whats the date today?
    <B> I can't help you with that, I'm sorry. Are you migrating from another tool?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> yes please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> hi friend
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes 
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yes
    <B> Are you migrating from another tool?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> who are you
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot Would you like to know more about Rasa Open Source or Rasa X?
<U> yup
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> hi friend
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes 
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yes
    <B> Are you migrating from another tool?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> yup
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.
<U> what can u do
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain NLU and try it out
<U> hi there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what is [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> Hi how are you
    <B> I'm great! Thanks for asking. And have you built a contextual assistant or a bot before?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> what is [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Would you also like me to explain Core?
<U> yes
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain NLU and try it out
<U> yo
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
## new to rasa/bots, explain NLU and try it out
<U> hey there boy
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to know how to get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> what are you doing
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. Would you like to know more about Rasa Open Source or Rasa X?
<U> yes
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain stack and try it out
<U> howdy
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How can I get started with [rasa](product)?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes i am
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> No, not really.
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> yeah
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) You can click here if you want to go to the next step to try out Rasa Open Source.
<U> who created you?
    <B> Engineers at Rasa. You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core and try out stack
<U> Hi there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ye
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> Yes, please tell me about Rasa [Core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Would you also like me to explain NLU?
<U> what do you usually do every night
    <B> I can't help you with that, I'm sorry. Would you also like me to explain NLU?
<U> no
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core and try out stack
<U> Whats up
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Help me get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i am!
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> I want know about Rasa [Core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Would you also like me to explain NLU?
<U> no
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> I am hungry, find me some place to go
    <B> I can't help you with that, I'm sorry. You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core, then nlu and try out stack
<U> hey bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> Oh yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> definitely not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> I want to know more about [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Would you also like me to explain NLU?
<U> Do you know my name?
    <B> I can't help you with that, I'm sorry. Would you also like me to explain NLU?
<U> yup
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core, then nlu and try out stack
<U> greetings
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> What are the components of [rasa](product)?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> exactly
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> [core](product) please
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Would you also like me to explain NLU?
<U> yep
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). You can click here if you want to go to the next step to try out Rasa Open Source.
<U> whats the weather today
    <B> I can't help you with that, I'm sorry. You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu and try out stack
<U> heyo
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ya
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> never
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> how about [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Would you also like me to explain Core?
<U> tell me who I am?
    <B> I can't help you with that, I'm sorry. Would you also like me to explain Core?
<U> no
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu and try out stack
<U> hi pal!
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you help me build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> Yep!
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> [nlu](product) please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Would you also like me to explain Core?
<U> no
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> who are you
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, don't explain and try out stack
<U> Heya
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you help me to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> what's your name
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot Would you like to know more about Rasa Open Source or Rasa X?
<U> nope
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, don't explain and try out stack
<U> hello sara
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> help with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no never
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> nope
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> how are you doing
    <B> I'm great! Thanks for asking. You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + not interested in products
<U> Hi, bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> who built you?
    <B> Engineers at Rasa.
<U> what are the components of [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> neither
    <B> Got it.
## not new to rasa + not interested in products
<U> hello friend
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to learn more about [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> what is sara?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> none
    <B> Got it.
## not new to rasa + not interested in products
<U> heya
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> what is the weather today? 
    <B> I can't help you with that, I'm sorry. Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> nah I'm good
    <B> Got it.
## not new to rasa + nlu + nothing special
<U> hey there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How can I get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> [NLU](product) please
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> not realy
    <B> Rasa Masterclass episodes 2-4 focus on NLU. Check out episode 2 [here](https://www.youtube.com/watch?v=k5UeywXA28k). Is there anything else I can help you with?
## not new to rasa + rasa x + nothing special
<U> hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How can I get started with [rasa](product)?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> not
    - slot{"onboarding": true}
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> how to get started with rasa [x](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. Would you also like me to explain NLU and Core?
<U> i need help
    <B> Would you also like me to explain NLU and Core?
<U> please explain
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + rasa x + nothing special
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How can I get started with [rasa](product)?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no I'm not
    - slot{"onboarding": false}
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> how can i get started with rasa [x](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. Would you also like me to explain NLU and Core?
<U> no
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + rasa x + nothing special
<U> heya
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> how old are you
    <B> Old enough to be a bot! Are you new to Rasa?
<U> yeah
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> how to get started with rasa [x](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. Would you also like me to explain NLU and Core?
<U> no thank you
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> what is your name
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + nlu + unknown topic
<U> hello bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How can I develop a bot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how do i get started with nlu
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> what is an entity
    <B> Hmm, seems like I haven't learned about that part of NLU yet! Try using the search bar above for now. Is there anything else I can help you with?
## not new to rasa + nlu + intent + no recommendation
<U> hello bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you help me build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> not really
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> rasa [nlu](product) please
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> I don't understand [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> no thanks
    <B> Got it. Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> hey there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> help me to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> rasa [nlu](product) please
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> I don't understand how you handle [intent classification](nlu_part) at Rasa
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> give me a recommendation
    <B> Which (human) language is your assistant in?
<U> [English](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> hey bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How can I get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i'm interested in [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> Can you explain [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> please recommend
    <B> Which (human) language is your assistant in?
<U> [Russian](language) language
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> hello bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how can i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how do I start with [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intent classification](nlu_part) please
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yes please
    <B> Which (human) language is your assistant in?
<U> my bot is in [dutch](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline duckling
<U> hey there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> I'm interested in rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> how does the [entity extraction](nlu_part) work?
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
<U> how old are you? 
    <B> Old enough to be a bot! Would you like me to try to recommend a pipeline?
<U> yep
    <B> What kind of entities do you want to extract?
<U> [date ranges](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline duckling
<U> heya
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i get get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> not
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> I'm interested in [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> how does the [entity extraction](nlu_part) work?
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
<U> please
    <B> What kind of entities do you want to extract?
<U> who built you?
    <B> Engineers at Rasa. What kind of entities do you want to extract? 
<U> [date ranges](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component Is there anything else I can help you with?
## how to get started without privacy policy
<U> how do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> what's the date today? 
    <B> I can't help you with that, I'm sorry. Are you new to Rasa?
<U> yeah
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> yes
    <B> Are you migrating from another tool?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> [both](product) please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.
## chitchat interrupting step 2 flow
<U> get_started_step2
    <B> I can show you how to start building your first AI assistant with Rasa. Are you ready?
<U> could you tell me the time
    <B> I can't help you with that, I'm sorry. Do you want to go ahead with the steps to create a contextual assistant?
<U> yes please
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training.
## chitchat interrupting and stopping step 2 flow
<U> get_started_step2
    <B> I can show you how to start building your first AI assistant with Rasa. Are you ready?
<U> what's up? 
    <B> I'm great! Thanks for asking. Do you want to go ahead with the steps to create a contextual assistant?
<U> nope
    <B> Got it. Is there anything else I can help you with?
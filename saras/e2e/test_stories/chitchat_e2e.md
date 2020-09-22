## chitchat
<U> whatsapp
    <B> I'm great! Thanks for asking.
## deny chitchat
<U> you know nothing.
    <B> That's not very nice
<U> i don't want to
    <B> For now that's all I can do, so feel free to try it out. If you have any suggestions for what I should learn, let me know!
## more chitchat
<U> salut
    <B> Hey there, my name is Sara.
<U> how's it going?
    <B> I'm great! Thanks for asking.
<U> can you tell me your age?
    <B> Old enough to be a bot!
## greet chitchat
<U> hello it is me again
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> do you feel good?
    <B> I'm great! Thanks for asking.
## just newsletter + confirm
<U> hhola
    <B> Hey there, my name is Sara.
<U> ok who are you ?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> subscription newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
<U> absolutely
    <B> Awesome! 
    <B> Is there anything else I can help you with?
## just newsletter, continue, + confirm
<U> ayyyy whaddup
    <B> Hey there, my name is Sara.
<U> now i want to signup for the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> how was your day?
    <B> I'm great! Thanks for asking. 
    <B> Do you still want to subscribe to the newsletter?
<U> yeeees
    <B> Cool, let's do it! 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> sure thing
    <B> Okay. 
    <B> Is there anything else I can help you with?
## just newsletter, don't continue, + confirm
<U> guten morgen
    <B> Hey there, my name is Sara.
<U> sign me up for that newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> nah, i'm good - how are you doing?
    <B> I'm great! Thanks for asking. 
    <B> Do you still want to subscribe to the newsletter?
<U> i dont want to accept :p lol
    <B> Got it. 
    <B> How is this conversation going?
    - form{"name": null}
<U> yes it was okay
    <B> Okay.
    <B> Is there anything else I can help you with?
## just newsletter (with email already) + confirm
<U> wasssup!
    <B> Hey there, my name is Sara.
<U> who am i talking to
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> please sign me up for the newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> fine
    <B> Okay. 
    <B> Is there anything else I can help you with?
## just newsletter (with email already)
<U> hai
    <B> Hey there, my name is Sara.
<U> how do i identify myself?
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
<U> i would like to subscribe
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just newsletter
<U> halo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what else can i do?
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
<U> newsletter please
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just newsletter, continue
<U> hello sweatheart
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you subscribe me to the newsletter
    <B> You got it.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> options
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. 
    <B> Do you still want to subscribe to the newsletter?
<U> i accept
    <B> That's great!
    - subscribe_newsletter_form
    - form{"name": null} 
    <B> How is this conversation going?
## just newsletter, don't continue
<U> hi folks
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> add me as your subscriber
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
<U> how is your day going
    <B> I'm great! Thanks for asking. 
    <B> Do you still want to subscribe to the newsletter?
<U> nopes
    <B> Got it.
    - form{"name": null}
    <B> How is this conversation going?
## just sales
<U> good evening
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> are u a real person?
    <B> I am a bot
<U> i wanna talk to your sales people.
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null}
## just sales, continue
<U> i said, helllllloooooo!!!!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how can i contact the team ?
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
<U> who am i speaking with
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot. 
    <B> Do you still want to talk to sales?
<U> cool story bro
    - sales_form
    - form{"name": null}
## just sales, don't continue
<U> wasssup!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hm ok then i want to talk to the sales dude
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
<U> how you help me
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
    <B> Do you still want to talk to sales?
<U> i dont want to
    <B> Got it.
    - form{"name": null}
## new to rasa + not new to chatbots + not migrating
<U> hieeeeeeeeeeeee
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hey what is your name
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
<U> how can i learn rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
    - slot{"onboarding": true}
<U> y
    <B> And have you built a contextual assistant or a bot before?
<U> i changed my mind. i want to accept it
    <B> Are you migrating from another tool?
<U> i don't want to give it to you
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> its okay
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API.
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> hi mrs rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> tell me more about get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> what can i ask you?
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call.
    <B> Are you new to Rasa?
<U> i will!
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> a little
    <B> Are you migrating from another tool?
<U> no way
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [open source rasa](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> hai
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> tell me how to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ofcoure i do
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> how's life treating you friend?
    <B> I'm great! Thanks for asking. 
    <B> And have you built a contextual assistant or a bot before?
<U> oh super
    <B> Are you migrating from another tool?
<U> no ma'am
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> lets do this
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> halo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what is a component in rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> it is ok
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> okay cool
    <B> Are you migrating from another tool?
<U> hey, i contacted you a couple of days ago but didn't get any response, any news?
    <B> I can't help you with that, I'm sorry. 
    <B> Are you migrating from another tool?
<U> nah not for me
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yes, i accept
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> hhola
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how can i start with rasa on a legacy windows without python?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yeeeeezzzzz
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> absolutely
    <B> Are you migrating from another tool?
<U> noooooooooooooooooooooooooooooooooooooooo
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> who ar eyou
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yep that's nice
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + not new to chatbots + not migrating
<U> hello sweet boy
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to create a basic chat bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> accept
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> how nice!
    <B> Are you migrating from another tool?
<U> no i won't
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> nice
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> what do you do
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain NLU and try it out
<U> hallo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to learn about rasa [core](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yeeees
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> how's life treating you friend?
    <B> I'm great! Thanks for asking. 
    <B> And have you built a contextual assistant or a bot before?
<U> i'm afraid not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how do i learn rasa [core](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
<U> yes pls
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain NLU and try it out
<U> salut
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> ow to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> jo
    <B> And have you built a contextual assistant or a bot before?
    - slot{"onboarding": true}
<U> no. u r idiot
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
## new to rasa/bots, explain NLU and try it out
<U> hi there it's me
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> lets do it
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> still dont want to tell
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> so what can you do
    <B> You can ask me about how to get started with Rasa, the difference between Rasa and Rasa X, subscribing to our newsletter or booking a sales call. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> kk
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain stack and try it out
<U> ayyyy whaddup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i wanna build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes that's what i want
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i dont want to
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yes pleae
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> who build yoi
    <B> Engineers at Rasa. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core and try out stack
<U> hi again
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i download rasa ?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> that's great
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i dont want to accept :p lol
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> 4 + 2 = ?
    <B> I can't help you with that, I'm sorry.
    <B> Would you also like me to explain NLU?
<U> absolutely not
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core and try out stack
<U> hola
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> tell me how i can get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> cool beans
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no. u r idiot
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to use [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> nevermind
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> what restaurant would you recommend for dinner?
    <B> I can't help you with that, I'm sorry. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core, then nlu and try out stack
<U> what up
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how can i build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ys
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nopes
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to know about [rasa open source](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> what exactly is my name?
    <B> I can't help you with that, I'm sorry.
    <B> Would you also like me to explain NLU?
<U> okay sure
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain core, then nlu and try out stack
<U> hi can you speak ?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> where do i find instructions
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> thats good
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> nehi
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to get started with rasa [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> alright
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> is the sun out where you are?
    <B> I can't help you with that, I'm sorry. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu and try out stack
<U> jojojo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do you build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ok great
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> stop it, i do not care!!!
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i would like to know more about rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
<U> do you know who am i?
    <B> I can't help you with that, I'm sorry.
    <B> Would you also like me to explain Core?
<U> absolutely not
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, explain nlu and try out stack
<U> good mourning
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you explain how can i make chatbot like you
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> si
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no, i hate it
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [rasa open source](product) first
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
<U> sorry not right now
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> what is rasa mascot
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, don't explain and try out stack
<U> konichiwa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> tell me more about get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i agree
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i dont want to tell
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what is your job?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> no, i hate it
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa/bots, don't explain and try out stack
<U> jojojo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep that's fine
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> no go
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i don not like this
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> whatchya upto ?
    <B> I'm great! Thanks for asking. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + not interested in products
<U> hi mrs rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> who was the person that built you?
    <B> Engineers at Rasa.
<U> how do i set up a chatbot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no way
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> never mind
    <B> Got it.
## not new to rasa + not interested in products
<U> halo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> where do i start?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> who am i speaking with
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
    <B> Are you new to Rasa?
<U> i don't want to say
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> one
    <B> Got it.
## not new to rasa + not interested in products
<U> hallo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> components in rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i'm not going to give it to you
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how's weather
    <B> I can't help you with that, I'm sorry.
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> no bots at all
    <B> Got it.
## not new to rasa + nlu + nothing special
<U> hey ther
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can i just test features without having to deal with your predefined conversation
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> thanks but no thanks
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> get started with [rasa open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> i'm not sure
    <B> Rasa Masterclass episodes 2-4 focus on NLU. Check out episode 2 [here](https://www.youtube.com/watch?v=k5UeywXA28k). 
    <B> Is there anything else I can help you with?
## not new to rasa + rasa x + nothing special
<U> halloooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how easy is it to use rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nopes
    - slot{"onboarding": true}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> rasa components
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
<U> what can we talk about?
    <B> Would you also like me to explain NLU and Core?
<U> why is that necessary
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + rasa x + nothing special
<U> hell9o
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how can i start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no i don't accept
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i wanna try rasa [nlu](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
<U> non
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## new to rasa + rasa x + nothing special
<U> hallo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to build chatbot using rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> how old?
    <B> Old enough to be a bot! 
    <B> Are you new to Rasa?
<U> ya i want
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> it sucks
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> the components of rasa
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
<U> i'm not going to give it to you
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
<U> what's your name ?
    <B> I'm Sara, the Rasa bot! At the same time I'm also the Rasa mascot.
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## not new to rasa + nlu + unknown topic
<U> hallo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what are the components?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no, i hate it
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> just rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> it would be helpful to learn more about [intent classification](nlu_part)
    <B> Hmm, seems like I haven't learned about that part of NLU yet! Try using the search bar above for now. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + no recommendation
<U> heeey
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what do i need for rasa implementation?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no i don't accept
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> about [both](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intents](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> no, thankyou
    <B> Got it. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> hi mrs rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what is a component in rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i do not need help installing
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to use [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> what is [duckling](nlu_part)?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> can you help me with the pipeline?
    <B> Which (human) language is your assistant in?
<U> it’s trained only in [english](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> hi mister
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> hi, can you help in understanding nlu
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nop
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what's rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> does rasa work with [duckling](nlu_part)?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> should i better start with the tensorflow pipeline or spacy?
    <B> Which (human) language is your assistant in?
<U> i wrote it in [italian](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> hellooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> need help in finding information about rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nevermind
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [both](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> i don’t understand how you handle [entity recognition](nlu_part) at rasa
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> i guess so
    <B> Which (human) language is your assistant in?
<U> it’s trained only in [italian](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline duckling
<U> hola
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> whar are the components of rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no i dont want
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what about [nlu](product)?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> how is [intent classification](nlu_part) managed in rasa?
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
<U> can you tell me what number represents your age?
    <B> Old enough to be a bot! 
    <B> Would you like me to try to recommend a pipeline?
<U> jezz
    <B> What kind of entities do you want to extract?
<U> [numbers](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + entity + pipeline duckling
<U> hey dude
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how this rasa works
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i'm not going to give it to you
    - slot{"onboarding": false}
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> rada [open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> hm [intents](nlu_part)?
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
<U> yes pleae
    <B> What kind of entities do you want to extract?
<U> who had the idea to generate you?
    <B> Engineers at Rasa. 
    <B> What kind of entities do you want to extract? 
<U> how much [money](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?
## how to get started without privacy policy
<U> more information on rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ok one then
    <B> I can't help you with that, I'm sorry. 
    <B> Are you new to Rasa?
<U> oh good !!
    - slot{"onboarding": true}
    <B> And have you built a contextual assistant or a bot before?
<U> i accept.
    <B> Are you migrating from another tool?
<U> absolutely not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to train [rasa open source](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
## chitchat interrupting step 2 flow
* get_started_step2
    <B> I can show you how to start building your first AI assistant with Rasa. Are you ready?
<U> what time do you have?
    <B> I can't help you with that, I'm sorry.
    <B> Do you want to go ahead with the steps to create a contextual assistant?
<U> yes i do
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## chitchat interrupting and stopping step 2 flow
* get_started_step2
    <B> I can show you how to start building your first AI assistant with Rasa. Are you ready?
<U> hiihihi
    <B> I'm great! Thanks for asking. 
    <B> Do you want to go ahead with the steps to create a contextual assistant?
<U> i do not need help installing
    <B> Got it. 
    <B> Is there anything else I can help you with?
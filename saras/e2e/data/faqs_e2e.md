## next step prompt
<U> next step
    - action_next_step
## faqs
* faq
    - respond_faq
## more faqs
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
* faq
    - respond_faq
## just newsletter
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> I want to sign up for newsletter
    <B> Can do.
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> Good!
    <B> Got it. Is there anything else I can help you with?
## just newsletter, continue + confirm
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> newsletter signup
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq
    - respond_faq
    <B> Do you still want to subscribe to the newsletter?
<U> yeah
    <B> perfect! 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> affirm
    <B> Got it. Is there anything else I can help you with?
## just newsletter, don't continue + confirm
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> signup to newsletter
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq
    - respond_faq
    <B> Do you still want to subscribe to the newsletter?
<U> not really
    <B> Got it. How is this conversation going?
<U> affirm
    <B> Got it. Is there anything else I can help you with?
## just newsletter (with email already) + confirm
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> can i sign up to newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> it's good!
    <B> Got it. Is there anything else I can help you with?
## just newsletter (with email already)
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> I would like to signup to newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just sales
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/). 
* faq
    - respond_faq
<U> talk to sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## just sales, continue
<U> hey bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to talk to sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
* faq
    - respond_faq
    <B> Do you still want to talk to sales?
<U> yes
    - sales_form
    - form{"name": null} 
    <B> How is this conversation going?
## just sales, don't continue
<U> hey there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> I want to talk to sales team
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
* faq
    - respond_faq 
    <B> Do you still want to talk to sales?
<U> nope
    - sales_form
    - form{"name": null}
## not new to rasa + not interested in products
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* faq
    - respond_faq 
    <B> Are you new to Rasa?
<U> nope I'm not
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> neither
    <B> Got it.

## not new to rasa + not interested in products
<U> greet
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
* faq
    - respond_faq 
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> I'm not interested
    <B> Got it. 
## not new to rasa + nlu + nothing special
<U> heya
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you tell me how to create a new rasa project
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> I'm not
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> what is this [nlu](product) thing?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* faq
    - respond_faq 
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> not really
    <B> Rasa Masterclass episodes 2-4 focus on NLU. Check out episode 2 [here](https://www.youtube.com/watch?v=k5UeywXA28k). Is there anything else I can help you with?
## not new to rasa + nlu + unknown topic
<U> hi hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> i want to know more about [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* faq
    - respond_faq 
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> not sure
    <B> Hmm, seems like I haven't learned about that part of NLU yet! Try using the search bar above for now. Is there anything else I can help you with?

## not new to rasa + nlu + intent + no recommendation
<U> good afternoon
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i would like to know how to get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope i'm not
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> more about [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> what is [intent classification](nlu_part)?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* faq
    - respond_faq 
    <B> Would you like me to try to recommend a pipeline?
<U> no thanks
    <B> Got it. Is there anything else I can help you with?

## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> hellio
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i'm not
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> Rasa [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [intent classification](nlu_part)?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yes please
    <B> Which (human) language is your assistant in?
* faq
    - respond_faq 
    <B> Which (human) language is your assistant in?
<U> The training data is in [English](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. Is there anything else I can help you with?

## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> hey
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get strated with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> interested in [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> tell me more about [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> please do
    <B> Which (human) language is your assistant in?
* faq
    - respond_faq 
    <B> Which (human) language is your assistant in?
<U> [Spanish](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. Is there anything else I can help you with?

## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> hey
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nope
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> I want to learn more about [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> tell me about [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> yes please recommend me a pipeline
    <B> Which (human) language is your assistant in?
* faq
    - respond_faq 
    <B> Which (human) language is your assistant in?
<U> in [Russian](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. Is there anything else I can help you with?

## not new to rasa + nlu + entity + pipeline duckling
<U> hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> tell me about the different parts of [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> [NLU](product) please
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> how does [entity recognition](nlu_part) work?
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
* faq
    - respond_faq 
    <B> Would you like me to try to recommend a pipeline?
<U> yes please
    <B> What kind of entities do you want to extract?
<U> [date ranges](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component Is there anything else I can help you with?

## not new to rasa + nlu + entity + pipeline duckling
<U> good afternoon
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> [NLU](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> tell me more about [entity recognition](nlu_part)
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
<U> yeah
    <B> What kind of entities do you want to extract?
* faq
    - respond_faq 
    <B> What kind of entities do you want to extract?
<U> [date ranges](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component Is there anything else I can help you with?

## FAQ - tell more about languages
* faq
    - respond_faq
* explain
    - action_explain_faq

## FAQ - Step 2 flow interrupted
<U> greetings
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to install [rasa](product)
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
## FAQ - Step 2 flow interrupted
<U> how can I install [rasa](product)
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?

## FAQ - Step 2 flow interrupted
<U> How to install rasa stack
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq  
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?
<U> yeah I'm ready
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## FAQ - Step 2 flow interrupted
<U> install [rasa](product)
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?
<U> not yet
    <B> Did you encounter any problems during installation?
<U> there is error
    <B> If you haven't already read the detailed instruction guide, you can look it up here. If you're still stuck, please ask for help on our [forum](https://forum.rasa.com/). You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## technical_question - docs_found - deny
<U> how to run sdk endpoint in background
    - action_docs_search 
    <B> Did that help?
<U> nope
    - action_tag_docs_search 
    - action_forum_search
## technical_question - docs_found - affirm
<U> Hello, where can I find the paper about DIET?
    - action_docs_search 
    <B> Did that help?
<U> yes
    - action_tag_docs_search 
    <B> Awesome!
## technical_question     <B> no docs_found
<U> installation of tensorflow-text
    - action_docs_search
    - action_forum_search
## FAQ - Step 2 flow interrupted
<U> i want to install [rasa](product)
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
<U> no
    <B> Did you encounter any problems during installation?
* faq
    - respond_faq 
    <B> If you haven't already read the detailed instruction guide, you can look it up here. If you're still stuck, please ask for help on our [forum](https://forum.rasa.com/). You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## FAQ - Step 2 flow interrupted
<U> [rasa](product) installation
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
<U> no
    <B> Did you encounter any problems during installation?
<U> yeah I did
    <B> Can you describe the problem you're having?
* faq
    - respond_faq 
    <B> If you haven't already read the detailed instruction guide, you can look it up here. If you're still stuck, please ask for help on our [forum](https://forum.rasa.com/). You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## FAQ - Step 2 flow interrupted
<U> get_started_step2
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> yeah
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?

## Just don't want to continue
<U> help me install [rasa](product)
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?

## new to rasa + not new to chatbots + not migrating
<U> hello bot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> how do I get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep
    <B> And have you built a contextual assistant or a bot before?
<U> sure
    <B> Are you migrating from another tool?
<U> no not really
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> [both](product) please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
<U> hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yep I am
    <B> And have you built a contextual assistant or a bot before?
<U> yeah I have
    <B> Are you migrating from another tool?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> [rasa open source](product) please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)

## new to rasa + not new to chatbots + not migrating
<U> hey there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i wanna get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* faq
    - respond_faq 
    <B> Are you new to Rasa?
<U> yep
    <B> And have you built a contextual assistant or a bot before?
<U> I have
    <B> Are you migrating from another tool?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> please explain
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)

## new to rasa + not new to chatbots + not migrating
<U> hello sara
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How can I get started with [rasa](product)?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i am
    <B> And have you built a contextual assistant or a bot before?
* faq
    - respond_faq 
    <B> And have you built a contextual assistant or a bot before?
<U> yeah I have
    <B> Are you migrating from another tool?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> yes sure
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
<U> hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
<U> yeah I am
    <B> Are you migrating from another tool?
* faq
    - respond_faq 
    <B> Are you migrating from another tool?
<U> no i'm not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> please explain
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
<U> hey there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to know about the parts of [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yeah I have
    <B> And have you built a contextual assistant or a bot before?
<U> yes
    <B> Are you migrating from another tool?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
* faq
    - respond_faq 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> please explain [both](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
<U> good morning
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
<U> Yea
    <B> Are you migrating from another tool?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> Yeah sure
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B>You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain stack and try it out
<U> hiii
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
* faq
    - respond_faq 
    <B> And have you built a contextual assistant or a bot before?
<U> no never
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> yes please
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain stack and try it out
<U> hello hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How do I get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> Yep
    <B> And have you built a contextual assistant or a bot before?
<U> Not really
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
* faq
    - respond_faq 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> explain it to me
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain x and try it out
<U> hey hey
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> How to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
<U> no :(
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> how do i get started with rasa [x](product) ?
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. Would you also like me to explain NLU and Core?
<U> no thanks
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain core and direct to step2
<U> hi there
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> Please tell me how to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes,i am
    <B> And have you built a contextual assistant or a bot before?
<U> nope !
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> how to start with rasa [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Would you also like me to explain NLU?
* faq
    - respond_faq 
    <B> Would you also like me to explain NLU?
<U> nope. i am good
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain core and direct to step2
<U> hi!
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can i try it out
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yas
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> could you explain rasa [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Would you also like me to explain NLU?
<U> no thanks
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain core, then nlu and direct to step2
<U> Hi rasa
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how can i get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ya
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> explain [core](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Would you also like me to explain NLU?
* faq
    - respond_faq 
    <B> Would you also like me to explain NLU?
<U> Yes please!
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain x, then direct to step2
<U> Hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do I get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> Yeah
    <B> And have you built a contextual assistant or a bot before?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> where to start with rasa [x](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. Would you also like me to explain NLU and Core?
* faq
    - respond_faq

## new to rasa/bots, explain core, then nlu and direct to step2
<U> hallo
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> Yepp
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> can you explain [core](product) please
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) Would you also like me to explain NLU?
<U> go for it
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain nlu and direct to step2
<U> hello robot
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i am!
    <B> And have you built a contextual assistant or a bot before?
<U> no never
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> please explain [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email).  Would you also like me to explain Core?
* faq
    - respond_faq  
    <B> Would you also like me to explain Core?
<U> no thanks
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain x and direct to step2
<U> hi !
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do I get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> Yea
    <B> And have you built a contextual assistant or a bot before?
<U> nope
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> Rasa [x](product) please
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. Would you also like me to explain NLU and Core?
* faq
    - respond_faq
<U> no thanks
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain nlu and direct to step2
<U> hi hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do I get started with [rasa](product)
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes
    <B> And have you built a contextual assistant or a bot before?
<U> nah
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email).  Would you also like me to explain Core?
<U> no thanks
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, don't explain and direct to step2
<U> hi
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes i am
    <B> And have you built a contextual assistant or a bot before?
<U> no
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
* faq
    - respond_faq
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> nope thank s
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, don't explain and direct to step2
<U> hello
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i am!
    <B> And have you built a contextual assistant or a bot before?
<U> no i have not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. Would you like to know more about Rasa Open Source or Rasa X?
<U> no
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## not new to rasa + not interested in products
<U> heya
    <B> Hi! My name's Sara. By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> can you help me get started?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no
    <B> Ok, which product would you like to know more about? Rasa Open Source or Rasa X?
<U> neither
    <B> Got it.
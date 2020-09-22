## next step prompt
<U> i have to go to the next step
    - action_next_step
## faqs
* faq
    - respond_faq
## more faqs
<U> hello rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
* faq
    - respond_faq
## just newsletter
<U> whats popping
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> newslettwr
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> yes it is
    <B> Got it. 
    <B> Is there anything else I can help you with?
## just newsletter, continue + confirm
<U> what up
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> sign me up for that newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq
    - respond_faq
    <B> Do you still want to subscribe to the newsletter?
<U> alright, cool
    <B> perfect! 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
<U> yes, i accept
    <B> Got it. 
    <B> Is there anything else I can help you with?
## just newsletter, don't continue + confirm
<U> rasa hello
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> subcribe
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq
    - respond_faq
    <B> Do you still want to subscribe to the newsletter?
<U> no i haven't decided yet if i want to sign up
    <B> Got it. 
    <B> How is this conversation going?
<U> yesh
    <B> Got it. 
    <B> Is there anything else I can help you with?
## just newsletter (with email already) + confirm
<U> hey let's talk
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> yes subscribe me
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
<U> hèhè
    <B> Got it.
    <B> Is there anything else I can help you with?
## just newsletter (with email already)
<U> hey, let's talk
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> lets do the newsletter signup
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just sales
<U> halloo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/). 
* faq
    - respond_faq
<U> i'd like to talk to a sales person
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## just sales, continue
<U> helloooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how can i book a sales call ?
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
* faq
    - respond_faq
    <B> Do you still want to talk to sales?
<U> ok fine
    - sales_form
    - form{"name": null} 
    <B> How is this conversation going?
## just sales, don't continue
<U> hello is anybody there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> bookin
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
* faq
    - respond_faq 
    <B> Do you still want to talk to sales?
<U> no, thankyou
    - sales_form
    - form{"name": null}
## not new to rasa + not interested in products
<U> hieeeeeeeeeeeee
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can you shw me some information about intallation?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* faq
    - respond_faq 
    <B> Are you new to Rasa?
<U> i guess it means - no
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> no i dont want
    <B> Got it.

## not new to rasa + not interested in products
<U> hi friends
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> where do i find instructions
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> absolutely not
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq
    - respond_faq 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> no ma'am
    <B> Got it. 
## not new to rasa + nlu + nothing special
<U> halloo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i don't agree
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> the [rasa open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* faq
    - respond_faq 
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> sorry not right now
    <B> Rasa Masterclass episodes 2-4 focus on NLU. Check out episode 2 [here](https://www.youtube.com/watch?v=k5UeywXA28k). 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + unknown topic
<U> hey, let's talk
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to build a bot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nehi
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [open source rasa](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* faq
    - respond_faq 
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> where am i right now?
    <B> Hmm, seems like I haven't learned about that part of NLU yet! Try using the search bar above for now. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + intent + no recommendation
<U> hi friends
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> find out how to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no, i hate it
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to learn about [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> what is [entity recognition](nlu_part)?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* faq
    - respond_faq 
    <B> Would you like me to try to recommend a pipeline?
<U> i don't want to give it to you
    <B> Got it. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> ssup?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what is componenbts
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no go
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i would like to know more about rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> i still don’t get how [intent classification](nlu_part) works
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> ok i accept
    <B> Which (human) language is your assistant in?
* faq
    - respond_faq 
    <B> Which (human) language is your assistant in?
<U> the language of the ai assistant is [english](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + intent + pipeline recommendation, spacy
<U> wasssup!
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to build assistant?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> no i dont want
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to get started with rasa [core](product)?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> what are [intents](nlu_part)?
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> which pipeline is better?
    <B> Which (human) language is your assistant in?
* faq
    - respond_faq 
    <B> Which (human) language is your assistant in?
<U> so far it only speaks [german](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + intent + pipeline recommendation, not spacy
<U> wasssup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> the components of rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> na
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to use [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> [recognition](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
<U> how to build a pipelin
    <B> Which (human) language is your assistant in?
* faq
    - respond_faq 
    <B> Which (human) language is your assistant in?
<U> the language of the ai assistant is [english](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + entity + pipeline duckling
<U> yoo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to create chatbot using rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i don not like this
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> more about [rasa open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> i checked the documentation on [entity recognition](nlu_part) but i still don’t understand it
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
* faq
    - respond_faq 
    <B> Would you like me to try to recommend a pipeline?
<U> yep i want that
    <B> What kind of entities do you want to extract?
<U> amounts of [money](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + entity + pipeline duckling
<U> hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to build a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i don't want to say
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i wanna try rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
<U> i was looking for [duckling](nlu_part) integration
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
<U> considering
    <B> What kind of entities do you want to extract?
* faq
    - respond_faq 
    <B> What kind of entities do you want to extract?
<U> how much [money](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?

## FAQ - tell more about languages
* faq
    - respond_faq
* explain
    - action_explain_faq

## FAQ - Step 2 flow interrupted
<U> heyho
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to install rasa in my system
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
## FAQ - Step 2 flow interrupted
<U> let me install rasa stack.
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?

## FAQ - Step 2 flow interrupted
<U> i'm installing rasa stack
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq  
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?
<U> ook
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## FAQ - Step 2 flow interrupted
<U> how do i install rasa in windows
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?
<U> no you did it wrong
    <B> Did you encounter any problems during installation?
<U> hi the command rasa init doesn't do anything in windows
    <B> If you haven't already read the detailed instruction guide, you can look it up here. If you're still stuck, please ask for help on our [forum](https://forum.rasa.com/). 
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## technical_question - docs_found - deny
<U> technical side of things?
    - action_docs_search 
    <B> Did that help?
<U> noooooooooooooooooooooooooooooooooooooooo
    - action_tag_docs_search 
    - action_forum_search
## technical_question - docs_found - affirm
<U> what is the difference between you and luis
    - action_docs_search 
    <B> Did that help?
<U> i will!
    - action_tag_docs_search 
    <B> Awesome!
## technical_question     <B> no docs_found
<U> where is the api for rasa x
    - action_docs_search
    - action_forum_search
## FAQ - Step 2 flow interrupted
<U> i want to use rasa stack
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
<U> no way
    <B> Did you encounter any problems during installation?
* faq
    - respond_faq 
    <B> If you haven't already read the detailed instruction guide, you can look it up here. If you're still stuck, please ask for help on our [forum](https://forum.rasa.com/). 
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## FAQ - Step 2 flow interrupted
<U> installing rasa
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
<U> no ma'am
    <B> Did you encounter any problems during installation?
<U> yesssss
    <B> Can you describe the problem you're having?
* faq
    - respond_faq 
    <B> If you haven't already read the detailed instruction guide, you can look it up here. If you're still stuck, please ask for help on our [forum](https://forum.rasa.com/). 
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## FAQ - Step 2 flow interrupted
* get_started_step2
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> it is ok
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?

## Just don't want to continue
<U> please install rasa stack
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?

## new to rasa + not new to chatbots + not migrating
<U> hey let's talk
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> how do i download rasa ?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i am using it
    <B> And have you built a contextual assistant or a bot before?
<U> yep. :/
    <B> Are you migrating from another tool?
<U> no thank s
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> what's rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
<U> hi mister
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> can you guide me know to create knowledge base chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> nice
    <B> And have you built a contextual assistant or a bot before?
<U> ya cool
    <B> Are you migrating from another tool?
<U> n
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i wanna try rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)

## new to rasa + not new to chatbots + not migrating
<U> hi folks
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> more info on components of rasa pls
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* faq
    - respond_faq 
    <B> Are you new to Rasa?
<U> yes great
    <B> And have you built a contextual assistant or a bot before?
<U> go
    <B> Are you migrating from another tool?
<U> no thank s
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> why is that necessary
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)

## new to rasa + not new to chatbots + not migrating
<U> hhola
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to download?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes give me information
    <B> And have you built a contextual assistant or a bot before?
* faq
    - respond_faq 
    <B> And have you built a contextual assistant or a bot before?
<U> yes, i accept
    <B> Are you migrating from another tool?
<U> no, i hate it
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> very much
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
<U> hola
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> tell me how to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i have used it in the past
    <B> And have you built a contextual assistant or a bot before?
<U> top
    <B> Are you migrating from another tool?
* faq
    - respond_faq 
    <B> Are you migrating from another tool?
<U> no i dont want to
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> why is that necessary
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
<U> sup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to implement rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> ok, i behave now
    <B> And have you built a contextual assistant or a bot before?
<U> i accept.
    <B> Are you migrating from another tool?
<U> it sux
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq
    - respond_faq 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> yes i wanna know more about rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
<U> hi friends
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to use
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> lets do it
    <B> And have you built a contextual assistant or a bot before?
<U> yaah
    <B> Are you migrating from another tool?
<U> thanks but no thanks
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> a little
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B>You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain stack and try it out
<U> hell9o
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to start
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> that's great
    <B> And have you built a contextual assistant or a bot before?
* faq
    - respond_faq 
    <B> And have you built a contextual assistant or a bot before?
<U> i don't want either of those
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> ja thats great
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain stack and try it out
<U> heyho
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how this rasa works
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yeah, why not
    <B> And have you built a contextual assistant or a bot before?
<U> this sucks
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq
    - respond_faq 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> could you tell me more
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain x and try it out
<U> ayyyy whaddup
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to develop a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> hm, i'd like that
    <B> And have you built a contextual assistant or a bot before?
<U> nein
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
<U> never mind
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain core and direct to step2
<U> jojojo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do you build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> thats fine
    <B> And have you built a contextual assistant or a bot before?
<U> nop
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [rasa open source](product) then lets talk about [rasa x](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* faq
    - respond_faq 
    <B> Would you also like me to explain NLU?
<U> not right now
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain core and direct to step2
<U> hi there it's me
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to make intelligence chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes it was okay
    <B> And have you built a contextual assistant or a bot before?
<U> not right now
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how to get started with rasa [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> i'm not sure
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain core, then nlu and direct to step2
<U> hi mister
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to get sarted
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> oh cool
    <B> And have you built a contextual assistant or a bot before?
<U> nevermind
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> how do i train rasa [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* faq
    - respond_faq 
    <B> Would you also like me to explain NLU?
<U> ya thats cool
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain x, then direct to step2
<U> hellooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what are rasa's components?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes great
    <B> And have you built a contextual assistant or a bot before?
<U> i don't want to say
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
* faq
    - respond_faq

## new to rasa/bots, explain core, then nlu and direct to step2
<U> hallo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> where can i learn more about rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> okie
    <B> And have you built a contextual assistant or a bot before?
<U> absolutely not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i want to learn about [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
<U> yessoo
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain nlu and direct to step2
<U> hi friends
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> rasa components
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> fine
    <B> And have you built a contextual assistant or a bot before?
<U> nein
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [both](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
* faq
    - respond_faq  
    <B> Would you also like me to explain Core?
<U> i dont want to tell
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain x and direct to step2
<U> hey dude
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how to build a bot?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yes i accept
    <B> And have you built a contextual assistant or a bot before?
<U> nop
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
* faq
    - respond_faq
<U> no. u r idiot
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain nlu and direct to step2
<U> hell9o
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> can give tell me about components of rosa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> definitely yes without a doubt
    <B> And have you built a contextual assistant or a bot before?
<U> no. u r idiot
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> get started with [rasa open source](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
<U> no go
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, don't explain and direct to step2
<U> halloooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> help me build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> i accept.
    <B> And have you built a contextual assistant or a bot before?
<U> i do not need help installing
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq
    - respond_faq
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> no, my frst time
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, don't explain and direct to step2
<U> hi can you speak ?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> components of rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> yaah
    <B> And have you built a contextual assistant or a bot before?
<U> no i dont want to
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i decline
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## not new to rasa + not interested in products
<U> whats popping
    <B> Hey there, my name is Sara.  
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq
    - respond_faq
<U> how can i learn rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
<U> it sucks
    <B> Would you like to know more about Rasa Open Source or Rasa X?
<U> i don't want to give it to you
    <B> Got it.
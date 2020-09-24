## next step prompt
* next_step: what is the next step ?
    - action_next_step
## faqs
* faq: How many people are in that community?
    - respond_faq
## more faqs
* greet: hello is anybody there
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq: what is rasa x enterprise
    - respond_faq
* faq: what programming languages does Rasa support?
    - respond_faq
## just newsletter
* greet: hi there it's me
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq: What language you support?
    - respond_faq
* signup_newsletter: i want on this dope newsletter - my email is r_grove@gmail.com
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* affirm: a little
    <B> Got it. 
    <B> Is there anything else I can help you with?
## just newsletter, continue + confirm
* greet: hello it is me again
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: subscribe to your newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq: what language list can I find for rasa
    - respond_faq
    <B> Do you still want to subscribe to the newsletter?
* affirm: ja
    <B> perfect! 
    - subscribe_newsletter_form
    - form{"name": null}
    <B> How is this conversation going?
* affirm: okay
    <B> Got it. 
    <B> Is there anything else I can help you with?
## just newsletter, don't continue + confirm
* greet: hey ther
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* signup_newsletter: do the newsletter then
    <B> Can do! 
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
* faq: are you really free
    - respond_faq
    <B> Do you still want to subscribe to the newsletter?
* deny: no ma'am
    <B> Got it. 
    <B> How is this conversation going?
* affirm: ok friend
    <B> Got it. 
    <B> Is there anything else I can help you with?
## just newsletter (with email already) + confirm
* greet: hello rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq: how cost to install Rasa?
    - respond_faq
* signup_newsletter: newsletter - my email is mabel@brown.com
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
* react_positive: love you
    <B> Got it.
    <B> Is there anything else I can help you with?
## just newsletter (with email already)
* greet: ey boss
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq: how much does Rasa cost ?
    - respond_faq
* signup_newsletter: i just want to signup for your newsletter
    <B> Can do!
    - subscribe_newsletter_form
    - form{"name": "subscribe_newsletter_form"}
    - form{"name": null}
    <B> How is this conversation going?
## just sales
* greet: hi can you speak ?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/). 
* faq: what is the enterprise pricing schedule?
    - respond_faq
* contact_sales: i want to talk to your sales guys
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
    - form{"name": null} 
    <B> How is this conversation going?
## just sales, continue
* greet: heelio
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: i want to talk to someone about your pricing system
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
* faq: whats the cost of rasa
    - respond_faq
    <B> Do you still want to talk to sales?
* affirm: a little
    - sales_form
    - form{"name": null} 
    <B> How is this conversation going?
## just sales, don't continue
* greet: hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* contact_sales: can someone call me please?
    <B> Sure, we can book a sales call! Let's get to know each other first . 
    - sales_form
    - form{"name": "sales_form"}
* faq: is Rasa open source
    - respond_faq 
    <B> Do you still want to talk to sales?
* deny: none of them
    - sales_form
    - form{"name": null}
## not new to rasa + not interested in products
* greet: hellooo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: components in rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* faq: is Rasa open sourced?
    - respond_faq 
    <B> Are you new to Rasa?
* deny: noooooooooooooooooooooooooooooooooooooooo
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: stop it, i do not care!!!
    <B> Got it.

## not new to rasa + not interested in products
* greet: hi can you speak ?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i want to implement rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no you did it wrong
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq: which python do you support?
    - respond_faq 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: i decline
    <B> Got it. 
## not new to rasa + nlu + nothing special
* greet: yoo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i want to get started
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no, i hate it
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: get started with [rasa open source](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* faq: which python version should i install
    - respond_faq 
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* deny: sorry not right now
    <B> Rasa Masterclass episodes 2-4 focus on NLU. Check out episode 2 [here](https://www.youtube.com/watch?v=k5UeywXA28k). 
    <B> Is there anything else I can help you with?
## not new to rasa + nlu + unknown topic
* greet: hieeeeeeeeeeeee
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how do i download rasa ?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no don"t want this!
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: [rasa open source](product) then lets talk about [rasa x](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* faq: Help me to figure out the meaning of slots.
    - respond_faq 
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* out_of_scope: what about wheather
    <B> Hmm, seems like I haven't learned about that part of NLU yet! Try using the search bar above for now. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + intent + no recommendation
* greet: halo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i like to build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no ma'am
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: just [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: how do you integrate [duckling](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* faq: I bet you can tell me all about slots.
    - respond_faq 
    <B> Would you like me to try to recommend a pipeline?
* deny: none of them
    <B> Got it. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + intent + pipeline recommendation, spacy
* greet: halo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i wanna build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: no. u r idiot
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: tell me how to get started with [core](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: it would be helpful to learn more about [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* affirm: ofcourse
    <B> Which (human) language is your assistant in?
* faq: I need a expert opinion on slots.
    - respond_faq 
    <B> Which (human) language is your assistant in?
* enter_data: the assistant is in [french](language)
    <B> If your training data is in English, a good starting point is this [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer), which uses both pretrained and supervised embeddings. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + intent + pipeline recommendation, spacy
* greet: bom dia
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: components of rasa?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't want either of those
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: and rasa [nlu](product)?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: [intent recognition](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* pipeline_recommendation: which pipeline is better?
    <B> Which (human) language is your assistant in?
* faq: Not sure what slots are.
    - respond_faq 
    <B> Which (human) language is your assistant in?
* enter_data: it’s an [portuguese](language) bot
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + intent + pipeline recommendation, not spacy
* greet: salut
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: tell me how i can get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i dont want to tell
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: just rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: i want to learn about [intent classification](nlu_part)
    <B> Check out [part 1](https://blog.rasa.com/rasa-nlu-in-depth-part-1-intent-classification/) of the Rasa NLU In Depth series from our software engineer Tobias to learn more about intent classification. Would you like me to try to recommend a pipeline?
* pipeline_recommendation: what i a good pipeline to start with?
    <B> Which (human) language is your assistant in?
* faq: What are slots?
    - respond_faq 
    <B> Which (human) language is your assistant in?
* enter_data: it’s in [portuguese](language)
    <B> For <language>, a good starting point is the [recommended pipeline](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#the-short-answer) for languages other than English. 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + entity + pipeline duckling
* greet: hi mister
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i want to develop a chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't want either of those
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to get started with rasa [core](product)?
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: i want to learn about [intent classification](nlu_part)
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
* faq: do you have a core tutorial i can follow
    - respond_faq 
    <B> Would you like me to try to recommend a pipeline?
* affirm: yop
    <B> What kind of entities do you want to extract?
* enter_data: get [dates](entity) from messages
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?

## not new to rasa + nlu + entity + pipeline duckling
* greet: hii
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: start rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: it sucks
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: yes i wanna know more about rasa [nlu](product)
    <B> Anything in particular you'd like to know about, e.g. intent classification or entity recognition?
* nlu_info: [intent](nlu_part) please
    <B> [Part 2](https://blog.rasa.com/rasa-nlu-in-depth-part-2-entity-recognition/) of the Rasa NLU In Depth series from our software engineer Tobias is a great tutorial about entity recognition. Would you like me to try to recommend a pipeline?
* affirm: yeah do that
    <B> What kind of entities do you want to extract?
* faq: i need a core tutorial
    - respond_faq 
    <B> What kind of entities do you want to extract?
* enter_data: [numbers](entity)
    <B> For that entity I'd suggest using the rule-based [DucklingHTTPExtractor](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor) component 
    <B> Is there anything else I can help you with?

## FAQ - tell more about languages
* faq: i need a tutorial on how to use rasa core
    - respond_faq
* explain: could you explain why you need that
    - action_explain_faq

## FAQ - Step 2 flow interrupted
* greet: yoo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* install_rasa: how to download rasa
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
## FAQ - Step 2 flow interrupted
* install_rasa: do you mind helping me install rasa?
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq: i am looking for a nlu tutorial
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?

## FAQ - Step 2 flow interrupted
* install_rasa: can you help me to install rasa?
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq: get me some tutorials
    - respond_faq  
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?
* affirm: ya go for it
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## FAQ - Step 2 flow interrupted
* install_rasa: i need to install rasa
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq: get me some tutorials
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?
* deny: i guess it means - no
    <B> Did you encounter any problems during installation?
* technical_question: how to train model
    <B> If you haven't already read the detailed instruction guide, you can look it up here. If you're still stuck, please ask for help on our [forum](https://forum.rasa.com/). 
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## technical_question - docs_found - deny
* technical_question: can you help me with this problem
    - action_docs_search 
    <B> Did that help?
* deny: no, thankyou
    - action_tag_docs_search 
    - action_forum_search
## technical_question - docs_found - affirm
* technical_question: getting some error
    - action_docs_search 
    <B> Did that help?
* affirm: ofcoure i do
    - action_tag_docs_search 
    <B> Awesome!
## technical_question     <B> no docs_found
* technical_question: credentials
    - action_docs_search
    - action_forum_search
## FAQ - Step 2 flow interrupted
* install_rasa: installing rasa
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* deny: no i haven't decided yet if i want to sign up
    <B> Did you encounter any problems during installation?
* faq: Can one make a voice bot with rasa?
    - respond_faq 
    <B> If you haven't already read the detailed instruction guide, you can look it up here. If you're still stuck, please ask for help on our [forum](https://forum.rasa.com/). 
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## FAQ - Step 2 flow interrupted
* install_rasa: how do i run rasa on windows
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* deny: no sorry
    <B> Did you encounter any problems during installation?
* affirm: yes i have built a bot before
    <B> Can you describe the problem you're having?
* faq: How to build a voice bot with rasa?
    - respond_faq 
    <B> If you haven't already read the detailed instruction guide, you can look it up here. If you're still stuck, please ask for help on our [forum](https://forum.rasa.com/). 
    <B> You can now create a new Rasa assistant and also talk to it by simply running rasa init. This comes with some sample training data and the required configuration files to get you started. You can also talk to the assistant after it is done training. If you have no questions, click here when you're ready to go to the next step. Or, at any point after you're done asking questions, say 'take me to the next step'.

## FAQ - Step 2 flow interrupted
* get_started_step2: go to step 2
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* affirm: kk
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq: It is possible to build a voice bot with rasa?
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?

## Just don't want to continue
* install_rasa: i need help setting up
    <B> You will first need to install Rasa Open Source using this installation command. pip install rasa. If you are having any trouble installing, you can look up detailed installation instructions here. If you got it successfully installed, are you ready to start building a contextual assistant?
* faq: can a voice bot be built using rasa?
    - respond_faq 
    <B> If you got it successfully installed, are you ready to start building a contextual assistant?

## new to rasa + not new to chatbots + not migrating
* greet: heelio
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq: with rasa can I construct a voice bot?
    - respond_faq
* how_to_get_started: what do i need for rasa implementation?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes pleae
    <B> And have you built a contextual assistant or a bot before?
* affirm: sweet
    <B> Are you migrating from another tool?
* deny: i'm not giving you my email address
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: i want to use [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
* greet: hei
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq: what can you put in the forum?
    - respond_faq
* how_to_get_started: how to get starter?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: accepted
    <B> And have you built a contextual assistant or a bot before?
* affirm: i want that
    <B> Are you migrating from another tool?
* deny: none of them
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: the [rasa open source](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)

## new to rasa + not new to chatbots + not migrating
* greet: hello rasa
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: hi, can you help in understanding nlu
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* faq: What is the scope of the Rasa forum?
    - respond_faq 
    <B> Are you new to Rasa?
* affirm: yes give me information
    <B> And have you built a contextual assistant or a bot before?
* affirm: yeah do that
    <B> Are you migrating from another tool?
* deny: nah not for me
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* explain: what does that mean
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)

## new to rasa + not new to chatbots + not migrating
* greet: halo sara
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: where can i find rasa document
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes i have!
    <B> And have you built a contextual assistant or a bot before?
* faq: tell me about the forum
    - respond_faq 
    <B> And have you built a contextual assistant or a bot before?
* affirm: yep thats cool
    <B> Are you migrating from another tool?
* deny: no thank s
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: ok cool
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
* greet: salut
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how do i build a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: ye splease
    <B> And have you built a contextual assistant or a bot before?
* affirm: i do
    <B> Are you migrating from another tool?
* faq: Could you please explain the Rasa forum to me?
    - respond_faq 
    <B> Are you migrating from another tool?
* deny: n
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* explain: give me more details
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
* greet: bom dia
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: tell me how i can get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: i will!
    <B> And have you built a contextual assistant or a bot before?
* affirm: ah ok
    <B> Are you migrating from another tool?
* deny: absolutely not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq: can I invent a voice bot with rasa?
    - respond_faq 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: what's rasa [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa + not new to chatbots + not migrating
* greet: heelio
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: where do i find instructions
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yeah do that
    <B> And have you built a contextual assistant or a bot before?
* affirm: cool story bro
    <B> Are you migrating from another tool?
* deny: it sux
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: ok, sara
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq: Could I build a rasa voice bot?
    - respond_faq 
    <B>You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain stack and try it out
* greet: ey boss
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: i want to create chatbot using rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: sweet
    <B> And have you built a contextual assistant or a bot before?
* faq: Is it possible to use rasa to build a voice bot?
    - respond_faq 
    <B> And have you built a contextual assistant or a bot before?
* deny: no way
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* affirm: that is cool
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain stack and try it out
* greet: bonjour
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: rasa basics
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: okay..
    <B> And have you built a contextual assistant or a bot before?
* deny: no way
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq: can I create a voice bot with rasa?
    - respond_faq 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* explain: give me more details
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain x and try it out
* greet: hi mister
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how to download?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: ok, i behave now
    <B> And have you built a contextual assistant or a bot before?
* deny: i don't want to say
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* enter_data: [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
* deny: noooooooooooooooooooooooooooooooooooooooo
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq: can a voice bot be built using rasa?
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain core and direct to step2
* greet: guten morgen
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: from where i should start?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yes that's great
    <B> And have you built a contextual assistant or a bot before?
* deny: absolutely not
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: get started with [rasa open source](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* faq: rasa can't be used to build a voice bot, can it?
    - respond_faq 
    <B> Would you also like me to explain NLU?
* deny: i don't want to say
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain core and direct to step2
* greet: ey boss
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: where to start the development of rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: that would be great
    <B> And have you built a contextual assistant or a bot before?
* deny: no bots at all
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: rasa [core](product) quickstart
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* deny: nehi
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq: What is the purpose of the Rasa forum?
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain core, then nlu and direct to step2
* greet: good mourning
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: hw to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: that ok
    <B> And have you built a contextual assistant or a bot before?
* deny: i don't think so
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how do i get rasa [nlu](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* faq: tell me about the forum
    - respond_faq 
    <B> Would you also like me to explain NLU?
* affirm: top
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain x, then direct to step2
* greet: yoo
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how to learn rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: ok fine
    <B> And have you built a contextual assistant or a bot before?
* deny: nah not for me
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* enter_data: [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
* faq: what can people do in the forum
    - respond_faq

## new to rasa/bots, explain core, then nlu and direct to step2
* greet: hello sweatheart
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: show me learning resources about rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: si
    <B> And have you built a contextual assistant or a bot before?
* deny: no bots at all
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: how to train [rasa open source](product)
    <B> Core decides what happens next in this conversation. Its machine learning-based dialogue management predicts the next best action based on the input from NLU, the conversation history and your training data. (Example: Core has a confidence of 87% that ask_primary_change is the next best action to confirm with the user if they want to change their primary contact information.) 
    <B> Would you also like me to explain NLU?
* affirm: yaah
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq: what do people do in the forum
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain nlu and direct to step2
* greet: ssup?
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how to make a bot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: jezz
    <B> And have you built a contextual assistant or a bot before?
* deny: n
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: i want to learn about [nlu](product)
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
* faq: what is the forum used for?
    - respond_faq  
    <B> Would you also like me to explain Core?
* deny: no i dont want
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain x and direct to step2
* greet: hai
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: need help in finding information about rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: ofcoure i do
    <B> And have you built a contextual assistant or a bot before?
* deny: i don't want to give you my email
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* enter_data: [rasa](product)
    <B> Rasa X is a toolset used to improve contextual assistants built using Rasa Open Source by helping you turn real user conversations into training data. Rasa X includes a user interface and a REST API. 
    <B> Would you also like me to explain NLU and Core?
* faq: Is rasa open source?
    - respond_faq
* deny: i'm not going to give it to you
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, explain nlu and direct to step2
* greet: hello everybody
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: where do i find instructions
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yop
    <B> And have you built a contextual assistant or a bot before?
* deny: thanks but no thanks
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* how_to_get_started: what about [nlu](product)?
    <B> NLU understands the user's message based on your previous training data:     - Intent classification: Interpreting meaning based on predefined intents (Example: Please send the confirmation to amy@example.com? is a send_confirmation intent with 93% confidence)     - Entity extraction: Recognizing structured data (Example: amy@example.com is an email). 
    <B> Would you also like me to explain Core?
* deny: no and no again
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq: how much costs the rasa platform
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, don't explain and direct to step2
* greet: hi friends
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: tell me how to get started with rasa
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: yap
    <B> And have you built a contextual assistant or a bot before?
* deny: non
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* faq: what language would rasa use
    - respond_faq
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: no you did it wrong
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## new to rasa/bots, don't explain and direct to step2
* greet: good evening
    <B> Hey there, my name is Sara. 
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* how_to_get_started: how about building chatbot
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* affirm: considering
    <B> And have you built a contextual assistant or a bot before?
* deny: no don"t want this!
    <B> Alright, I'll briefly explain the components that make up Rasa then. Rasa Open Source is a machine learning framework for automated text and voice-based conversations. Rasa X is a tool that helps you build, improve, and deploy AI Assistants that are powered by the Rasa Open Source framework. 
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: no bots at all
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.
* faq: do you know what language rasa uses
    - respond_faq 
    <B> You can click here if you want to go to the next step to try out Rasa Open Source.

## not new to rasa + not interested in products
* greet: hi mister
    <B> Hey there, my name is Sara.  
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* faq: What languages can rasa be relied upon to support?
    - respond_faq
* how_to_get_started: how toget strated?
    <B> To determine how I can help you best, I'm going to ask you a few questions. Let's go. Are you new to Rasa?
* deny: i don't wanna tell the name of my company
    <B> Would you like to know more about Rasa Open Source or Rasa X?
* deny: no bots at all
    <B> Got it.
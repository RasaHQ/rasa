## Step 4 prompt
* get_started_step4: go to step 4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).

## Happy path: Prompt -> Get event locations -> get next event
* get_started_step4: go to step 4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* ask_which_events: i want to meet rasa
    <B> Looks like we don't have currently have any Rasa events planned.
* ask_which_events: can you tell all of the events?
    <B> Looks like we don't have currently have any Rasa events planned. 
    <B> You should join the [Rasa Community forum](https://forum.rasa.com), where you can ask questions about Rasa Open Source and Rasa X, find resources, and stay up-to-date with what's happening at Rasa.

## Happy path: Prompt -> Get event locations -> get next event for location
* get_started_step4: go to step 4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* ask_which_events: so what events are there?
    <B> Looks like we don't have currently have any Rasa events planned.
* ask_which_events: is there an event in [montreal](location)
    <B> Looks like we don't have currently have any Rasa events planned.
    <B> You should join the [Rasa Community forum](https://forum.rasa.com), where you can ask questions about Rasa Open Source and Rasa X, find resources, and stay up-to-date with what's happening at Rasa.

## Happy path: Get event locations -> get next event
* ask_which_events: what kind of events are there?
    <B> Looks like we don't have currently have any Rasa events planned.
* ask_which_events: what events are there?
    <B> Looks like we don't have currently have any Rasa events planned. 
    <B> Is there anything else I can help you with?

## Happy path: Get event locations -> get next event for location
* ask_which_events: at which date the next community event will take place?
    <B> Looks like we don't have currently have any Rasa events planned.
* ask_which_events: what are the events for [new york](location)?
    <B> Looks like we don't have currently have any Rasa events planned. 
## Prompt -> Get next event
* get_started_step4: go to step 4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* ask_which_events: what do you know about rasa meetups?
    <B> Here are the upcoming Rasa events. 
    <B> You should join the [Rasa Community forum](https://forum.rasa.com), where you can ask questions about Rasa Open Source and Rasa X, find resources, and stay up-to-date with what's happening at Rasa.

## Prompt -> Get next event for location
* get_started_step4: go to step 4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* ask_which_events: when are the events for [paris](location)?
    <B> Here are the upcoming Rasa events. 
    <B> You should join the [Rasa Community forum](https://forum.rasa.com), where you can ask questions about Rasa Open Source and Rasa X, find resources, and stay up-to-date with what's happening at Rasa.

## Get next event
* ask_which_events: will there be an event in my city?
    <B> Here are the upcoming Rasa events. 
    <B> Is there anything else I can help you with?
## Get next event for location
* ask_which_events: is there an event in [montreal](location)
    <B> Here are the upcoming Rasa events. 
    <B> Is there anything else I can help you with?
## Prompt -> Contribute
* get_started_step4: go to step 4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* ask_why_contribute: why add to your business?
    <B> Here a few reasons why you should contribute:     - Explore the source code in depth     - Contribute to open source software     - Earn [Rasa swag])(https://rasa.com/community/contribute/)!
* ask_how_contribute: in what manner can one contribute?
    <B> Check out the [Contribute to Rasa](https://github.com/orgs/RasaHQ/projects/23) board for ideas on how to contribute code or content. You can also help by answering other members' questions on the [forum](http://forum.rasa.com).

## Contribute
* ask_why_contribute: what can i bring to help your code
    <B> Here a few reasons why you should contribute:     - Explore the source code in depth     - Contribute to open source software     - Earn [Rasa swag])(https://rasa.com/community/contribute/)!
* ask_how_contribute: tell me how i can contribute
    <B> Check out the [Contribute to Rasa](https://github.com/orgs/RasaHQ/projects/23) board for ideas on how to contribute code or content. You can also help by answering other members' questions on the [forum](http://forum.rasa.com).
## Prompt -> How can I contribute
* get_started_step4: go to step 4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* ask_how_contribute: what should i work on?
    <B> Check out the [Contribute to Rasa](https://github.com/orgs/RasaHQ/projects/23) board for ideas on how to contribute code or content. You can also help by answering other members' questions on the [forum](http://forum.rasa.com).

## How can I contribute
* ask_how_contribute: how can i contribute to your code
    <B> Check out the [Contribute to Rasa](https://github.com/orgs/RasaHQ/projects/23) board for ideas on how to contribute code or content. You can also help by answering other members' questions on the [forum](http://forum.rasa.com).
## Prompt -> Get help in forum
* get_started_step4: go to step 4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
* ask_question_in_forum: where can i post on the forum?
    <B> Here's the link to the forum: [https://forum.rasa.com](https://forum.rasa.com)
## Get help in the forum
* ask_question_in_forum: i need to ask something of the forum
    <B> Here's the link to the forum: [https://forum.rasa.com](https://forum.rasa.com)
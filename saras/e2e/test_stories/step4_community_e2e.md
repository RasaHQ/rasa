## Step 4 prompt
* get_started_step4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).

## Happy path: Prompt -> Get event locations -> get next event
* get_started_step4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> do you know the exact date for the next community event?
    <B> Looks like we don't have currently have any Rasa events planned.
<U> when is our next group event going to take place?
    <B> Looks like we don't have currently have any Rasa events planned. 
    <B> You should join the [Rasa Community forum](https://forum.rasa.com), where you can ask questions about Rasa Open Source and Rasa X, find resources, and stay up-to-date with what's happening at Rasa.

## Happy path: Prompt -> Get event locations -> get next event for location
* get_started_step4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> when will the next event occur in the community?
    <B> Looks like we don't have currently have any Rasa events planned.
<U> when is the next event for [detroit](location)?
    <B> Looks like we don't have currently have any Rasa events planned.
    <B> You should join the [Rasa Community forum](https://forum.rasa.com), where you can ask questions about Rasa Open Source and Rasa X, find resources, and stay up-to-date with what's happening at Rasa.

## Happy path: Get event locations -> get next event
<U> what do you know about rasa meetups?
    <B> Looks like we don't have currently have any Rasa events planned.
<U> so what exactly are these events?
    <B> Looks like we don't have currently have any Rasa events planned. 
    <B> Is there anything else I can help you with?

## Happy path: Get event locations -> get next event for location
<U> give me the events you have.
    <B> Looks like we don't have currently have any Rasa events planned.
<U> what is the next event for [paris](location)?
    <B> Looks like we don't have currently have any Rasa events planned. 
## Prompt -> Get next event
* get_started_step4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> which events you got?
    <B> Here are the upcoming Rasa events. 
    <B> You should join the [Rasa Community forum](https://forum.rasa.com), where you can ask questions about Rasa Open Source and Rasa X, find resources, and stay up-to-date with what's happening at Rasa.

## Prompt -> Get next event for location
* get_started_step4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> what is the next event for [seattle](location)?
    <B> Here are the upcoming Rasa events. 
    <B> You should join the [Rasa Community forum](https://forum.rasa.com), where you can ask questions about Rasa Open Source and Rasa X, find resources, and stay up-to-date with what's happening at Rasa.

## Get next event
<U> where can i see a calendar of community events?
    <B> Here are the upcoming Rasa events. 
    <B> Is there anything else I can help you with?
## Get next event for location
<U> what are the events for [new york](location)?
    <B> Here are the upcoming Rasa events. 
    <B> Is there anything else I can help you with?
## Prompt -> Contribute
* get_started_step4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> why be a part of your mission?
    <B> Here a few reasons why you should contribute:     - Explore the source code in depth     - Contribute to open source software     - Earn [Rasa swag])(https://rasa.com/community/contribute/)!
<U> how can i add code to rasa
    <B> Check out the [Contribute to Rasa](https://github.com/orgs/RasaHQ/projects/23) board for ideas on how to contribute code or content. You can also help by answering other members' questions on the [forum](http://forum.rasa.com).

## Contribute
<U> why do i want to help with your code
    <B> Here a few reasons why you should contribute:     - Explore the source code in depth     - Contribute to open source software     - Earn [Rasa swag])(https://rasa.com/community/contribute/)!
<U> how can i be a contributor?
    <B> Check out the [Contribute to Rasa](https://github.com/orgs/RasaHQ/projects/23) board for ideas on how to contribute code or content. You can also help by answering other members' questions on the [forum](http://forum.rasa.com).
## Prompt -> How can I contribute
* get_started_step4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> i want to make rasa better
    <B> Check out the [Contribute to Rasa](https://github.com/orgs/RasaHQ/projects/23) board for ideas on how to contribute code or content. You can also help by answering other members' questions on the [forum](http://forum.rasa.com).

## How can I contribute
<U> how to improve rasa
    <B> Check out the [Contribute to Rasa](https://github.com/orgs/RasaHQ/projects/23) board for ideas on how to contribute code or content. You can also help by answering other members' questions on the [forum](http://forum.rasa.com).
## Prompt -> Get help in forum
* get_started_step4
    <B> Hey there, my name is Sara.
    <B> By chatting to me you agree to our [privacy policy](https://rasa.com/privacy-policy/).
<U> how do i post my question on the forum?
    <B> Here's the link to the forum: [https://forum.rasa.com](https://forum.rasa.com)
## Get help in the forum
<U> how do i post my question on the forum?
    <B> Here's the link to the forum: [https://forum.rasa.com](https://forum.rasa.com)
# Rasa Core Examples

## moodbot: Simple getting started example

Getting started example. It is a simple example showcasing the creation of a 
bot for facebook (or for any other channel for that matter). 
Documentation can be found [here](https://core.rasa.com/tutorial_basics.html).

## restaurantbot: Restaurant suggestion example

Highlights advanced features, like a custom policy as well as custom action
code. It is an example of supervised learning with stories, and includes a NLU
model. Based on the bAbI and DSTC datasets. 
Documentation can be found [here](https://core.rasa.com/tutorial_supervised.html).

## concertbot: Interactive learning example

Create stories using interactive learning. Does not contain NLU training data,
hence uses a simple regex interpreter to parse user messages. 
Documentation can be found [here](https://core.rasa.com/tutorial_interactive_learning.html).

## remotebot: Using Rasa Core as a HTTP server

Showcase for using Rasa over HTTP. Instead of defining custom action code in 
python, the Rasa Core server will call a custom endpoint to run the action code
written in an arbitrary programming language. 
Documentation can be found [here](https://core.rasa.com/tutorial_remote.html).


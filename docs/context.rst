.. _section_context:

Context-aware Dialogue
======================

Rasa NLU allows you to turn natural language into structured data,
but this might not be enough if you want to build a bot that handles what
has been said in context and adjusts the flow of the conversation
accordingly. Rasa's open-source solution to handle contextual dialogue is
`Rasa Core <https://github.com/RasaHQ/rasa_core>`_, but there are other tools
out there such as `Dialogflow <https://dialogflow.com>`_ (not open-sourced).

Rasa Core uses machine learning to predict the evolution of a conversation,
and does away with the need for tedious and poorly-scaling ``if/else`` logic.
It also allows you to implement custom actions in response to the
user message, such as saying something back, modifying a database, calling an
API or handing over to a human. It is by design the natural companion of
Rasa NLU if you want to build conversational bots.
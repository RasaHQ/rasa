.. _section_context:

Context-aware Dialogue
======================

Rasa NLU allows you to turn natural language into structured data,
but this is just one part of building a chat or voice app. 

Rasa's open-source solution to handle contextual dialogue is
`Rasa Core <https://github.com/RasaHQ/rasa_core>`_ with documentation `here <https://core.rasa.com>`_.

Rasa Core uses machine learning to predict the evolution of a conversation,
so you don't have to write poorly-scaling ``if/else`` logic (read `this post <https://medium.com/rasa-blog/a-new-approach-to-conversational-software-2e64a5d05f2a>`_).
It allows you to implement custom actions that should be executed, 
such as saying something , modifying a database, calling an
API or handing over to a human. 

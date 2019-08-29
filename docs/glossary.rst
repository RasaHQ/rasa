:desc: Glossary for all Rasa-related terms

.. _glossary:

Glossary
========

.. glossary::

    Action
        A single step that a bot takes in a conversation (e.g. calling an API or sending a response back to the user).

    Annotation
        Adding labels to messages and conversations so that they can be used to train a model.

    CMS
        A Content Management System (CMS) can be used to store bot responses externally instead of directly including it as part of the domain.  This provides more flexibility in changing them as they are not tightly-coupled with the training data.

    Custom Action
        An action written by a Rasa developer that can run arbitrary code mainly to interact with the outside world.

    Default Action
        A built-in action that comes with predefined functionality.

    Domain
        Defines the inputs and outputs of an assistant.

        It includes a list of all the intents, entities, slots, actions, and forms that the assistant knows about.

    Entity
        Structured information that can be extracted from a user message.

        For example a telephone number, a person's name, a location, the name of a product

    Event
        All conversations in Rasa are represented as a sequence of events. For instance, a ``UserUttered`` represents a user entering a message, and an ``ActionExecuted`` represents the assistant executing an action. You can learn more about them :ref:`here <events>`.

    Form
        A type of custom action that asks the user for multiple pieces of information.

        For example, if you need a city, a cuisine, and a price range to recommend a restaurant, you can create  a restaurant form to do that. You can describe any business logic inside a form. For example, if you want to ask for a particular neighbourhood if a user mentions a large city like Los Angeles, you can write that logic inside the form.

    Happy / Unhappy Paths
        If your assistant asks a user for some information and the user provides it, we call that a happy path. Unhappy paths are all the possible edge cases of a bot. For example, the user refusing to give some input, changing the topic of conversation, or correcting something they said earlier.

    Intent
        Something that a user is trying to convey or accomplish (e,g., greeting, specifying a location).

    Interactive Learning
        A mode of training the bot where the user provides feedback to the bot while talking to it.

        This is a powerful way to write complicated stories by enabling users to explore what a bot can do and easily fix any mistakes it makes.

    NLG
        Natural Language Generation (NLG) is the process of generating natural language messages to send to a user.

        Rasa uses a simple template-based approach for NLG. Data-driven approaches (such as neural NLG) can be implemented by creating a custom NLG component.

    Rasa NLU
        Natural Language Understanding (NLU) deals with parsing and understanding human language into a structured format.

        Rasa NLU is the part of Rasa that performs intent classification and entity extraction.

    Pipeline
        A Rasa bot's NLU system is defined by a pipeline, which is a list of NLU components (see "Rasa NLU Component") in a particular order. A user input is processed by each component one by one before finally giving out the structured output.

    Policy
        Policies make decisions on how conversation flow should proceed. At every turn, the policy which predicts the next action with the highest confidence will be used.  A Core model can have multiple policies included, and the policy whose prediction has the highest confidence decides the next action to be taken.

    Rasa Core
        The dialogue engine that decides on what to do next in a conversation based on the context.

    Rasa NLU Component
        An element in the Rasa NLU pipeline (see "Pipeline").

        Incoming messages are processed by a sequence of components called a pipeline. A component can perform tasks ranging from entity extraction to intent classification to pre-processing.

    Slot
        A key-value store that Rasa uses to track information over the course of a conversation.

        For the different types of slots, see :ref:`here <slot-classes>`.

    Story
        A conversation between a user and a bot annotated with the intent / entities of the users' messages as well as the sequence of actions to be performed by the bot

    Template / Response / Utterance
        A message template that is used to respond to a user. This can include text, buttons, images, and other attachments.

    User Goal
        A goal that a user wants to achieve.

        For example, a user may have the goal of booking a table at a restaurant. Another user may just want to make small talk.  Sometimes, the user expresses their goal with a single message, e.g. "I want to book a table at a restaurant". Other times the assistant may have to ask a few questions to understand how to help the user.  Note: Many other places refer to the user goal as the "intent", but in Rasa terminology, an intent is associated with every user message.

    Utterance
        A type of bot action that only involves sending a pre-defined message back to the user (without running any other code)

    Word embedding / Word vector
        A vector of floating point numbers which represent the meaning of a word. Words which have similar meanings should have vectors which point in almost the same direction.  Word embeddings are often used as an input to machine learning algorithms.


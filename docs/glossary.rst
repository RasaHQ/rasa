:desc: Glossary for all Rasa-related terms

.. _glossary:

Glossary
========

.. glossary::

    Action
        a single step a bot takes in a conversation (e.g. call an API or send a response back to the user)

    Annotation
        could either refer to the process of labelling data or the label itself

    CMS
        A Content Management System (CMS) can be used to store responses externally instead of hardcoding it as part of the domain.

    Conversation
        a back and forth between a user and a bot (so usually just messages - unlabeled conversation)

    Custom Action
        an action written by a Rasa developer that can run arbitrary code mainly to interact with the outside world

    Default Actions
        built-in actions that carry a special meaning to Rasa

    Domain
        outlines the capabilities of a conversational assistant

        While this primarily includes the intents that it can recognize and the actions it can perform, it also encompasses entities, slots, forms, and templates for the bot.

    Entities
        structured information that can be extracted from a user message (e.g. a telephone number)

    Event
        All conversations in Rasa are represented as a sequence of events. For instance, a ``UserUttered`` represents a user entering a message, and ``SlotSet`` represents the act of setting a slot to a certain value. You can learn more about them :ref:`here <events>`.

    Forms
        One of the most common conversation patterns is slot filling - that is, collecting a few pieces of information from a user in order to do something (book a restaurant, call an API, search a database, etc.). A Rasa form enables bot developers to declaratively define the fields and validation rules for each of them.

    Happy / Unhappy Paths
        Happy paths refer to conversations where everything proceeds smoothly as intended, and there are no deviations. Unhappy paths involve things like the user changing the topic of conversation, correcting a previous input, etc.

    Intent
        meaning of a user's message selected as one of a set of predefined meanings (e.g. greeting).

    Interactive Learning
        a mode of training the bot where the user provides feedback to the bot while talking to it

        This is a powerful way to explore what a bot can do, and the easiest way to fix any mistakes it make.

    NLG
        Natural Language Generation (NLG) is the component that generates natural language output from structured data.

    NLU
        Natural Language Understanding (NLU) deals with parsing and understanding human language into a structured format. Rasa NLU is the part of Rasa that performs intent classification and entity extraction.

    Pipeline
        A Rasa bot's NLU system is defined by a pipeline, which is a list of NLU components in a particular order. A user input is processed by each component one by one before finally giving out the structured output.

    Policy
        Policies make decisions on how conversation flow should proceed. A Core model can have multiple policies included, and the policy whose prediction has the highest confidence decides the next action to be taken.

    Rasa Core
        the dialogue engine that decides on what to do next in a conversation

    Rasa Core policy
        decides which action to take at every step in the conversation

        There are different policies to choose from, and multiple policies can be included in a single agent. At every turn, the policy which predicts the next action with the highest confidence will be used. If two policies predict with equal confidence, the policy with the higher priority will be used.

    Rasa NLU component
        an element in the Rasa NLU pipeline

        Incoming messages are processed by a sequence of components called a pipeline. These components are executed one after another in a so-called processing pipeline. A component can perform tasks ranging from entity extraction to intent classification to pre-processing.

    Slots
        memory of the bot where structured information is stores (e.g. extracted entities or results from API calls)

    spaCy
        a free, open-source library for advanced Natural Language Processing (NLP) in Python

    Story
        a user/bot conversation annotated with the intent / entities of the users' messages as well as the sequence of actions performed by the bot

    Template / Response
        a message a bot can respond back to the user with

        This is not just limited to text, which means it can include buttons, images, and other attachments.

    Training
        using labeled data to train a model

    User Goal
        small reusable parts of stories (e.g. setting a reminder, filling out a form, answering FAQs)

    User goal
        a goal that a user wants to achieve

        This can also be ambiguous / unclear in the beginning (e.g. buying an insurance policy). You might need multiple skills to achieve a user goal.

    Utterance
        a type of bot action that only involves sending a pre-defined message back to the user (without running any other code)

    Word embedding
        numeric representations of natural language words in a high-dimensional space

        They are often used as an input to machine learning algorithms.


:desc: Glossary for all Rasa-related terms

.. _glossary:

Glossary
========

.. glossary::

    (Un)Happy Paths
        Happy paths refer to conversations where everything proceeds smoothly as intended, and there are no deviations. Unhappy paths involve things like the user changing the topic conversation, correcting a previous input, etc.

    Action
        A single step a bot takes in a conversation (e.g. call an API or send a response).

    Annotation
        Labeling data.

    CMS
        Content Management System (CMS) can be used to store responses externally instead of hardcoding it as part of the domain.

    Command Line Interface
        Easy-to-remember commands for common tasks

    Conversation
        Actual user-bot messages - a back and forth between a user and a bot (so usually just messages - unlabeled conversation).

    Custom Action
        An action written by a Rasa developer that can run arbitrary code mainly to interact with the outside world.

    Default Actions
        Built-in actions that carry a special meaning to Rasa.

    Domain
        A domain 

    Entities
        Structured information that can be extracted from a user message (e.g. telephone number).

    Event
        Conversations in Rasa are represented as a sequence of events. 

    Forms
        One of the most common conversation patterns is to collect a few pieces of information from a user in order to do something (book a restaurant, call an API, search a database, etc.). This is also called slot filling.

    Intent
        Meaning of a users message selected as one of a set of predefined meanings (e.g. greeting).

    Interactive Learning
        You provide feedback to your bot while you talk to it. This is a powerful way to explore what your bot can do, and the easiest way to fix any mistakes it make.

    NLG
        Natural Language Generation (NLG) is the component that generates natural language output from structured data.

    NLU
        Natural Language Understanding (NLU) deals with parsing and understanding human language into a structured format. Rasa NLU is the part of Rasa that performs intent classification and entity extraction.

    Open Source Software
        Software with source code that anyone can inspect, modify, and enhance.

    Pipeline
        A Rasa bot's NLU system is defined by a pipeline, which is a list of NLU components in a particular order. A user input is processed by each component one by one before finally giving out the structured output.

    Policy
        Policies make decisions on how conversation flow should proceed. A Core model can have multiple policies included, and the policy whose prediction has the highest confidence decides the next action to be taken.

    Rasa Core
        The dialogue engine that decides on what to do next in a conversation.

    Rasa Core policy
        Decides which action to take at every step in the conversation. There are different policies to choose from, and you can include multiple policies in a single agent. At every turn, the policy which predicts the next action with the highest confidence will be used. If two policies predict with equal confidence, the policy with the higher priority will be used.

    Rasa NLU component
        Allows you to customize your model and finetune it on your dataset. Incoming messages are processed by a sequence of components called a pipeline. These components are executed one after another in a so-called processing pipeline. There are components for entity extraction, for intent classification, pre-processing, and others.

    Repository
        A folder for your project. Your project's repository contains all of your project's files and stores each file's revision history. You can also discuss and manage your project's work within the repository.

    Skill
        Small reusable parts of stories (e.g. setting a reminder, filling out a form, answering FAQs).

    Slots
        Memory of the bot, used to store structured information (e.g. extracted entities or results from API calls).

    spaCy
        A free, open-source library for advanced Natural Language Processing (NLP) in Python.

    Story
        Labeled conversation - an annotated / labeled conversation (so this includes the the intent / entities of the user as well as the sequence of actions and slots set) - stories can span across multiple skills.

    Template / Response
        An utterance is defined by a template, which could contain text, images, buttons, and other attachments.

    Training
        Using labeled data to train a model.

    User goal
        A goal that a user wants to achieve - this can also be ambiguous / unclear in the beginning (e.g. buying an insurance policy) - to achieve a user goal, you might need multiple skills.

    Utterance
        A type of bot action that only involves sending a pre-defined message back to the user (without running any other code).

    Word embedding
        A dense representation of a word often used as an input to machine learning algorithms.


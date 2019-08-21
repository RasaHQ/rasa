:desc: Glossary for all Rasa-related terms

.. _glossary:

Glossary
========

.. glossary::

    User goal
        A goal that a user wants to achieve - this can also be ambiguous / unclear in the beginning (e.g. buying an insurance policy) - to achieve a user goal, you might need multiple skills.

    Conversation
        Actual user-bot messages - a back and forth between a user and a bot (so usually just messages - unlabeled conversation).

    Story
        Labeled conversation - an annotated / labeled conversation (so this includes the the intent / entities of the user as well as the sequence of actions and slots set) - stories can span across multiple skills.

    Skill
        Small reusable parts of stories (e.g. setting a reminder, filling out a form, answering FAQs).

    Action
        A single step a bot takes in a conversation (e.g. call an API or send a response).

    Annotation
        Labeling data.

    Training
        Using labeled data to train a model.

    Intent
        Meaning of a users message selected as one of a set of predefined meanings (e.g. greeting).

    Entities
        Structured information that can be extracted from a user message (e.g. telephone number).

    Slots
        Memory of the bot, used to store structured information (e.g. extracted entities or results from API calls).

    Open source software
        Software with source code that anyone can inspect, modify, and enhance.

    Repository
        A folder for your project. Your project's repository contains all of your project's files and stores each file's revision history. You can also discuss and manage your project's work within the repository.

    Interactive Learning
        You provide feedback to your bot while you talk to it. This is a powerful way to explore what your bot can do, and the easiest way to fix any mistakes it make.

    Forms
        One of the most common conversation patterns is to collect a few pieces of information from a user in order to do something (book a restaurant, call an API, search a database, etc.). This is also called slot filling.

    Rasa NLU component
        Allows you to customize your model and finetune it on your dataset. Incoming messages are processed by a sequence of components called a pipeline. These components are executed one after another in a so-called processing pipeline. There are components for entity extraction, for intent classification, pre-processing, and others.

    Rasa Core policy
        Decides which action to take at every step in the conversation. There are different policies to choose from, and you can include multiple policies in a single agent. At every turn, the policy which predicts the next action with the highest confidence will be used. If two policies predict with equal confidence, the policy with the higher priority will be used.

    Command Line Interface
        Easy-to-remember commands for common tasks




:desc: Training data format for Rasa 2.0

.. _training-data-format:

====================
Training Data Format
====================

.. edit-link::

.. contents::
   :local:


YAML Format
-----------

Rasa Open Source 2.0 uses `YAML <https://yaml.org/spec/1.2/spec.html>`_ as a unified and extendable way to
manage all kinds of training data, including NLU data, stories, and responses. 
With the YAML format, users are free to distribute training data among any number of YAML files; only the top level keys determine what kind of training data is in a section. That means that if any one component of training data gets too large (e.g. a lookup table with many entries), it can be moved to a separate file.


High Level Structure
~~~~~~~~~~~~~~~~~~~~

Each file can contain one or more **keys** with corresponding training data. One file can contain multiple keys, as long as there is not more than one of a certain key in a single file. The available keys are:

- ``version``
- ``stories``
- ``nlu``
- ``responses``
- ``rules``
- ``e2e_tests``

All YAML training data files should specify the ``version`` key in order to be parsed correctly:

- Training data files with no ``version`` specified will be assumed to be in the format of the latest version of Rasa Open Source.
- Training data files with a version greater than is currently available for Rasa Open Source will be skipped.


Example
~~~~~~~

Here's a short example which keeps all training data in a single file:

.. code::

    version: "2.0"

    stories:
    - story: greet and faq
      steps:
      - intent: greet
      - action: utter_greet
      - intent: faq
      - action: respond_faq

    rules:
    - rule: Greet user
      steps:
      - intent: greet
      - action: utter_greet

    responses:
      faq/language:
      - text |
        I can only do English at the moment
  
      utter_greet:
      - text: |
        Hallo there!
  
    nlu:
    - intent: greet
      examples: |
      - Hallo
      - Hi
    
    - intent: faq/language
      examples: |
      - What language do you speak?
      - Do you only handle english?

    e2e_tests:
    - user: |
        hello
      intent: greet
    - action: utter_greet
    - user: |
       what language do you speak
      intent: faq/language
    - action: respond_faq

.. _story-training-data-format:

Stories
-------

A story is a representation of a conversation between a user and an AI assistant. 
User messages are expressed as corresponding :ref:`intents` (and entities where necessary) while the assistant's responses are expressed with the corresponding :ref:`action` names.

Stories are formatted as follows:

.. code:: YAML

    stories:
    - story: <story name>
      metadata:
        <any_key>: <any value>
        <another_key>: <another value>
      steps:
      - intent: <user's intent>
        entities: 
         - <entity_name>: <entity_value>
         - <entity_name>
      - action: <bot action>

What makes up a story?
~~~~~~~~~~~~~~~~~~~~~~

User Messages
*************

While writing stories, you do not have to deal with the specific contents of
the messages that the users send. Instead, you can take advantage of the output
from the NLU pipeline, which lets you use just the combination of an intent and
entities to refer to all the possible messages the users can send to mean the
same thing.

It is important to include the entities here as well because the policies learn
to predict the next action based on a *combination* of both the intent and
entities (you can, however, change this behavior using the
:ref:`use_entities <use_entities>` attribute).

Actions
~~~~~~~
While writing stories, you will encounter two types of actions: utterance actions
and custom actions. Utterance actions are hardcoded messages that a bot can respond
with. Custom actions, on the other hand, involve custom code being executed.

All actions (both utterance actions and custom actions) executed by the bot are specified with the ``action:`` key followed by the name of the action.

The responses for utterance actions must begin with the prefix ``utter_``, and must match the name
of the response defined in the domain.

For custom actions, the action name is the string you choose to return from
the ``name`` method of the custom action class. Although there is no restriction
on naming your custom actions (unlike utterance actions), the best practice here is to
prefix the name with ``action_``

If you use the :ref:`ReponseSelector <>`, you'll also encounter `respond_` actions. 


Entities and Slots
~~~~~~~~~~~~~~~~~~




Checkpoints
~~~~~~~~~~~

ORs
~~~



.. _rule-training-data-format:

Rules
-----

.. code:: YAML

    rules:
    - rule: Greet user
      steps:
      - intent: greet
      - action: utter_greet



.. _nlu-training-data-format:

NLU Training Data
-----------------

NLU training data is defined under the ``nlu`` key.
This section includes
  
  - Training examples grouped by :ref:`intent-training-data-format`, optionally with annotated :ref:`entity-training-data-format`
  - :ref:`synonym-training-data-format`
  - :ref:`regex-training-data-format`
  - :ref:`lookup-table-training-data-format`
  
Here is a simple example of each item type:

  .. code:: YAML

    nlu:
    - intent: greet
      examples: |
        - hey
        - hello

    - synonym: credit
      examples: |
      - credit card account
      - credit account

    - regex: zipcode
      examples: |
        - [0-9]{5}

    - lookup: additional_currencies
      examples: |
        - Peso
        - Euro
        - Dollar

.. _intent-training-data-format:

Intents
~~~~~~~

Training examples are grouped by :ref:`intent <>` and listed under the ``examples`` key. Examples can be provided in one of two formats:

  1. As a list of text values. For example:

  .. code:: YAML

    nlu:
    - intent: greet
      examples: |
        - hey
        - hallo
        - whats up

  2. As a list of dictionaries, with at least the ``text`` key specified. For example:

  .. code:: YAML

    nlu:
    - intent: greet
      examples: 
        - text: |
            hallo
          metadata:
            sentiment: neutral
        - text: |
            hey there!


  Note the ``metadata`` key inside  the ``examples`` dictionary. The Rasa Open Source parser will not read its value and you can use it to store any information relevant to the example.

.. _entity-training-data-format:

Entities
~~~~~~~~

Where applicable, entites are annotated in training examples using the syntax: 

.. code:: YAML

    [<entity-text>]{"entity": "<entity name>"}


In a training example, this would look like:

.. code:: YAML

    nlu:
    - intent: check_balance
      examples: |
        - how much do I have on my [savings]{"entity": "account"} account
        - how much money is in my [checking]{"entity": "account"} account


You can also assign synonyms, roles, or groups to an entity using the syntax:

.. code:: YAML

    [<entity-text>]{"entity": "<entity name>", "role": "<role name>", "group": "<group name>", "value": "<entity synonym>"}

The keywords ``role``, ``group``, and ``value`` are optional in this notation.
The ``value`` keyword refers to synonyms, which are explained in the following section.
To understand what the labels ``role`` and ``group`` are for, see section :ref:`entities-roles-groups`. 


.. _synonym-training-data-format:

Synonyms
~~~~~~~~

Synonyms provide a way to normalize your training data by mapping an extracted entity to a value other than the literal text extracted. Synonyms can be defined in the format:

.. code:: YAML

    - synonym: <synonym value>
      examples: |
      - <a synonym variation>
      - <another synonym variation>

Synonyms can also be definedin-line in your training examples by specifying the ``value`` of the entity:

.. code:: YAML

    nlu:
    - intent: check_balance
      examples: |
        - how much do I have on my [credit card account]{"entity": "account", "value": "credit"}
        - how much do I owe on my [credit account]{"entity": "account", "value": "credit"}

To use the synonyms defined in your training data, you need to make sure the pipeline contains the
``EntitySynonymMapper`` component (see :ref:`components`).
You should define synonyms when there are multiple ways users refer to the same thing. 

For example, let's say you had an entity ``account_type``, and you expect the value "credit".  Your users also refer to their "credit" account as "credit account" and "credit card account". 
  
In this case, you could define "credit card account" and "credit account" as **synonyms** to "credit":

.. code:: YAML

    - synonym: credit
      examples: |
      - credit card account
      - credit account

Then, if either of these phrases is extracted as an entity, it will be mapped to the **value** ``credit``.

Synonym mapping only happens **after** entities have been extracted. That means that in addition to defining your synonyms, you need to provide examples of the variations on a synonym 
as :ref:`entities in your training examples <entity-training-data-format>` so that Rasa Open Source can learn to pick them up.


.. _regex-training-data-format:

Regular Expression Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regular expressions can be used in two different ways:

1. They can be used to support intent classification and entity extraction when using the :ref:`RegexFeaturizer`
component in the pipeline.

2. They can be used to directly extract entities from a user messages when using the :ref:`RegexEntityExtractor`
component in the pipeline.

For example, if your entity has a deterministic structure (like a zipcode or an email address), you can use a regular
expression to ease detection of that entity (using the :ref:`RegexFeaturizer`) or to directly extract the entities from
the user message (using the :ref:`RegexEntityExtractor`). For the zipcode example it might look like this:

.. code:: YAML

    - regex: zipcode
      examples: |
        - [0-9]{5}


If you are using regular expressions to directly extract entities using the :ref:`RegexEntityExtractor`, the name
of the regular expression should match the name of the entity you want to extract.

If you are using the regular expressions for the :ref:`RegexFeaturizer` the name of the regular expression does
not matter. If does not define the entity nor the intent, it is just a human readable description for you to remember
what this regex is used for and is the title of the corresponding pattern feature.

If you want to use the :ref:`RegexFeaturizer` you can also use the regex features to improve the intent
classification performance, for example, by defining a greet clause:

.. code:: YAML

    - regex: greet
      examples: |
        - hey[^\\s]*

Try to create your regular expressions in a way that they match as few words as possible. E.g. using ``hey[^\\s]*``
instead of ``hey.*``, as the later one might match the whole message whereas the first one only matches a single word.

When using the :ref:`RegexFeaturizer`, the regex features for entity extraction are currently only supported by the
``CRFEntityExtractor`` and the ``DIETClassifier`` component! Hence, other entity extractors, like
``MitieEntityExtractor`` or ``SpacyEntityExtractor`` won't use the generated features and their
presence will not improve entity recognition for these extractors. Currently, all intent classifiers make use of
available regex features.

.. note::
    Regex features only define entities when used in combination with the :ref:`RegexEntityExtractor`. Otherwise they
    don't define entities nor intents! They simply provide patterns to help the classifier
    recognize entities and related intents. Hence, you still need to provide intent & entity examples as part of your
    training data!


.. _lookup-table-training-data-format:

Lookup Tables
~~~~~~~~~~~~~

Lookup tables provide a convenient way to supply a list of entity examples. The format is as follows:

.. code:: 

    - lookup: <lookup table name>
      examples: |
        - <an entity>
        - <another entity>

The name of the lookup table is subject to the same constraints as the name of a regex feature.

When lookup tables are supplied in training data, the contents are combined
into a large regex pattern that looks for exact matches in
the training examples. These regexes match over multiple tokens, so
``lettuce wrap`` would match ``get me a lettuce wrap ASAP`` as ``[0 0 0 1 1 0]``.
These regexes are processed identically to the regular regex patterns
directly specified in the training data.

.. note::
    If you are using lookup tables in combination with the :ref:`RegexFeaturizer`, there must be a few examples of matches
    in your training data. Otherwise the model will not learn to use the lookup table match features.

.. warning::
    You have to be careful when you add data to the lookup table.
    For example, if there are false positives or other noise in the table,
    this can hurt performance. So make sure your lookup tables contain
    clean data.



.. _responses-training-data-format:

Responses
---------



.. _test-conversation-training-data-format:

Test Conversations
------------------
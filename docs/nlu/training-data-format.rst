:desc: Read more about how to format training data with Rasa NLU for open a
       source natural language processing. a
 a
.. _training-data-format: a
 a
Training Data Format a
==================== a
 a
.. edit-link:: a
 a
.. contents:: a
   :local: a
 a
Data Formats a
~~~~~~~~~~~~ a
 a
 a
You can provide training data as Markdown or as JSON, as a single file or as a directory containing multiple files. a
Note that Markdown is usually easier to work with. a
 a
 a
Markdown Format a
--------------- a
 a
Markdown is the easiest Rasa NLU format for humans to read and write. a
Examples are listed using the unordered list syntax, e.g. minus ``-``, asterisk ``*``, or plus ``+``. a
Examples are grouped by intent, and entities are annotated as Markdown links, a
e.g. ``[<entity text>](<entity name>)``, or by using the following syntax ``[<entity-text>]{"entity": "<entity name>"}``. a
Using the latter syntax, you can also assign synonyms, roles, or groups to an entity, e.g. a
``[<entity-text>]{"entity": "<entity name>", "role": "<role name>", "group": "<group name>", "value": "<entity synonym>"}``. a
The keywords ``role``, ``group``, and ``value`` are optional in this notation. a
To understand what the labels ``role`` and ``group`` are for, see section :ref:`entities-roles-groups`. a
 a
.. code-block:: md a
 a
    ## intent:check_balance a
    - what is my balance <!-- no entity --> a
    - how much do I have on my [savings](source_account) <!-- entity "source_account" has value "savings" --> a
    - how much do I have on my [savings account]{"entity": "source_account", "value": "savings"} <!-- synonyms, method 1--> a
    - Could I pay in [yen](currency)?  <!-- entity matched by lookup table --> a
 a
    ## intent:greet a
    - hey a
    - hello a
 a
    ## synonym:savings   <!-- synonyms, method 2 --> a
    - pink pig a
 a
    ## regex:zipcode a
    - [0-9]{5} a
 a
    ## lookup:additional_currencies  <!-- specify lookup tables in an external file --> a
    path/to/currencies.txt a
 a
The training data for Rasa NLU is structured into different parts: a
 a
- common examples a
- synonyms a
- regex features and a
- lookup tables a
 a
While common examples is the only part that is mandatory, including the others will help the NLU model a
learn the domain with fewer examples and also help it be more confident of its predictions. a
 a
Synonyms will map extracted entities to the same name, for example mapping "my savings account" to simply "savings". a
However, this only happens *after* the entities have been extracted, so you need to provide examples with the synonyms a
present so that Rasa can learn to pick them up. a
 a
Lookup tables may be specified as plain text files containing newline-separated words or  a
phrases. Upon loading the training data, these files are used to generate a
case-insensitive regex patterns that are added to the regex features. a
 a
.. note:: a
    The common theme here is that common examples, regex features and lookup tables merely act as cues to the final NLU a
    model by providing additional features to the machine learning algorithm during training. Therefore, it must not be a
    assumed that having a single example would be enough for the model to robustly identify intents and/or entities a
    across all variants of that example. a
 a
.. note:: a
    ``/`` symbol is reserved as a delimiter to separate retrieval intents from response text identifiers. Make sure not a
    to use it in the name of your intents. a
 a
.. warning:: a
    The synonym format to specify synonyms ``[savings account](source_account:savings)`` is deprecated. Please use the a
    new format ``[savings account]{"entity": "source_account", "value": "savings"}``. a
 a
    To update your training data file execute the following command on the terminal of your choice: a
    ``sed -i -E 's/\[([^)]+)\]\(([^)]+):([^)]+)\)/[\1]{"entity": "\2", "value": "\3"}/g' <nlu training data file>`` a
    Your NLU training data file will contain the new training data format after you executed the above command. a
    Depending on your OS you might need to update the syntax of the sed command. a
 a
JSON Format a
----------- a
 a
The JSON format consists of a top-level object called ``rasa_nlu_data``, with the keys a
``common_examples``, ``entity_synonyms`` and ``regex_features``. a
The most important one is ``common_examples``. a
 a
.. code-block:: json a
 a
    { a
        "rasa_nlu_data": { a
            "common_examples": [], a
            "regex_features" : [], a
            "lookup_tables"  : [], a
            "entity_synonyms": [] a
        } a
    } a
 a
The ``common_examples`` are used to train your model. You should put all of your training a
examples in the ``common_examples`` array. a
Regex features are a tool to help the classifier detect entities or intents and improve the performance. a
 a
 a
Improving Intent Classification and Entity Recognition a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
Common Examples a
--------------- a
 a
Common examples have three components: ``text``, ``intent`` and ``entities``. The first two are strings while the last a
one is an array. a
 a
 - The *text* is the user message [required] a
 - The *intent* is the intent that should be associated with the text [optional] a
 - The *entities* are specific parts of the text which need to be identified [optional] a
 a
Entities are specified with a ``start`` and an ``end`` value, which together make a range a
to apply to the string, e.g. in the example below, with ``text="show me chinese restaurants"``, then a
``text[8:15] == 'chinese'``. Entities can span multiple words, and in a
fact the ``value`` field does not have to correspond exactly to the substring in your example. a
That way you can map synonyms, or misspellings, to the same ``value``. a
 a
.. code-block:: md a
 a
    ## intent:restaurant_search a
    - show me [chinese](cuisine) restaurants a
 a
 a
Regular Expression Features a
--------------------------- a
Regular expressions can be used to support the intent classification and entity extraction. For example, if your entity a
has a deterministic structure (like a zipcode or an email address), you can use a regular expression to ease detection a
of that entity. For the zipcode example it might look like this: a
 a
.. code-block:: md a
 a
    ## regex:zipcode a
    - [0-9]{5} a
 a
    ## regex:greet a
    - hey[^\\s]* a
 a
The name doesn't define the entity nor the intent, it is just a human readable description for you to remember what a
this regex is used for and is the title of the corresponding pattern feature. As you can see in the above example, you a
can also use the regex features to improve the intent a
classification performance. a
 a
Try to create your regular expressions in a way that they match as few words as possible. E.g. using ``hey[^\s]*`` a
instead of ``hey.*``, as the later one might match the whole message whereas the first one only matches a single word. a
 a
Regex features for entity extraction are currently only supported by the ``CRFEntityExtractor`` component! Hence, other a
entity extractors, like ``MitieEntityExtractor`` or ``SpacyEntityExtractor`` won't use the generated features and their a
presence will not improve entity recognition for these extractors. Currently, all intent classifiers make use of a
available regex features. a
 a
.. note:: a
    Regex features don't define entities nor intents! They simply provide patterns to help the classifier a
    recognize entities and related intents. Hence, you still need to provide intent & entity examples as part of your a
    training data! a
 a
.. _lookup-tables: a
 a
Lookup Tables a
------------- a
Lookup tables provide a convenient way to supply a list of entity examples. a
The supplied lookup table files must be in a newline-delimited format. a
For example, ``data/test/lookup_tables/plates.txt`` may contain: a
 a
.. literalinclude:: ../../data/test/lookup_tables/plates.txt a
 a
And can be loaded and used as shown here: a
 a
.. code-block:: md a
 a
    ## lookup:plates a
    data/test/lookup_tables/plates.txt a
 a
    ## intent:food_request a
    - I'd like beef [tacos](plates) and a [burrito](plates) a
    - How about some [mapo tofu](plates) a
 a
When lookup tables are supplied in training data, the contents are combined a
into a large, case-insensitive regex pattern that looks for exact matches in a
the training examples. These regexes match over multiple tokens, so a
``lettuce wrap`` would match ``get me a lettuce wrap ASAP`` as ``[0 0 0 1 1 0]``. a
These regexes are processed identically to the regular regex patterns a
directly specified in the training data. a
 a
.. note:: a
    For lookup tables to be effective, there must be a few examples of matches in your training data. Otherwise the a
    model will not learn to use the lookup table match features. a
 a
 a
.. warning:: a
    You have to be careful when you add data to the lookup table. a
    For example if there are false positives or other noise in the table, a
    this can hurt performance. So make sure your lookup tables contain a
    clean data. a
 a
 a
Normalizing Data a
~~~~~~~~~~~~~~~~ a
 a
.. _entity_synonyms: a
 a
Entity Synonyms a
--------------- a
If you define entities as having the same value they will be treated as synonyms. Here is an example of that: a
 a
.. code-block:: md a
 a
    ## intent:search a
    - in the center of [NYC]{"entity": "city", "value": "New York City") a
    - in the centre of [New York City](city) a
 a
 a
As you can see, the entity ``city`` has the value ``New York City`` in both examples, even though the text in the first a
example states ``NYC``. By defining the value attribute to be different from the value found in the text between start a
and end index of the entity, you can define a synonym. Whenever the same text will be found, the value will use the a
synonym instead of the actual text in the message. a
 a
To use the synonyms defined in your training data, you need to make sure the pipeline contains the a
``EntitySynonymMapper`` component (see :ref:`components`). a
 a
Alternatively, you can add an "entity_synonyms" array to define several synonyms to one entity value. Here is an a
example of that: a
 a
.. code-block:: md a
 a
    ## synonym:New York City a
    - NYC a
    - nyc a
    - the big apple a
 a
.. note:: a
    Please note that adding synonyms using the above format does not improve the model's classification of those entities. a
    **Entities must be properly classified before they can be replaced with the synonym value.** a
 a
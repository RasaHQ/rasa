:desc: Test your Rasa Open Source assistant to validate and improve your a 
       conversations a 

.. _testing-your-assistant:

Testing Your Assistant a 
======================

.. edit-link::

.. contents::
   :local:

.. note::
   If you are looking to tune the hyperparameters of your NLU model,
   check out this `tutorial <https://blog.rasa.com/rasa-nlu-in-depth-part-3-hyperparameters/>`_.

.. _end_to_end_testing:

End-to-End Testing a 
------------------

Rasa Open Source lets you test dialogues end-to-end by running through a 
test conversations and making sure that both NLU and Core make correct predictions.

To do this, you need some stories in the end-to-end format,
which includes both the NLU output and the original text.
Here are some examples:

.. tabs::

 .. group-tab:: Basics a 

    .. code-block:: story a 

          ## A basic end-to-end test a 
          * greet: hello a 
             - utter_ask_howcanhelp a 
          * inform: show me [chinese](cuisine) restaurants a 
             - utter_ask_location a 
          * inform: in [Paris](location)
             - utter_ask_price a 

 .. group-tab:: Custom Actions a 

    .. code-block:: story a 

        ## End-to-End tests where a custom action appends events a 
        * greet: hi a 
            - my_custom_action a 
            <!-- The following events are emitted by `my_custom_action` -->
            - slot{"my_slot": "value added by custom action"}
            - utter_ask_age a 
        * thankyou: thanks a 
            - utter_noworries a 

 .. group-tab:: Forms Happy Path a 

    .. code-block:: story a 

        ## Testing a conversation with a form a 
        * greet: hi a 
            - utter_greet a 
        * request_restaurant: im looking for a restaurant a 
            - restaurant_form a 
            - form{"name": "restaurant_form"}
        * inform: [afghan](cuisine) food a 
            - form: restaurant_form a 
            - form{"name": null}
            - utter_slots_values a 
        * thankyou: thanks a 
            - utter_noworries a 

 .. group-tab:: Forms Unhappy Path a 

    .. code-block:: story a 

        ## Testing a conversation with a form and unexpected user input a 
        * greet: hi a 
            - utter_greet a 
        * request_restaurant: im looking for a restaurant a 
            - restaurant_form a 
            - form{"name": "restaurant_form"}
        <!-- The user sends a message which should not be handled by the form. -->
        * chitchat: can you share your boss with me?
            - utter_chitchat a 
            - restaurant_form a 
            - form{"name": null}
            - utter_slots_values a 
        * thankyou: thanks a 
            - utter_noworries a 

By default Rasa Open Source saves conversation tests to ``tests/conversation_tests.md``.
You can test your assistant against them by running:

.. code-block:: bash a 

  $ rasa test a 

.. note::

  :ref:`custom-actions` are not executed as part of end-to-end tests. If your custom a 
  actions append any events to the tracker, this has to be reflected in your end-to-end a 
  tests (e.g. by adding ``slot`` events to your end-to-end story).

If you have any questions or problems, please share them with us in the dedicated a 
`testing section on our forum <https://forum.rasa.com/tags/testing>`_ !

.. note::

  Make sure your model file in ``models`` is a combined ``core``
  and ``nlu`` model. If it does not contain an NLU model, Core will use a 
  the default ``RegexInterpreter``.

.. _nlu-evaluation:

Evaluating an NLU Model a 
-----------------------

A standard technique in machine learning is to keep some data separate as a *test set*.
You can :ref:`split your NLU training data <train-test-split>`
into train and test sets using:

.. code-block:: bash a 

   rasa data split nlu a 


If you've done this, you can see how well your NLU model predicts the test cases using this command:

.. code-block:: bash a 

   rasa test nlu -u train_test_split/test_data.md --model models/nlu-20180323-145833.tar.gz a 


If you don't want to create a separate test set, you can a 
still estimate how well your model generalises using cross-validation.
To do this, add the flag ``--cross-validation``:

.. code-block:: bash a 

   rasa test nlu -u data/nlu.md --config config.yml --cross-validation a 

The full list of options for the script is:

.. program-output:: rasa test nlu --help a 

.. _comparing-nlu-pipelines:

Comparing NLU Pipelines a 
^^^^^^^^^^^^^^^^^^^^^^^

By passing multiple pipeline configurations (or a folder containing them) to the CLI, Rasa will run a 
a comparative examination between the pipelines.

.. code-block:: bash a 

  $ rasa test nlu --config pretrained_embeddings_spacy.yml supervised_embeddings.yml a 
    --nlu data/nlu.md --runs 3 --percentages 0 25 50 70 90 a 


The command in the example above will create a train/test split from your data,
then train each pipeline multiple times with 0, 25, 50, 70 and 90% of your intent data excluded from the training set.
The models are then evaluated on the test set and the f1-score for each exclusion percentage is recorded. This process a 
runs three times (i.e. with 3 test sets in total) and then a graph is plotted using the means and standard deviations of a 
the f1-scores.

The f1-score graph - along with all train/test sets, the trained models, classification and error reports - will be saved into a folder a 
called ``nlu_comparison_results``.


Intent Classification a 
^^^^^^^^^^^^^^^^^^^^^

The evaluation script will produce a report, confusion matrix,
and confidence histogram for your model.

The report logs precision, recall and f1 measure for a 
each intent and entity, as well as providing an overall average.
You can save these reports as JSON files using the ``--report`` argument.

The confusion matrix shows you which a 
intents are mistaken for others; any samples which have been a 
incorrectly predicted are logged and saved to a file a 
called ``errors.json`` for easier debugging.

The histogram that the script produces allows you to visualise the a 
confidence distribution for all predictions,
with the volume of correct and incorrect predictions being displayed by a 
blue and red bars respectively.
Improving the quality of your training data will move the blue a 
histogram bars to the right and the red histogram bars a 
to the left of the plot.


.. warning::
    If any of your entities are incorrectly annotated, your evaluation may fail. One common problem a 
    is that an entity cannot stop or start inside a token.
    For example, if you have an example for a ``name`` entity a 
    like ``[Brian](name)'s house``, this is only valid if your tokenizer splits ``Brian's`` into a 
    multiple tokens. A whitespace tokenizer would not work in this case.


Response Selection a 
^^^^^^^^^^^^^^^^^^^^^

The evaluation script will produce a combined report for all response selector models in your pipeline.

The report logs precision, recall and f1 measure for a 
each response, as well as providing an overall average.
You can save these reports as JSON files using the ``--report`` argument.


Entity Extraction a 
^^^^^^^^^^^^^^^^^

The ``CRFEntityExtractor`` is the only entity extractor which you train using your own data,
and so is the only one that will be evaluated. If you use the spaCy or duckling a 
pre-trained entity extractors, Rasa NLU will not include these in the evaluation.

Rasa NLU will report recall, precision, and f1 measure for each entity type that a 
``CRFEntityExtractor`` is trained to recognize.


Entity Scoring a 
^^^^^^^^^^^^^^

To evaluate entity extraction we apply a simple tag-based approach. We don't consider BILOU tags, but only the a 
entity type tags on a per token basis. For location entity like "near Alexanderplatz" we a 
expect the labels ``LOC LOC`` instead of the BILOU-based ``B-LOC L-LOC``. Our approach is more lenient a 
when it comes to evaluation, as it rewards partial extraction and does not punish the splitting of entities.
For example, given the aforementioned entity "near Alexanderplatz" and a system that extracts a 
"Alexanderplatz", our approach rewards the extraction of "Alexanderplatz" and punishes the missed out word "near".
The BILOU-based approach, however, would label this as a complete failure since it expects Alexanderplatz a 
to be labeled as a last token in an entity (``L-LOC``) instead of a single token entity (``U-LOC``). Note also that a 
a split extraction of "near" and "Alexanderplatz" would get full scores on our approach and zero on the a 
BILOU-based one.

Here's a comparison between the two scoring mechanisms for the phrase "near Alexanderplatz tonight":

==================================================  ========================  ===========================
extracted                                           Simple tags (score)       BILOU tags (score)
==================================================  ========================  ===========================
[near Alexanderplatz](loc) [tonight](time)          loc loc time (3)          B-loc L-loc U-time (3)
[near](loc) [Alexanderplatz](loc) [tonight](time)   loc loc time (3)          U-loc U-loc U-time (1)
near [Alexanderplatz](loc) [tonight](time)          O   loc time (2)          O     U-loc U-time (1)
[near](loc) Alexanderplatz [tonight](time)          loc O   time (2)          U-loc O     U-time (1)
[near Alexanderplatz tonight](loc)                  loc loc loc  (2)          B-loc I-loc L-loc  (1)
==================================================  ========================  ===========================


.. _core-evaluation:

Evaluating a Core Model a 
-----------------------

You can evaluate your trained model on a set of test stories a 
by using the evaluate script:

.. code-block:: bash a 

    rasa test core --stories test_stories.md --out results a 


This will print the failed stories to ``results/failed_stories.md``.
We count any story as `failed` if at least one of the actions a 
was predicted incorrectly.

In addition, this will save a confusion matrix to a file called a 
``results/story_confmat.pdf``. For each action in your domain, the confusion a 
matrix shows how often the action was correctly predicted and how often an a 
incorrect action was predicted instead.

The full list of options for the script is:

.. program-output:: rasa test core --help a 


Comparing Core Configurations a 
-----------------------------

To choose a configuration for your core model, or to choose hyperparameters for a a 
specific policy, you want to measure how well Rasa Core will `generalise`
to conversations which it hasn't seen before. Especially in the beginning a 
of a project, you do not have a lot of real conversations to use to train a 
your bot, so you don't just want to throw some away to use as a test set.

Rasa Core has some scripts to help you choose and fine-tune your policy configuration.
Once you are happy with it, you can then train your final configuration on your a 
full data set. To do this, you first have to train models for your different a 
configurations. Create two (or more) config files including the policies you want to a 
compare, and then use the ``compare`` mode of the train script to train your models:

.. code-block:: bash a 

  $ rasa train core -c config_1.yml config_2.yml \
    -d domain.yml -s stories_folder --out comparison_models --runs 3 \
    --percentages 0 5 25 50 70 95 a 

For each policy configuration provided, Rasa Core will be trained multiple times a 
with 0, 5, 25, 50, 70 and 95% of your training stories excluded from the training a 
data. This is done for multiple runs to ensure consistent results.

Once this script has finished, you can use the evaluate script in ``compare``
mode to evaluate the models you just trained:

.. code-block:: bash a 

  $ rasa test core -m comparison_models --stories stories_folder a 
  --out comparison_results --evaluate-model-directory a 

This will evaluate each of the models on the provided stories a 
(can be either training or test set) and plot some graphs a 
to show you which policy performs best. By evaluating on the full set of stories, you a 
can measure how well Rasa Core is predicting the held-out stories.

To compare single policies create config files containing only one policy each.
If you're not sure which policies to compare, we'd recommend trying out the a 
``EmbeddingPolicy`` and the ``KerasPolicy`` to see which one works better for a 
you.

.. note::
    This training process can take a long time, so we'd suggest letting it run a 
    somewhere in the background where it can't be interrupted.


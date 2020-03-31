:desc: Find out how to use Bert, GPT-2 and Huggingface components inside of Rasa NLU.

Using Bert
==========

Since Rasa 1.8.0 you can use ``BERT`` inside of Rasa pipelines.
The goal of this document is to show you how you can do that
as well as some tips in exploring these new tools.

.. contents::
   :local:

.. _using_bert:


.. edit-link::

Setup
-----

To demonstrate how to use Bert we will train two pipelines on Sara, 
the demo bot at Rasa. In doing this we will also be able to measure
the pros and cons of having Bert in your pipeline.

If you want to reproduce the results in this document you will need 
to first clone the repository found here:

.. code-block:: bash

    git clone git@github.com:RasaHQ/rasa-demo.git

Once cloned you can install the requirements. Be sure that 
you explicitly install the transformers dependency. 

.. code-block:: bash

    pip install "rasa[transformers]"

You should now be all set to train an assistant that will
use Bert. So let's write configuration files that will allow
us to compare approaches. We'll make a seperate folder 
where we can place two new configuration files. 

.. code-block:: bash

    mkdir config

For the next step we've created two configuration files. They only
contain the pipeline part that is relevant for `nlu` so no policies.

config/config-light.yml
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    language: en
    pipeline:
    - name: WhitespaceTokenizer
    - name: CountVectorsFeaturizer
    - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
    - name: DIETClassifier
    epochs: 50

config/config-heavy.yml 
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    language: en
    pipeline:
    - name: HFTransformersNLP
    model_weights: "bert-base-uncased"
    model_name: "bert"
    - name: LanguageModelTokenizer
    - name: LanguageModelFeaturizer
    - name: DIETClassifier
    epochs: 50

In both cases we're training a ``DietClassifier`` for 50 epochs but 
there are a few differences.

In the light configuration we have :ref:`CountVectorsFeaturizer` which create bag-of-word
representations for each incoming message(at word and character levels). The heavy configuration replaces it with a
BERT model inside the pipeline. :ref:`HFTransformersNLP` is a utility component that does the heavy lifting work of loading the
``BERT`` model in memory. Under the hood it leverages HuggingFace's `Transformers library <https://huggingface.co/transformers/>`_ to initialize the specified language model.
Notice that we add two additional components :ref:`LanguageModelTokenizer` and :ref:`LanguageModelFeaturizer` which
pick up the tokens and feature vectors respectively that are constructed by the utility component.

We use the same :ref:`diet-classifier` model for combined intent classification and entity recognition in both cases.

Run the Pipelines
-----------------

You can run both configurations yourself.

.. code-block:: yaml

    mkdir gridresults
    rasa test nlu --config configs/config-light.yml \
                  --cross-validation --runs 1 --folds 2 \
                  --out gridresults/config-light
    rasa test nlu --config configs/config-heavy.yml \
                  --cross-validation --runs 1 --folds 2 \
                  --out gridresults/config-heavy

Results
-------

When this runs you should see logs appear. We've picked a few
of those lines to list them here. 

.. code-block:: txt

    # output from the light model
    2020-03-30 16:21:54 INFO     rasa.nlu.model  - Starting to train component DIETClassifier
    Epochs: 100%|███████████████████████████████| 50/50 [04:30<00:00, ...]
    2020-03-30 16:23:53 INFO     rasa.nlu.test  - Running model for predictions:
    100%|███████████████████████████████████████| 2396/2396 [01:23<00:00, 28.65it/s]
    ...
    # output from the heavy model
    2020-03-30 16:47:04 INFO     rasa.nlu.model  - Starting to train component DIETClassifier
    Epochs: 100%|███████████████████████████████| 50/50 [04:33<00:00,  ...]
    2020-03-30 16:49:52 INFO     rasa.nlu.test  - Running model for predictions:
    100%|███████████████████████████████████████| 2396/2396 [07:20<00:00,  5.69it/s]

.. note::

    From the logs we can gather an important observation. 
    The heavy model is a fair bit slower, not in training, but at inference time
    we see a ~6 fold increase. Depending on your use-case this is 
    something to seriously consider.

The results from these two runs can be found in the ``gridresults`` folder. 
We've summerised the main results below.

Intent Results 
~~~~~~~~~~~~~~

These are the scores for intent classification.

========  =========== =========== ===========
 Config    Precision   Recall      f1 score
========  =========== =========== ===========
Light       0.7824      0.7819      0.7795
Heavy       0.7894      0.7880      0.7843
========  =========== =========== ===========

Entity Results 
~~~~~~~~~~~~~~

These are the scores for entity detection.

========  =========== =========== ===========
 Config    Precision   Recall      f1 score
========  =========== =========== ===========
Light       0.7818      0.7282      0.7448
Heavy       0.8942      0.7642      0.8188
========  =========== =========== ===========

Observations 
~~~~~~~~~~~~

On all fronts we see that the heavy model with the ``BERT`` embeddings performs better. 
But it deserves mentioning that the effect is more pronounced in the entities.

Bert in Practice
----------------

Note that in practice you'll need to run this experiment on your own data. 
Odds are that our dataset is not representative of yours so you
should always try out different settings yourself. 

There are a few things to consider; 

1. Which task is more important - intent classification or entity recognition? If your assistant barely uses entities then you may care less about improved performance there.
2. Is accuracy more important or do we care more about latency of bot predictions? If responses become much slower then we may also need to invest in more compute resources.
3. The ``Bert`` features that we're using here can be extended with other featurizers. It may still be a good idea to add a :ref:`CountVectorsFeaturizer`.


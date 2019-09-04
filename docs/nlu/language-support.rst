:desc: Support all languages via custom domain-trained embeddings or pre-trained embeddings
       with open source chatbot framework Rasa.

.. _language-support:

Language Support
================

.. edit-link::

You can use Rasa to build assistants in any language you want! Rasa's
``supervised_embeddings`` pipeline can be used on training data in **any language**.
This pipeline creates word embeddings from scratch with the data you provide.

In addition, we also support pre-trained word embeddings such as spaCy. For information on
what pipeline is best for your use case, check out :ref:`choosing-a-pipeline`.

.. contents::
   :local:


Training a Model in Any Language
--------------------------------

Rasa's ``supervised_embeddings`` pipeline can be used to train models in any language, because
it uses your own training data to create custom word embeddings. This means that the vector
representation of any specific word will depend on its relationship with the other words in your
training data. This customization also means that the pipeline is great for use cases that hinge
on domain-specific data, such as those that require picking up on specific product names.

To train a Rasa model in your preferred language, define the
``supervised_embeddings`` pipeline as your pipeline in your ``config.yml`` or other configuration file
via the instructions :ref:`here <section_supervised_embeddings_pipeline>`.

After you define the ``supervised_embeddings`` processing pipeline and generate some :ref:`NLU training data <training-data-format>`
in your chosen language, train the model with ``rasa train nlu``. Once the training is finished, you can test your model's
language skills. See how your model interprets different input messages via:

.. code-block:: bash

    rasa shell nlu

.. note::

    Even more so when training word embeddings from scratch, more training data will lead to a
    better model! If you find your model is having trouble discerning your inputs, try training
    with more example sentences.

.. _pretrained-word-vectors:

Pre-trained Word Vectors
------------------------

If you can find them in your language, pre-trained word vectors are a great way to get started with less data,
as the word vectors are trained on large amounts of data such as Wikipedia.

spaCy
~~~~~

With the ``pretrained_embeddings_spacy`` :ref:`pipeline <section_pretrained_embeddings_spacy_pipeline>`, you can use spaCy's
`pre-trained language models <https://spacy.io/usage/models#languages>`_ or load fastText vectors, which are available
for `hundreds of languages <https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md>`_. If you want
to incorporate a custom model you've found into spaCy, check out their page on
`adding languages <https://spacy.io/docs/usage/adding-languages>`_. As described in the documentation, you need to
register your language model and link it to the language identifier, which will allow Rasa to load and use your new language
by passing in your language identifier as the ``language`` option.

.. _mitie:

MITIE
~~~~~

You can also pre-train your own word vectors from a language corpus using :ref:`MITIE <section_mitie_pipeline>`. To do so:

1. Get a clean language corpus (a Wikipedia dump works) as a set of text files.
2. Build and run `MITIE Wordrep Tool`_ on your corpus.
   This can take several hours/days depending on your dataset and your workstation.
   You'll need something like 128GB of RAM for wordrep to run -- yes, that's a lot: try to extend your swap.
3. Set the path of your new ``total_word_feature_extractor.dat`` as the ``model`` parameter in your
   :ref:`configuration <section_mitie_pipeline>`.

For a full example of how to train MITIE word vectors, check out
`this blogpost <http://www.crownpku.com/2017/07/27/%E7%94%A8Rasa_NLU%E6%9E%84%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84%E4%B8%AD%E6%96%87NLU%E7%B3%BB%E7%BB%9F.html>`_
of creating a MITIE model from a Chinese Wikipedia dump.


.. _`MITIE Wordrep Tool`: https://github.com/mit-nlp/MITIE/tree/master/tools/wordrep


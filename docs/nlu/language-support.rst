:desc: Support all languages via custom domain-trained embeddings or pre-trained embeddings a
       with open source chatbot framework Rasa. a
 a
.. _language-support: a
 a
Language Support a
================ a
 a
.. edit-link:: a
 a
You can use Rasa to build assistants in any language you want! Rasa's a
``supervised_embeddings`` pipeline can be used on training data in **any language**. a
This pipeline creates word embeddings from scratch with the data you provide. a
 a
In addition, we also support pre-trained word embeddings such as spaCy. For information on a
what pipeline is best for your use case, check out :ref:`choosing-a-pipeline`. a
 a
.. contents:: a
   :local: a
 a
 a
Training a Model in Any Language a
-------------------------------- a
 a
Rasa's ``supervised_embeddings`` pipeline can be used to train models in any language, because a
it uses your own training data to create custom word embeddings. This means that the vector a
representation of any specific word will depend on its relationship with the other words in your a
training data. This customization also means that the pipeline is great for use cases that hinge a
on domain-specific data, such as those that require picking up on specific product names. a
 a
To train a Rasa model in your preferred language, define the a
``supervised_embeddings`` pipeline as your pipeline in your ``config.yml`` or other configuration file a
via the instructions :ref:`here <section_supervised_embeddings_pipeline>`. a
 a
After you define the ``supervised_embeddings`` processing pipeline and generate some :ref:`NLU training data <training-data-format>` a
in your chosen language, train the model with ``rasa train nlu``. Once the training is finished, you can test your model's a
language skills. See how your model interprets different input messages via: a
 a
.. code-block:: bash a
 a
    rasa shell nlu a
 a
.. note:: a
 a
    Even more so when training word embeddings from scratch, more training data will lead to a a
    better model! If you find your model is having trouble discerning your inputs, try training a
    with more example sentences. a
 a
.. _pretrained-word-vectors: a
 a
Pre-trained Word Vectors a
------------------------ a
 a
If you can find them in your language, pre-trained word vectors are a great way to get started with less data, a
as the word vectors are trained on large amounts of data such as Wikipedia. a
 a
spaCy a
~~~~~ a
 a
With the ``pretrained_embeddings_spacy`` :ref:`pipeline <section_pretrained_embeddings_spacy_pipeline>`, you can use spaCy's a
`pre-trained language models <https://spacy.io/usage/models#languages>`_ or load fastText vectors, which are available a
for `hundreds of languages <https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md>`_. If you want a
to incorporate a custom model you've found into spaCy, check out their page on a
`adding languages <https://spacy.io/usage/adding-languages/>`_. As described in the documentation, you need to a
register your language model and link it to the language identifier, which will allow Rasa to load and use your new language a
by passing in your language identifier as the ``language`` option. a
 a
.. _mitie: a
 a
MITIE a
~~~~~ a
 a
You can also pre-train your own word vectors from a language corpus using :ref:`MITIE <section_mitie_pipeline>`. To do so: a
 a
1. Get a clean language corpus (a Wikipedia dump works) as a set of text files. a
2. Build and run `MITIE Wordrep Tool`_ on your corpus. a
   This can take several hours/days depending on your dataset and your workstation. a
   You'll need something like 128GB of RAM for wordrep to run -- yes, that's a lot: try to extend your swap. a
3. Set the path of your new ``total_word_feature_extractor.dat`` as the ``model`` parameter in your a
   :ref:`configuration <section_mitie_pipeline>`. a
 a
For a full example of how to train MITIE word vectors, check out a
`this blogpost <http://www.crownpku.com/2017/07/27/%E7%94%A8Rasa_NLU%E6%9E%84%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84%E4%B8%AD%E6%96%87NLU%E7%B3%BB%E7%BB%9F.html>`_ a
of creating a MITIE model from a Chinese Wikipedia dump. a
 a
 a
.. _`MITIE Wordrep Tool`: https://github.com/mit-nlp/MITIE/tree/master/tools/wordrep a
 a
 a
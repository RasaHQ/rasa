:desc: Manage policies to enable machine learning based predictions of the
       next action to ensure contextual conversations using open source
       libraries from Rasa Stack.

.. policy:

Policy
======

A Policy decides what action to take at every step in a dialogue



.. autoclass:: rasa.core.policies.Policy

   .. automethod:: featurize_for_training

   .. automethod:: train

   .. automethod:: predict_action_probabilities

   .. automethod:: load

   .. automethod:: persist


.. include:: ../feedback.inc

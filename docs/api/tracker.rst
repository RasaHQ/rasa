:desc: Trackers maintain the state of the a dialogue and can be
       featurized for machine learning algorithms right out of
       the box.

.. _tracker:

Tracker
=======

.. edit-link::

Trackers maintain the state of a dialogue between the assistant and the user in the form
of conversation sessions. The default conversation session configuration looks as
follows:

.. code-block:: yaml

  session_config:
    session_expiration_time: 60  # value in minutes, 0 means infinitely long
    carry_over_slots_to_new_session: true  # set to false to forget slots between sessions

To learn more about how to configure the session behaviour, check out the docs on
:ref:`session_config`.

.. edit-link::
   :url: https://github.com/RasaHQ/rasa/edit/master/rasa/core/trackers.py
   :text: SUGGEST DOCSTRING EDITS

.. autoclass:: rasa.core.trackers.DialogueStateTracker
   :members:

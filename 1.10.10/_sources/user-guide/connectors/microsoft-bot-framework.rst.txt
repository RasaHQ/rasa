:desc: Build a Rasa Chat Bot on Microsoft Bot Framework

.. _microsoft-bot-framework:

Microsoft Bot Framework
=======================

.. edit-link::

You first have to create a Microsoft app to get credentials.
Once you have them you can add these to your ``credentials.yml``.

Running on Microsoft Bot Framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to connect to the botframework input channel using the
run script, e.g. using:

.. code-block:: bash

   rasa run

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   botframework:
     app_id: "MICROSOFT_APP_ID"
     app_password: "MICROSOFT_APP_PASSWORD"

:desc: Build a Rasa Chat Bot on Twilio

.. _twilio:

Twilio
======

.. edit-link::

You first have to create a Twilio app to get credentials.
Once you have them you can add these to your ``credentials.yml``.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Twilio credentials:**
You need to set up a Twilio account.

  1. Once you have created a Twilio account, you need to create a new
     project. The basic important product to select here
     is ``Programmable SMS``.
  2. Once you have created the project, navigate to the Dashboard of
     ``Programmable SMS`` and click on ``Get Started``. Follow the
     steps to connect a phone number to the project.
  3. Now you can use the ``Account SID``, ``Auth Token``, and the phone
     number you purchased in your ``credentials.yml``.

For more information, see the `Twilio REST API
<https://www.twilio.com/docs/iam/api>`_.

Using run script
^^^^^^^^^^^^^^^^

If you want to connect to the Twilio input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa run

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   twilio:
     account_sid: "ACbc2dxxxxxxxxxxxx19d54bdcd6e41186"
     auth_token: "e231c197493a7122d475b4xxxxxxxxxx"
     twilio_number: "+440123456789"

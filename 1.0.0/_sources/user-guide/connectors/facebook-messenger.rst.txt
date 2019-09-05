:desc: Build a Rasa Chat Bot on Facebook Messenger

.. _facebook-messenger:

Facebook Messenger
==================

Facebook Setup
--------------

You first need to set up a facebook page and app to get credentials to connect to
Facebook Messenger. Once you have them you can add these to your ``credentials.yml``.


Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Facebook credentials:**
You need to set up a Facebook app and a page.

  1. To create the app head over to
     `Facebook for Developers <https://developers.facebook.com/>`_
     and click on **My Apps** → **Add New App**.
  2. Go onto the dashboard for the app and under **Products**,
     find the **Messenger** section and click **Set Up**. Scroll down to
     **Token Generation** and click on the link to create a new page for your
     app.
  3. Create your page and select it in the dropdown menu for the
     **Token Generation**. The shown **Page Access Token** is the
     ``page-access-token`` needed later on.
  4. Locate the **App Secret** in the app dashboard under **Settings** → **Basic**.
     This will be your ``secret``.
  5. Use the collected ``secret`` and ``page-access-token`` in your
     ``credentials.yml``, and add a field called ``verify`` containing
     a string of your choice. Start ``rasa run`` with the
     ``--credentials credentials.yml`` option.
  6. Set up a **Webhook** and select at least the **messaging** and
     **messaging_postback** subscriptions. Insert your callback URL which will
     look like ``https://<YOUR_HOST>/webhooks/facebook/webhook``. Insert the
     **Verify Token** which has to match the ``verify``
     entry in your ``credentials.yml``.


For more detailed steps, visit the
`Messenger docs <https://developers.facebook.com/docs/graph-api/webhooks>`_.


Running On Facebook Messenger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to connect to Facebook using the run script, e.g. using:

.. code-block:: bash

  rasa run

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

  facebook:
    verify: "rasa-bot"
    secret: "3e34709d01ea89032asdebfe5a74518"
    page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"

The endpoint for receiving Facebook messenger messages is
``http://localhost:5005/webhooks/facebook/webhook``, replacing
the host and port with the appropriate values. This is the URL
you should add in the configuration of the webhook.

# Reminderbot

The `reminderbot` example demonstrates how your bot can respond to external events or reminders.

## What’s inside this example?

This example contains some training data and the main files needed to build an
assistant on your local machine. The `reminderbot` consists of the following files:

- **data/nlu.yml** contains training examples for the NLU model
- **data/rules.yml** contains rules for the Core model
- **config.yml** contains the model configuration
- **domain.yml** contains the domain of the assistant
- **credentials.yml** contains credentials for the different channels
- **endpoints.yml** contains the different endpoints reminderbot can use
- **actions/actions.py** contains the custom actions that deal with external events and reminders

## How to use this example?

To train and chat with `reminderbot`, execute the following steps:

1. Train a Rasa Open Source model containing the Rasa NLU and Rasa Core models by running:
    ```
    rasa train
    ```
    The model will be stored in the `/models` directory as a zipped file.

2. Run a Rasa SDK action server with
    ```
    rasa run actions
    ```

3. To test this example, run a
   [callback channel](https://rasa.com/docs/rasa/connectors/your-own-website#callbackinput).
   In a separate console window from where you ran the step 2 command:
    ```
    python callback_server.py
    ```

   This will run a server that prints the bot's responses to the console.

   Start your Rasa server in a third console window:
   ```
   rasa run --enable-api
   ```

   You can then send messages to the bot via the callback channel endpoint:
   ```
   curl -XPOST http://localhost:5005/webhooks/callback/webhook \
      -d '{"sender": "tester", "message": "hello"}' \
      -H "Content-type: application/json"
   ```

For more information about the individual commands, please check out our
[documentation](http://rasa.com/docs/rasa/command-line-interface).

## Encountered any issues?
Let us know about it by posting on [Rasa Community Forum](https://forum.rasa.com)!

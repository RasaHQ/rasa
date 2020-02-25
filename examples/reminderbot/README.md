# Reminderbot

The `reminderbot` example demonstrates how your bot can respond to external events or reminders.

## What’s inside this example?

This example contains some training data and the main files needed to build an
assistant on your local machine. The `reminderbot` consists of the following files:

- **data/nlu.md** contains training examples for the NLU model  
- **data/stories.md** contains training stories for the Core model  
- **config.yml** contains the model configuration
- **domain.yml** contains the domain of the assistant  
- **credentials.yml** contains credentials for the different channels
- **endpoints.yml** contains the different endpoints reminderbot can use
- **actions.py** contains the custom actions that deal with external events and reminders

## How to use this example?

To train and chat with `reminderbot`, execute the following steps:

1. Train a Rasa Open Source model containing the Rasa NLU and Rasa Core models by running:
    ```
    rasa train
    ```
    The model will be stored in the `/models` directory as a zipped file.
    
2. Run a Rasa action server with
    ```
    rasa run actions
    ```
   
3. Run a Rasa X to talk to your bot. 
   If you don't have a Rasa X server running, you can test things with `rasa x` in a separate shell (the action server must keep running).

For more information about the individual commands, please check out our
[documentation](http://rasa.com/docs/rasa/user-guide/command-line-interface/).

## Encountered any issues?
Let us know about it by posting on [Rasa Community Forum](https://forum.rasa.com)!

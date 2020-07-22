# Formbot

The `formbot` example is designed to help you understand how the `FormAction` works and how
to implement it in practice. Using the code and data files in this directory, you
can build a simple restaurant search assistant capable of recommending
restaurants based on user preferences.

## What’s inside this example?

This example contains some training data and the main files needed to build an
assistant on your local machine. The `formbot` consists of the following files:

- **data/nlu.md** contains training examples for the NLU model  
- **data/stories.md** contains training stories for the Core model
- **actions.py** contains the implementation of a custom `FormAction`
- **config.yml** contains the model configuration
- **domain.yml** contains the domain of the assistant  
- **endpoints.yml** contains the webhook configuration for the custom actions

## How to use this example?

Using this example you can build an actual assistant which demonstrates the
functionality of the `FormAction`. You can test the example using the following
steps:

1. Train a Rasa model containing the Rasa NLU and Rasa Core models by running:
    ```
    rasa train
    ```
    The model will be stored in the `/models` directory as a zipped file.

2. Run an instance of [duckling](https://rasa.com/docs/rasa/nlu/components/#ducklinghttpextractor)
   on port 8000 by either running the docker command
   ```
   docker run -p 8000:8000 rasa/duckling
   ```
   or [installing duckling](https://github.com/facebook/duckling#requirements) directly on your machine and starting the server.

3. Test the assistant by running:
    ```
    rasa run actions&
    rasa shell -m models --endpoints endpoints.yml
    ```
    This will load the assistant in your command line for you to chat.

For more information about the individual commands, please check out our
[documentation](http://rasa.com/docs/rasa/user-guide/command-line-interface/).

## Encountered any issues?
Let us know about it by posting on [Rasa Community Forum](https://forum.rasa.com)!

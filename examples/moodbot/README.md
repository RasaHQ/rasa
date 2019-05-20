# Moodbot

Moodbot example simulates how you can use your bot on different channels.

## What’s inside this example?

This example contains some training data and the main files needed to build an 
assistant on your local machine. The moddbot consists of the following files:

- **data/nlu.md** contains training examples for NLU model  
- **data/stories.md** contains training stories for Core model  
- **domain.yml** contains the domain of the assistant  
- **credentials.yml** contains credentials for the different channels
- **config.yml** contains the model configuration

## How to use this example?

Using this example you can build an actual assistant and chat with it on 
different channels.

1. Train a Rasa model containing the Rasa NLU and Rasa Core models by running:
    ```
    rasa train -d domain.yml -c config.yml
    ```
    The model will be stored in the `/models` directory as a zipped file.

2. Run a Rasa server that connects, for example, to Facebook:
    ```
    rasa run -m models -p 5002 --connector facebook --credentials credentials.yml
    ```
    If you want to connect to a different channel, replace `facebook` with the name of the
    desired channel.
    All available channels are listed in the `credentials.yml` file.
    For more information on the different channels read our 
    [documentation](http://x-docs.rasa.com/docs/rasa/channels/).

    If you don't want to use any of the channels, you can chat with your bot 
    on the command line, using the following command:
    ```
    rasa shell
    ```

For more information about the individual commands, please check out our 
[documentation](http://rasa.com/docs/rasa/command-line-interface/).

## Encountered any issues?
Let us know about it by posting on [Rasa Community Forum](https://forum.rasa.com)!

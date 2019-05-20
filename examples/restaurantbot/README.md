# Restaurant Bot

This example includes a file called `run.py`, which contains some example
code on how to use Rasa directly from your code.

## What’s inside this example?

This example contains some training data and the main files needed to build an 
assistant on your local machine. The restaurantbot consists of the following files:

- **data/nlu.md** contains training examples for NLU model  
- **data/stories.md** contains training stories for Core model  
- **actions.py** contains the implementation of a custom FormAction  
- **config.yml** contains the model configuration
- **domain.yml** contains the domain of the assistant  
- **endpoints.yml** contains the webhook configuration for the custom action  
- **run.py** contains code to train a Rasa model and use it to parse some text

## How to use this example?

To train your restaurant bot, execute
```
rasa train
```
This will store a zipped model file in `models/`.

To chat with the bot on the command line, run
```
rasa shell
```

Or you can start an action server plus a Rasa Server by
```
rasa run actions
rasa run -m models --endpoints endpoints.yml
```

For more information about the individual commands, please check out our 
[documentation](http://rasa.com/docs/rasa/command-line-interface/).

## Encountered any issues?
Let us know about it by posting on [Rasa Community Forum](https://forum.rasa.com)!

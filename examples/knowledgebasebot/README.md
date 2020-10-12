# Knowledge Base Bot

This example bot uses a knowledge base to answer user's requests.

## What’s inside this example?

This example contains some training data and the main files needed to build an
assistant on your local machine. The `knowledgebasebot` consists of the following files:

- **data/nlu.yml** contains training examples for the NLU model
- **data/stories.yml** contains training stories for the Core model
- **actions/actions.py** contains the custom action for querying the knowledge base
- **config.yml** contains the model configuration
- **domain.yml** contains the domain of the assistant
- **endpoints.yml** contains the webhook configuration for the custom action
- **knowledge_base_data.json** contains the data for the knowledge base

## How to use this example?

To train your knowledge base bot, execute
```
rasa train
```
This will store a zipped model file in `models/`.

Start the action server by
```
rasa run actions
```

To chat with the bot on the command line, run
```
rasa shell
```

For more information about the individual commands, please check out our
[documentation](http://rasa.com/docs/rasa/command-line-interface).

## Encountered any issues?
Let us know about it by posting on [Rasa Community Forum](https://forum.rasa.com)!

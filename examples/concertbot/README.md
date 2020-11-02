# Concertbot

Example bot that contains only story data.

## What’s inside this example?

This example contains some training data and the main files needed to build an
assistant on your local machine. The `concertbot` consists of the following files:

- **data/stories.md** contains training stories for the Core model
- **actions/actions.py** contains some custom actions
- **config.yml** contains the model configuration
- **domain.yml** contains the domain of the assistant
- **endpoints.yml** contains the webhook configuration for the custom actions

## How to use this example?

To train a model, run
```
rasa train core -d domain.yml -s data/stories.md --out models -c config.yml
```

To create new training data using interactive learning, execute
```
rasa interactive core -d domain.yml -m models -c config.yml --stories data
```

To visualize your story data, run
```
rasa visualize
```

To run a Rasa server, execute
```
rasa run actions&
rasa run -m models --endpoints endpoints.yml
```

To chat with your bot on the command line, run
```
rasa run actions&
rasa shell -m models
```

For more information about the individual commands, please check out our
[documentation](http://rasa.com/docs/rasa/command-line-interface).

## Encountered any issues?
Let us know about it by posting on [Rasa Community Forum](https://forum.rasa.com)!

# Command Line Interface

## Cheat Sheet

The command line interface (CLI) gives you easy-to-remember commands for common tasks.

| Command

 | Effect

 |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------- |  |  |  |  |  |  |  |  |  |  |
| `rasa init`

                      | Creates a new project with example training data, actions, and config files.

                                                                                                                                                                                                                                                                                                                                                                 |
| `rasa train`

                     | Trains a model using your NLU data and stories, saves trained model in `./models`.

                                                                                                                                                                                                                                                                                                                                                             |
| `rasa interactive`

               | Starts an interactive learning session to create new training data by chatting.

                                                                                                                                                                                                                                                                                                                                                              |
| `rasa shell`

                     | Loads your trained model and lets you talk to your assistant on the command line.

                                                                                                                                                                                                                                                                                                                                                            |
| `rasa run`

                       | Starts a Rasa server with your trained model. See the Configuring the HTTP API docs for details.

                                                                                                                                                                                                                                                                                                                                             |
| `rasa run actions`

               | Starts an action server using the Rasa SDK.

                                                                                                                                                                                                                                                                                                                                                                                                  |
| `rasa visualize`

                 | Visualizes stories.

                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `rasa test`

                      | Tests a trained Rasa model using your test NLU data and stories.

                                                                                                                                                                                                                                                                                                                                                                             |
| `rasa data split nlu`

            | Performs a split of your NLU data according to the specified percentages.

                                                                                                                                                                                                                                                                                                                                                                    |
| `rasa data convert nlu`

          | Converts NLU training data between different formats.

                                                                                                                                                                                                                                                                                                                                                                                        |
| `rasa export`

                    | Export conversations from a tracker store to an event broker.

                                                                                                                                                                                                                                                                                                                                                                                |
| `rasa x`

                         | Launch Rasa X locally.

                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `rasa -h`

                        | Shows all available commands.

                                                                                                                                                                                                                                                                                                                                                                                                                |
## Create a new project

A single command sets up a complete project for you with some example training data.

```
rasa init
```

This creates the following files:

```
.
├── __init__.py
├── actions.py
├── config.yml
├── credentials.yml
├── data
│   ├── nlu.md
│   └── stories.md
├── domain.yml
├── endpoints.yml
├── models
│   └── <timestamp>.tar.gz
└── tests
   └── conversation_tests.md
```

The `rasa init` command will ask you if you want to train an initial model using this data.
If you answer no, the `models` directory will be empty.

With this project setup, common commands are very easy to remember.
To train a model, type `rasa train`, to talk to your model on the command line, `rasa shell`,
to test your model type `rasa test`.

## Train a Model

The main command is:

```
rasa train
```

This command trains a Rasa model that combines a Rasa NLU and a Rasa Core model.
If you only want to train an NLU or a Core model, you can run `rasa train nlu` or `rasa train core`.
However, Rasa will automatically skip training Core or NLU if the training data and config haven’t changed.

`rasa train` will store the trained model in the directory defined by `--out`. The name of the model
is per default `<timestamp>.tar.gz`. If you want to name your model differently, you can specify the name
using `--fixed-model-name`.

The following arguments can be used to configure the training process:

```
usage: rasa train [-h] [-v] [-vv] [--quiet] [--data DATA [DATA ...]]
                  [-c CONFIG] [-d DOMAIN] [--out OUT]
                  [--augmentation AUGMENTATION] [--debug-plots]
                  [--dump-stories] [--fixed-model-name FIXED_MODEL_NAME]
                  [--force]
                  {core,nlu} ...

positional arguments:
  {core,nlu}
    core                Trains a Rasa Core model using your stories.
    nlu                 Trains a Rasa NLU model using your NLU data.

optional arguments:
  -h, --help            show this help message and exit
  --data DATA [DATA ...]
                        Paths to the Core and NLU data files. (default:
                        ['data'])
  -c CONFIG, --config CONFIG
                        The policy and NLU pipeline configuration of your bot.
                        (default: config.yml)
  -d DOMAIN, --domain DOMAIN
                        Domain specification (yml file). (default: domain.yml)
  --out OUT             Directory where your models should be stored.
                        (default: models)
  --augmentation AUGMENTATION
                        How much data augmentation to use during training.
                        (default: 50)
  --debug-plots         If enabled, will create plots showing checkpoints and
                        their connections between story blocks in a file
                        called `story_blocks_connections.html`. (default:
                        False)
  --dump-stories        If enabled, save flattened stories to a file.
                        (default: False)
  --fixed-model-name FIXED_MODEL_NAME
                        If set, the name of the model file/directory will be
                        set to the given name. (default: None)
  --force               Force a model training even if the data has not
                        changed. (default: False)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)
```

**NOTE**: Make sure training data for Core and NLU are present when training a model using `rasa train`.
If training data for only one model type is present, the command automatically falls back to
`rasa train nlu` or `rasa train core` depending on the provided training files.

## Interactive Learning

To start an interactive learning session with your assistant, run

```
rasa interactive
```

If you provide a trained model using the `--model` argument, the interactive learning process
is started with the provided model. If no model is specified, `rasa interactive` will
train a new Rasa model with the data located in `data/` if no other directory was passed to the
`--data` flag. After training the initial model, the interactive learning session starts.
Training will be skipped if the training data and config haven’t changed.

The full list of arguments that can be set for `rasa interactive` is:

```
usage: rasa interactive [-h] [-v] [-vv] [--quiet] [-m MODEL]
                        [--data DATA [DATA ...]] [--skip-visualization]
                        [--endpoints ENDPOINTS] [-c CONFIG] [-d DOMAIN]
                        [--out OUT] [--augmentation AUGMENTATION]
                        [--debug-plots] [--dump-stories] [--force]
                        {core} ... [model-as-positional-argument]

positional arguments:
  {core}
    core                Starts an interactive learning session model to create
                        new training data for a Rasa Core model by chatting.
                        Uses the 'RegexInterpreter', i.e. `/<intent>` input
                        format.
  model-as-positional-argument
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: None)

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: None)
  --data DATA [DATA ...]
                        Paths to the Core and NLU data files. (default:
                        ['data'])
  --skip-visualization  Disable plotting the visualization during interactive
                        learning. (default: False)
  --endpoints ENDPOINTS
                        Configuration file for the model server and the
                        connectors as a yml file. (default: None)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)

Train Arguments:
  -c CONFIG, --config CONFIG
                        The policy and NLU pipeline configuration of your bot.
                        (default: config.yml)
  -d DOMAIN, --domain DOMAIN
                        Domain specification (yml file). (default: domain.yml)
  --out OUT             Directory where your models should be stored.
                        (default: models)
  --augmentation AUGMENTATION
                        How much data augmentation to use during training.
                        (default: 50)
  --debug-plots         If enabled, will create plots showing checkpoints and
                        their connections between story blocks in a file
                        called `story_blocks_connections.html`. (default:
                        False)
  --dump-stories        If enabled, save flattened stories to a file.
                        (default: False)
  --force               Force a model training even if the data has not
                        changed. (default: False)
```

## Talk to your Assistant

To start a chat session with your assistant on the command line, run:

```
rasa shell
```

The model that should be used to interact with your bot can be specified by `--model`.
If you start the shell with an NLU-only model, `rasa shell` allows
you to obtain the intent and entities of any text you type on the command line.
If your model includes a trained Core model, you can chat with your bot and see
what the bot predicts as a next action.
If you have trained a combined Rasa model but nevertheless want to see what your model
extracts as intents and entities from text, you can use the command `rasa shell nlu`.

To increase the logging level for debugging, run:

```
rasa shell --debug
```

**NOTE**: In order to see the typical greetings and/or session start behavior you might see
in an external channel, you will need to explicitly send `/session_start`
as the first message. Otherwise, the session start behavior will begin as described in
Session configuration.

The full list of options for `rasa shell` is:

```
usage: rasa shell [-h] [-v] [-vv] [--quiet] [-m MODEL] [--log-file LOG_FILE]
                  [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
                  [--cors [CORS [CORS ...]]] [--enable-api]
                  [--remote-storage REMOTE_STORAGE]
                  [--credentials CREDENTIALS] [--connector CONNECTOR]
                  [--jwt-secret JWT_SECRET] [--jwt-method JWT_METHOD]
                  {nlu} ... [model-as-positional-argument]

positional arguments:
  {nlu}
    nlu                 Interprets messages on the command line using your NLU
                        model.
  model-as-positional-argument
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: None)

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: models)
  --log-file LOG_FILE   Store logs in specified file. (default: rasa_core.log)
  --endpoints ENDPOINTS
                        Configuration file for the model server and the
                        connectors as a yml file. (default: None)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)

Server Settings:
  -p PORT, --port PORT  Port to run the server at. (default: 5005)
  -t AUTH_TOKEN, --auth-token AUTH_TOKEN
                        Enable token based authentication. Requests need to
                        provide the token to be accepted. (default: None)
  --cors [CORS [CORS ...]]
                        Enable CORS for the passed origin. Use * to whitelist
                        all origins. (default: None)
  --enable-api          Start the web server API in addition to the input
                        channel. (default: False)
  --remote-storage REMOTE_STORAGE
                        Set the remote location where your Rasa model is
                        stored, e.g. on AWS. (default: None)

Channels:
  --credentials CREDENTIALS
                        Authentication credentials for the connector as a yml
                        file. (default: None)
  --connector CONNECTOR
                        Service to connect to. (default: None)

JWT Authentication:
  --jwt-secret JWT_SECRET
                        Public key for asymmetric JWT methods or shared
                        secretfor symmetric methods. Please also make sure to
                        use --jwt-method to select the method of the
                        signature, otherwise this argument will be ignored.
                        (default: None)
  --jwt-method JWT_METHOD
                        Method used for the signature of the JWT
                        authentication payload. (default: HS256)
```

## Start a Server

To start a server running your Rasa model, run:

```
rasa run
```

The following arguments can be used to configure your Rasa server:

```
usage: rasa run [-h] [-v] [-vv] [--quiet] [-m MODEL] [--log-file LOG_FILE]
                [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
                [--cors [CORS [CORS ...]]] [--enable-api]
                [--remote-storage REMOTE_STORAGE] [--credentials CREDENTIALS]
                [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
                [--jwt-method JWT_METHOD]
                {actions} ... [model-as-positional-argument]

positional arguments:
  {actions}
    actions             Runs the action server.
  model-as-positional-argument
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: None)

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: models)
  --log-file LOG_FILE   Store logs in specified file. (default: rasa_core.log)
  --endpoints ENDPOINTS
                        Configuration file for the model server and the
                        connectors as a yml file. (default: None)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)

Server Settings:
  -p PORT, --port PORT  Port to run the server at. (default: 5005)
  -t AUTH_TOKEN, --auth-token AUTH_TOKEN
                        Enable token based authentication. Requests need to
                        provide the token to be accepted. (default: None)
  --cors [CORS [CORS ...]]
                        Enable CORS for the passed origin. Use * to whitelist
                        all origins. (default: None)
  --enable-api          Start the web server API in addition to the input
                        channel. (default: False)
  --remote-storage REMOTE_STORAGE
                        Set the remote location where your Rasa model is
                        stored, e.g. on AWS. (default: None)

Channels:
  --credentials CREDENTIALS
                        Authentication credentials for the connector as a yml
                        file. (default: None)
  --connector CONNECTOR
                        Service to connect to. (default: None)

JWT Authentication:
  --jwt-secret JWT_SECRET
                        Public key for asymmetric JWT methods or shared
                        secretfor symmetric methods. Please also make sure to
                        use --jwt-method to select the method of the
                        signature, otherwise this argument will be ignored.
                        (default: None)
  --jwt-method JWT_METHOD
                        Method used for the signature of the JWT
                        authentication payload. (default: HS256)
```

For more information on the additional parameters, see Configuring the HTTP API.
See the Rasa HTTP API docs for detailed documentation of all the endpoints.

## Start an Action Server

To run your action server run

```
rasa run actions
```

The following arguments can be used to adapt the server settings:

```
usage: rasa run actions [-h] [-v] [-vv] [--quiet] [-p PORT]
                        [--cors [CORS [CORS ...]]] [--actions ACTIONS]

optional arguments:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  port to run the server at (default: 5055)
  --cors [CORS [CORS ...]]
                        enable CORS for the passed origin. Use * to whitelist
                        all origins (default: None)
  --actions ACTIONS     name of action package to be loaded (default: None)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)
```

## Visualize your Stories

To open a browser tab with a graph showing your stories:

```
rasa visualize
```

Normally, training stories in the directory `data` are visualized. If your stories are located
somewhere else, you can specify their location with `--stories`.

Additional arguments are:

```
usage: rasa visualize [-h] [-v] [-vv] [--quiet] [-d DOMAIN] [-s STORIES]
                      [-c CONFIG] [--out OUT] [--max-history MAX_HISTORY]
                      [-u NLU]

optional arguments:
  -h, --help            show this help message and exit
  -d DOMAIN, --domain DOMAIN
                        Domain specification (yml file). (default: domain.yml)
  -s STORIES, --stories STORIES
                        File or folder containing your training stories.
                        (default: data)
  -c CONFIG, --config CONFIG
                        The policy and NLU pipeline configuration of your bot.
                        (default: config.yml)
  --out OUT             Filename of the output path, e.g. 'graph.html'.
                        (default: graph.html)
  --max-history MAX_HISTORY
                        Max history to consider when merging paths in the
                        output graph. (default: 2)
  -u NLU, --nlu NLU     File or folder containing your NLU data, used to
                        insert example messages into the graph. (default:
                        None)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)
```

## Evaluating a Model on Test Data

To evaluate your model on test data, run:

```
rasa test
```

Specify the model to test using `--model`.
Check out more details in Evaluating an NLU Model and Evaluating a Core Model.

The following arguments are available for `rasa test`:

```
usage: rasa test [-h] [-v] [-vv] [--quiet] [-m MODEL] [-s STORIES]
                 [--max-stories MAX_STORIES] [--out OUT] [--e2e]
                 [--endpoints ENDPOINTS] [--fail-on-prediction-errors]
                 [--url URL] [-u NLU] [--report [REPORT]]
                 [--successes [SUCCESSES]] [--errors ERRORS]
                 [--histogram HISTOGRAM] [--confmat CONFMAT]
                 [--cross-validation] [-c CONFIG] [-f FOLDS]
                 {core,nlu} ...

positional arguments:
  {core,nlu}
    core                Tests a trained Rasa Core model using your test
                        stories.
    nlu                 Tests a trained Rasa NLU model using your test NLU
                        data.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: models)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)

Core Test Arguments:
  -s STORIES, --stories STORIES
                        File or folder containing your test stories. (default:
                        data)
  --max-stories MAX_STORIES
                        Maximum number of stories to test on. (default: None)
  --out OUT             Output path for any files created during the
                        evaluation. (default: results)
  --e2e, --end-to-end   Run an end-to-end evaluation for combined action and
                        intent prediction. Requires a story file in end-to-end
                        format. (default: False)
  --endpoints ENDPOINTS
                        Configuration file for the connectors as a yml file.
                        (default: None)
  --fail-on-prediction-errors
                        If a prediction error is encountered, an exception is
                        thrown. This can be used to validate stories during
                        tests, e.g. on travis. (default: False)
  --url URL             If supplied, downloads a story file from a URL and
                        trains on it. Fetches the data by sending a GET
                        request to the supplied URL. (default: None)

NLU Test Arguments:
  -u NLU, --nlu NLU     File or folder containing your NLU data. (default:
                        data)
  --report [REPORT]     Output path to save the intent/entity metrics report.
                        (default: None)
  --successes [SUCCESSES]
                        Output path to save successful predictions. (default:
                        None)
  --errors ERRORS       Output path to save model errors. (default:
                        errors.json)
  --histogram HISTOGRAM
                        Output path for the confidence histogram. (default:
                        hist.png)
  --confmat CONFMAT     Output path for the confusion matrix plot. (default:
                        confmat.png)
```

## Create a Train-Test Split

To create a split of your NLU data, run:

```
rasa data split nlu
```

You can specify the training data, the fraction, and the output directory using the following arguments:

```
usage: rasa data split nlu [-h] [-v] [-vv] [--quiet] [-u NLU]
                           [--training-fraction TRAINING_FRACTION] [--out OUT]

optional arguments:
  -h, --help            show this help message and exit
  -u NLU, --nlu NLU     File or folder containing your NLU data. (default:
                        data)
  --training-fraction TRAINING_FRACTION
                        Percentage of the data which should be in the training
                        data. (default: 0.8)
  --out OUT             Directory where the split files should be stored.
                        (default: train_test_split)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)
```

This command will attempt to keep the proportions of intents the same in train and test.
If you have NLG data for retrieval actions, this will be saved to seperate files:

```
ls train_test_split

      nlg_test_data.md     test_data.json
      nlg_training_data.md training_data.json
```

## Convert Data Between Markdown and JSON

To convert NLU data from LUIS data format, WIT data format, Dialogflow data format, JSON, or Markdown
to JSON or Markdown, run:

```
rasa data convert nlu
```

You can specify the input file, output file, and the output format with the following arguments:

```
usage: rasa data convert nlu [-h] [-v] [-vv] [--quiet] --data DATA --out OUT
                             [-l LANGUAGE] -f {json,md}

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Path to the file or directory containing Rasa NLU
                        data. (default: None)
  --out OUT             File where to save training data in Rasa format.
                        (default: None)
  -l LANGUAGE, --language LANGUAGE
                        Language of data. (default: en)
  -f {json,md}, --format {json,md}
                        Output format the training data should be converted
                        into. (default: None)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)
```

## Export Conversations to an Event Broker

To export events from a tracker store using an event broker, run:

```
rasa export
```

You can specify the location of the environments file, the minimum and maximum
timestamps of events that should be published, as well as the conversation IDs that
should be published.

```
usage: rasa [-h] [--version]
            {init,run,shell,train,interactive,test,visualize,data,x} ...
rasa: error: invalid choice: 'export' (choose from 'init', 'run', 'shell', 'train', 'interactive', 'test', 'visualize', 'data', 'x')
```

## Start Rasa X

Rasa X is a toolset that helps you leverage conversations to improve your assistant.
You can find more information about it <a class="reference external" href="https://rasa.com/docs/rasa-x/" target="_blank">here</a>.You can start Rasa X locally by executing

```
rasa x
```

To be able to start Rasa X you need to have Rasa X local mode installed
and you need to be in a Rasa project.**NOTE**: By default Rasa X runs on the port 5002. Using the argument `--rasa-x-port` allows you to change it to
any other port.

The following arguments are available for `rasa x`:

```
usage: rasa x [-h] [-v] [-vv] [--quiet] [-m MODEL] [--data DATA] [--no-prompt]
              [--production] [--rasa-x-port RASA_X_PORT] [--log-file LOG_FILE]
              [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
              [--cors [CORS [CORS ...]]] [--enable-api]
              [--remote-storage REMOTE_STORAGE] [--credentials CREDENTIALS]
              [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
              [--jwt-method JWT_METHOD]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: models)
  --data DATA           Path to the file or directory containing stories and
                        Rasa NLU data. (default: data)
  --no-prompt           Automatic yes or default options to prompts and
                        oppressed warnings. (default: False)
  --production          Run Rasa X in a production environment. (default:
                        False)
  --rasa-x-port RASA_X_PORT
                        Port to run the Rasa X server at. (default: 5002)
  --log-file LOG_FILE   Store logs in specified file. (default: rasa_core.log)
  --endpoints ENDPOINTS
                        Configuration file for the model server and the
                        connectors as a yml file. (default: None)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)

Server Settings:
  -p PORT, --port PORT  Port to run the server at. (default: 5005)
  -t AUTH_TOKEN, --auth-token AUTH_TOKEN
                        Enable token based authentication. Requests need to
                        provide the token to be accepted. (default: None)
  --cors [CORS [CORS ...]]
                        Enable CORS for the passed origin. Use * to whitelist
                        all origins. (default: None)
  --enable-api          Start the web server API in addition to the input
                        channel. (default: False)
  --remote-storage REMOTE_STORAGE
                        Set the remote location where your Rasa model is
                        stored, e.g. on AWS. (default: None)

Channels:
  --credentials CREDENTIALS
                        Authentication credentials for the connector as a yml
                        file. (default: None)
  --connector CONNECTOR
                        Service to connect to. (default: None)

JWT Authentication:
  --jwt-secret JWT_SECRET
                        Public key for asymmetric JWT methods or shared
                        secretfor symmetric methods. Please also make sure to
                        use --jwt-method to select the method of the
                        signature, otherwise this argument will be ignored.
                        (default: None)
  --jwt-method JWT_METHOD
                        Method used for the signature of the JWT
                        authentication payload. (default: HS256)
```

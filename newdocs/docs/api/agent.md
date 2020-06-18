# Agent


### class rasa.core.agent.Agent(domain=None, policies=None, interpreter=None, generator=None, tracker_store=None, action_endpoint=None, fingerprint=None, model_directory=None, model_server=None, remote_storage=None)
The Agent class provides a convenient interface for the most important
Rasa functionality.

This includes training, handling messages, loading a dialogue model,
getting the next action, and handling a channel.


#### create_processor(preprocessor=None)
Instantiates a processor based on the set state of the agent.


* **Return type**

    `MessageProcessor`



#### async execute_action(sender_id, action, output_channel, policy, confidence)
Handle a single message.


* **Return type**

    `DialogueStateTracker`



#### handle_channels(channels, http_port=5005, route='/webhooks/', cors=None)
Start a webserver attaching the input channels and handling msgs.


* **Return type**

    `Sanic`



#### async handle_message(message, message_preprocessor=None, \*\*kwargs)
Handle a single message.


* **Return type**

    `Optional`[`List`[`str`]]



#### async handle_text(text_message, message_preprocessor=None, output_channel=None, sender_id='default')
Handle a single message.

If a message preprocessor is passed, the message will be passed to that
function first and the return value is then used as the
input for the dialogue engine.

The return value of this function depends on the `output_channel`. If
the output channel is not set, set to `None`, or set
to `CollectingOutputChannel` this function will return the messages
the bot wants to respond.


* **Example**

    ```python
    >>> from rasa.core.agent import Agent
    >>> from rasa.core.interpreter import RasaNLUInterpreter
    >>> agent = Agent.load("examples/restaurantbot/models/current")
    >>> await agent.handle_text("hello")
    [u'how can I help you?']
    ```



* **Return type**

    `Optional`[`List`[`Dict`[`str`, `Any`]]]



#### is_ready()
Check if all necessary components are instantiated to use agent.


#### classmethod load(unpacked_model_path, interpreter=None, generator=None, tracker_store=None, action_endpoint=None, model_server=None, remote_storage=None)
Load a persisted model from the passed path.


* **Return type**

    `Agent`



#### async load_data(resource_name, remove_duplicates=True, unique_last_num_states=None, augmentation_factor=20, tracker_limit=None, use_story_concatenation=True, debug_plots=False, exclusion_percentage=None)
Load training data from a resource.


* **Return type**

    `List`[`DialogueStateTracker`]



#### async log_message(message, message_preprocessor=None, \*\*kwargs)
Append a message to a dialogue - does not predict actions.


* **Return type**

    `DialogueStateTracker`



#### persist(model_path, dump_flattened_stories=False)
Persists this agent into a directory for later loading and usage.


* **Return type**

    `None`



#### predict_next(sender_id, \*\*kwargs)
Handle a single message.


* **Return type**

    `Dict`[`str`, `Any`]



#### toggle_memoization(activate)
Toggles the memoization on and off.

If a memoization policy is present in the ensemble, this will toggle
the prediction of that policy. When set to `False` the Memoization
policies present in the policy ensemble will not make any predictions.
Hence, the prediction result from the ensemble always needs to come
from a different policy (e.g. `KerasPolicy`). Useful to test
prediction
capabilities of an ensemble when ignoring memorized turns from the
training data.


* **Return type**

    `None`



#### train(training_trackers, \*\*kwargs)
Train the policies / policy ensemble using dialogue data from file.


* **Parameters**

    
    * **training_trackers** – trackers to train on


    * **\*\*kwargs** – additional arguments passed to the underlying ML
    trainer (e.g. keras parameters)



* **Return type**

    `None`

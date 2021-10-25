---
sidebar_label: rasa.core.agent
title: rasa.core.agent
---
#### load\_from\_server

```python
async def load_from_server(agent: Agent, model_server: EndpointConfig) -> Agent
```

Load a persisted model from a server.

#### load\_agent

```python
async def load_agent(model_path: Optional[Text] = None, model_server: Optional[EndpointConfig] = None, remote_storage: Optional[Text] = None, endpoints: Optional[AvailableEndpoints] = None, loop: Optional[AbstractEventLoop] = None) -> Agent
```

Loads agent from server, remote storage or disk.

**Arguments**:

- `model_path` - Path to the model if it&#x27;s on disk.
- `model_server` - Configuration for a potential server which serves the model.
- `remote_storage` - URL of remote storage for model.
- `endpoints` - Endpoint configuration.
- `loop` - Optional async loop to pass to broker creation.
  

**Returns**:

  The instantiated `Agent` or `None`.

#### agent\_must\_be\_ready

```python
def agent_must_be_ready(f: Callable[..., Any]) -> Callable[..., Any]
```

Any Agent method decorated with this will raise if the agent is not ready.

## Agent Objects

```python
class Agent()
```

The Agent class provides an interface for the most important Rasa functionality.

This includes training, handling messages, loading a dialogue model,
getting the next action, and handling a channel.

#### \_\_init\_\_

```python
def __init__(domain: Optional[Union[Text, Domain]] = None, generator: Union[EndpointConfig, NaturalLanguageGenerator, None] = None, tracker_store: Optional[TrackerStore] = None, lock_store: Optional[LockStore] = None, action_endpoint: Optional[EndpointConfig] = None, fingerprint: Optional[Text] = None, model_server: Optional[EndpointConfig] = None, remote_storage: Optional[Text] = None, http_interpreter: Optional[RasaNLUHttpInterpreter] = None)
```

Initializes an `Agent`.

#### load

```python
@classmethod
def load(cls, model_path: Union[Text, Path], domain: Optional[Union[Text, Domain]] = None, generator: Union[EndpointConfig, NaturalLanguageGenerator, None] = None, tracker_store: Optional[TrackerStore] = None, lock_store: Optional[LockStore] = None, action_endpoint: Optional[EndpointConfig] = None, fingerprint: Optional[Text] = None, model_server: Optional[EndpointConfig] = None, remote_storage: Optional[Text] = None, http_interpreter: Optional[RasaNLUHttpInterpreter] = None) -> Agent
```

Constructs a new agent and loads the processer and model.

#### load\_model

```python
def load_model(model_path: Union[Text, Path], fingerprint: Optional[Text] = None) -> None
```

Loads the agent&#x27;s model and processor given a new model path.

#### model\_id

```python
@property
def model_id() -> Optional[Text]
```

Returns the model_id from processor&#x27;s model_metadata.

#### model\_name

```python
@property
def model_name() -> Optional[Text]
```

Returns the model name from processor&#x27;s model_path.

#### is\_ready

```python
def is_ready() -> bool
```

Check if all necessary components are instantiated to use agent.

#### parse\_message

```python
@agent_must_be_ready
async def parse_message(message_data: Text) -> Dict[Text, Any]
```

Handles message text and intent payload input messages.

The return value of this function is parsed_data.

**Arguments**:

- `message_data` _Text_ - Contain the received message in text or\
  intent payload format.
  

**Returns**:

  The parsed message.
  

**Example**:

  
  {\
- `&quot;text&quot;` - &#x27;/greet{&quot;name&quot;:&quot;Rasa&quot;}&#x27;,\
- `&quot;intent&quot;` - {&quot;name&quot;: &quot;greet&quot;, &quot;confidence&quot;: 1.0},\
- `&quot;intent_ranking&quot;` - [{&quot;name&quot;: &quot;greet&quot;, &quot;confidence&quot;: 1.0}],\
- `&quot;entities&quot;` - [{&quot;entity&quot;: &quot;name&quot;, &quot;start&quot;: 6,\
- `&quot;end&quot;` - 21, &quot;value&quot;: &quot;Rasa&quot;}],\
  }

#### handle\_message

```python
async def handle_message(message: UserMessage) -> Optional[List[Dict[Text, Any]]]
```

Handle a single message.

#### predict\_next\_for\_sender\_id

```python
@agent_must_be_ready
async def predict_next_for_sender_id(sender_id: Text) -> Optional[Dict[Text, Any]]
```

Predict the next action for a sender id.

#### predict\_next\_with\_tracker

```python
@agent_must_be_ready
def predict_next_with_tracker(tracker: DialogueStateTracker, verbosity: EventVerbosity = EventVerbosity.AFTER_RESTART) -> Optional[Dict[Text, Any]]
```

Predicts the next action.

#### log\_message

```python
@agent_must_be_ready
async def log_message(message: UserMessage) -> DialogueStateTracker
```

Append a message to a dialogue - does not predict actions.

#### execute\_action

```python
@agent_must_be_ready
async def execute_action(sender_id: Text, action: Text, output_channel: OutputChannel, policy: Optional[Text], confidence: Optional[float]) -> Optional[DialogueStateTracker]
```

Executes an action.

#### trigger\_intent

```python
@agent_must_be_ready
async def trigger_intent(intent_name: Text, entities: List[Dict[Text, Any]], output_channel: OutputChannel, tracker: DialogueStateTracker) -> None
```

Trigger a user intent, e.g. triggered by an external event.

#### handle\_text

```python
@agent_must_be_ready
async def handle_text(text_message: Union[Text, Dict[Text, Any]], output_channel: Optional[OutputChannel] = None, sender_id: Optional[Text] = DEFAULT_SENDER_ID) -> Optional[List[Dict[Text, Any]]]
```

Handle a single message.

If a message preprocessor is passed, the message will be passed to that
function first and the return value is then used as the
input for the dialogue engine.

The return value of this function depends on the ``output_channel``. If
the output channel is not set, set to ``None``, or set
to ``CollectingOutputChannel`` this function will return the messages
the bot wants to respond.

:Example:

&gt;&gt;&gt; from rasa.core.agent import Agent
&gt;&gt;&gt; agent = Agent.load(&quot;examples/moodbot/models&quot;)
&gt;&gt;&gt; await agent.handle_text(&quot;hello&quot;)
[u&#x27;how can I help you?&#x27;]

#### load\_model\_from\_remote\_storage

```python
def load_model_from_remote_storage(model_name: Text) -> None
```

Loads an Agent from remote storage.


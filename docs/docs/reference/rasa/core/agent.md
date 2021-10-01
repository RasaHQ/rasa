---
sidebar_label: rasa.core.agent
title: rasa.core.agent
---
#### load\_from\_server

```python
async def load_from_server(agent: "Agent", model_server: EndpointConfig) -> "Agent"
```

Load a persisted model from a server.

#### create\_agent

```python
def create_agent(model: Text, endpoints: Text = None) -> "Agent"
```

Create an agent instance based on a stored model.

**Arguments**:

- `model` - file path to the stored model
- `endpoints` - file path to the used endpoint configuration

#### load\_agent

```python
async def load_agent(model_path: Optional[Text] = None, model_server: Optional[EndpointConfig] = None, remote_storage: Optional[Text] = None, interpreter: Optional[NaturalLanguageInterpreter] = None, generator: Union[EndpointConfig, NaturalLanguageGenerator] = None, tracker_store: Optional[TrackerStore] = None, lock_store: Optional[LockStore] = None, action_endpoint: Optional[EndpointConfig] = None) -> Optional["Agent"]
```

Loads agent from server, remote storage or disk.

**Arguments**:

- `model_path` - Path to the model if it&#x27;s on disk.
- `model_server` - Configuration for a potential server which serves the model.
- `remote_storage` - URL of remote storage for model.
- `interpreter` - NLU interpreter to parse incoming messages.
- `generator` - Optional response generator.
- `tracker_store` - TrackerStore for persisting the conversation history.
- `lock_store` - LockStore to avoid that a conversation is modified by concurrent
  actors.
- `action_endpoint` - Action server configuration for executing custom actions.
  

**Returns**:

  The instantiated `Agent` or `None`.

## Agent Objects

```python
class Agent()
```

The Agent class provides a convenient interface for the most important
Rasa functionality.

This includes training, handling messages, loading a dialogue model,
getting the next action, and handling a channel.

#### load

```python
@classmethod
def load(cls, model_path: Union[Text, Path], interpreter: Optional[NaturalLanguageInterpreter] = None, generator: Union[EndpointConfig, NaturalLanguageGenerator] = None, tracker_store: Optional[TrackerStore] = None, lock_store: Optional[LockStore] = None, action_endpoint: Optional[EndpointConfig] = None, model_server: Optional[EndpointConfig] = None, remote_storage: Optional[Text] = None, path_to_model_archive: Optional[Text] = None, new_config: Optional[Dict] = None, finetuning_epoch_fraction: float = 1.0) -> "Agent"
```

Load a persisted model from the passed path.

#### is\_core\_ready

```python
def is_core_ready() -> bool
```

Check if all necessary components and policies are ready to use the agent.

#### is\_ready

```python
def is_ready() -> bool
```

Check if all necessary components are instantiated to use agent.

Policies might not be available, if this is an NLU only agent.

#### parse\_message\_using\_nlu\_interpreter

```python
async def parse_message_using_nlu_interpreter(message_data: Text, tracker: DialogueStateTracker = None) -> Dict[Text, Any]
```

Handles message text and intent payload input messages.

The return value of this function is parsed_data.

**Arguments**:

- `message_data` _Text_ - Contain the received message in text or\
  intent payload format.
- `tracker` _DialogueStateTracker_ - Contains the tracker to be\
  used by the interpreter.
  

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
async def handle_message(message: UserMessage, message_preprocessor: Optional[Callable[[Text], Text]] = None, **kwargs: Any, ,) -> Optional[List[Dict[Text, Any]]]
```

Handle a single message.

#### predict\_next

```python
async def predict_next(sender_id: Text, **kwargs: Any) -> Optional[Dict[Text, Any]]
```

Handle a single message.

#### log\_message

```python
async def log_message(message: UserMessage, message_preprocessor: Optional[Callable[[Text], Text]] = None, **kwargs: Any, ,) -> DialogueStateTracker
```

Append a message to a dialogue - does not predict actions.

#### execute\_action

```python
async def execute_action(sender_id: Text, action: Text, output_channel: OutputChannel, policy: Optional[Text], confidence: Optional[float]) -> Optional[DialogueStateTracker]
```

Handle a single message.

#### trigger\_intent

```python
async def trigger_intent(intent_name: Text, entities: List[Dict[Text, Any]], output_channel: OutputChannel, tracker: DialogueStateTracker) -> None
```

Trigger a user intent, e.g. triggered by an external event.

#### handle\_text

```python
async def handle_text(text_message: Union[Text, Dict[Text, Any]], message_preprocessor: Optional[Callable[[Text], Text]] = None, output_channel: Optional[OutputChannel] = None, sender_id: Optional[Text] = DEFAULT_SENDER_ID) -> Optional[List[Dict[Text, Any]]]
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
&gt;&gt;&gt; from rasa.core.interpreter import RasaNLUInterpreter
&gt;&gt;&gt; agent = Agent.load(&quot;examples/moodbot/models&quot;)
&gt;&gt;&gt; await agent.handle_text(&quot;hello&quot;)
[u&#x27;how can I help you?&#x27;]

#### load\_data

```python
def load_data(training_resource: Union[Text, TrainingDataImporter], remove_duplicates: bool = True, unique_last_num_states: Optional[int] = None, augmentation_factor: int = 50, tracker_limit: Optional[int] = None, use_story_concatenation: bool = True, debug_plots: bool = False, exclusion_percentage: Optional[int] = None) -> List["TrackerWithCachedStates"]
```

Load training data from a resource.

#### train

```python
def train(training_trackers: List[DialogueStateTracker], **kwargs: Any) -> None
```

Train the policies / policy ensemble using dialogue data from file.

**Arguments**:

- `training_trackers` - trackers to train on
- `**kwargs` - additional arguments passed to the underlying ML
  trainer (e.g. keras parameters)

#### persist

```python
def persist(model_path: Text) -> None
```

Persists this agent into a directory for later loading and usage.

#### visualize

```python
async def visualize(resource_name: Text, output_file: Text, max_history: Optional[int] = None, nlu_training_data: Optional[TrainingData] = None, should_merge_nodes: bool = True, fontsize: int = 12) -> None
```

Visualize the loaded training data from the resource.

#### create\_processor

```python
def create_processor(preprocessor: Optional[Callable[[Text], Text]] = None) -> MessageProcessor
```

Instantiates a processor based on the set state of the agent.


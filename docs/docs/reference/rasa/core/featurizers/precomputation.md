---
sidebar_label: rasa.core.featurizers.precomputation
title: rasa.core.featurizers.precomputation
---
## MessageContainerForCoreFeaturization Objects

```python
class MessageContainerForCoreFeaturization()
```

A key-value store for specific `Messages`.

This container can be only be used to store messages that contain exactly
one of the following attributes: `ACTION_NAME`, `ACTION_TEXT`, `TEXT`, or `INTENT`.
A combination of the key attribute and the corresponding value will be used as
key for the respective message.

Background/Motivation:
- Our policies only require these attributes to be tokenized and/or featurized
  via NLU graph components, which is why we don&#x27;t care about storing anything else.
- Our tokenizers and featurizers work independently for each attribute,
  which is why we can separate them and ask for &quot;exactly one&quot; of the key
  attributes.
- Our tokenizers add attributes (e.g. token sequences) and not just `Features`,
  which is why we need messages and why we allow messages to contain more than
  just the key attributes.
- Due to the way we use this datastructure, it won&#x27;t contain all features that the
  policies need (cf. `rasa.core.featurizers.SingleStateFeaturizer`) and sometimes
  the messages will contain no features at all, which is the motivation for the
  name of this class.
- Values for different attributes might coincide (e.g. &#x27;greet&#x27; can appear as user
  text as well as name of an intent), but attributes are not all tokenized and
  featurized in the same way, which is why we use the combination of key attribute
  and value to identify a message.

Usage:
- At the start of core&#x27;s featurization pipeline, we use this container to
  de-duplicate the given story data during training (e.g. &quot;Hello&quot; might appear very
  often but it will end up in the training data only once) and to de-duplicate
  the data given in the tracker (e.g. if a text appears repeatedly in the
  dialogue, it will only be featurized once later).
  See: `rasa.core.featurizers.precomputation.CoreFeaturizationInputConverter`.
- At the end of core&#x27;s featurization pipeline, we wrap all resulting
  (training data) messages into this container again.
  See: `rasa.core.featurizers.precomputation.CoreFeaturizationCollector`.

#### \_\_init\_\_

```python
def __init__() -> None
```

Creates an empty container for precomputations.

#### fingerprint

```python
def fingerprint() -> Text
```

Fingerprint the container.

**Returns**:

  hex string as a fingerprint of the container.

#### messages

```python
def messages(key_attribute: Text = None) -> ValuesView
```

Returns a view of all messages.

#### all\_messages

```python
def all_messages() -> List[Message]
```

Returns a list containing all messages.

#### keys

```python
def keys(key_attribute: Text) -> KeysView
```

Returns a view of the value keys for the given key attribute.

#### num\_collisions\_ignored

```python
@property
def num_collisions_ignored() -> int
```

Returns the number of collisions that have been ignored.

#### add

```python
def add(message_with_one_key_attribute: Message) -> None
```

Adds the given message if it is not already present.

**Arguments**:

- `message_with_one_key_attribute` - The message we want to add to the lookup
  table. It must have exactly one key attribute.
  

**Raises**:

  `ValueError` if the given message does not contain exactly one key
  attribute or if there is a collision with a message that has a different
  hash value

#### add\_all

```python
def add_all(messages_with_one_key_attribute: List[Message]) -> None
```

Adds the given messages.

**Arguments**:

- `messages_with_one_key_attribute` - The messages that we want to add.
  Each one must have exactly one key attribute.
  

**Raises**:

  `ValueError` if we cannot create a key for the given message or if there is
  a collisions with a message that has a different hash value

#### collect\_features

```python
def collect_features(sub_state: SubState, attributes: Optional[Iterable[Text]] = None) -> Dict[Text, List[Features]]
```

Collects features for all attributes in the given substate.

There might be be multiple messages in the container that contain features
relevant for the given substate, e.g. this is the case if `TEXT` and
`INTENT` are present in the given substate. All of those messages will be
collected and their features combined.

**Arguments**:

- `sub_state` - substate for which we want to extract the relevent features
- `attributes` - if not `None`, this specifies the list of the attributes of the
  `Features` that we&#x27;re interested in (i.e. all other `Features` contained
  in the relevant messages will be ignored)
  

**Returns**:

  a dictionary that maps all the (requested) attributes to a list of `Features`
  

**Raises**:

- ``ValueError`` - if there exists some key pair (i.e. key attribute and
  corresponding value) from the given substate cannot be found
- ``RuntimeError`` - if features for the same attribute are found in two
  different messages that are associated with the given substate

#### lookup\_message

```python
def lookup_message(user_text: Text) -> Message
```

Returns a message that contains the given user text.

**Arguments**:

- `user_text` - the text of a user utterance

**Raises**:

  `ValueError` if there is no message associated with the given user text

#### derive\_messages\_from\_domain\_and\_add

```python
def derive_messages_from_domain_and_add(domain: Domain) -> None
```

Adds all lookup table entries that can be derived from the domain.

That is, all action names, action texts, and intents defined in the domain
will be turned into a (separate) messages and added to this lookup table.

**Arguments**:

- `domain` - the domain from which we extract the substates

#### derive\_messages\_from\_events\_and\_add

```python
def derive_messages_from_events_and_add(events: Iterable[Event]) -> None
```

Adds all relevant messages that can be derived from the given events.

That is, each action name, action text, user text and intent that can be
found in the given events will be turned into a (separate) message and added
to this container.

**Arguments**:

- `events` - list of events to extract the substate from

## CoreFeaturizationInputConverter Objects

```python
class CoreFeaturizationInputConverter(GraphComponent)
```

Provides data for the featurization pipeline.

During training as well as during inference, the converter de-duplicates the given
data (i.e. story graph or list of messages) such that each text and intent from a
user message and each action name and action text appears exactly once.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> CoreFeaturizationInputConverter
```

Creates a new instance (see parent class for full docstring).

#### convert\_for\_training

```python
def convert_for_training(domain: Domain, story_graph: StoryGraph) -> TrainingData
```

Creates de-duplicated training data.

Each possible user text and intent and each action name and action text
that can be found in the given domain and story graph appears exactly once
in the resulting training data. Moreover, each item is contained in a separate
messsage.

**Arguments**:

- `domain` - the domain
- `story_graph` - a story graph

**Returns**:

  training data

#### convert\_for\_inference

```python
def convert_for_inference(tracker: DialogueStateTracker) -> List[Message]
```

Creates a list of messages containing single user and action attributes.

Each possible user text and intent and each action name and action text
that can be found in the events of the given tracker will appear exactly once
in the resulting messages. Moreover, each item is contained in a separate
messsage.

**Arguments**:

- `tracker` - a dialogue state tracker containing events

**Returns**:

  a list of messages

## CoreFeaturizationCollector Objects

```python
class CoreFeaturizationCollector(GraphComponent)
```

Collects featurized messages for use by a policy.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> CoreFeaturizationCollector
```

Creates a new instance (see parent class for full docstring).

#### collect

```python
def collect(messages: Union[TrainingData, List[Message]]) -> MessageContainerForCoreFeaturization
```

Collects messages.


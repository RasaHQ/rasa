---
sidebar_label: rasa.shared.core.domain
title: rasa.shared.core.domain
---
## InvalidDomain Objects

```python
class InvalidDomain(RasaException)
```

Exception that can be raised when domain is not valid.

## ActionNotFoundException Objects

```python
class ActionNotFoundException(ValueError,  RasaException)
```

Raised when an action name could not be found.

## SessionConfig Objects

```python
class SessionConfig(NamedTuple)
```

The Session Configuration.

#### default

```python
 | @staticmethod
 | default() -> "SessionConfig"
```

Returns the SessionConfig with the default values.

#### are\_sessions\_enabled

```python
 | are_sessions_enabled() -> bool
```

Returns a boolean value depending on the value of session_expiration_time.

## Domain Objects

```python
class Domain()
```

The domain specifies the universe in which the bot&#x27;s policy acts.

A Domain subclass provides the actions the bot can take, the intents
and entities it can recognise.

#### from\_file

```python
 | @classmethod
 | from_file(cls, path: Text) -> "Domain"
```

Loads the `Domain` from a YAML file.

#### from\_yaml

```python
 | @classmethod
 | from_yaml(cls, yaml: Text, original_filename: Text = "") -> "Domain"
```

Loads the `Domain` from YAML text after validating it.

#### from\_dict

```python
 | @classmethod
 | from_dict(cls, data: Dict) -> "Domain"
```

Deserializes and creates domain.

**Arguments**:

- `data` - The serialized domain.
  

**Returns**:

  The instantiated `Domain` object.

#### from\_directory

```python
 | @classmethod
 | from_directory(cls, path: Text) -> "Domain"
```

Loads and merges multiple domain files recursively from a directory tree.

#### merge

```python
 | merge(domain: Optional["Domain"], override: bool = False) -> "Domain"
```

Merge this domain with another one, combining their attributes.

List attributes like ``intents`` and ``actions`` will be deduped
and merged. Single attributes will be taken from `self` unless
override is `True`, in which case they are taken from `domain`.

#### retrieval\_intents

```python
 | @rasa.shared.utils.common.lazy_property
 | retrieval_intents() -> List[Text]
```

List retrieval intents present in the domain.

#### collect\_entity\_properties

```python
 | @classmethod
 | collect_entity_properties(cls, domain_entities: List[Union[Text, Dict[Text, Any]]]) -> Tuple[List[Text], Dict[Text, List[Text]], Dict[Text, List[Text]]]
```

Get entity properties for a domain from what is provided by a domain file.

**Arguments**:

- `domain_entities` - The entities as provided by a domain file.
  

**Returns**:

  A list of entity names.
  A dictionary of entity names to roles.
  A dictionary of entity names to groups.

#### collect\_intent\_properties

```python
 | @classmethod
 | collect_intent_properties(cls, intents: List[Union[Text, Dict[Text, Any]]], entities: List[Text], roles: Dict[Text, List[Text]], groups: Dict[Text, List[Text]]) -> Dict[Text, Dict[Text, Union[bool, List]]]
```

Get intent properties for a domain from what is provided by a domain file.

**Arguments**:

- `intents` - The intents as provided by a domain file.
- `entities` - All entities as provided by a domain file.
- `roles` - The roles of entities as provided by a domain file.
- `groups` - The groups of entities as provided by a domain file.
  

**Returns**:

  The intent properties to be stored in the domain.

#### \_\_init\_\_

```python
 | __init__(intents: Union[Set[Text], List[Text], List[Dict[Text, Any]]], entities: List[Union[Text, Dict[Text, Any]]], slots: List[Slot], responses: Dict[Text, List[Dict[Text, Any]]], action_names: List[Text], forms: Union[Dict[Text, Any], List[Text]], action_texts: Optional[List[Text]] = None, store_entities_as_slots: bool = True, session_config: SessionConfig = SessionConfig.default()) -> None
```

Creates a `Domain`.

**Arguments**:

- `intents` - Intent labels.
- `entities` - The names of entities which might be present in user messages.
- `slots` - Slots to store information during the conversation.
- `responses` - Bot responses. If an action with the same name is executed, it
  will send the matching response to the user.
- `action_names` - Names of custom actions.
- `forms` - Form names and their slot mappings.
- `action_texts` - End-to-End bot utterances from end-to-end stories.
- `store_entities_as_slots` - If `True` Rasa will automatically create `SlotSet`
  events for entities if there are slots with the same name as the entity.
- `session_config` - Configuration for conversation sessions. Conversations are
  restarted at the end of a session.

#### \_\_deepcopy\_\_

```python
 | __deepcopy__(memo: Optional[Dict[int, Any]]) -> "Domain"
```

Enables making a deep copy of the `Domain` using `copy.deepcopy`.

See https://docs.python.org/3/library/copy.html#copy.deepcopy
for more implementation.

**Arguments**:

- `memo` - Optional dictionary of objects already copied during the current
  copying pass.
  

**Returns**:

  A deep copy of the current domain.

#### count\_conditional\_response\_variations

```python
 | count_conditional_response_variations() -> int
```

Returns count of conditional response variations.

#### \_\_hash\_\_

```python
 | __hash__() -> int
```

Returns a unique hash for the domain.

#### fingerprint

```python
 | fingerprint() -> Text
```

Returns a unique hash for the domain which is stable across python runs.

**Returns**:

  fingerprint of the domain

#### user\_actions\_and\_forms

```python
 | @rasa.shared.utils.common.lazy_property
 | user_actions_and_forms() -> List[Text]
```

Returns combination of user actions and forms.

#### action\_names

```python
 | @rasa.shared.utils.common.lazy_property
 | action_names() -> List[Text]
```

Returns action names or texts.

#### num\_actions

```python
 | @rasa.shared.utils.common.lazy_property
 | num_actions() -> int
```

Returns the number of available actions.

#### num\_states

```python
 | @rasa.shared.utils.common.lazy_property
 | num_states() -> int
```

Number of used input states for the action prediction.

#### retrieval\_intent\_templates

```python
 | @rasa.shared.utils.common.lazy_property
 | retrieval_intent_templates() -> Dict[Text, List[Dict[Text, Any]]]
```

Return only the responses which are defined for retrieval intents.

#### retrieval\_intent\_responses

```python
 | @rasa.shared.utils.common.lazy_property
 | retrieval_intent_responses() -> Dict[Text, List[Dict[Text, Any]]]
```

Return only the responses which are defined for retrieval intents.

#### templates

```python
 | @rasa.shared.utils.common.lazy_property
 | templates() -> Dict[Text, List[Dict[Text, Any]]]
```

Temporary property before templates become completely deprecated.

#### is\_retrieval\_intent\_template

```python
 | @staticmethod
 | is_retrieval_intent_template(response: Tuple[Text, List[Dict[Text, Any]]]) -> bool
```

Check if the response is for a retrieval intent.

These templates have a `/` symbol in their name. Use that to filter them from
the rest.

#### is\_retrieval\_intent\_response

```python
 | @staticmethod
 | is_retrieval_intent_response(response: Tuple[Text, List[Dict[Text, Any]]]) -> bool
```

Check if the response is for a retrieval intent.

These responses have a `/` symbol in their name. Use that to filter them from
the rest.

#### add\_categorical\_slot\_default\_value

```python
 | add_categorical_slot_default_value() -> None
```

See `_add_categorical_slot_default_value` for docstring.

#### add\_requested\_slot

```python
 | add_requested_slot() -> None
```

See `_add_categorical_slot_default_value` for docstring.

#### add\_knowledge\_base\_slots

```python
 | add_knowledge_base_slots() -> None
```

See `_add_categorical_slot_default_value` for docstring.

#### index\_for\_action

```python
 | index_for_action(action_name: Text) -> int
```

Looks up which action index corresponds to this action name.

#### raise\_action\_not\_found\_exception

```python
 | raise_action_not_found_exception(action_name_or_text: Text) -> NoReturn
```

Raises exception if action name or text not part of the domain or stories.

**Arguments**:

- `action_name_or_text` - Name of an action or its text in case it&#x27;s an
  end-to-end bot utterance.
  

**Raises**:

- `ActionNotFoundException` - If `action_name_or_text` are not part of this
  domain.

#### random\_template\_for

```python
 | random_template_for(utter_action: Text) -> Optional[Dict[Text, Any]]
```

Returns a random response for an action name.

**Arguments**:

- `utter_action` - The name of the utter action.
  

**Returns**:

  A response for an utter action.

#### slot\_states

```python
 | @rasa.shared.utils.common.lazy_property
 | slot_states() -> List[Text]
```

Returns all available slot state strings.

#### entity\_states

```python
 | @rasa.shared.utils.common.lazy_property
 | entity_states() -> List[Text]
```

Returns all available entity state strings.

#### concatenate\_entity\_labels

```python
 | @staticmethod
 | concatenate_entity_labels(entity_labels: Dict[Text, List[Text]], entity: Optional[Text] = None) -> List[Text]
```

Concatenates the given entity labels with their corresponding sub-labels.

If a specific entity label is given, only this entity label will be
concatenated with its corresponding sub-labels.

**Arguments**:

- `entity_labels` - A map of an entity label to its sub-label list.
- `entity` - If present, only this entity will be considered.
  

**Returns**:

  A list of labels.

#### input\_state\_map

```python
 | @rasa.shared.utils.common.lazy_property
 | input_state_map() -> Dict[Text, int]
```

Provide a mapping from state names to indices.

#### input\_states

```python
 | @rasa.shared.utils.common.lazy_property
 | input_states() -> List[Text]
```

Returns all available states.

#### get\_active\_state

```python
 | get_active_state(tracker: "DialogueStateTracker", omit_unset_slots: bool = False) -> State
```

Given a dialogue tracker, makes a representation of current dialogue state.

**Arguments**:

- `tracker` - dialog state tracker containing the dialog so far
- `omit_unset_slots` - If `True` do not include the initial values of slots.
  

**Returns**:

  A representation of the dialogue&#x27;s current state.

#### states\_for\_tracker\_history

```python
 | states_for_tracker_history(tracker: "DialogueStateTracker", omit_unset_slots: bool = False, ignore_rule_only_turns: bool = False, rule_only_data: Optional[Dict[Text, Any]] = None) -> List[State]
```

List of states for each state of the trackers history.

**Arguments**:

- `tracker` - Dialogue state tracker containing the dialogue so far.
- `omit_unset_slots` - If `True` do not include the initial values of slots.
- `ignore_rule_only_turns` - If True ignore dialogue turns that are present
  only in rules.
- `rule_only_data` - Slots and loops,
  which only occur in rules but not in stories.
  

**Returns**:

  A list of states.

#### slots\_for\_entities

```python
 | slots_for_entities(entities: List[Dict[Text, Any]]) -> List[SlotSet]
```

Creates slot events for entities if auto-filling is enabled.

**Arguments**:

- `entities` - The list of entities.
  

**Returns**:

  A list of `SlotSet` events.

#### persist\_specification

```python
 | persist_specification(model_path: Text) -> None
```

Persist the domain specification to storage.

#### load\_specification

```python
 | @classmethod
 | load_specification(cls, path: Text) -> Dict[Text, Any]
```

Load a domains specification from a dumped model directory.

#### compare\_with\_specification

```python
 | compare_with_specification(path: Text) -> bool
```

Compare the domain spec of the current and the loaded domain.

Throws exception if the loaded domain specification is different
to the current domain are different.

#### as\_dict

```python
 | as_dict() -> Dict[Text, Any]
```

Return serialized `Domain`.

#### get\_responses\_with\_multilines

```python
 | @staticmethod
 | get_responses_with_multilines(responses: Dict[Text, List[Dict[Text, Any]]]) -> Dict[Text, List[Dict[Text, Any]]]
```

Returns `responses` with preserved multilines in the `text` key.

**Arguments**:

- `responses` - Original `responses`.
  

**Returns**:

  `responses` with preserved multilines in the `text` key.

#### cleaned\_domain

```python
 | cleaned_domain() -> Dict[Text, Any]
```

Fetch cleaned domain to display or write into a file.

The internal `used_entities` property is replaced by `use_entities` or
`ignore_entities` and redundant keys are replaced with default values
to make the domain easier readable.

**Returns**:

  A cleaned dictionary version of the domain.

#### persist

```python
 | persist(filename: Union[Text, Path]) -> None
```

Write domain to a file.

#### persist\_clean

```python
 | persist_clean(filename: Union[Text, Path]) -> None
```

Write cleaned domain to a file.

#### as\_yaml

```python
 | as_yaml(clean_before_dump: bool = False) -> Text
```

Dump the `Domain` object as a YAML string.
This function preserves the orders of the keys in the domain.

**Arguments**:

- `clean_before_dump` - When set to `True`, this method returns
  a version of the domain without internal
  information. Defaults to `False`.

**Returns**:

  A string in YAML format representing the domain.

#### intent\_config

```python
 | intent_config(intent_name: Text) -> Dict[Text, Any]
```

Return the configuration for an intent.

#### intents

```python
 | @rasa.shared.utils.common.lazy_property
 | intents() -> List[Text]
```

Returns sorted list of intents.

#### domain\_warnings

```python
 | domain_warnings(intents: Optional[Union[List[Text], Set[Text]]] = None, entities: Optional[Union[List[Text], Set[Text]]] = None, actions: Optional[Union[List[Text], Set[Text]]] = None, slots: Optional[Union[List[Text], Set[Text]]] = None) -> Dict[Text, Dict[Text, Set[Text]]]
```

Generate domain warnings from intents, entities, actions and slots.

Returns a dictionary with entries for `intent_warnings`,
`entity_warnings`, `action_warnings` and `slot_warnings`. Excludes domain slots
from domain warnings in case they are not featurized.

#### check\_missing\_templates

```python
 | check_missing_templates() -> None
```

Warn user of utterance names which have no specified response.

#### check\_missing\_responses

```python
 | check_missing_responses() -> None
```

Warn user of utterance names which have no specified response.

#### is\_empty

```python
 | is_empty() -> bool
```

Check whether the domain is empty.

#### is\_domain\_file

```python
 | @staticmethod
 | is_domain_file(filename: Text) -> bool
```

Checks whether the given file path is a Rasa domain file.

**Arguments**:

- `filename` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a domain file, otherwise `False`.
  

**Raises**:

- `YamlException` - if the file seems to be a YAML file (extension) but
  can not be read / parsed.

#### slot\_mapping\_for\_form

```python
 | slot_mapping_for_form(form_name: Text) -> Dict[Text, Any]
```

Retrieve the slot mappings for a form which are defined in the domain.

Options:
- an extracted entity
- intent: value pairs
- trigger_intent: value pairs
- a whole message
or a list of them, where the first match will be picked

**Arguments**:

- `form_name` - The name of the form.
  

**Returns**:

  The slot mapping or an empty dictionary in case no mapping was found.

## SlotMapping Objects

```python
class SlotMapping(Enum)
```

Defines the available slot mappings.

#### \_\_str\_\_

```python
 | __str__() -> Text
```

Returns a string representation of the object.

#### validate

```python
 | @staticmethod
 | validate(mapping: Dict[Text, Any], form_name: Text, slot_name: Text) -> None
```

Validates a slot mapping.

**Arguments**:

- `mapping` - The mapping which is validated.
- `form_name` - The name of the form which uses this slot mapping.
- `slot_name` - The name of the slot which is mapped by this mapping.
  

**Raises**:

- `InvalidDomain` - In case the slot mapping is not valid.


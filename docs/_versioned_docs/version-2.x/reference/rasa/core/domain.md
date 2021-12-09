---
sidebar_label: rasa.core.domain
title: rasa.core.domain
---
## InvalidDomain Objects

```python
class InvalidDomain(Exception)
```

Exception that can be raised when domain is not valid.

## Domain Objects

```python
class Domain()
```

The domain specifies the universe in which the bot&#x27;s policy acts.

A Domain subclass provides the actions the bot can take, the intents
and entities it can recognise.

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

#### collect\_intent\_properties

```python
 | @classmethod
 | collect_intent_properties(cls, intents: List[Union[Text, Dict[Text, Any]]], entities: List[Text]) -> Dict[Text, Dict[Text, Union[bool, List]]]
```

Get intent properties for a domain from what is provided by a domain file.

**Arguments**:

- `intents` - The intents as provided by a domain file.
- `entities` - All entities as provided by a domain file.
  

**Returns**:

  The intent properties to be stored in the domain.

#### user\_actions\_and\_forms

```python
 | @lazy_property
 | user_actions_and_forms()
```

Returns combination of user actions and forms.

#### num\_actions

```python
 | @lazy_property
 | num_actions()
```

Returns the number of available actions.

#### num\_states

```python
 | @lazy_property
 | num_states()
```

Number of used input states for the action prediction.

#### add\_categorical\_slot\_default\_value

```python
 | add_categorical_slot_default_value() -> None
```

Add a default value to all categorical slots.

All unseen values found for the slot will be mapped to this default value
for featurization.

#### add\_requested\_slot

```python
 | add_requested_slot() -> None
```

Add a slot called `requested_slot` to the list of slots.

The value of this slot will hold the name of the slot which the user
needs to fill in next (either explicitly or implicitly) as part of a form.

#### add\_knowledge\_base\_slots

```python
 | add_knowledge_base_slots() -> None
```

Add slots for the knowledge base action to the list of slots, if the
default knowledge base action name is present.

As soon as the knowledge base action is not experimental anymore, we should
consider creating a new section in the domain file dedicated to knowledge
base slots.

#### action\_for\_name

```python
 | action_for_name(action_name: Text, action_endpoint: Optional[EndpointConfig]) -> Optional[Action]
```

Look up which action corresponds to this action name.

#### action\_for\_index

```python
 | action_for_index(index: int, action_endpoint: Optional[EndpointConfig]) -> Optional[Action]
```

Integer index corresponding to an actions index in the action list.

This method resolves the index to the actions name.

#### index\_for\_action

```python
 | index_for_action(action_name: Text) -> Optional[int]
```

Look up which action index corresponds to this action name.

#### slot\_states

```python
 | @lazy_property
 | slot_states() -> List[Text]
```

Returns all available slot state strings.

#### input\_state\_map

```python
 | @lazy_property
 | input_state_map() -> Dict[Text, int]
```

Provide a mapping from state names to indices.

#### input\_states

```python
 | @lazy_property
 | input_states() -> List[Text]
```

Returns all available states.

#### get\_active\_states

```python
 | get_active_states(tracker: "DialogueStateTracker") -> State
```

Return a bag of active states from the tracker state.

#### states\_for\_tracker\_history

```python
 | states_for_tracker_history(tracker: "DialogueStateTracker") -> List[State]
```

Array of states for each state of the trackers history.

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

#### persist

```python
 | persist(filename: Union[Text, Path]) -> None
```

Write domain to a file.

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

#### persist\_clean

```python
 | persist_clean(filename: Text) -> None
```

Write cleaned domain to a file.

#### intent\_config

```python
 | intent_config(intent_name: Text) -> Dict[Text, Any]
```

Return the configuration for an intent.

#### domain\_warnings

```python
 | domain_warnings(intents: Optional[Union[List[Text], Set[Text]]] = None, entities: Optional[Union[List[Text], Set[Text]]] = None, actions: Optional[Union[List[Text], Set[Text]]] = None, slots: Optional[Union[List[Text], Set[Text]]] = None) -> Dict[Text, Dict[Text, Set[Text]]]
```

Generate domain warnings from intents, entities, actions and slots.

Returns a dictionary with entries for `intent_warnings`,
`entity_warnings`, `action_warnings` and `slot_warnings`. Excludes domain slots
of type `UnfeaturizedSlot` from domain warnings.

#### check\_missing\_templates

```python
 | check_missing_templates() -> None
```

Warn user of utterance names which have no specified template.

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


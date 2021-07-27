---
sidebar_label: rasa.core.actions.forms
title: rasa.core.actions.forms
---
## FormAction Objects

```python
class FormAction(LoopAction)
```

Action which implements and executes the form logic.

#### \_\_init\_\_

```python
 | __init__(form_name: Text, action_endpoint: Optional[EndpointConfig]) -> None
```

Creates a `FormAction`.

**Arguments**:

- `form_name` - Name of the form.
- `action_endpoint` - Endpoint to execute custom actions.

#### name

```python
 | name() -> Text
```

Return the form name.

#### required\_slots

```python
 | required_slots(domain: Domain) -> List[Text]
```

A list of required slots that the form has to fill.

**Returns**:

  A list of slot names.

#### from\_entity

```python
 | from_entity(entity: Text, intent: Optional[Union[Text, List[Text]]] = None, not_intent: Optional[Union[Text, List[Text]]] = None, role: Optional[Text] = None, group: Optional[Text] = None) -> Dict[Text, Any]
```

A dictionary for slot mapping to extract slot value.

From:
- an extracted entity
- conditioned on
    - intent if it is not None
    - not_intent if it is not None,
        meaning user intent should not be this intent
    - role if it is not None
    - group if it is not None

#### get\_mappings\_for\_slot

```python
 | get_mappings_for_slot(slot_to_fill: Text, domain: Domain) -> List[Dict[Text, Any]]
```

Get mappings for requested slot.

If None, map requested slot to an entity with the same name

#### get\_ignored\_intents

```python
 | get_ignored_intents(domain: Domain) -> List[Text]
```

Returns a list of ignored intents.

**Arguments**:

- `domain` - The current model domain.
  

**Returns**:

  The value/s found in `ignored_intents` parameter in the `domain.yml`
  (under forms).

#### intent\_is\_desired

```python
 | intent_is_desired(requested_slot_mapping: Dict[Text, Any], tracker: "DialogueStateTracker", domain: Domain) -> bool
```

Check whether user intent matches intent conditions.

#### entity\_is\_desired

```python
 | entity_is_desired(slot_mapping: Dict[Text, Any], slot: Text, entity_type_of_slot_to_fill: Optional[Text], tracker: DialogueStateTracker, domain: Domain) -> bool
```

Check whether slot should be filled by an entity in the input or not.

**Arguments**:

- `slot_mapping` - Slot mapping.
- `slot` - The slot to be filled.
- `entity_type_of_slot_to_fill` - Entity type of slot to fill.
- `tracker` - The tracker.
- `domain` - The domain.
  

**Returns**:

  True, if slot should be filled, false otherwise.

#### get\_entity\_value\_for\_slot

```python
 | @staticmethod
 | get_entity_value_for_slot(name: Text, tracker: "DialogueStateTracker", slot_to_be_filled: Text, role: Optional[Text] = None, group: Optional[Text] = None) -> Any
```

Extract entities for given name and optional role and group.

**Arguments**:

- `name` - entity type (name) of interest
- `tracker` - the tracker
- `slot_to_be_filled` - Slot which is supposed to be filled by this entity.
- `role` - optional entity role of interest
- `group` - optional entity group of interest
  

**Returns**:

  Value of entity.

#### extract\_other\_slots

```python
 | extract_other_slots(tracker: DialogueStateTracker, domain: Domain) -> Dict[Text, Any]
```

Extract the values of the other slots
if they are set by corresponding entities from the user input
else return `None`.

#### get\_slot\_to\_fill

```python
 | get_slot_to_fill(tracker: "DialogueStateTracker") -> Optional[str]
```

Gets the name of the slot which should be filled next.

When switching to another form, the requested slot setting is still from the
previous form and must be ignored.

**Returns**:

  The slot name or `None`

#### extract\_requested\_slot

```python
 | extract_requested_slot(tracker: "DialogueStateTracker", domain: Domain, slot_to_fill: Text) -> Dict[Text, Any]
```

Extract the value of requested slot from a user input else return `None`.

**Arguments**:

- `tracker` - a DialogueStateTracker instance
- `domain` - the current domain
- `slot_to_fill` - the name of the slot to fill
  

**Returns**:

  a dictionary with one key being the name of the slot to fill
  and its value being the slot value, or an empty dictionary
  if no slot value was found.

#### validate\_slots

```python
 | async validate_slots(slot_candidates: Dict[Text, Any], tracker: "DialogueStateTracker", domain: Domain, output_channel: OutputChannel, nlg: NaturalLanguageGenerator) -> List[Union[SlotSet, Event]]
```

Validate the extracted slots.

If a custom action is available for validating the slots, we call it to validate
them. Otherwise there is no validation.

**Arguments**:

- `slot_candidates` - Extracted slots which are candidates to fill the slots
  required by the form.
- `tracker` - The current conversation tracker.
- `domain` - The current model domain.
- `output_channel` - The output channel which can be used to send messages
  to the user.
- `nlg` - `NaturalLanguageGenerator` to use for response generation.
  

**Returns**:

  The validation events including potential bot messages and `SlotSet` events
  for the validated slots.

#### validate

```python
 | async validate(tracker: "DialogueStateTracker", domain: Domain, output_channel: OutputChannel, nlg: NaturalLanguageGenerator) -> List[Union[SlotSet, Event]]
```

Extract and validate value of requested slot.

If nothing was extracted reject execution of the form action.
Subclass this method to add custom validation and rejection logic

#### request\_next\_slot

```python
 | async request_next_slot(tracker: "DialogueStateTracker", domain: Domain, output_channel: OutputChannel, nlg: NaturalLanguageGenerator, events_so_far: List[Event]) -> List[Union[SlotSet, Event]]
```

Request the next slot and response if needed, else return `None`.

#### activate

```python
 | async activate(output_channel: "OutputChannel", nlg: "NaturalLanguageGenerator", tracker: "DialogueStateTracker", domain: "Domain") -> List[Event]
```

Activate form if the form is called for the first time.

If activating, validate any required slots that were filled before
form activation and return `Form` event with the name of the form, as well
as any `SlotSet` events from validation of pre-filled slots.

**Arguments**:

- `output_channel` - The output channel which can be used to send messages
  to the user.
- `nlg` - `NaturalLanguageGenerator` to use for response generation.
- `tracker` - Current conversation tracker of the user.
- `domain` - Current model domain.
  

**Returns**:

  Events from the activation.


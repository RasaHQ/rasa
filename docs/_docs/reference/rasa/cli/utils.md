---
sidebar_label: rasa.cli.utils
title: rasa.cli.utils
---
#### get\_validated\_path

```python
get_validated_path(current: Optional[Text], parameter: Text, default: Optional[Text] = None, none_is_valid: bool = False) -> Optional[Text]
```

Check whether a file path or its default value is valid and returns it.

**Arguments**:

- `current` - The parsed value.
- `parameter` - The name of the parameter.
- `default` - The default value of the parameter.
- `none_is_valid` - `True` if `None` is valid value for the path,
  else `False``
  

**Returns**:

  The current value if it was valid, else the default value of the
  argument if it is valid, else `None`.

#### cancel\_cause\_not\_found

```python
cancel_cause_not_found(current: Optional[Text], parameter: Text, default: Optional[Text]) -> None
```

Exits with an error because the given path was not valid.

**Arguments**:

- `current` - The path given by the user.
- `parameter` - The name of the parameter.
- `default` - The default value of the parameter.

#### parse\_last\_positional\_argument\_as\_model\_path

```python
parse_last_positional_argument_as_model_path() -> None
```

Fixes the parsing of a potential positional model path argument.

#### button\_to\_string

```python
button_to_string(button: Dict[Text, Any], idx: int = 0) -> Text
```

Create a string representation of a button.

#### element\_to\_string

```python
element_to_string(element: Dict[Text, Any], idx: int = 0) -> Text
```

Create a string representation of an element.

#### button\_choices\_from\_message\_data

```python
button_choices_from_message_data(message: Dict[Text, Any], allow_free_text_input: bool = True) -> List[Text]
```

Return list of choices to present to the user.

If allow_free_text_input is True, an additional option is added
at the end along with the response buttons that allows the user
to type in free text.

#### payload\_from\_button\_question

```python
payload_from_button_question(button_question: "Question") -> Text
```

Prompt user with a button question and returns the nlu payload.

#### signal\_handler

```python
signal_handler(_: int, __: FrameType) -> None
```

Kills Rasa when OS signal is received.


---
sidebar_label: rasa.cli.utils
title: rasa.cli.utils
---
#### get\_validated\_path

```python
def get_validated_path(
        current: Optional[Union["Path", Text]],
        parameter: Text,
        default: Optional[Union["Path", Text]] = None,
        none_is_valid: bool = False) -> Optional[Union["Path", Text]]
```

Checks whether a file path or its default value is valid and returns it.

**Arguments**:

- `current` - The parsed value.
- `parameter` - The name of the parameter.
- `default` - The default value of the parameter.
- `none_is_valid` - `True` if `None` is valid value for the path,
  else `False``
  

**Returns**:

  The current value if it was valid, else the default value of the
  argument if it is valid, else `None`.

#### missing\_config\_keys

```python
def missing_config_keys(path: Union["Path", Text],
                        mandatory_keys: List[Text]) -> List[Text]
```

Checks whether the config file at `path` contains the `mandatory_keys`.

**Arguments**:

- `path` - The path to the config file.
- `mandatory_keys` - A list of mandatory config keys.
  

**Returns**:

  The list of missing config keys.

#### validate\_assistant\_id\_in\_config

```python
def validate_assistant_id_in_config(config_file: Union["Path", Text]) -> None
```

Verifies that the assistant_id key exists and has a unique value in config.

Issues a warning if the key does not exist or has the default value and replaces it
with a pseudo-random string value.

#### validate\_config\_path

```python
def validate_config_path(config: Optional[Union[Text, "Path"]],
                         default_config: Text = DEFAULT_CONFIG_PATH) -> Text
```

Verifies that the config path exists.

Exit if the config file does not exist.

**Arguments**:

- `config` - Path to the config file.
- `default_config` - default config to use if the file at `config` doesn&#x27;t exist.
  
- `Returns` - The path to the config file.

#### validate\_mandatory\_config\_keys

```python
def validate_mandatory_config_keys(config: Union[Text, "Path"],
                                   mandatory_keys: List[Text]) -> Text
```

Get a config from a config file and check if it is valid.

Exit if the config isn&#x27;t valid.

**Arguments**:

- `config` - Path to the config file.
- `mandatory_keys` - The keys that have to be specified in the config file.
  
- `Returns` - The path to the config file if the config is valid.

#### get\_validated\_config

```python
def get_validated_config(config: Optional[Union[Text, "Path"]],
                         mandatory_keys: List[Text],
                         default_config: Text = DEFAULT_CONFIG_PATH) -> Text
```

Validates config and returns path to validated config file.

#### cancel\_cause\_not\_found

```python
def cancel_cause_not_found(current: Optional[Union["Path",
                                                   Text]], parameter: Text,
                           default: Optional[Union["Path", Text]]) -> None
```

Exits with an error because the given path was not valid.

**Arguments**:

- `current` - The path given by the user.
- `parameter` - The name of the parameter.
- `default` - The default value of the parameter.

#### parse\_last\_positional\_argument\_as\_model\_path

```python
def parse_last_positional_argument_as_model_path() -> None
```

Fixes the parsing of a potential positional model path argument.

#### button\_to\_string

```python
def button_to_string(button: Dict[Text, Any], idx: int = 0) -> Text
```

Create a string representation of a button.

#### element\_to\_string

```python
def element_to_string(element: Dict[Text, Any], idx: int = 0) -> Text
```

Create a string representation of an element.

#### button\_choices\_from\_message\_data

```python
def button_choices_from_message_data(
        message: Dict[Text, Any],
        allow_free_text_input: bool = True) -> List[Text]
```

Return list of choices to present to the user.

If allow_free_text_input is True, an additional option is added
at the end along with the response buttons that allows the user
to type in free text.

#### payload\_from\_button\_question

```python
async def payload_from_button_question(button_question: "Question") -> Text
```

Prompt user with a button question and returns the nlu payload.

#### signal\_handler

```python
def signal_handler(_: int, __: FrameType) -> None
```

Kills Rasa when OS signal is received.


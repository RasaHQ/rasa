---
sidebar_label: rasa.core.nlg.interpolator
title: rasa.core.nlg.interpolator
---
#### interpolate\_text

```python
def interpolate_text(response: Text, values: Dict[Text, Text]) -> Text
```

Interpolate values into responses with placeholders.

Transform response tags from &quot;{tag_name}&quot; to &quot;{0[tag_name]}&quot; as described here:
https://stackoverflow.com/questions/7934620/python-dots-in-the-name-of-variable-in-a-format-string#comment9695339_7934969
Block characters, making sure not to allow:
(a) newline in slot name
(b) { or } in slot name

**Arguments**:

- `response` - The piece of text that should be interpolated.
- `values` - A dictionary of keys and the values that those
  keys should be replaced with.
  

**Returns**:

  The piece of text with any replacements made.

#### interpolate

```python
def interpolate(response: Union[List[Any], Dict[Text, Any], Text], values: Dict[Text, Text]) -> Union[List[Any], Dict[Text, Any], Text]
```

Recursively process response and interpolate any text keys.

**Arguments**:

- `response` - The response that should be interpolated.
- `values` - A dictionary of keys and the values that those
  keys should be replaced with.
  

**Returns**:

  The response with any replacements made.


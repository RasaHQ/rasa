---
sidebar_label: interpolator
title: rasa.core.nlg.interpolator
---

#### interpolate\_text

```python
interpolate_text(template: Text, values: Dict[Text, Text]) -> Text
```

Interpolate values into templates with placeholders.

Transform template tags from &quot;{tag_name}&quot; to &quot;{0[tag_name]}&quot; as described here:
https://stackoverflow.com/questions/7934620/python-dots-in-the-name-of-variable-in-a-format-string#comment9695339_7934969
Block characters, making sure not to allow:
(a) newline in slot name
(b) { or } in slot name

**Arguments**:

- `template` - The piece of text that should be interpolated.
- `values` - A dictionary of keys and the values that those
  keys should be replaced with.
  

**Returns**:

  The piece of text with any replacements made.

#### interpolate

```python
interpolate(template: Union[List[Any], Dict[Text, Any], Text], values: Dict[Text, Text]) -> Union[List[Any], Dict[Text, Any], Text]
```

Recursively process template and interpolate any text keys.

**Arguments**:

- `template` - The template that should be interpolated.
- `values` - A dictionary of keys and the values that those
  keys should be replaced with.
  

**Returns**:

  The template with any replacements made.


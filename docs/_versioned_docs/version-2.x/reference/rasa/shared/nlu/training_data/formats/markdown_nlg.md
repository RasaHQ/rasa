---
sidebar_label: rasa.shared.nlu.training_data.formats.markdown_nlg
title: rasa.shared.nlu.training_data.formats.markdown_nlg
---
## NLGMarkdownReader Objects

```python
class NLGMarkdownReader(MarkdownReader)
```

Reads markdown training data with NLG stories and creates TrainingData object.

#### \_\_init\_\_

```python
 | __init__(ignore_deprecation_warning: bool = False) -> None
```

Creates reader. See parent class docstring for more information.

#### reads

```python
 | reads(s: Text, **kwargs: Any) -> "TrainingData"
```

Read markdown string and create TrainingData object.

#### is\_markdown\_nlg\_file

```python
 | @staticmethod
 | is_markdown_nlg_file(filename: Union[Text, Path]) -> bool
```

Checks if given file contains NLG training data.

**Arguments**:

- `filename` - Path to the training data file.
  

**Returns**:

  `True` if file contains NLG training data, `False` otherwise.

## NLGMarkdownWriter Objects

```python
class NLGMarkdownWriter(MarkdownWriter)
```

Converts retrieval intent data to Markdown.

#### dumps

```python
 | dumps(training_data: "TrainingData") -> Text
```

Transforms the NlG part of TrainingData object into a markdown string.


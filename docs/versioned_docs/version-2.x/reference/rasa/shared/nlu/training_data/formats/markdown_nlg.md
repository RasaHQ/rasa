---
sidebar_label: markdown_nlg
title: rasa.shared.nlu.training_data.formats.markdown_nlg
---

## NLGMarkdownReader Objects

```python
class NLGMarkdownReader(TrainingDataReader)
```

Reads markdown training data containing NLG stories and creates a TrainingData object.

#### reads

```python
 | reads(s: Text, **kwargs: Any) -> "TrainingData"
```

Read markdown string and create TrainingData object

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
class NLGMarkdownWriter(TrainingDataWriter)
```

#### dumps

```python
 | dumps(training_data: "TrainingData") -> Text
```

Transforms the NlG part of TrainingData object into a markdown string.


---
sidebar_label: rasa.shared.nlu.training_data.formats.markdown
title: rasa.shared.nlu.training_data.formats.markdown
---
## MarkdownReader Objects

```python
class MarkdownReader(TrainingDataReader)
```

Reads markdown training data and creates a TrainingData object.

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

## MarkdownWriter Objects

```python
class MarkdownWriter(TrainingDataWriter)
```

Converts NLU data to Markdown.

#### \_\_init\_\_

```python
 | __init__(ignore_deprecation_warning: bool = False) -> None
```

Creates writer.

**Arguments**:

- `ignore_deprecation_warning` - `True` if deprecation warning for Markdown
  format should be suppressed.

#### dumps

```python
 | dumps(training_data: "TrainingData") -> Text
```

Transforms a TrainingData object into a markdown string.


---
sidebar_label: rasa.nlu.training_data.formats.markdown
title: rasa.nlu.training_data.formats.markdown
---
## MarkdownReader Objects

```python
class MarkdownReader(TrainingDataReader)
```

Reads markdown training data and creates a TrainingData object.

#### reads

```python
 | reads(s: Text, **kwargs: Any) -> "TrainingData"
```

Read markdown string and create TrainingData object

## MarkdownWriter Objects

```python
class MarkdownWriter(TrainingDataWriter)
```

#### dumps

```python
 | dumps(training_data: "TrainingData") -> Text
```

Transforms a TrainingData object into a markdown string.


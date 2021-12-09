---
sidebar_label: rasa.nlu.training_data.formats.markdown_nlg
title: rasa.nlu.training_data.formats.markdown_nlg
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

## NLGMarkdownWriter Objects

```python
class NLGMarkdownWriter(TrainingDataWriter)
```

#### dumps

```python
 | dumps(training_data: "TrainingData") -> Text
```

Transforms the NlG part of TrainingData object into a markdown string.


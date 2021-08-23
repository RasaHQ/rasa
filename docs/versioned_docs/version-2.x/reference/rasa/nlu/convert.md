---
sidebar_label: rasa.nlu.convert
title: rasa.nlu.convert
---
#### convert\_training\_data

```python
convert_training_data(data_file: Union[list, Text], out_file: Text, output_format: Text, language: Text) -> None
```

Convert training data.

**Arguments**:

- `data_file` _Union[list, Text]_ - Path to the file or directory
  containing Rasa data.
- `out_file` _Text_ - File or existing path where to save
  training data in Rasa format.
- `output_format` _Text_ - Output format the training data
  should be converted into.
- `language` _Text_ - Language of the data.


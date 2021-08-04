---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.convert_featurizer
title: rasa.nlu.featurizers.dense_featurizer.convert_featurizer
---

## ConveRTFeaturizer Objects

```python
class ConveRTFeaturizer(DenseFeaturizer)
```

Featurizer using ConveRT model.

Loads the ConveRT(https://github.com/PolyAI-LDN/polyai-models#convert)
model from TFHub and computes sentence and sequence level feature representations
for dense featurizable attributes of each message object.


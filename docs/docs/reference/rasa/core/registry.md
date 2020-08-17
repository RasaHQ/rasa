---
sidebar_label: rasa.core.registry
title: rasa.core.registry
---

This module imports all of the components. To avoid cycles, no component
should import this in module scope.

#### policy\_from\_module\_path

```python
policy_from_module_path(module_path: Text) -> Type["Policy"]
```

Given the name of a policy module tries to retrieve the policy.

#### featurizer\_from\_module\_path

```python
featurizer_from_module_path(module_path: Text) -> Type["TrackerFeaturizer"]
```

Given the name of a featurizer module tries to retrieve it.


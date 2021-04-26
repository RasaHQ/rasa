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

**Arguments**:

- `module_path` - a path to a policy
  

**Returns**:

  a :class:`rasa.core.policies.policy.Policy`

#### featurizer\_from\_module\_path

```python
featurizer_from_module_path(module_path: Text) -> Type["TrackerFeaturizer"]
```

Given the name of a featurizer module tries to retrieve it.

**Arguments**:

- `module_path` - a path to a featurizer
  

**Returns**:

  a :class:`rasa.core.featurizers.tracker_featurizers.TrackerFeaturizer`

#### state\_featurizer\_from\_module\_path

```python
state_featurizer_from_module_path(module_path: Text) -> Type["SingleStateFeaturizer"]
```

Given the name of a single state featurizer module tries to retrieve it.

**Arguments**:

- `module_path` - a path to a single state featurizer
  

**Returns**:

  a :class:`rasa.core.featurizers.single_state_featurizer.SingleStateFeaturizer`


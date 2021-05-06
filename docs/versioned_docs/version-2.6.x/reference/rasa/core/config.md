---
sidebar_label: rasa.core.config
title: rasa.core.config
---
#### load

```python
load(config_file: Union[Text, Dict]) -> List["Policy"]
```

Load policy data stored in the specified file.

#### migrate\_fallback\_policies

```python
migrate_fallback_policies(config: Dict) -> Tuple[Dict, Optional["StoryStep"]]
```

Migrate the deprecated fallback policies to their `RulePolicy` counterpart.

**Arguments**:

- `config` - The model configuration containing deprecated policies.
  

**Returns**:

  The updated configuration and the required fallback rules.

#### migrate\_mapping\_policy\_to\_rules

```python
migrate_mapping_policy_to_rules(config: Dict[Text, Any], domain: "Domain") -> Tuple[Dict[Text, Any], "Domain", List["StoryStep"]]
```

Migrate `MappingPolicy` to its `RulePolicy` counterparts.

This migration will update the config, domain and generate the required rules.

**Arguments**:

- `config` - The model configuration containing deprecated policies.
- `domain` - The domain which potentially includes intents with the `triggers`
  property.
  

**Returns**:

  The updated model configuration, the domain without trigger intents, and the
  generated rules.


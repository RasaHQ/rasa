---
sidebar_label: rasa.engine.recipes.graph_recipe
title: rasa.engine.recipes.graph_recipe
---
## GraphV1Recipe Objects

```python
class GraphV1Recipe(Recipe)
```

Recipe which converts the graph model config to train and predict graph.

#### get\_targets

```python
 | get_targets(config: Dict, training_type: TrainingType) -> Tuple[Text, Any]
```

Return NLU and core targets from config dictionary.

Note that default recipe has `nlu_target` and `core_target` as
fixed values of `run_RegexMessageHandler` and `select_prediction`
respectively. For graph recipe, target values are customizable. These
can be used in validation (default recipe does this validation check)
and during execution (all recipes use targets during execution).

#### graph\_config\_for\_recipe

```python
 | graph_config_for_recipe(config: Dict, cli_parameters: Dict[Text, Any], training_type: TrainingType = TrainingType.BOTH, is_finetuning: bool = False) -> GraphModelConfiguration
```

Converts the default config to graphs (see interface for full docstring).


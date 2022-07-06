---
sidebar_label: rasa.core.train
title: rasa.core.train
---
#### train\_comparison\_models

```python
def train_comparison_models(story_file: Text, domain: Text, output_path: Text = "", exclusion_percentages: Optional[List] = None, policy_configs: Optional[List] = None, runs: int = 1, additional_arguments: Optional[Dict] = None) -> None
```

Trains multiple models for comparison of policies.

#### get\_no\_of\_stories

```python
def get_no_of_stories(story_file: Text, domain: Text) -> int
```

Gets number of stories in a file.

#### do\_compare\_training

```python
def do_compare_training(args: argparse.Namespace, story_file: Text, additional_arguments: Optional[Dict] = None) -> None
```

Train multiple models for comparison of policies and dumps the result.


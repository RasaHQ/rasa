---
sidebar_label: rasa.core.train
title: rasa.core.train
---
#### train

```python
train(domain_file: Union[Domain, Text], training_resource: Union[Text, "TrainingDataImporter"], output_path: Text, interpreter: Optional["NaturalLanguageInterpreter"] = None, endpoints: "AvailableEndpoints" = None, policy_config: Optional[Union[Text, Dict]] = None, exclusion_percentage: Optional[int] = None, additional_arguments: Optional[Dict] = None, model_to_finetune: Optional["Agent"] = None) -> "Agent"
```

Trains the model.

#### train\_comparison\_models

```python
train_comparison_models(story_file: Text, domain: Text, output_path: Text = "", exclusion_percentages: Optional[List] = None, policy_configs: Optional[List] = None, runs: int = 1, additional_arguments: Optional[Dict] = None) -> None
```

Train multiple models for comparison of policies

#### get\_no\_of\_stories

```python
get_no_of_stories(story_file: Text, domain: Text) -> int
```

Get number of stories in a file.

#### do\_compare\_training

```python
do_compare_training(args: argparse.Namespace, story_file: Text, additional_arguments: Optional[Dict] = None) -> None
```

Train multiple models for comparison of policies and dumps the result.


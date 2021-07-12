---
sidebar_label: rasa.shared.core.training_data.visualization
title: rasa.shared.core.training_data.visualization
---
## UserMessageGenerator Objects

```python
class UserMessageGenerator()
```

#### message\_for\_data

```python
 | message_for_data(structured_info: Dict[Text, Any]) -> Any
```

Find a data sample with the same intent and entities.

Given the parsed data from a message (intent and entities) finds a
message in the data that has the same intent and entities.

#### persist\_graph

```python
persist_graph(graph: "networkx.Graph", output_file: Text) -> None
```

Plots the graph and persists it into a html file.

#### visualize\_neighborhood

```python
async visualize_neighborhood(current: Optional[List[Event]], event_sequences: List[List[Event]], output_file: Optional[Text] = None, max_history: int = 2, interpreter: NaturalLanguageInterpreter = RegexInterpreter(), nlu_training_data: Optional["TrainingData"] = None, should_merge_nodes: bool = True, max_distance: int = 1, fontsize: int = 12) -> "networkx.MultiDiGraph"
```

Given a set of event lists, visualizing the flows.

#### visualize\_stories

```python
async visualize_stories(story_steps: List[StoryStep], domain: Domain, output_file: Optional[Text], max_history: int, interpreter: NaturalLanguageInterpreter = RegexInterpreter(), nlu_training_data: Optional["TrainingData"] = None, should_merge_nodes: bool = True, fontsize: int = 12) -> "networkx.MultiDiGraph"
```

Given a set of stories, generates a graph visualizing the flows in the stories.

Visualization is always a trade off between making the graph as small as
possible while
at the same time making sure the meaning doesn&#x27;t change to &quot;much&quot;. The
algorithm will
compress the graph generated from the stories to merge nodes that are
similar. Hence,
the algorithm might create paths through the graph that aren&#x27;t actually
specified in the
stories, but we try to minimize that.

Output file defines if and where a file containing the plotted graph
should be stored.

The history defines how much &#x27;memory&#x27; the graph has. This influences in
which situations the
algorithm will merge nodes. Nodes will only be merged if they are equal
within the history, this
means the larger the history is we take into account the less likely it
is we merge any nodes.

The training data parameter can be used to pass in a Rasa NLU training
data instance. It will
be used to replace the user messages from the story file with actual
messages from the training data.


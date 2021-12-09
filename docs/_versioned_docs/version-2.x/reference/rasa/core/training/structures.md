---
sidebar_label: rasa.core.training.structures
title: rasa.core.training.structures
---
## Checkpoint Objects

```python
class Checkpoint()
```

#### filter\_trackers

```python
 | filter_trackers(trackers: List[DialogueStateTracker]) -> List[DialogueStateTracker]
```

Filters out all trackers that do not satisfy the conditions.

## StoryStep Objects

```python
class StoryStep()
```

A StoryStep is a section of a story block between two checkpoints.

NOTE: Checkpoints are not only limited to those manually written
in the story file, but are also implicitly created at points where
multiple intents are separated in one line by chaining them with &quot;OR&quot;s.

#### explicit\_events

```python
 | explicit_events(domain: Domain, should_append_final_listen: bool = True) -> List[Union[Event, List[Event]]]
```

Returns events contained in the story step including implicit events.

Not all events are always listed in the story dsl. This
includes listen actions as well as implicitly
set slots. This functions makes these events explicit and
returns them with the rest of the steps events.

## Story Objects

```python
class Story()
```

#### from\_events

```python
 | @staticmethod
 | from_events(events: List[Event], story_name: Optional[Text] = None) -> "Story"
```

Create a story from a list of events.

## StoryGraph Objects

```python
class StoryGraph()
```

Graph of the story-steps pooled from all stories in the training data.

#### ordered\_steps

```python
 | ordered_steps() -> List[StoryStep]
```

Returns the story steps ordered by topological order of the DAG.

#### cyclic\_edges

```python
 | cyclic_edges() -> List[Tuple[Optional[StoryStep], Optional[StoryStep]]]
```

Returns the story steps ordered by topological order of the DAG.

#### overlapping\_checkpoint\_names

```python
 | @staticmethod
 | overlapping_checkpoint_names(cps: List[Checkpoint], other_cps: List[Checkpoint]) -> Set[Text]
```

Find overlapping checkpoints names

#### with\_cycles\_removed

```python
 | with_cycles_removed() -> "StoryGraph"
```

Create a graph with the cyclic edges removed from this graph.

#### get

```python
 | get(step_id: Text) -> Optional[StoryStep]
```

Looks a story step up by its id.

#### as\_story\_string

```python
 | as_story_string() -> Text
```

Convert the graph into the story file format.

#### order\_steps

```python
 | @staticmethod
 | order_steps(story_steps: List[StoryStep]) -> Tuple[deque, List[Tuple[Text, Text]]]
```

Topological sort of the steps returning the ids of the steps.

#### topological\_sort

```python
 | @staticmethod
 | topological_sort(graph: Dict[Text, Set[Text]]) -> Tuple[deque, List[Tuple[Text, Text]]]
```

Creates a top sort of a directed graph. This is an unstable sorting!

The function returns the sorted nodes as well as the edges that need
to be removed from the graph to make it acyclic (and hence, sortable).

The graph should be represented as a dictionary, e.g.:

&gt;&gt;&gt; example_graph = {
...         &quot;a&quot;: set(&quot;b&quot;, &quot;c&quot;, &quot;d&quot;),
...         &quot;b&quot;: set(),
...         &quot;c&quot;: set(&quot;d&quot;),
...         &quot;d&quot;: set(),
...         &quot;e&quot;: set(&quot;f&quot;),
...         &quot;f&quot;: set()}
&gt;&gt;&gt; StoryGraph.topological_sort(example_graph)
(deque([u&#x27;e&#x27;, u&#x27;f&#x27;, u&#x27;a&#x27;, u&#x27;c&#x27;, u&#x27;d&#x27;, u&#x27;b&#x27;]), [])

#### is\_empty

```python
 | is_empty() -> bool
```

Checks if `StoryGraph` is empty.


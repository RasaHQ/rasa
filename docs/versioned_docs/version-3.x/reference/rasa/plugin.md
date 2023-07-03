---
sidebar_label: rasa.plugin
title: rasa.plugin
---
#### plugin\_manager

```python
@functools.lru_cache(maxsize=2)
def plugin_manager() -> pluggy.PluginManager
```

Initialises a plugin manager which registers hook implementations.

#### refine\_cli

```python
@hookspec
def refine_cli(subparsers: SubParsersAction,
               parent_parsers: List[argparse.ArgumentParser]) -> None
```

Customizable hook for adding CLI commands.

#### handle\_space\_args

```python
@hookspec(firstresult=True)  # type: ignore[misc]
def handle_space_args(args: argparse.Namespace) -> Dict[Text, Any]
```

Extracts space from the command line arguments.

#### modify\_default\_recipe\_graph\_train\_nodes

```python
@hookspec
def modify_default_recipe_graph_train_nodes(
        train_config: Dict[Text, Any], train_nodes: Dict[Text, "SchemaNode"],
        cli_parameters: Dict[Text, Any]) -> None
```

Hook specification to modify the default recipe graph for training.

Modifications are made in-place.

#### modify\_default\_recipe\_graph\_predict\_nodes

```python
@hookspec
def modify_default_recipe_graph_predict_nodes(
        predict_nodes: Dict[Text, "SchemaNode"]) -> None
```

Hook specification to modify the default recipe graph for prediction.

Modifications are made in-place.

#### get\_version\_info

```python
@hookspec
def get_version_info() -> Tuple[Text, Text]
```

Hook specification for getting plugin version info.

#### configure\_commandline

```python
@hookspec
def configure_commandline(
        cmdline_arguments: argparse.Namespace) -> Optional[Text]
```

Hook specification for configuring plugin CLI.

#### init\_telemetry

```python
@hookspec
def init_telemetry(endpoints_file: Optional[Text]) -> None
```

Hook specification for initialising plugin telemetry.

#### mock\_tracker\_for\_evaluation

```python
@hookspec
def mock_tracker_for_evaluation(
        example: Message, model_metadata: Optional[ModelMetadata]
) -> Optional[DialogueStateTracker]
```

Generate a mocked tracker for NLU evaluation.

#### clean\_entity\_targets\_for\_evaluation

```python
@hookspec
def clean_entity_targets_for_evaluation(merged_targets: List[str],
                                        extractor: str) -> List[str]
```

Remove entity targets for space-based entity extractors.

#### prefix\_stripping\_for\_custom\_actions

```python
@hookspec(firstresult=True)  # type: ignore[misc]
def prefix_stripping_for_custom_actions(
        json_body: Dict[Text, Any]) -> Dict[Text, Any]
```

Remove namespacing introduced by spaces before custom actions call.

#### prefixing\_custom\_actions\_response

```python
@hookspec
def prefixing_custom_actions_response(json_body: Dict[Text, Any],
                                      response: Dict[Text, Any]) -> None
```

Add namespacing to the response from custom actions.

#### init\_managers

```python
@hookspec
def init_managers(endpoints_file: Optional[Text]) -> None
```

Hook specification for initialising managers.

#### create\_tracker\_store

```python
@hookspec(firstresult=True)  # type: ignore[misc]
def create_tracker_store(
        endpoint_config: Union["TrackerStore",
                               "EndpointConfig"], domain: "Domain",
        event_broker: Optional["EventBroker"]) -> "TrackerStore"
```

Hook specification for wrapping with AuthRetryTrackerStore.

#### init\_anonymization\_pipeline

```python
@hookspec(firstresult=True)  # type: ignore[misc]
def init_anonymization_pipeline(endpoints_file: Optional[Text]) -> None
```

Hook specification for initialising the anonymization pipeline.

#### get\_anonymization\_pipeline

```python
@hookspec(firstresult=True)  # type: ignore[misc]
def get_anonymization_pipeline() -> Optional[Any]
```

Hook specification for getting the anonymization pipeline.

#### get\_license\_hash

```python
@hookspec(firstresult=True)  # type: ignore[misc]
def get_license_hash() -> Optional[Text]
```

Hook specification for getting the license hash.

#### after\_server\_stop

```python
@hookspec
def after_server_stop() -> None
```

Hook specification for stopping the server.

Use this hook to de-initialize any resources that require explicit cleanup like,
thread shutdown, closing connections, etc.


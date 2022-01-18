---
sidebar_label: rasa.telemetry
title: rasa.telemetry
---
#### print\_telemetry\_reporting\_info

```python
print_telemetry_reporting_info() -> None
```

Print telemetry information to std out.

#### is\_telemetry\_enabled

```python
is_telemetry_enabled() -> bool
```

Check if telemetry is enabled either in configuration or environment.

**Returns**:

  `True`, if telemetry is enabled, `False` otherwise.

#### initialize\_telemetry

```python
initialize_telemetry() -> bool
```

Read telemetry configuration from the user&#x27;s Rasa config file in $HOME.

Creates a default configuration if no configuration exists.

**Returns**:

  `True`, if telemetry is enabled, `False` otherwise.

#### ensure\_telemetry\_enabled

```python
ensure_telemetry_enabled(f: Callable[..., Any]) -> Callable[..., Any]
```

Function decorator for telemetry functions that ensures telemetry is enabled.

WARNING: does not work as a decorator for async generators.

**Arguments**:

- `f` - function to call if telemetry is enabled

**Returns**:

  Return wrapped function

#### telemetry\_write\_key

```python
telemetry_write_key() -> Optional[Text]
```

Read the Segment write key from the segment key text file.
The segment key text file should by present only in wheel/sdist packaged
versions of Rasa Open Source. This avoids running telemetry locally when
developing on Rasa or when running CI builds.

In local development, this should always return `None` to avoid logging telemetry.

**Returns**:

  Segment write key, if the key file was present.

#### sentry\_write\_key

```python
sentry_write_key() -> Optional[Text]
```

Read the sentry write key from the sentry key text file.

**Returns**:

  Sentry write key, if the key file was present.

#### segment\_request\_header

```python
segment_request_header(write_key: Text) -> Dict[Text, Any]
```

Use a segment write key to create authentication headers for the segment API.

**Arguments**:

- `write_key` - Authentication key for segment.
  

**Returns**:

  Authentication headers for segment.

#### segment\_request\_payload

```python
segment_request_payload(distinct_id: Text, event_name: Text, properties: Dict[Text, Any], context: Dict[Text, Any]) -> Dict[Text, Any]
```

Compose a valid payload for the segment API.

**Arguments**:

- `distinct_id` - Unique telemetry ID.
- `event_name` - Name of the event.
- `properties` - Values to report along the event.
- `context` - Context information about the event.
  

**Returns**:

  Valid segment payload.

#### in\_continuous\_integration

```python
in_continuous_integration() -> bool
```

Returns `True` if currently running inside a continuous integration context.

#### print\_telemetry\_event

```python
print_telemetry_event(payload: Dict[Text, Any]) -> None
```

Print a telemetry events payload to the commandline.

**Arguments**:

- `payload` - payload of the event

#### with\_default\_context\_fields

```python
with_default_context_fields(context: Optional[Dict[Text, Any]] = None) -> Dict[Text, Any]
```

Return a new context dictionary with default and provided field values merged.

The default fields contain only the OS information for now.

**Arguments**:

- `context` - Context information about the event.
  

**Returns**:

  A new context.

#### get\_telemetry\_id

```python
get_telemetry_id() -> Optional[Text]
```

Return the unique telemetry identifier for this Rasa Open Source install.
The identifier can be any string, but it should be a UUID.

**Returns**:

  The identifier, if it is configured correctly.

#### toggle\_telemetry\_reporting

```python
toggle_telemetry_reporting(is_enabled: bool) -> None
```

Write to the configuration if telemetry tracking should be enabled or disabled.

**Arguments**:

- `is_enabled` - `True` if the telemetry reporting should be enabled,
  `False` otherwise.

#### filter\_errors

```python
filter_errors(event: Dict[Text, Any], hint: Optional[Dict[Text, Any]] = None) -> Optional[Dict[Text, Any]]
```

Filter errors.

**Arguments**:

- `event` - event to be logged to sentry
- `hint` - some hinting information sent alongside of the event
  

**Returns**:

  the event without any sensitive / PII data or `None` if the event constitutes
  an `ImportError` which should be discarded.

#### before\_send

```python
before_send(event: Dict[Text, Any], _unused_hint: Optional[Dict[Text, Any]] = None) -> Optional[Dict[Text, Any]]
```

Strips the sensitive data and filters errors before sending to sentry.

**Arguments**:

- `event` - event to be logged to sentry
- `_unused_hint` - some hinting information sent alongside of the event
  

**Returns**:

  the event without any sensitive / PII data or `None` if the event should
  be discarded.

#### strip\_sensitive\_data\_from\_sentry\_event

```python
strip_sensitive_data_from_sentry_event(event: Dict[Text, Any], _unused_hint: Optional[Dict[Text, Any]] = None) -> Optional[Dict[Text, Any]]
```

Remove any sensitive data from the event (e.g. path names).

**Arguments**:

- `event` - event to be logged to sentry
- `_unused_hint` - some hinting information sent alongside of the event
  

**Returns**:

  the event without any sensitive / PII data or `None` if the event should
  be discarded.

#### initialize\_error\_reporting

```python
@ensure_telemetry_enabled
initialize_error_reporting() -> None
```

Sets up automated error reporting.

Exceptions are reported to sentry. We avoid sending any metadata (local
variables, paths, ...) to make sure we don&#x27;t compromise any data. Only the
exception and its stacktrace is logged and only if the exception origins
from the `rasa` package.

#### track\_model\_training

```python
@contextlib.contextmanager
track_model_training(training_data: "TrainingDataImporter", model_type: Text, is_finetuning: bool = False) -> typing.Generator[None, None, None]
```

Track a model training started.

WARNING: since this is a generator, it can&#x27;t use the ensure telemetry
decorator. We need to manually add these checks here. This can be
fixed as soon as we drop python 3.6 support.

**Arguments**:

- `training_data` - Training data used for the training.
- `model_type` - Specifies the type of training, should be either &quot;rasa&quot;, &quot;core&quot;
  or &quot;nlu&quot;.
- `is_finetuning` - `True` if the model is trained by finetuning another model.

#### track\_telemetry\_disabled

```python
@ensure_telemetry_enabled
track_telemetry_disabled() -> None
```

Track when a user disables telemetry.

#### track\_data\_split

```python
@ensure_telemetry_enabled
track_data_split(fraction: float, data_type: Text) -> None
```

Track when a user splits data.

**Arguments**:

- `fraction` - How much data goes into train and how much goes into test
- `data_type` - Is this core, nlu or nlg data

#### track\_validate\_files

```python
@ensure_telemetry_enabled
track_validate_files(validation_success: bool) -> None
```

Track when a user validates data files.

**Arguments**:

- `validation_success` - Whether the validation was successful

#### track\_data\_convert

```python
@ensure_telemetry_enabled
track_data_convert(output_format: Text, data_type: Text) -> None
```

Track when a user converts data.

**Arguments**:

- `output_format` - Target format for the converter
- `data_type` - Is this core, nlu or nlg data

#### track\_tracker\_export

```python
@ensure_telemetry_enabled
track_tracker_export(number_of_exported_events: int, tracker_store: "TrackerStore", event_broker: "EventBroker") -> None
```

Track when a user exports trackers.

**Arguments**:

- `number_of_exported_events` - Number of events that got exported
- `tracker_store` - Store used to retrieve the events from
- `event_broker` - Broker the events are getting published towards

#### track\_interactive\_learning\_start

```python
@ensure_telemetry_enabled
track_interactive_learning_start(skip_visualization: bool, save_in_e2e: bool) -> None
```

Track when a user starts an interactive learning session.

**Arguments**:

- `skip_visualization` - Is visualization skipped in this session
- `save_in_e2e` - Is e2e used in this session

#### track\_server\_start

```python
@ensure_telemetry_enabled
track_server_start(input_channels: List["InputChannel"], endpoints: Optional["AvailableEndpoints"], model_directory: Optional[Text], number_of_workers: int, is_api_enabled: bool) -> None
```

Tracks when a user starts a rasa server.

**Arguments**:

- `input_channels` - Used input channels
- `endpoints` - Endpoint configuration for the server
- `model_directory` - directory of the running model
- `number_of_workers` - number of used Sanic workers
- `is_api_enabled` - whether the rasa API server is enabled

#### track\_project\_init

```python
@ensure_telemetry_enabled
track_project_init(path: Text) -> None
```

Track when a user creates a project using rasa init.

**Arguments**:

- `path` - Location of the project

#### track\_shell\_started

```python
@ensure_telemetry_enabled
track_shell_started(model_type: Text) -> None
```

Track when a user starts a bot using rasa shell.

**Arguments**:

- `model_type` - Type of the model, core / nlu or rasa.

#### track\_rasa\_x\_local

```python
@ensure_telemetry_enabled
track_rasa_x_local() -> None
```

Track when a user runs Rasa X in local mode.

#### track\_visualization

```python
@ensure_telemetry_enabled
track_visualization() -> None
```

Track when a user runs the visualization.

#### track\_core\_model\_test

```python
@ensure_telemetry_enabled
track_core_model_test(num_story_steps: int, e2e: bool, agent: "Agent") -> None
```

Track when a user tests a core model.

**Arguments**:

- `num_story_steps` - Number of test stories used for the comparison
- `e2e` - indicator if tests running in end to end mode
- `agent` - Agent of the model getting tested

#### track\_nlu\_model\_test

```python
@ensure_telemetry_enabled
track_nlu_model_test(test_data: "TrainingData") -> None
```

Track when a user tests an nlu model.

**Arguments**:

- `test_data` - Data used for testing

#### track\_markers\_extraction\_initiated

```python
@ensure_telemetry_enabled
track_markers_extraction_initiated(strategy: Text, only_extract: bool, seed: bool, count: Optional[int]) -> None
```

Track when a user tries to extract success markers.

**Arguments**:

- `strategy` - The strategy the user is using for tracker selection
- `only_extract` - Indicates if the user is only extracting markers or also
  producing stats
- `seed` - Indicates if the user used a seed for this attempt
- `count` - (Optional) The number of trackers the user is trying to select.

#### track\_markers\_extracted

```python
@ensure_telemetry_enabled
track_markers_extracted(trackers_count: int) -> None
```

Track when markers have been extracted by a user.

**Arguments**:

- `trackers_count` - The actual number of trackers processed

#### track\_markers\_stats\_computed

```python
@ensure_telemetry_enabled
track_markers_stats_computed(trackers_count: int) -> None
```

Track when stats over markers have been computed by a user.

**Arguments**:

- `trackers_count` - The actual number of trackers processed

#### track\_markers\_parsed\_count

```python
@ensure_telemetry_enabled
track_markers_parsed_count(marker_count: int, max_depth: int, branching_factor: int) -> None
```

Track when markers have been successfully parsed from config.

**Arguments**:

- `marker_count` - The number of markers found in the config
- `max_depth` - The maximum depth of any marker in the config
- `branching_factor` - The maximum number of children of any marker in the config.


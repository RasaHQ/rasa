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

Return a new context dictionary that contains the default field values merged
with the provided ones. The default fields contain only the OS information for now.

**Arguments**:

- `context` - Context information about the event.
  

**Returns**:

  A new context.

#### track

```python
@ensure_telemetry_enabled
async track(event_name: Text, properties: Optional[Dict[Text, Any]] = None, context: Optional[Dict[Text, Any]] = None) -> None
```

Tracks a telemetry event.

It is OK to use this function from outside telemetry.py, but note that it
is recommended to create a new track_xyz() function for complex telemetry
events, or events that are generated from many parts of the Rasa Open Source code.

**Arguments**:

- `event_name` - Name of the event.
- `properties` - Dictionary containing the event&#x27;s properties.
- `context` - Dictionary containing some context for this event.

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

#### strip\_sensitive\_data\_from\_sentry\_event

```python
strip_sensitive_data_from_sentry_event(event: Dict[Text, Any], _unused_hint: Optional[Dict[Text, Any]] = None) -> Dict[Text, Any]
```

Remove any sensitive data from the event (e.g. path names).

**Arguments**:

- `event` - event to be logged to sentry
- `_unused_hint` - some hinting information sent alongside of the event
  

**Returns**:

  the event without any sensitive / PII data.

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
@ensure_telemetry_enabled
@async_generator.asynccontextmanager
async track_model_training(training_data: TrainingDataImporter, model_type: Text) -> None
```

Track a model training started.

**Arguments**:

- `training_data` - Training data used for the training.
- `model_type` - Specifies the type of training, should be either &quot;rasa&quot;, &quot;core&quot;
  or &quot;nlu&quot;.

#### track\_telemetry\_disabled

```python
@ensure_telemetry_enabled
async track_telemetry_disabled() -> None
```

Track when a user disables telemetry.


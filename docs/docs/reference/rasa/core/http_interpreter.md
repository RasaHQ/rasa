---
sidebar_label: rasa.core.http_interpreter
title: rasa.core.http_interpreter
---
## RasaNLUHttpInterpreter Objects

```python
class RasaNLUHttpInterpreter()
```

Allows for an HTTP endpoint to be used to parse messages.

#### \_\_init\_\_

```python
 | __init__(endpoint_config: Optional[EndpointConfig] = None) -> None
```

Initializes a `RasaNLUHttpInterpreter`.

#### parse

```python
 | async parse(message: UserMessage) -> Dict[Text, Any]
```

Parse a text message.

Return a default value if the parsing of the text failed.


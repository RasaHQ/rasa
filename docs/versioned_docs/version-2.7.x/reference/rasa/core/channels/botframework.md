---
sidebar_label: rasa.core.channels.botframework
title: rasa.core.channels.botframework
---
## BotFramework Objects

```python
class BotFramework(OutputChannel)
```

A Microsoft Bot Framework communication channel.

## BotFrameworkInput Objects

```python
class BotFrameworkInput(InputChannel)
```

Bot Framework input channel implementation.

#### \_\_init\_\_

```python
 | __init__(app_id: Text, app_password: Text) -> None
```

Create a Bot Framework input channel.

**Arguments**:

- `app_id` - Bot Framework&#x27;s API id
- `app_password` - Bot Framework application secret

#### add\_attachments\_to\_metadata

```python
 | @staticmethod
 | add_attachments_to_metadata(postdata: Dict[Text, Any], metadata: Optional[Dict[Text, Any]]) -> Optional[Dict[Text, Any]]
```

Merge the values of `postdata[&#x27;attachments&#x27;]` with `metadata`.


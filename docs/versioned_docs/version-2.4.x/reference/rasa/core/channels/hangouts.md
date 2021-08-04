---
sidebar_label: hangouts
title: rasa.core.channels.hangouts
---

## HangoutsOutput Objects

```python
class HangoutsOutput(OutputChannel)
```

#### send\_custom\_json

```python
 | async send_custom_json(recipient_id: Text, json_message: Dict, **kwargs) -> None
```

Custom json payload is simply forwarded to Google Hangouts without
any modifications. Use this for more complex cards, which can be created
in actions.py.

## HangoutsInput Objects

```python
class HangoutsInput(InputChannel)
```

Channel that uses Google Hangouts Chat API to communicate.


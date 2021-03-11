---
sidebar_label: console
title: rasa.core.channels.console
---

#### record\_messages

```python
async record_messages(sender_id, server_url=DEFAULT_SERVER_URL, auth_token="", max_message_limit=None, use_response_stream=True) -> int
```

Read messages from the command line and print bot responses.


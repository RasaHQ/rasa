---
sidebar_label: rasa.core.channels.console
title: rasa.core.channels.console
---
#### record\_messages

```python
async record_messages(sender_id: Text, server_url: Text = DEFAULT_SERVER_URL, auth_token: Text = "", max_message_limit: Optional[int] = None, use_response_stream: bool = True) -> int
```

Read messages from the command line and print bot responses.


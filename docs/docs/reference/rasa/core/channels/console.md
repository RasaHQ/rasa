---
sidebar_label: rasa.core.channels.console
title: rasa.core.channels.console
---
#### print\_buttons

```python
print_buttons(message: Dict[Text, Any], is_latest_message: bool = False, color: Text = rasa.shared.utils.io.bcolors.OKBLUE) -> Optional[questionary.Question]
```

Create CLI buttons from message data.

#### record\_messages

```python
async record_messages(sender_id: Text, server_url: Text = DEFAULT_SERVER_URL, auth_token: Text = "", max_message_limit: Optional[int] = None, use_response_stream: bool = True, request_timeout: Optional[int] = None) -> int
```

Read messages from the command line and print bot responses.


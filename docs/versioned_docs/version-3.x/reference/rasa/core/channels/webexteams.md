---
sidebar_label: rasa.core.channels.webexteams
title: rasa.core.channels.webexteams
---
## WebexTeamsBot Objects

```python
class WebexTeamsBot(OutputChannel)
```

A Cisco WebexTeams communication channel.

## WebexTeamsInput Objects

```python
class WebexTeamsInput(InputChannel)
```

WebexTeams input channel. Based on the HTTPInputChannel.

#### \_\_init\_\_

```python
 | __init__(access_token: Text, room: Optional[Text] = None) -> None
```

Create a Cisco Webex Teams input channel.

Needs a couple of settings to properly authenticate and validate
messages. Details here https://developer.webex.com/authentication.html

**Arguments**:

- `access_token` - Cisco WebexTeams bot access token.
- `room` - the string identifier for a room to which the bot posts


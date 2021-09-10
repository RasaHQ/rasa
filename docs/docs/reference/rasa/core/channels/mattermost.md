---
sidebar_label: rasa.core.channels.mattermost
title: rasa.core.channels.mattermost
---
## MattermostBot Objects

```python
class MattermostBot(OutputChannel)
```

A Mattermost communication channel

#### token\_from\_login

```python
@classmethod
def token_from_login(cls, url: Text, user: Text, password: Text) -> Optional[Text]
```

Retrieve access token for mattermost user.

#### send\_image\_url

```python
async def send_image_url(recipient_id: Text, image: Text, **kwargs: Any) -> None
```

Sends an image.

#### send\_text\_with\_buttons

```python
async def send_text_with_buttons(recipient_id: Text, text: Text, buttons: List[Dict[Text, Any]], **kwargs: Any, ,) -> None
```

Sends buttons to the output.

## MattermostInput Objects

```python
class MattermostInput(InputChannel)
```

Mattermost input channel implemenation.

#### \_\_init\_\_

```python
def __init__(url: Text, token: Text, webhook_url: Text) -> None
```

Create a Mattermost input channel.
Needs a couple of settings to properly authenticate and validate
messages.

**Arguments**:

- `url` - Your Mattermost team url including /v4 example
  https://mysite.example.com/api/v4
- `token` - Your mattermost bot token
- `webhook_url` - The mattermost callback url as specified
  in the outgoing webhooks in mattermost example
  https://mysite.example.com/webhooks/mattermost/webhook


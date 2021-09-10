---
sidebar_label: rasa.core.channels.facebook
title: rasa.core.channels.facebook
---
## Messenger Objects

```python
class Messenger()
```

Implement a fbmessenger to parse incoming webhooks and send msgs.

#### message

```python
async def message(message: Dict[Text, Any], metadata: Optional[Dict[Text, Any]]) -> None
```

Handle an incoming event from the fb webhook.

#### postback

```python
async def postback(message: Dict[Text, Any], metadata: Optional[Dict[Text, Any]]) -> None
```

Handle a postback (e.g. quick reply button).

## MessengerBot Objects

```python
class MessengerBot(OutputChannel)
```

A bot that uses fb-messenger to communicate.

#### send

```python
def send(recipient_id: Text, element: Any) -> None
```

Sends a message to the recipient using the messenger client.

#### send\_text\_message

```python
async def send_text_message(recipient_id: Text, text: Text, **kwargs: Any) -> None
```

Send a message through this channel.

#### send\_image\_url

```python
async def send_image_url(recipient_id: Text, image: Text, **kwargs: Any) -> None
```

Sends an image. Default will just post the url as a string.

#### send\_action

```python
async def send_action(recipient_id: Text, sender_action: Text) -> None
```

Sends a sender action to facebook (e.g. &quot;typing_on&quot;).

**Arguments**:

- `recipient_id` - recipient
- `sender_action` - action to send, e.g. &quot;typing_on&quot; or &quot;mark_seen&quot;

#### send\_text\_with\_buttons

```python
async def send_text_with_buttons(recipient_id: Text, text: Text, buttons: List[Dict[Text, Any]], **kwargs: Any, ,) -> None
```

Sends buttons to the output.

#### send\_quick\_replies

```python
async def send_quick_replies(recipient_id: Text, text: Text, quick_replies: List[Dict[Text, Any]], **kwargs: Any, ,) -> None
```

Sends quick replies to the output.

#### send\_elements

```python
async def send_elements(recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any) -> None
```

Sends elements to the output.

#### send\_custom\_json

```python
async def send_custom_json(recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any) -> None
```

Sends custom json data to the output.

## FacebookInput Objects

```python
class FacebookInput(InputChannel)
```

Facebook input channel implementation. Based on the HTTPInputChannel.

#### \_\_init\_\_

```python
def __init__(fb_verify: Text, fb_secret: Text, fb_access_token: Text) -> None
```

Create a facebook input channel.

Needs a couple of settings to properly authenticate and validate
messages. Details to setup:

https://github.com/rehabstudio/fbmessenger#facebook-app-setup

**Arguments**:

- `fb_verify` - FB Verification string
  (can be chosen by yourself on webhook creation)
- `fb_secret` - facebook application secret
- `fb_access_token` - access token to post in the name of the FB page

#### validate\_hub\_signature

```python
@staticmethod
def validate_hub_signature(app_secret: Text, request_payload: bytes, hub_signature_header: Text) -> bool
```

Make sure the incoming webhook requests are properly signed.

**Arguments**:

- `app_secret` - Secret Key for application
- `request_payload` - request body
- `hub_signature_header` - X-Hub-Signature header sent with request
  

**Returns**:

- `bool` - indicated that hub signature is validated


---
sidebar_label: rasa.core.channels.twilio
title: rasa.core.channels.twilio
---
## TwilioOutput Objects

```python
class TwilioOutput(Client,  OutputChannel)
```

Output channel for Twilio

#### send\_text\_message

```python
async def send_text_message(recipient_id: Text, text: Text, **kwargs: Any) -> None
```

Sends text message

#### send\_image\_url

```python
async def send_image_url(recipient_id: Text, image: Text, **kwargs: Any) -> None
```

Sends an image.

#### send\_custom\_json

```python
async def send_custom_json(recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any) -> None
```

Send custom json dict

## TwilioInput Objects

```python
class TwilioInput(InputChannel)
```

Twilio input channel


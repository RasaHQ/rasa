---
sidebar_label: socketio
title: rasa.core.channels.socketio
---

## SocketIOOutput Objects

```python
class SocketIOOutput(OutputChannel)
```

#### send\_text\_message

```python
 | async send_text_message(recipient_id: Text, text: Text, **kwargs: Any) -> None
```

Send a message through this channel.

#### send\_image\_url

```python
 | async send_image_url(recipient_id: Text, image: Text, **kwargs: Any) -> None
```

Sends an image to the output

#### send\_text\_with\_buttons

```python
 | async send_text_with_buttons(recipient_id: Text, text: Text, buttons: List[Dict[Text, Any]], **kwargs: Any, ,) -> None
```

Sends buttons to the output.

#### send\_elements

```python
 | async send_elements(recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any) -> None
```

Sends elements to the output.

#### send\_custom\_json

```python
 | async send_custom_json(recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any) -> None
```

Sends custom json to the output

#### send\_attachment

```python
 | async send_attachment(recipient_id: Text, attachment: Dict[Text, Any], **kwargs: Any) -> None
```

Sends an attachment to the user.

## SocketIOInput Objects

```python
class SocketIOInput(InputChannel)
```

A socket.io input channel.


---
sidebar_label: rasa.core.channels.telegram
title: rasa.core.channels.telegram
---
## TelegramOutput Objects

```python
class TelegramOutput(Bot,  OutputChannel)
```

Output channel for Telegram.

#### send\_text\_message

```python
 | async send_text_message(recipient_id: Text, text: Text, **kwargs: Any) -> None
```

Sends text message.

#### send\_image\_url

```python
 | async send_image_url(recipient_id: Text, image: Text, **kwargs: Any) -> None
```

Sends an image.

#### send\_text\_with\_buttons

```python
 | async send_text_with_buttons(recipient_id: Text, text: Text, buttons: List[Dict[Text, Any]], button_type: Optional[Text] = "inline", **kwargs: Any, ,) -> None
```

Sends a message with keyboard.

For more information: https://core.telegram.org/bots#keyboards

:button_type inline: horizontal inline keyboard

:button_type vertical: vertical inline keyboard

:button_type reply: reply keyboard

#### send\_custom\_json

```python
 | async send_custom_json(recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any) -> None
```

Sends a message with a custom json payload.

## TelegramInput Objects

```python
class TelegramInput(InputChannel)
```

Telegram input channel

#### get\_output\_channel

```python
 | get_output_channel() -> TelegramOutput
```

Loads the telegram channel.


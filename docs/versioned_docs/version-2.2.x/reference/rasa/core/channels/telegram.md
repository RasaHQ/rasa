---
sidebar_label: telegram
title: rasa.core.channels.telegram
---

## TelegramOutput Objects

```python
class TelegramOutput(Bot,  OutputChannel)
```

Output channel for Telegram

#### send\_text\_with\_buttons

```python
 | async send_text_with_buttons(recipient_id: Text, text: Text, buttons: List[Dict[Text, Any]], button_type: Optional[Text] = "inline", **kwargs: Any, ,) -> None
```

Sends a message with keyboard.

For more information: https://core.telegram.org/bots#keyboards

:button_type inline: horizontal inline keyboard

:button_type vertical: vertical inline keyboard

:button_type reply: reply keyboard

## TelegramInput Objects

```python
class TelegramInput(InputChannel)
```

Telegram input channel


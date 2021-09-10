---
sidebar_label: rasa.core.channels.twilio_voice
title: rasa.core.channels.twilio_voice
---
## TwilioVoiceInput Objects

```python
class TwilioVoiceInput(InputChannel)
```

Input channel for Twilio Voice.

#### name

```python
@classmethod
def name(cls) -> Text
```

Name of channel.

#### from\_credentials

```python
@classmethod
def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel
```

Load custom configurations.

#### \_\_init\_\_

```python
def __init__(initial_prompt: Optional[Text], reprompt_fallback_phrase: Optional[Text], assistant_voice: Optional[Text], speech_timeout: Optional[Text], speech_model: Optional[Text], enhanced: Optional[Text]) -> None
```

Creates a connection to Twilio voice.

**Arguments**:

- `initial_prompt` - text to use to prompt a conversation when call is answered.
- `reprompt_fallback_phrase` - phrase to use if no user response.
- `assistant_voice` - name of the assistant voice to use.
- `speech_timeout` - how long to pause when user finished speaking.
- `speech_model` - type of transcription model to use from Twilio.
- `enhanced` - toggle to use Twilio&#x27;s premium speech transcription model.

#### blueprint

```python
def blueprint(on_new_message: Callable[[UserMessage], Awaitable[None]]) -> Blueprint
```

Defines endpoints for Twilio voice channel.

## TwilioVoiceCollectingOutputChannel Objects

```python
class TwilioVoiceCollectingOutputChannel(CollectingOutputChannel)
```

Output channel that collects send messages in a list.

(doesn&#x27;t send them anywhere, just collects them).

#### name

```python
@classmethod
def name(cls) -> Text
```

Name of the output channel.

#### send\_text\_message

```python
async def send_text_message(recipient_id: Text, text: Text, **kwargs: Any) -> None
```

Sends the text message after removing emojis.

#### send\_text\_with\_buttons

```python
async def send_text_with_buttons(recipient_id: Text, text: Text, buttons: List[Dict[Text, Any]], **kwargs: Any, ,) -> None
```

Convert buttons into a voice representation.

#### send\_image\_url

```python
async def send_image_url(recipient_id: Text, image: Text, **kwargs: Any) -> None
```

For voice channel do not send images.


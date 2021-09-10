---
sidebar_label: rasa.core.channels.slack
title: rasa.core.channels.slack
---
## SlackBot Objects

```python
class SlackBot(OutputChannel)
```

A Slack communication channel

## SlackInput Objects

```python
class SlackInput(InputChannel)
```

Slack input channel implementation. Based on the HTTPInputChannel.

#### \_\_init\_\_

```python
def __init__(slack_token: Text, slack_channel: Optional[Text] = None, proxy: Optional[Text] = None, slack_retry_reason_header: Optional[Text] = None, slack_retry_number_header: Optional[Text] = None, errors_ignore_retry: Optional[List[Text]] = None, use_threads: Optional[bool] = False, slack_signing_secret: Text = "") -> None
```

Create a Slack input channel.

Needs a couple of settings to properly authenticate and validate
messages. Details to setup:

https://github.com/slackapi/python-slackclient

**Arguments**:

- `slack_token` - Your Slack Authentication token. You can create a
  Slack app and get your Bot User OAuth Access Token
  `here &lt;https://api.slack.com/slack-apps&gt;`_.
- `slack_channel` - the string identifier for a channel to which
  the bot posts, or channel name (e.g. &#x27;#bot-test&#x27;)
  If not set, messages will be sent back
  to the &quot;App&quot; DM channel of your bot&#x27;s name.
- `proxy` - A Proxy Server to route your traffic through
- `slack_retry_reason_header` - Slack HTTP header name indicating reason
  that slack send retry request.
- `slack_retry_number_header` - Slack HTTP header name indicating
  the attempt number.
- `errors_ignore_retry` - Any error codes given by Slack
  included in this list will be ignored.
  Error codes are listed
  `here &lt;https://api.slack.com/events-api#errors&gt;`_.
- `use_threads` - If set to `True`, your bot will send responses in Slack as
  a threaded message. Responses will appear as a normal Slack message
  if set to `False`.
- `slack_signing_secret` - Slack creates a unique string for your app and
  shares it with you. This allows us to verify requests from Slack
  with confidence by verifying signatures using your signing secret.

#### process\_message

```python
async def process_message(request: Request, on_new_message: Callable[[UserMessage], Awaitable[Any]], text: Text, sender_id: Optional[Text], metadata: Optional[Dict]) -> Any
```

Slack retries to post messages up to 3 times based on
failure conditions defined here:
https://api.slack.com/events-api#failure_conditions

#### get\_metadata

```python
def get_metadata(request: Request) -> Dict[Text, Any]
```

Extracts the metadata from a slack API event.

Slack Documentation: https://api.slack.com/types/event

**Arguments**:

- `request` - A `Request` object that contains a slack API event in the body.
  

**Returns**:

  Metadata extracted from the sent event payload. This includes the output
  channel for the response, and users that have installed the bot.

#### is\_request\_from\_slack\_authentic

```python
def is_request_from_slack_authentic(request: Request) -> bool
```

Validate a request from Slack for its authenticity.

Checks if the signature matches the one we expect from Slack. Ensures
we don&#x27;t process request from a third-party disguising as slack.

**Arguments**:

- `request` - incoming request to be checked
  

**Returns**:

  `True` if the request came from Slack.


---
sidebar_label: rasa.utils.llm
title: rasa.utils.llm
---
#### tracker\_as\_readable\_transcript

```python
def tracker_as_readable_transcript(tracker: DialogueStateTracker,
                                   human_prefix: str = USER,
                                   ai_prefix: str = AI,
                                   max_turns: Optional[int] = 20) -> str
```

Creates a readable dialogue from a tracker.

**Arguments**:

- `tracker` - the tracker to convert
- `human_prefix` - the prefix to use for human utterances
- `ai_prefix` - the prefix to use for ai utterances
- `max_turns` - the maximum number of turns to include in the transcript
  

**Example**:

  &gt;&gt;&gt; tracker = Tracker(
  ...     sender_id=&quot;test&quot;,
  ...     slots=[],
  ...     events=[
  ...         UserUttered(&quot;hello&quot;),
  ...         BotUttered(&quot;hi&quot;),
  ...     ],
  ... )
  &gt;&gt;&gt; tracker_as_readable_transcript(tracker)
- `USER` - hello
- `AI` - hi
  

**Returns**:

  A string representing the transcript of the tracker

#### sanitize\_message\_for\_prompt

```python
def sanitize_message_for_prompt(text: Optional[str]) -> str
```

Removes new lines from a string.

**Arguments**:

- `text` - the text to sanitize
  

**Returns**:

  A string with new lines removed.


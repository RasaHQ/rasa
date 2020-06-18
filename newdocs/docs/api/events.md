# Events

Conversations in Rasa are represented as a sequence of events.
This page lists the event types defined in Rasa Core.

**NOTE**: If you are using the Rasa SDK to write custom actions in python,
you need to import the events from `rasa_sdk.events`, not from
`rasa.core.events`. If you are writing actions in another language,
your events should be formatted like the JSON objects on this page.

## General Purpose Events

### Set a Slot


* **Short**

    Event to set a slot on a tracker



* **JSON**

    ```
    evt = {"event": "slot", "name": "departure_airport", "value": "BER"}
    ```



* **Class**

    
    ### class rasa.core.events.SlotSet(key, value=None, timestamp=None)
    The user has specified their preference for the value of a `slot`.

    Every slot has a name and a value. This event can be used to set a
    value for a slot on a conversation.

    As a side effect the `Tracker`’s slots will be updated so
    that `tracker.slots[key]=value`.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        tracker._set_slot(self.key, self.value)
    ```


### Restart a conversation


* **Short**

    Resets anything logged on the tracker.



* **JSON**

    ```
    evt = {"event": "restart"}
    ```



* **Class**

    
    ### class rasa.core.events.Restarted(timestamp=None)
    Conversation should start over & history wiped.

    Instead of deleting all events, this event can be used to reset the
    trackers state (e.g. ignoring any past user messages & resetting all
    the slots).



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        from rasa.core.actions.action import (  # pytype: disable=pyi-error
            ACTION_LISTEN_NAME,
        )

        tracker._reset()
        tracker.trigger_followup_action(ACTION_LISTEN_NAME)
    ```


### Reset all Slots


* **Short**

    Resets all the slots of a conversation.



* **JSON**

    ```
    evt = {"event": "reset_slots"}
    ```



* **Class**

    
    ### class rasa.core.events.AllSlotsReset(timestamp=None)
    All Slots are reset to their initial values.

    If you want to keep the dialogue history and only want to reset the
    slots, you can use this event to set all the slots to their initial
    values.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker) -> None:
        tracker._reset_slots()
    ```


### Schedule a reminder


* **Short**

    Schedule an intent to be triggered in the future.



* **JSON**

    ```
    evt = {
      "event": "reminder",
      "intent": "my_intent",
      "entities": {"entity1": "value1", "entity2": "value2"},
      "date_time": "2018-09-03T11:41:10.128172",
      "name": "my_reminder",
      "kill_on_user_msg": True,
    }
    ```



* **Class**

    
    ### class rasa.core.events.ReminderScheduled(action_name, trigger_date_time, name=None, kill_on_user_message=True, timestamp=None)
    Allows asynchronous scheduling of action execution.

    As a side effect the message processor will schedule an action to be run
    at the trigger date.



* **Effect**

    When added to a tracker, Rasa Core will schedule the intent (and entities) to be
    triggered in the future, in place of a user input. You can link
    this intent to an action of your choice using the Mapping Policy.


### Cancel a reminder


* **Short**

    Cancel one or more reminders.



* **JSON**

    ```
    evt = {
      "event": "cancel_reminder",
      "name": "my_reminder",
      "intent": "my_intent",
      "entities": [
            {"entity": "entity1", "value": "value1"},
            {"entity": "entity2", "value": "value2"},
        ],
      "date_time": "2018-09-03T11:41:10.128172",
    }
    ```



* **Class**

    
    ### class rasa.core.events.ReminderCancelled(action_name, timestamp=None)
    Cancel all jobs with a specific name.



* **Effect**

    When added to a tracker, Rasa Core will cancel any outstanding reminders that
    match the `ReminderCancelled` event. For example,


    * `ReminderCancelled(intent="greet")` cancels all reminders with intent `greet`


    * `ReminderCancelled(entities={...})` cancels all reminders with the given entities


    * `ReminderCancelled("...")` cancels the one unique reminder with the given name


    * `ReminderCancelled()` cancels all reminders


### Pause a conversation


* **Short**

    Stops the bot from responding to messages. Action prediction
    will be halted until resumed.



* **JSON**

    ```
    evt = {"event": "pause"}
    ```



* **Class**

    
    ### class rasa.core.events.ConversationPaused(timestamp=None)
    Ignore messages from the user to let a human take over.

    As a side effect the `Tracker`’s `paused` attribute will
    be set to `True`.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker) -> None:
        tracker._paused = True
    ```


### Resume a conversation


* **Short**

    Resumes a previously paused conversation. The bot will start
    predicting actions again.



* **JSON**

    ```
    evt = {"event": "resume"}
    ```



* **Class**

    
    ### class rasa.core.events.ConversationResumed(timestamp=None)
    Bot takes over conversation.

    Inverse of `PauseConversation`. As a side effect the `Tracker`’s
    `paused` attribute will be set to `False`.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker) -> None:
        tracker._paused = False
    ```


### Force a followup action


* **Short**

    Instead of predicting the next action, force the next action
    to be a fixed one.



* **JSON**

    ```
    evt = {"event": "followup", "name": "my_action"}
    ```



* **Class**

    
    ### class rasa.core.events.FollowupAction(name, timestamp=None)
    Enqueue a followup action.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        tracker.trigger_followup_action(self.action_name)
    ```


## Automatically tracked events

### User sent message


* **Short**

    Message a user sent to the bot.



* **JSON**

    ```
    evt = {
        "event": "user",
        "text": "Hey",
        "parse_data": {
            "intent": {
                "name": "greet",
                "confidence": 0.9
            },
            "entities": []
        },
        "metadata": {},
    }
    ```



* **Class**

    
    ### class rasa.core.events.UserUttered(text, intent=None, entities=None, parse_data=None, timestamp=None, input_channel=None, message_id=None)
    The user has said something to the bot.

    As a side effect a new `Turn` will be created in the `Tracker`.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        tracker.latest_message = self
        tracker.clear_followup_action()
    ```


### Bot responded message


* **Short**

    Message a bot sent to the user.



* **JSON**

    ```
    evt = {"event": "bot", "text": "Hey there!", "data": {}}
    ```



* **Class**

    
    ### class rasa.core.events.BotUttered(text=None, data=None, metadata=None, timestamp=None)
    The bot has said something to the user.

    This class is not used in the story training as it is contained in the

    `ActionExecuted` class. An entry is made in the `Tracker`.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker: "DialogueStateTracker") -> None:

        tracker.latest_bot_utterance = self
    ```


### Undo a user message


* **Short**

    Undoes all side effects that happened after the last user message
    (including the `user` event of the message).



* **JSON**

    ```
    evt = {"event": "rewind"}
    ```



* **Class**

    
    ### class rasa.core.events.UserUtteranceReverted(timestamp=None)
    Bot reverts everything until before the most recent user message.

    The bot will revert all events after the latest UserUttered, this
    also means that the last event on the tracker is usually action_listen
    and the bot is waiting for a new user message.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        tracker._reset()
        tracker.replay_events()
    ```


### Undo an action


* **Short**

    Undoes all side effects that happened after the last action
    (including the `action` event of the action).



* **JSON**

    ```
    evt = {"event": "undo"}
    ```



* **Class**

    
    ### class rasa.core.events.ActionReverted(timestamp=None)
    Bot undoes its last action.

    The bot reverts everything until before the most recent action.
    This includes the action itself, as well as any events that
    action created, like set slot events - the bot will now
    predict a new action using the state before the most recent
    action.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        tracker._reset()
        tracker.replay_events()
    ```


### Log an executed action


* **Short**

    Logs an action the bot executed to the conversation. Events that
    action created are logged separately.



* **JSON**

    ```
    evt = {"event": "action", "name": "my_action"}
    ```



* **Class**

    
    ### class rasa.core.events.ActionExecuted(action_name, policy=None, confidence=None, timestamp=None)
    An operation describes an action taken + its result.

    It comprises an action and a list of events. operations will be appended
    to the latest `Turn` in the `Tracker.turns`.



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker: "DialogueStateTracker") -> None:

        tracker.set_latest_action_name(self.action_name)
        tracker.clear_followup_action()
    ```


### Start a new conversation session


* **Short**

    Marks the beginning of a new conversation session. Resets the tracker and
    triggers an `ActionSessionStart` which by default applies the existing
    `SlotSet` events to the new session.



* **JSON**

    ```
    evt = {"event": "session_started"}
    ```



* **Class**



* **Effect**

    When added to a tracker, this is the code used to update the tracker:

    ```
    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        # noinspection PyProtectedMember
        tracker._reset()
    ```

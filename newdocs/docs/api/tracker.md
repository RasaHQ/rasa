# Tracker

Trackers maintain the state of a dialogue between the assistant and the user in the form
of conversation sessions. To learn more about how to configure the session behavior,
check out the docs on Session configuration.


### class rasa.core.trackers.DialogueStateTracker(sender_id, slots, max_event_history=None)
Maintains the state of a conversation.

The field max_event_history will only give you these last events,
it can be set in the tracker_store


#### applied_events()
Returns all actions that should be applied - w/o reverted events.


* **Return type**

    `List`[`Event`]



#### as_dialogue()
Return a `Dialogue` object containing all of the turns.

This can be serialised and later used to recover the state
of this tracker exactly.


* **Return type**

    `Dialogue`



#### change_form_to(form_name)
Activate or deactivate a form


* **Return type**

    `None`



#### clear_followup_action()
Clears follow up action when it was executed.


* **Return type**

    `None`



#### copy()
Creates a duplicate of this tracker


#### current_slot_values()
Return the currently set values of the slots


* **Return type**

    [typing.Dict[str, typing.Any]]



#### current_state(event_verbosity=<EventVerbosity.NONE: 1>)
Return the current tracker state as an object.


* **Return type**

    `Dict`[`str`, `Any`]



#### events_after_latest_restart()
Return a list of events after the most recent restart.


* **Return type**

    `List`[`Event`]



#### export_stories(e2e=False)
Dump the tracker as a story in the Rasa Core story format.

Returns the dumped tracker as a string.


* **Return type**

    `str`



#### export_stories_to_file(export_path='debug.md')
Dump the tracker as a story to a file.


* **Return type**

    `None`



#### classmethod from_dict(sender_id, events_as_dict, slots=None, max_event_history=None)
Create a tracker from dump.

The dump should be an array of dumped events. When restoring
the tracker, these events will be replayed to recreate the state.


* **Return type**

    `DialogueStateTracker`



#### generate_all_prior_trackers()
Returns a generator of the previous trackers of this tracker.

The resulting array is representing
the trackers before each action.


* **Return type**

    Generator[DialogueStateTracker, None, None]



#### get_last_event_for(event_type, action_names_to_exclude=None, skip=0)
Gets the last event of a given type which was actually applied.


* **Parameters**

    
    * **event_type** – The type of event you want to find.


    * **action_names_to_exclude** – Events of type ActionExecuted which
    should be excluded from the results. Can be used to skip
    action_listen events.


    * **skip** – Skips n possible results before return an event.



* **Returns**

    event which matched the query or None if no event matched.



* **Return type**

    `Optional`[`Event`]



#### get_latest_entity_values(entity_type)
Get entity values found for the passed entity name in latest msg.

If you are only interested in the first entity of a given type use
next(tracker.get_latest_entity_values(“my_entity_name”), None).
If no entity is found None is the default result.


* **Return type**

    `Iterator`[`str`]



#### get_latest_input_channel()
Get the name of the input_channel of the latest UserUttered event


* **Return type**

    `Optional`[`str`]



#### get_slot(key)
Retrieves the value of a slot.


* **Return type**

    `Optional`[`Any`]



#### idx_after_latest_restart()
Return the idx of the most recent restart in the list of events.

If the conversation has not been restarted, `0` is returned.


* **Return type**

    `int`



#### init_copy()
Creates a new state tracker with the same initial values.


* **Return type**

    `DialogueStateTracker`



#### is_paused()
State whether the tracker is currently paused.


* **Return type**

    `bool`



#### last_executed_action_has(name, skip=0)
Returns whether last ActionExecuted event had a specific name.


* **Parameters**

    
    * **name** – Name of the event which should be matched.


    * **skip** – Skips n possible results in between.



* **Returns**

    True if last executed action had name name, otherwise False.



* **Return type**

    `bool`



#### past_states(domain)
Generate the past states of this tracker based on the history.


* **Return type**

    `deque`



#### recreate_from_dialogue(dialogue)
Use a serialised Dialogue to update the trackers state.

This uses the state as is persisted in a `TrackerStore`. If the
tracker is blank before calling this method, the final state will be
identical to the tracker from which the dialogue was created.


* **Return type**

    `None`



#### reject_action(action_name)
Notify active form that it was rejected


* **Return type**

    `None`



#### replay_events()
Update the tracker based on a list of events.


* **Return type**

    `None`



#### set_form_validation(validate)
Toggle form validation


* **Return type**

    `None`



#### set_latest_action_name(action_name)
Set latest action name
and reset form validation and rejection parameters


* **Return type**

    `None`



#### travel_back_in_time(target_time)
Creates a new tracker with a state at a specific timestamp.

A new tracker will be created and all events previous to the
passed time stamp will be replayed. Events that occur exactly
at the target time will be included.


* **Return type**

    `DialogueStateTracker`



#### trigger_followup_action(action)
Triggers another action following the execution of the current.


* **Return type**

    `None`



#### update(event, domain=None)
Modify the state of the tracker according to an `Event`.


* **Return type**

    `None`

.. _scheduling:

Scheduling Reminders
==================================

.. note::

    This is a python-only feature. For now, you can not use the scheduling
    feature if you are running the framework as a http server.


One of your users got halfway through a task and then stopped messaging your bot. 
Now you want to remind them to get back to you. 
Don't worry! Rasa has you covered.

To schedule an action for later execution, there is a special event called ``ReminderScheduled``. 
Let's make our example a bit more specific: 
We want a confirmation to book a restaurant table - without it the reservation won't be made.

.. code-block:: md
    :linenos:

    ## book a restaurant
    * start_booking
        - action_suggest_restaurant
    * select_restaurant{"name": "Papi's Pizza Place"}
        - action_confirm_booking


The last action in this story, ``action_confirm_booking``, will ask the user for a confirmation and usually
the user would directly send a confirmation - but sometimes they might forget to do that.

There are two things we need to do:
    1. schedule a reminder for a specified time in the future
    2. define what happens when that reminder is triggered


Scheduling the reminder
-----------------------

The reminder will be scheduled when it is returned as an event after executing an action.
Here is an example implementation for our ``action_confirm_booking``:

.. doctest::

    from rasa_core.actions import Action
    from rasa_core.events import ReminderScheduled
    import datetime
    from datetime import timedelta

    class ActionConfirmBooking(Action):
        def name(self):
            return "action_confirm_booking"

        def run(self, dispatcher, tracker, domain):
            dispatcher.utter_message("Do you want to confirm your booking at Papi's pizza?")
            return [ReminderScheduled("action_booking_reminder", datetime.now() + timedelta(hours=5)]

This action schedules a reminder in 5 hours. The reminder will trigger the action ``action_booking_reminder``.

What happens when the reminder is triggered
---------------------------------------------
First of all, the action whose name was part of the reminder will get executed. An implementation for our
example could look like this:

.. doctest::

    from rasa_core.actions import Action
    from rasa_core.events import ReminderScheduled


    class ActionBookingReminder(Action):
        def name(self):
            return "action_booking_reminder"

        def run(self, dispatcher, tracker, domain):
            dispatcher.utter_message("You have an unconfirmed booking at Papi's pizza, would you like to confirm it?")
            return []

By default, reminders will be cancelled if the user sends *any* message to the bot before the scheduled reminder time. 
If you do not want that to happen you need so set the ``kill_on_user_message`` flag when creating the reminder:
``ReminderScheduled("action_booking_reminder", datetime.now() + timedelta(hours=5), kill_on_user_message=False)``

After the action is triggered, the action execution continues as though the user had sent the bot an
empty message. 
So in our stories we need to define what happens after the reminder was executed:

.. code-block:: md
    :linenos:

    ## book a restaurant
    * start_booking
        - action_suggest_restaurant
    * select_restaurant{"name": "Papi's Pizza Place"}
        - action_confirm_booking

    ## reminder_confirm
        - action_booking_reminder
    * agree
        - action_book_restaurant

    ## reminder_cancel
        - action_booking_reminder
    * deny
        - action_cancel_booking


We have added two stories: One where the user agrees to the message we sent in the reminder and one where they
decide to cancel the booking.

.. warning::

    It is **very important** to specify what should happen after the reminder
    was triggered. Otherwise the bot won't know what to do after running the action of the reminder and
    it will run a seemingly random action.
    So make sure to add a story into your training data that starts with the action of the reminder.

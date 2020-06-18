# Guiding Users

## Implicit Confirmation

Implicit confirmation involves repeating details back to the user to reassure
them that they were understood correctly.
This also gives the user a chance to intervene if your assistant misunderstood.

## Explicit Confirmation

Explicit confirmation means asking the user to clarify how you should help them.
An important thing to remember about AI assistants is that **the user is never wrong**.
When a user tells you something like I just moved, they are being perfectly clear,
even if your assistant is not sure how to help them.
If the user goal is ambiguous to your assistant, ask for clarification.

You can configure the `TwoStageFallbackPolicy` to ask your user to clarify,
and present them with quick replies for the most likely intents.
To do this, configure the policy as in this example:

```
policies:
- name: TwoStageFallbackPolicy
  nlu_threshold: 0.3
  core_threshold: 0.3
  fallback_core_action_name: "action_default_fallback"
  fallback_nlu_action_name: "action_default_fallback"
  deny_suggestion_intent_name: "out_of_scope"
...
```

## Explaining Possibilities

AI assistants are always limited to helping users with a specific set of
tasks, and should be able to explain to a user what they can do.
That includes coherently responding to requests that are out of scope.

When to explain the limitations of your assistant is specific to your application,
but these example stories show some common cases:

```
## user asks whats possible
* ask_whatspossible
  - utter_explain_whatspossible

## user asks for something out of scope
* out_of_scope
  - utter_cannot_help
  - utter_explain_whatspossible
```

## Collecting User Feedback

Asking for feedback is one of the best tools you have to understand
your users and determine whether you solved their problem!
Storing this feedback is a powerful way to figure out how you can improve your assistant.

Use a form to collect user feedback. To do this, define a custom form action
(see Forms for more details about forms).

```
from rasa_sdk.action import FormAction

class FeedbackForm(FormAction):

    def name(self):
        return "feedback_form"

    @staticmethod
    def required_slots(tracker):
        return ["feedback", "negative_feedback_reason"]
```

Add the form and slots to your domain:

```
forms:
  - feedback_form
slots:
  feedback:
    type: bool
  feedback_reason:
    type: text
  requested_slot:
    type: text
```

And make sure the `FormPolicy` is present in your configuration file:

```
policies:
  - FormPolicy
  ...
```

## Handing off to a Human

Users will be very frustrated if your assistant cannot help them and there is no way to reroute
the conversation to a human agent. There should always be a way to break out of a conversation!
There are multiple reasons why you might trigger a human handoff:


* the user asks to speak to a human


* the assistant is struggling to understand the user


* the assistant understands what the user wants, but a human is required to resolve the issue.

The direct request to speak with a human can be handled using the mapping policy:

```
intents:
  - request_human: {"triggers": "action_human_handoff"}
```

## feedback1
    - utter_ask_feedback
* out_of_scope
    - utter_thumbsup
    - utter_anything_else

## feedback2
    - utter_ask_feedback
* enter_data
    - utter_thumbsup
    - utter_anything_else

## feedback3
    - utter_ask_feedback
<U> fuck yeah!
    - utter_great
    - utter_anything_else

## feedback deny
    - utter_ask_feedback
<U> not yet
    - utter_thumbsup
    - utter_anything_else

## feedback thank
    - utter_ask_feedback
* thank
    - utter_noworries
    - utter_anything_else

## feedback thumbsup
    - utter_ask_feedback
* feedback{"feedback_value": "negative"}
    - slot{"feedback_value": "negative"}
    - action_tag_feedback
    - utter_thumbsup
    - utter_anything_else

## feedback thumbsup
    - utter_ask_feedback
* feedback{"feedback_value": "positive"}
    - slot{"feedback_value": "positive"}
    - action_tag_feedback
    - utter_great
    - utter_anything_else

# coding=utf-8
import pytest

from rasa_core.channels import CollectingOutputChannel
from rasa_core.dispatcher import Button, Element, Dispatcher
from rasa_core.domain import Domain
from rasa_core.nlg import TemplatedNaturalLanguageGenerator
from rasa_core.trackers import DialogueStateTracker


def test_dispatcher_utter_attachment(default_dispatcher_collecting):
    default_dispatcher_collecting.utter_attachment("http://my-attachment")
    collected = default_dispatcher_collecting.output_channel.latest_output()
    assert {'recipient_id': 'my-sender',
            'image': 'http://my-attachment'} == collected


def test_dispatcher_utter_template(default_dispatcher_collecting,
                                   default_tracker):
    default_dispatcher_collecting.utter_template("utter_goodbye",
                                                 default_tracker)
    collected = default_dispatcher_collecting.output_channel.latest_output()
    assert collected['text'] in {"goodbye 😢", "bye bye 😢"}


def test_dispatcher_handle_unknown_template(default_dispatcher_collecting,
                                            default_tracker):
    default_dispatcher_collecting.utter_template("my_made_up_template",
                                                 default_tracker)

    collected = default_dispatcher_collecting.output_channel.latest_output()
    assert collected is None


def test_dispatcher_template_invalid_vars():
    templates = {
        "my_made_up_template": [{
            "text": "a template referencing an invalid {variable}."}]}
    bot = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator(templates)
    dispatcher = Dispatcher("my-sender", bot, nlg)
    tracker = DialogueStateTracker("my-sender", slots=[])
    dispatcher.utter_template("my_made_up_template", tracker)
    collected = dispatcher.output_channel.latest_output()
    assert collected['text'].startswith(
        "a template referencing an invalid {variable}.")


def test_dispatcher_utter_response(default_dispatcher_collecting):
    text_only_message = {"text": "hey"}
    image_only_message = {"image": "https://i.imgur.com/nGF1K8f.jpg"}
    text_and_image_message = {
        "text": "look at this",
        "image": "https://i.imgur.com/T5xVo.jpg"
    }

    default_dispatcher_collecting.utter_response(text_only_message)
    default_dispatcher_collecting.utter_response(image_only_message)
    default_dispatcher_collecting.utter_response(text_and_image_message)
    collected = default_dispatcher_collecting.output_channel.messages

    assert len(collected) == 4

    # text only message
    assert collected[0] == {
        "recipient_id": "my-sender",
        "text": "hey"}

    # image only message
    assert collected[1] == {
        "recipient_id": "my-sender",
        "image": "https://i.imgur.com/nGF1K8f.jpg"}

    # text & image combined - will result in two messages
    assert collected[2] == {
        "recipient_id": "my-sender",
        "text": "look at this"}
    assert collected[3] == {
        "recipient_id": "my-sender",
        "image": "https://i.imgur.com/T5xVo.jpg"}


def test_dispatcher_utter_buttons(default_dispatcher_collecting):
    buttons = [
        Button(title="Btn1", payload="/btn1"),
        Button(title="Btn2", payload="/btn2")
    ]
    default_dispatcher_collecting.utter_button_message("my message", buttons)
    collected = default_dispatcher_collecting.output_channel.messages
    assert len(collected) == 1
    assert collected[0]['text'] == "my message"
    assert collected[0]['buttons'] == [
        {'payload': u'/btn1', 'title': u'Btn1'},
        {'payload': u'/btn2', 'title': u'Btn2'}
    ]


def test_dispatcher_utter_buttons_from_domain_templ(default_tracker):
    domain_file = "examples/moodbot/domain.yml"
    domain = Domain.load(domain_file)
    bot = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    dispatcher = Dispatcher("my-sender", bot, nlg)
    dispatcher.utter_template("utter_greet", default_tracker)
    assert len(bot.messages) == 1
    assert bot.messages[0]['text'] == "Hey! How are you?"
    assert bot.messages[0]['buttons'] == [
        {'payload': 'great', 'title': 'great'},
        {'payload': 'super sad', 'title': 'super sad'}
    ]


def test_dispatcher_utter_custom_message(default_dispatcher_collecting):
    elements = [
        Element(title="hey there", subtitle="welcome", buttons=[
            Button(title="Btn1", payload="/btn1"),
            Button(title="Btn2", payload="/btn2")]),
        Element(title="another title", subtitle="another subtitle", buttons=[
            Button(title="Btn3", payload="/btn3"),
            Button(title="Btn4", payload="/btn4")])
    ]
    default_dispatcher_collecting.utter_custom_message(*elements)
    collected = default_dispatcher_collecting.output_channel.messages
    assert len(collected) == 2
    assert collected[0]['text'] == "hey there : welcome"
    assert collected[0]['buttons'] == [
        {'payload': u'/btn1', 'title': u'Btn1'},
        {'payload': u'/btn2', 'title': u'Btn2'}
    ]
    assert collected[1]['text'] == "another title : another subtitle"
    assert collected[1]['buttons'] == [
        {'payload': u'/btn3', 'title': u'Btn3'},
        {'payload': u'/btn4', 'title': u'Btn4'}
    ]

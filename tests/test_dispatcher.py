from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels.console import ConsoleOutputChannel
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.dispatcher import Button, Element, Dispatcher
from rasa_core.domain import TemplateDomain


def test_dispatcher_utter_attachment(default_dispatcher_collecting):
    default_dispatcher_collecting.utter_attachment("http://my-attachment")
    collected = default_dispatcher_collecting.output_channel.latest_output()
    assert {'recipient_id': 'my-sender',
            'text': 'Image: http://my-attachment'} == collected


def test_dispatcher_utter_template(default_dispatcher_collecting):
    default_dispatcher_collecting.utter_template("utter_goodbye")
    collected = default_dispatcher_collecting.output_channel.latest_output()
    assert collected['text'] in {"goodbye :(", "bye bye"}


def test_dispatcher_handle_unknown_template(default_dispatcher_collecting):
    default_dispatcher_collecting.utter_template("my_made_up_template")
    collected = default_dispatcher_collecting.output_channel.latest_output()
    assert collected['text'].startswith("Undefined utter template")


def test_dispatcher_template_invalid_vars():
    domain = TemplateDomain(
            [], [], [], {
                "my_made_up_template": [{
                    "text": "a template referencing an invalid {variable}."}]},
            [], [], None, [])
    bot = CollectingOutputChannel()
    dispatcher = Dispatcher("my-sender", bot, domain)
    dispatcher.utter_template("my_made_up_template")
    collected = dispatcher.output_channel.latest_output()
    assert collected['text'].startswith(
            "a template referencing an invalid {variable}.")


def test_dispatcher_utter_buttons(default_dispatcher_collecting):
    buttons = [
        Button(title="Btn1", payload="/btn1"),
        Button(title="Btn2", payload="/btn2")
    ]
    default_dispatcher_collecting.utter_button_message("my message", buttons)
    collected = default_dispatcher_collecting.output_channel.messages
    assert len(collected) == 1
    assert collected[0]['text'] == "my message"
    assert collected[0]['data'] == [
        {'payload': u'/btn1', 'title': u'Btn1'},
        {'payload': u'/btn2', 'title': u'Btn2'}
    ]


def test_dispatcher_utter_buttons_from_domain_templ():
    domain_file = "examples/moodbot/domain.yml"
    domain = TemplateDomain.load(domain_file)
    bot = CollectingOutputChannel()
    dispatcher = Dispatcher("my-sender", bot, domain)
    dispatcher.utter_template("utter_greet")
    assert len(bot.messages) == 1
    assert bot.messages[0]['text'] == "Hey! How are you?"
    assert bot.messages[0]['data'] == [
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
    assert collected[0]['data'] == [
        {'payload': u'/btn1', 'title': u'Btn1'},
        {'payload': u'/btn2', 'title': u'Btn2'}
    ]
    assert collected[1]['text'] == "another title : another subtitle"
    assert collected[1]['data'] == [
        {'payload': u'/btn3', 'title': u'Btn3'},
        {'payload': u'/btn4', 'title': u'Btn4'}
    ]

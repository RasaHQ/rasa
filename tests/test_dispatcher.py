from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels.console import ConsoleOutputChannel
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.dispatcher import Button, Element, Dispatcher
from rasa_core.domain import TemplateDomain


def test_dispatcher_utter_attachment(default_dispatcher, capsys):
    default_dispatcher.utter_attachment("http://my-attachment")
    out, _ = capsys.readouterr()
    assert "Image: http://my-attachment" in out


def test_dispatcher_utter_template(default_dispatcher, capsys):
    default_dispatcher.utter_template("utter_goodbye")
    out, _ = capsys.readouterr()
    assert "goodbye :(" in out or "bye bye" in out


def test_dispatcher_handle_unknown_template(default_dispatcher, capsys):
    default_dispatcher.utter_template("my_made_up_template")
    out, _ = capsys.readouterr()
    assert "Undefined utter template" in out


def test_dispatcher_utter_buttons(default_dispatcher, capsys):
    buttons = [
        Button(title="Btn1", payload="_btn1"),
        Button(title="Btn2", payload="_btn2")
    ]
    default_dispatcher.utter_button_message("my message", buttons)
    out, _ = capsys.readouterr()
    assert "my message" in out
    assert "Btn1" in out
    assert "Btn2" in out


def test_dispatcher_utter_buttons_from_domain_templ(capsys):
    domain_file = "examples/moodbot/domain.yml"
    domain = TemplateDomain.load(domain_file)
    bot = CollectingOutputChannel()
    dispatcher = Dispatcher("my-sender", bot, domain)
    dispatcher.utter_template("utter_greet")
    assert bot.messages[0][1] == "Hey! How are you?"
    assert bot.messages[1][1] == "1: great (great)"
    assert bot.messages[2][1] == "2: super sad (super sad)"


def test_dispatcher_utter_custom_message(default_dispatcher, capsys):
    elements = [
        Element(title="hey there", subtitle="welcome", buttons=[
            Button(title="Btn1", payload="_btn1"),
            Button(title="Btn2", payload="_btn2")]),
        Element(title="anoter title", subtitle="another subtitle", buttons=[
            Button(title="Btn3", payload="_btn3"),
            Button(title="Btn4", payload="_btn4")])
    ]
    default_dispatcher.utter_custom_message(*elements)
    out, _ = capsys.readouterr()
    assert "hey there" in out
    assert "welcome" in out
    assert "Btn1" in out
    assert "Btn2" in out
    assert "anoter title" in out
    assert "another subtitle" in out
    assert "Btn3" in out
    assert "Btn4" in out

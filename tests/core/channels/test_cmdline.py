from typing import List, Text

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from prompt_toolkit import PromptSession, Application
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rasa.core.channels.console import record_messages

from aioresponses import aioresponses

ENTER = "\x0a"


def mock_stdin(input_from_stdin: List[Text]):
    text = ""
    for line in input_from_stdin:
        text += line + ENTER + "\r"

    inp = create_pipe_input()
    inp.send_text(text)

    prompt_session_init = PromptSession.__init__

    def prompt_session_init_fake(self, *k, **kw):
        prompt_session_init(self, input=inp, output=DummyOutput(), *k, **kw)

    PromptSession.__init__ = prompt_session_init_fake

    application_init = Application.__init__

    def application_init_fake(self, *k, **kw):
        kw.pop("input", None)
        kw.pop("output", None)
        application_init(self, input=inp, output=DummyOutput(), *k, **kw)

    Application.__init__ = application_init_fake

    return inp


async def test_record_messages(monkeypatch: MonkeyPatch, capsys: CaptureFixture):
    input_output = [
        {
            "in": "Give me a question!",
            "out": [
                {
                    "buttons": [
                        {
                            "title": "button 1 title",
                            "payload": "button 1 payload",
                            "details": "button 1 details",
                        }
                    ],
                    "text": "This is a button 1",
                },
                {
                    "buttons": [
                        {
                            "title": "button 2 title",
                            "payload": "button 2 payload",
                            "details": "button 2 details",
                        }
                    ],
                    "text": "This is a button 2",
                },
                {
                    "buttons": [
                        {
                            "title": "button 3 title",
                            "payload": "button 3 payload",
                            "details": "button 3 details",
                        }
                    ],
                    "text": "This is a button 3",
                },
            ],
        },
        {"in": ENTER, "out": [{"text": "You've pressed the button"}]},
        {"in": "Dummy message", "out": [{"text": "Dummy response"}]},
    ]

    inp = mock_stdin([m["in"] for m in input_output])

    server_url = "http://example.com"
    endpoint = f"{server_url}/webhooks/rest/webhook"

    with aioresponses() as mocked:

        for output in [m["out"] for m in input_output]:
            if output:
                mocked.post(url=endpoint, payload=output)

        num_of_messages = await record_messages(
            "123",
            server_url=server_url,
            max_message_limit=len(input_output),
            use_response_stream=False,
        )

        assert num_of_messages == len(input_output)

    captured = capsys.readouterr()

    assert "button 1 payload" in captured.out
    assert "button 2 payload" in captured.out

    inp.close()

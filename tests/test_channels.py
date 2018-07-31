from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from httpretty import httpretty

from rasa_core.channels import console
from tests import utilities


def test_console_input():
    import rasa_core.channels.console

    # Overwrites the input() function and when someone else tries to read
    # something from the command line this function gets called.
    with utilities.mocked_cmd_input(rasa_core.channels.console,
                                    text="Test Input"):
        httpretty.register_uri(httpretty.POST,
                               'https://abc.defg/webhooks/rest/webhook',
                               body='')

        httpretty.enable()
        console.record_messages(
                server_url="https://abc.defg",
                max_message_limit=3)
        httpretty.disable()

        assert (httpretty.latest_requests[-1].path ==
                "/webhooks/rest/webhook?stream=true&token=")

        assert (httpretty.latest_requests[-1].body ==
                """{"message": "Test Input", "sender": "default"}""")

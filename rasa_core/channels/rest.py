from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Callable, Optional, Text

from rasa_core.channels.channel import InputChannel
from rasa_core.channels.channel import UserMessage


class HttpInputChannel(InputChannel):
    """An input channel that collects messages from an HTTP endpoint.

    There is no actual API definition of the HTTP endpoint here. Instead, the
    channel expects `listener_components` to be passed in. These components
    define API endpoints, e.g. there can be a rasa REST endpoint, a facebook
    REST endpoint and so on. This channel will then start a HTTP server for
    accepting the incoming HTTP requests and redirecting them to the appropriate
    listener components."""

    def __init__(self, http_port, url_prefix, *listener_components):
        # type: (int, Optional[Text], *HttpInputComponent) -> None
        self.listener_components = listener_components
        self.http_port = http_port
        self.url_prefix = url_prefix

    def start_async_listening(self, message_queue):
        # type: (Dequeue) -> None
        """Start to push the incoming messages from channel into the queue."""
        self._record_messages(message_queue.enqueue)

    def start_sync_listening(self, message_handler):
        # type: (Callable[[UserMessage], None]) -> None
        """Should call the message handler for every incoming message."""
        self._record_messages(message_handler)

    def _has_root_prefix(self):
        # type: () -> bool
        """Check if the stored url prefix corresponds to the root of an url."""
        return (not self.url_prefix or
                self.url_prefix == "" or
                self.url_prefix == "/")

    def _record_messages(self, on_message):
        # type: (Callable[[UserMessage], None]) -> None
        from flask import Flask

        app = Flask(__name__)
        for component in self.listener_components:
            if self._has_root_prefix():
                app.register_blueprint(component.blueprint(on_message))
            else:
                app.register_blueprint(component.blueprint(on_message),
                                       url_prefix=self.url_prefix)

        from gevent.wsgi import WSGIServer
        http_server = WSGIServer(('0.0.0.0', self.http_port), app)
        http_server.serve_forever()


class HttpInputComponent(object):
    def blueprint(self, on_new_message):
        # type: (Callable[[UserMessage], None])-> None
        """Defines a Flask blueprint.

        The blueprint will be attached to a running flask server and handel
        incoming routes it registered for."""
        raise NotImplementedError(
                "Component listener needs to provide blueprint.")

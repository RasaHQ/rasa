from rasa_core.channels import OutputChannel


class CollectingOutputChannel(OutputChannel):
    """Output channel that collects send messages in a list

    (doesn't send them anywhere, just collects them)."""

    def __init__(self):
        self.messages = []

    def latest_output(self):
        if self.messages:
            return self.messages[-1]
        else:
            return None

    def send_text_message(self, recipient_id, message):
        self.messages.append((recipient_id, message))

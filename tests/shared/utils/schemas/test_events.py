from typing import Type

import pytest
import jsonschema

from rasa.core.actions.action import RemoteAction
from rasa.shared.core.events import Event
import rasa.shared.utils.common


@pytest.mark.parametrize("event_class", rasa.shared.utils.common.all_subclasses(Event))
async def test_remote_action_validate_all_event_subclasses(event_class: Type[Event]):
    response = {"events": [{"event": event_class.type_name}], "responses": []}

    if (
        event_class.type_name != "wrong_utterance"
        and event_class.type_name != "wrong_action"
    ):
        jsonschema.validate(response, RemoteAction.action_response_format_spec())

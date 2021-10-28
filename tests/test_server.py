import asyncio
import json
import os
import time
import urllib.parse
import uuid
import sys
from http import HTTPStatus
from multiprocessing import Process, Manager
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import List, Text, Type, Generator, NoReturn, Dict, Optional
from unittest.mock import Mock, ANY

from _pytest.tmpdir import TempPathFactory
import pytest
import requests
from _pytest import pathlib
from _pytest.monkeypatch import MonkeyPatch
from aioresponses import aioresponses
from freezegun import freeze_time
from unittest.mock import MagicMock
from ruamel.yaml import StringIO
from sanic import Sanic

import rasa
import rasa.constants
import rasa.core.jobs
from rasa.engine.storage.local_model_storage import LocalModelStorage
import rasa.nlu
import rasa.server
import rasa.shared.constants
import rasa.shared.utils.io
import rasa.utils.io
from rasa.core import utils
from rasa.core.agent import Agent, load_agent
from rasa.core.channels import (
    channel,
    CollectingOutputChannel,
    RestInput,
    SlackInput,
    CallbackInput,
)
from rasa.core.channels.slack import SlackBot
from rasa.core.tracker_store import InMemoryTrackerStore
import rasa.nlu.test
from rasa.nlu.test import CVEvaluationResult
from rasa.shared.core import events
from rasa.shared.core.constants import (
    ACTION_SESSION_START_NAME,
    ACTION_LISTEN_NAME,
    REQUESTED_SLOT,
    SESSION_START_METADATA_SLOT,
)
from rasa.shared.core.domain import Domain, SessionConfig
from rasa.shared.core.events import (
    Event,
    UserUttered,
    SlotSet,
    BotUttered,
    ActionExecuted,
    SessionStarted,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.model_training import TrainingResult
from rasa.utils.endpoints import EndpointConfig
from tests.conftest import AsyncMock, with_model_id, with_model_ids
from tests.nlu.utilities import ResponseTest
from tests.utilities import json_of_latest_request, latest_request

# a couple of event instances that we can use for testing
test_events = [
    Event.from_parameters(
        {
            "event": UserUttered.type_name,
            "text": "/goodbye",
            "parse_data": {
                "intent": {"confidence": 1.0, INTENT_NAME_KEY: "greet"},
                "entities": [],
            },
        }
    ),
    BotUttered("Welcome!", {"test": True}),
    SlotSet("cuisine", 34),
    SlotSet("cuisine", "34"),
    SlotSet("location", None),
    SlotSet("location", [34, "34", None]),
]

# sequence of events expected at the beginning of trackers
session_start_sequence: List[Event] = [
    ActionExecuted(ACTION_SESSION_START_NAME),
    SessionStarted(),
    ActionExecuted(ACTION_LISTEN_NAME),
]


@pytest.fixture()
async def tear_down_scheduler() -> Generator[None, None, None]:
    yield None
    rasa.core.jobs.__scheduler = None


async def test_root(rasa_non_trained_server: Sanic):
    _, response = await rasa_non_trained_server.asgi_client.get("/")
    assert response.status == HTTPStatus.OK
    assert response.text.startswith("Hello from Rasa:")
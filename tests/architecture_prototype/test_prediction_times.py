import asyncio
import os
import time

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core.domain import Domain


async def test_predictions():
    tracker_store = InMemoryTrackerStore(Domain.empty())
    path_to_model = os.environ.get("MODEL", "graph_model.tar.gz")
    agent = Agent.load_local_model(
        path_to_model, generator=None, tracker_store=tracker_store
    )

    start = time.time()
    print(f"Started model predictions at {start}")
    for i in range(1000):
        await agent.handle_message(
            UserMessage(text="can you help me to build a bot", sender_id=f"sender_{i}")
        )

    end = time.time()
    print(f"Finished model predictions at {end}. Total time: {end - start}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_predictions())

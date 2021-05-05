import asyncio
import cProfile
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
    # profile = cProfile.Profile()
    # profile.enable()

    for i in range(1000):
        r = await agent.handle_message(
            UserMessage(text="why is Rasa useful", sender_id=f"sender_{i}")
        )
        assert r

    end = time.time()
    # profile.disable()
    #
    # profile.dump_stats("./test_inference.prof")
    print(f"Finished model predictions at {end}. Total time: {end - start}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_predictions())

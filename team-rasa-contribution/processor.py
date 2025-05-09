# Inside rasa/core/processor.py (method: handle_message)

import time
import logging

logger = logging.getLogger(__name__)

async def handle_message(self, message: UserMessage) -> Optional[List[Dict[Text, Any]]]:
    start_time = time.perf_counter()
    
    tracker = await self._get_tracker(message.sender_id)

    # existing processing steps (NLU, policy prediction, action execution)...
    await self._run_action(tracker, ...)

    end_time = time.perf_counter()
    duration = round(end_time - start_time, 3)
    logger.info(f"[Runtime] Message '{message.text}' processed in {duration}s")

    return output_channel.messages

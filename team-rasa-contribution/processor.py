# Inside rasa/core/processor.py (method: handle_message)

import time 
import logging

logger = logging.getLogger(__name__) 

async def handle_message(self, message: UserMessage) -> Optional[List[Dict[Text, Any]]]: #handle_message method
    """Process a message from a user and return the response messages."""
    start_time = time.perf_counter() # Start the timer
    
    tracker = await self._get_tracker(message.sender_id) # Retrieve the tracker for the user

    # existing processing steps (NLU, policy prediction, action execution)...
    await self._run_action(tracker, ...) # Run the action based on the tracker state

    end_time = time.perf_counter() # End the timer
    duration = round(end_time - start_time, 3) # Calculate the duration
    # Log the processing time
    logger.info(f"[Runtime] Message '{message.text}' processed in {duration}s")

    return output_channel.messages

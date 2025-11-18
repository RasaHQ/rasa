"""
Custom Rasa Actions with LLM Integration
Hybrid approach: Rasa + OpenAI/Claude
"""
import logging
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction

# Import LLM modules
from actions.llm.providers import get_provider
from actions.llm.fallback import LLMFallbackHandler
from actions.llm.response_enhancer import ResponseEnhancer
from actions.llm.intent_clarifier import IntentClarifier

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration - Change these to switch between providers
# ============================================================================

# Provider: 'openai' or 'claude'
LLM_PROVIDER = "openai"  # Change to 'claude' to use Claude

# Model names
OPENAI_MODEL = "gpt-4"  # or "gpt-3.5-turbo" for cheaper option
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # or "claude-3-haiku-20240307"

# Fallback threshold
CONFIDENCE_THRESHOLD = 0.7


# ============================================================================
# Initialize LLM Components
# ============================================================================

def get_llm_components():
    """Get initialized LLM components"""
    try:
        # Get provider
        if LLM_PROVIDER == "openai":
            provider = get_provider("openai", model=OPENAI_MODEL)
        elif LLM_PROVIDER == "claude":
            provider = get_provider("claude", model=CLAUDE_MODEL)
        else:
            raise ValueError(f"Unknown provider: {LLM_PROVIDER}")

        # Initialize components
        fallback_handler = LLMFallbackHandler(
            provider=provider,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        response_enhancer = ResponseEnhancer(provider=provider)
        intent_clarifier = IntentClarifier(provider=provider)

        return fallback_handler, response_enhancer, intent_clarifier

    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM components: {e}")
        return None, None, None


# Initialize once
FALLBACK_HANDLER, RESPONSE_ENHANCER, INTENT_CLARIFIER = get_llm_components()


# ============================================================================
# Action: LLM Fallback
# ============================================================================

class ActionLLMFallback(Action):
    """
    Fallback to LLM when Rasa confidence is low

    Usage in config.yml:
    policies:
      - name: RulePolicy
        core_fallback_threshold: 0.3
        core_fallback_action_name: "action_llm_fallback"
    """

    def name(self) -> Text:
        return "action_llm_fallback"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        if not FALLBACK_HANDLER or not FALLBACK_HANDLER.provider.is_available():
            # LLM not available
            dispatcher.utter_message(
                text="Xin lỗi, tôi không hiểu. Bạn có thể diễn đạt lại không?"
            )
            return []

        # Get user message
        user_message = tracker.latest_message.get("text", "")

        # Get intent info for logging
        intent_info = tracker.latest_message.get("intent", {})

        # Build context from conversation history
        context = FALLBACK_HANDLER.build_context(tracker.events)

        # Generate response using LLM
        try:
            llm_response = FALLBACK_HANDLER.generate_response(
                user_message=user_message,
                context=context,
                intent_info=intent_info
            )

            dispatcher.utter_message(text=llm_response)

            # Log metrics
            return [SlotSet("last_fallback_used", True)]

        except Exception as e:
            logger.error(f"❌ LLM fallback failed: {e}")
            dispatcher.utter_message(
                text="Xin lỗi, tôi đang gặp sự cố. Bạn có thể thử lại sau không?"
            )
            return []


# ============================================================================
# Action: Show Flights (with LLM enhancement)
# ============================================================================

class ActionShowFlights(Action):
    """
    Show flight search results with LLM-enhanced response
    """

    def name(self) -> Text:
        return "action_show_flights"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Get slots
        departure = tracker.get_slot("departure")
        destination = tracker.get_slot("destination")
        date = tracker.get_slot("travel_date")
        passengers = tracker.get_slot("num_passengers") or 1

        # Mock database query (replace with real DB call)
        flights_data = {
            "departure": departure,
            "destination": destination,
            "date": date,
            "passengers": passengers,
            "results": [
                {
                    "airline": "Vietnam Airlines",
                    "flight_number": "VN123",
                    "departure_time": "08:00",
                    "arrival_time": "10:15",
                    "price": "2,500,000",
                    "duration": "2h 15m",
                    "aircraft": "Boeing 787"
                },
                {
                    "airline": "VietJet Air",
                    "flight_number": "VJ456",
                    "departure_time": "14:30",
                    "arrival_time": "16:45",
                    "price": "1,800,000",
                    "duration": "2h 15m",
                    "aircraft": "Airbus A321"
                }
            ]
        }

        # Enhance response with LLM
        if RESPONSE_ENHANCER and RESPONSE_ENHANCER.provider.is_available():
            try:
                enhanced_response = RESPONSE_ENHANCER.enhance_flight_results(
                    flights_data=flights_data
                )
                dispatcher.utter_message(text=enhanced_response)

            except Exception as e:
                logger.error(f"❌ Response enhancement failed: {e}")
                # Fallback to simple template
                dispatcher.utter_message(
                    text=f"Tìm thấy {len(flights_data['results'])} chuyến bay "
                         f"từ {departure} đến {destination}."
                )
        else:
            # No LLM available, use simple response
            dispatcher.utter_message(
                text=f"Tìm thấy {len(flights_data['results'])} chuyến bay."
            )

        return [SlotSet("flight_search_done", True)]


# ============================================================================
# Action: Confirm Booking (with LLM enhancement)
# ============================================================================

class ActionConfirmBooking(Action):
    """
    Confirm booking with LLM-enhanced message
    """

    def name(self) -> Text:
        return "action_confirm_booking"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Mock booking (replace with real booking API)
        booking_data = {
            "booking_code": "ABC123XYZ",
            "passenger_name": tracker.get_slot("passenger_name") or "Quý khách",
            "flight_number": "VN123",
            "departure": tracker.get_slot("departure"),
            "destination": tracker.get_slot("destination"),
            "date": tracker.get_slot("travel_date"),
            "seat": "12A",
            "price": "2,500,000"
        }

        # Enhance confirmation with LLM
        if RESPONSE_ENHANCER and RESPONSE_ENHANCER.provider.is_available():
            try:
                enhanced_confirmation = RESPONSE_ENHANCER.enhance_booking_confirmation(
                    booking_data=booking_data
                )
                dispatcher.utter_message(text=enhanced_confirmation)

            except Exception as e:
                logger.error(f"❌ Confirmation enhancement failed: {e}")
                dispatcher.utter_message(
                    text=f"✅ Đặt vé thành công! "
                         f"Mã đặt chỗ: {booking_data['booking_code']}"
                )
        else:
            dispatcher.utter_message(
                text=f"✅ Đặt vé thành công! Mã: {booking_data['booking_code']}"
            )

        return [
            SlotSet("booking_code", booking_data["booking_code"]),
            SlotSet("booking_confirmed", True)
        ]


# ============================================================================
# Action: Clarify Intent
# ============================================================================

class ActionClarifyIntent(Action):
    """
    Use LLM to clarify ambiguous intents

    Triggered when top 2 intents have similar confidence
    """

    def name(self) -> Text:
        return "action_clarify_intent"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        if not INTENT_CLARIFIER or not INTENT_CLARIFIER.provider.is_available():
            # LLM not available, ask user directly
            dispatcher.utter_message(
                text="Xin lỗi, bạn có thể nói rõ hơn không?"
            )
            return []

        # Get intent ranking
        intent_ranking = tracker.latest_message.get("intent_ranking", [])

        if not INTENT_CLARIFIER.is_ambiguous(intent_ranking):
            # Not ambiguous, proceed normally
            return []

        user_message = tracker.latest_message.get("text", "")
        context = FALLBACK_HANDLER.build_context(tracker.events) if FALLBACK_HANDLER else ""

        try:
            # Clarify intent using LLM
            clarified_intent = INTENT_CLARIFIER.clarify_intent(
                user_message=user_message,
                intent_ranking=intent_ranking,
                context=context
            )

            # Or ask user for clarification
            # clarification_question = INTENT_CLARIFIER.get_clarification_question(
            #     intent_ranking=intent_ranking
            # )
            # dispatcher.utter_message(text=clarification_question)

            return [SlotSet("clarified_intent", clarified_intent)]

        except Exception as e:
            logger.error(f"❌ Intent clarification failed: {e}")
            dispatcher.utter_message(
                text="Xin lỗi, bạn có thể diễn đạt lại không?"
            )
            return []


# ============================================================================
# Action: Handle Out of Scope
# ============================================================================

class ActionHandleOutOfScope(Action):
    """
    Handle out-of-scope questions with LLM
    """

    def name(self) -> Text:
        return "action_handle_out_of_scope"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text", "")

        if FALLBACK_HANDLER and FALLBACK_HANDLER.provider.is_available():
            try:
                response = FALLBACK_HANDLER.handle_out_of_scope(user_message)
                dispatcher.utter_message(text=response)

            except Exception as e:
                logger.error(f"❌ Out-of-scope handling failed: {e}")
                dispatcher.utter_message(
                    text="Xin lỗi, tôi chỉ hỗ trợ đặt vé máy bay. "
                         "Tôi có thể giúp bạn tìm chuyến bay không? ✈️"
                )
        else:
            dispatcher.utter_message(
                text="Xin lỗi, tôi chỉ hỗ trợ đặt vé máy bay. "
                     "Bạn cần giúp gì về vé máy bay không?"
            )

        return []

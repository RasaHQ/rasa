"""
LLM Fallback Handler - Handle low confidence intents
"""
import logging
from typing import List, Dict, Optional, Any
from .providers import LLMProvider

logger = logging.getLogger(__name__)


class LLMFallbackHandler:
    """
    Handles fallback to LLM when Rasa confidence is low
    """

    def __init__(
        self,
        provider: LLMProvider,
        confidence_threshold: float = 0.7,
        context_length: int = 5
    ):
        """
        Args:
            provider: LLM provider instance
            confidence_threshold: Trigger fallback below this confidence
            context_length: Number of previous messages to include as context
        """
        self.provider = provider
        self.confidence_threshold = confidence_threshold
        self.context_length = context_length

        self.system_prompt = self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """Bạn là trợ lý đặt vé máy bay thông minh và thân thiện.

QUY TẮC QUAN TRỌNG:
1. CHỈ giúp về đặt vé máy bay (booking, hủy vé, đổi lịch, hỏi giá)
2. Nếu user hỏi ngoài topic → Lịch sự từ chối và chuyển hướng về đặt vé
3. Trả lời NGẮN GỌN, rõ ràng (tối đa 3 câu)
4. Luôn hỏi thông tin CẦN THIẾT để giúp user
5. KHÔNG BAO GIỜ bịa thông tin chuyến bay hoặc giá vé

PHONG CÁCH:
- Thân thiện, nhiệt tình
- Gọi user bằng "bạn"
- Dùng emoji phù hợp (✈️, 🛫, 💰, 📅)

VÍ DỤ TỐT:
User: "Thời tiết Paris thế nào?"
Bot: "Tôi không có thông tin thời tiết, nhưng tôi có thể giúp bạn đặt vé bay đến Paris! ✈️ Bạn muốn xem các chuyến bay không?"

VÍ DỤ XẤU:
Bot: "Xin lỗi tôi không biết." ← Không helpful, không redirect
"""

    def set_system_prompt(self, prompt: str):
        """Update system prompt"""
        self.system_prompt = prompt

    def should_fallback(self, confidence: float) -> bool:
        """Check if should fallback to LLM"""
        return confidence < self.confidence_threshold

    def build_context(self, events: List[Dict]) -> str:
        """
        Build conversation context from tracker events

        Args:
            events: List of tracker events

        Returns:
            Formatted context string
        """
        context_messages = []
        count = 0

        # Get last N messages
        for event in reversed(events):
            if count >= self.context_length:
                break

            if event.get("event") == "user":
                context_messages.append(f"User: {event.get('text', '')}")
                count += 1
            elif event.get("event") == "bot":
                text = event.get("text") or event.get("data", {}).get("text", "")
                if text:
                    context_messages.append(f"Bot: {text}")

        # Reverse to chronological order
        context_messages.reverse()

        return "\n".join(context_messages) if context_messages else "No previous context."

    def generate_response(
        self,
        user_message: str,
        context: str = "",
        intent_info: Optional[Dict] = None
    ) -> str:
        """
        Generate fallback response using LLM

        Args:
            user_message: Current user message
            context: Previous conversation context
            intent_info: Rasa intent detection info (for debugging)

        Returns:
            Generated response
        """
        if not self.provider.is_available():
            logger.warning("⚠️ LLM provider not available, using default fallback")
            return "Xin lỗi, tôi không hiểu. Bạn có thể diễn đạt lại không?"

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add context if available
        if context:
            messages.append({
                "role": "user",
                "content": f"Ngữ cảnh hội thoại trước:\n{context}"
            })
            messages.append({
                "role": "assistant",
                "content": "Tôi đã hiểu ngữ cảnh."
            })

        # Add current message
        messages.append({
            "role": "user",
            "content": user_message
        })

        # Log fallback
        logger.info(
            f"🔄 LLM Fallback triggered | "
            f"User: '{user_message[:50]}...' | "
            f"Intent: {intent_info.get('name') if intent_info else 'unknown'} "
            f"({intent_info.get('confidence', 0):.2f})"
        )

        try:
            response = self.provider.generate(messages)
            logger.info(f"✅ LLM fallback response generated")
            return response

        except Exception as e:
            logger.error(f"❌ LLM fallback failed: {e}")
            return "Xin lỗi, hệ thống đang gặp sự cố. Bạn có thể thử lại sau được không?"

    def handle_out_of_scope(self, user_message: str) -> str:
        """
        Handle out-of-scope questions

        Args:
            user_message: User's out-of-scope message

        Returns:
            Polite redirect response
        """
        messages = [
            {
                "role": "system",
                "content": """Bạn là trợ lý đặt vé máy bay.
                User vừa hỏi câu NGOÀI chức năng của bạn.

                Hãy:
                1. Lịch sự từ chối (không thô lỗ)
                2. Giải thích bạn CHỈ giúp đặt vé máy bay
                3. Hỏi xem có thể giúp họ đặt vé không

                NGẮN GỌN (2-3 câu), thân thiện."""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        try:
            return self.provider.generate(messages, temperature=0.5)
        except:
            return "Xin lỗi, tôi chỉ hỗ trợ đặt vé máy bay. Tôi có thể giúp bạn tìm chuyến bay không? ✈️"

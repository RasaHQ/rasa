"""
Intent Clarifier - Use LLM to clarify ambiguous intents
"""
import logging
from typing import List, Dict, Optional
from .providers import LLMProvider

logger = logging.getLogger(__name__)


class IntentClarifier:
    """
    Use LLM to clarify when Rasa has multiple similar confidence intents
    """

    def __init__(
        self,
        provider: LLMProvider,
        ambiguity_threshold: float = 0.2
    ):
        """
        Args:
            provider: LLM provider instance
            ambiguity_threshold: Clarify if top 2 intents differ by less than this
        """
        self.provider = provider
        self.ambiguity_threshold = ambiguity_threshold

        # Intent descriptions for LLM context
        self.intent_descriptions = self._get_default_intent_descriptions()

    def _get_default_intent_descriptions(self) -> Dict[str, str]:
        """Get default intent descriptions"""
        return {
            "greet": "User chào hỏi, bắt đầu hội thoại",
            "goodbye": "User chào tạm biệt, kết thúc hội thoại",
            "book_flight": "User muốn đặt vé máy bay mới",
            "cancel_booking": "User muốn hủy vé đã đặt",
            "reschedule_flight": "User muốn đổi lịch bay (change date/time)",
            "ask_price": "User hỏi về giá vé",
            "ask_flight_status": "User hỏi tình trạng chuyến bay",
            "provide_info": "User cung cấp thông tin (địa điểm, ngày giờ, v.v.)",
            "confirm": "User xác nhận (yes, ok, đồng ý)",
            "deny": "User từ chối (no, không, thôi)",
            "ask_help": "User yêu cầu trợ giúp, hướng dẫn",
        }

    def set_intent_descriptions(self, descriptions: Dict[str, str]):
        """Update intent descriptions"""
        self.intent_descriptions.update(descriptions)

    def is_ambiguous(self, intent_ranking: List[Dict]) -> bool:
        """
        Check if intent is ambiguous

        Args:
            intent_ranking: List of intents with confidence scores

        Returns:
            True if ambiguous
        """
        if len(intent_ranking) < 2:
            return False

        top1 = intent_ranking[0]["confidence"]
        top2 = intent_ranking[1]["confidence"]

        # Ambiguous if difference is small
        return (top1 - top2) < self.ambiguity_threshold

    def clarify_intent(
        self,
        user_message: str,
        intent_ranking: List[Dict],
        context: Optional[str] = None
    ) -> str:
        """
        Use LLM to clarify which intent is correct

        Args:
            user_message: User's message
            intent_ranking: Top intents from Rasa
            context: Conversation context

        Returns:
            Clarified intent name
        """
        if not self.provider.is_available():
            logger.warning("⚠️ LLM not available for clarification")
            return intent_ranking[0]["name"]  # Use Rasa's top prediction

        # Get top 3 intents
        top_intents = intent_ranking[:3]

        # Build intent descriptions
        intent_info = []
        for i, intent_data in enumerate(top_intents, 1):
            intent_name = intent_data["name"]
            confidence = intent_data["confidence"]
            description = self.intent_descriptions.get(
                intent_name,
                "Không có mô tả"
            )

            intent_info.append(
                f"{i}. {intent_name} (confidence: {confidence:.2f})\n"
                f"   Mô tả: {description}"
            )

        intent_info_text = "\n".join(intent_info)

        prompt = f"""
User nói: "{user_message}"

Rasa phát hiện các intents có thể:
{intent_info_text}

{"Ngữ cảnh: " + context if context else ""}

NHIỆM VỤ: Chọn intent PHÙ HỢP NHẤT với câu nói của user.

QUY TẮC:
1. Phân tích ý định CHÍNH của user
2. Xem xét ngữ cảnh (nếu có)
3. Chọn 1 trong các intents trên

TRẢ LỜI: Chỉ tên intent, KHÔNG giải thích.

Ví dụ output: book_flight
"""

        try:
            messages = [{"role": "user", "content": prompt}]

            clarified = self.provider.generate(
                messages,
                temperature=0.1,  # Low temp for consistency
                max_tokens=20
            ).strip()

            # Validate returned intent
            valid_intents = [i["name"] for i in top_intents]

            if clarified in valid_intents:
                logger.info(
                    f"🔍 Intent clarified: '{user_message[:30]}...' "
                    f"→ {clarified} "
                    f"(was: {intent_ranking[0]['name']})"
                )
                return clarified
            else:
                logger.warning(
                    f"⚠️ LLM returned invalid intent: {clarified}. "
                    f"Using Rasa's top: {intent_ranking[0]['name']}"
                )
                return intent_ranking[0]["name"]

        except Exception as e:
            logger.error(f"❌ Intent clarification failed: {e}")
            return intent_ranking[0]["name"]

    def get_clarification_question(
        self,
        intent_ranking: List[Dict]
    ) -> str:
        """
        Generate question to ask user for clarification

        Args:
            intent_ranking: Top intents

        Returns:
            Clarification question
        """
        if not self.provider.is_available():
            return "Xin lỗi, bạn có thể nói rõ hơn được không?"

        top_intents = intent_ranking[:2]

        intent_names = [
            self.intent_descriptions.get(i["name"], i["name"])
            for i in top_intents
        ]

        prompt = f"""
Tạo câu hỏi ngắn gọn để XÁC NHẬN ý định của user.

User có thể muốn:
1. {intent_names[0]}
2. {intent_names[1]}

YÊU CẦU:
- Hỏi user muốn làm gì trong 2 options trên
- Thân thiện, lịch sự
- Ngắn gọn (1 câu)
- Dễ trả lời

Ví dụ tốt:
"Bạn muốn đặt vé mới hay hủy vé đã đặt ạ?"
"""

        try:
            messages = [{"role": "user", "content": prompt}]
            question = self.provider.generate(
                messages,
                temperature=0.7,
                max_tokens=100
            )

            logger.info("❓ Clarification question generated")
            return question

        except:
            return "Xin lỗi, bạn muốn làm gì ạ?"

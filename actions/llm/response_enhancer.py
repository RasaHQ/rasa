"""
Response Enhancer - Make Rasa responses more natural using LLM
"""
import logging
import json
from typing import Dict, List, Any, Optional
from .providers import LLMProvider

logger = logging.getLogger(__name__)


class ResponseEnhancer:
    """
    Enhance Rasa responses to be more natural and conversational
    """

    def __init__(self, provider: LLMProvider):
        """
        Args:
            provider: LLM provider instance
        """
        self.provider = provider

    def enhance_flight_results(
        self,
        flights_data: Dict[str, Any],
        user_preferences: Optional[Dict] = None
    ) -> str:
        """
        Generate natural response for flight search results

        Args:
            flights_data: Flight search results
            user_preferences: User preferences (if any)

        Returns:
            Natural, formatted response
        """
        if not self.provider.is_available():
            logger.warning("⚠️ LLM not available, using template")
            return self._template_flight_results(flights_data)

        prompt = f"""
Tạo response thân thiện, chuyên nghiệp cho kết quả tìm vé máy bay.

THÔNG TIN TÌM KIẾM:
- Từ: {flights_data.get('departure', 'N/A')}
- Đến: {flights_data.get('destination', 'N/A')}
- Ngày: {flights_data.get('date', 'N/A')}
- Số hành khách: {flights_data.get('passengers', 1)}

KẾT QUẢ TÌM ĐƯỢC:
{json.dumps(flights_data.get('results', []), indent=2, ensure_ascii=False)}

YÊU CẦU:
1. Giới thiệu ngắn gọn (1 câu)
2. Trình bày TỪNG chuyến bay rõ ràng:
   - Hãng hàng không + mã chuyến
   - Giờ khởi hành → Giờ đến
   - Giá vé (định dạng VNĐ dễ đọc)
   - 1-2 ưu điểm nổi bật
3. Hỏi user muốn chọn chuyến nào
4. Dùng emoji phù hợp: ✈️, 🛫, ⏰, 💰
5. Ngắn gọn, dễ đọc

CHÚ Ý: Chỉ dùng thông tin có sẵn, KHÔNG bịa thêm.
"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.generate(
                messages,
                temperature=0.7,
                max_tokens=500
            )

            logger.info("✨ Flight results enhanced by LLM")
            return response

        except Exception as e:
            logger.error(f"❌ Enhancement failed: {e}")
            return self._template_flight_results(flights_data)

    def _template_flight_results(self, flights_data: Dict) -> str:
        """Fallback template for flight results"""
        results = flights_data.get('results', [])

        if not results:
            return "Xin lỗi, không tìm thấy chuyến bay phù hợp. Bạn có thể thử ngày khác không?"

        text = f"Tìm thấy {len(results)} chuyến bay:\n\n"

        for i, flight in enumerate(results, 1):
            text += f"{i}. {flight.get('airline', '')} {flight.get('flight_number', '')}\n"
            text += f"   {flight.get('departure_time', '')} → {flight.get('arrival_time', '')}\n"
            text += f"   Giá: {flight.get('price', '')}đ\n\n"

        text += "Bạn muốn đặt chuyến nào?"
        return text

    def enhance_booking_confirmation(
        self,
        booking_data: Dict[str, Any]
    ) -> str:
        """
        Generate natural booking confirmation

        Args:
            booking_data: Booking details

        Returns:
            Natural confirmation message
        """
        if not self.provider.is_available():
            return f"✅ Đặt vé thành công! Mã đặt chỗ: {booking_data.get('booking_code', 'N/A')}"

        prompt = f"""
Tạo response XÁC NHẬN ĐẶT VÉ thành công, vui vẻ và chuyên nghiệp.

THÔNG TIN ĐẶT VÉ:
{json.dumps(booking_data, indent=2, ensure_ascii=False)}

YÊU CẦU:
1. Chúc mừng ngắn gọn
2. Mã đặt chỗ (nổi bật)
3. Tóm tắt chuyến bay (1-2 dòng)
4. Hướng dẫn tiếp theo (check email, làm thủ tục, v.v.)
5. Dùng emoji: ✅, ✈️, 🎉, 📧
6. NGẮN GỌN (tối đa 5 câu)
"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.generate(
                messages,
                temperature=0.7,
                max_tokens=300
            )

            logger.info("✨ Booking confirmation enhanced")
            return response

        except:
            return f"✅ Đặt vé thành công! Mã đặt chỗ: {booking_data.get('booking_code', 'N/A')}"

    def enhance_error_message(
        self,
        error_type: str,
        error_details: Optional[str] = None
    ) -> str:
        """
        Generate friendly error message

        Args:
            error_type: Type of error (e.g., 'payment_failed', 'flight_full')
            error_details: Additional details

        Returns:
            Friendly error message
        """
        if not self.provider.is_available():
            return "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau."

        error_contexts = {
            "payment_failed": "Thanh toán không thành công",
            "flight_full": "Chuyến bay đã hết chỗ",
            "booking_expired": "Phiên đặt vé đã hết hạn",
            "invalid_date": "Ngày không hợp lệ",
            "system_error": "Lỗi hệ thống"
        }

        error_msg = error_contexts.get(error_type, "Có lỗi xảy ra")

        prompt = f"""
Tạo thông báo LỖI thân thiện, không làm user buồn.

LỖI: {error_msg}
Chi tiết: {error_details or 'Không có'}

YÊU CẦU:
1. Xin lỗi chân thành
2. Giải thích ngắn gọn vấn đề
3. Đề xuất giải pháp/bước tiếp theo
4. Thể hiện sẵn sàng hỗ trợ
5. Tích cực, không tiêu cực
6. 2-3 câu

Ví dụ TỐT:
"Rất xin lỗi, chuyến bay này đã hết chỗ 😔 Tôi có thể giúp bạn tìm chuyến bay khác cùng ngày hoặc ngày gần nhất được không?"
"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.provider.generate(
                messages,
                temperature=0.6,
                max_tokens=200
            )

            logger.info(f"✨ Error message enhanced: {error_type}")
            return response

        except:
            return f"Xin lỗi, {error_msg.lower()}. Bạn có muốn thử lại không?"

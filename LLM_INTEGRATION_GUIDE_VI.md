# Hướng Dẫn Tích Hợp LLM (OpenAI/Claude) Với Rasa

## Tổng Quan

Module này cung cấp tích hợp hoàn chỉnh giữa Rasa và các LLMs (OpenAI GPT-4, Claude AI) theo phương án **Hybrid** - kết hợp điểm mạnh của cả hai.

### Kiến Trúc

```
User Input
    ↓
┌─────────────────────────────────┐
│  Rasa NLU (DIETClassifier)      │
│  - Intent classification        │
│  - Entity extraction            │
└─────────────────────────────────┘
    ↓
Confidence Check
    ├─ High (>0.7) → Rasa Actions ⚡
    │                (Fast, controlled)
    │
    └─ Low (<0.7)  → LLM Fallback 🤖
                     (Smart, flexible)
```

---

## Cài Đặt

### Bước 1: Cài Dependencies

```bash
# Cài đặt dependencies cho LLM
pip install -r config/llm_requirements.txt

# Hoặc cài thủ công
pip install openai anthropic python-dotenv rasa-sdk
```

### Bước 2: Cấu Hình API Keys

```bash
# Copy file example
cp .env.example .env

# Edit .env và thêm API keys
nano .env
```

**File `.env`:**
```bash
# Chọn provider
LLM_PROVIDER=openai  # or 'claude'

# OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here

# Or Claude API Key
ANTHROPIC_API_KEY=your-claude-key-here

# Thresholds
CONFIDENCE_THRESHOLD=0.7
AMBIGUITY_THRESHOLD=0.2
```

**Lấy API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Claude: https://console.anthropic.com/

### Bước 3: Cấu Trúc Thư Mục

```
rasa_chatbot/
├── actions/
│   ├── llm/                    # LLM integration module
│   │   ├── __init__.py
│   │   ├── providers.py        # OpenAI + Claude providers
│   │   ├── fallback.py         # LLM fallback handler
│   │   ├── response_enhancer.py
│   │   └── intent_clarifier.py
│   └── actions_llm.py          # Custom Rasa actions
│
├── examples/hybrid_bot/        # Example hybrid bot
│   ├── config.yml
│   ├── domain.yml
│   ├── endpoints.yml
│   └── data/
│
├── config/
│   └── llm_requirements.txt
│
└── .env                        # API keys (create from .env.example)
```

---

## Sử Dụng

### Quick Start

```bash
# 1. Train Rasa model
cd examples/hybrid_bot
rasa train

# 2. Start action server (in terminal 1)
rasa run actions

# 3. Start Rasa (in terminal 2)
rasa shell

# Test conversation
You: Xin chào
Bot: Xin chào! Tôi là trợ lý đặt vé máy bay...

You: Thời tiết Paris thế nào?
Bot: [LLM Fallback] Tôi không có thông tin thời tiết,
     nhưng tôi có thể giúp bạn đặt vé bay đến Paris! ✈️
```

---

## Các Tính Năng

### 1. LLM Fallback ⭐

**Khi dùng:**
- Rasa confidence < 0.7
- User hỏi câu chưa train
- Cần xử lý linh hoạt

**Cách hoạt động:**

```yaml
# config.yml
policies:
  - name: RulePolicy
    core_fallback_threshold: 0.3
    core_fallback_action_name: "action_llm_fallback"
```

**Ví dụ:**
```
User: "Ủa sao đắt thế? So với tàu hỏa thì sao?"

Rasa:
  Intent: nlu_fallback (confidence: 0.25)
  → Trigger action_llm_fallback

LLM Response:
  "Tôi hiểu bạn quan tâm đến giá cả. Máy bay tuy đắt hơn
   tàu hỏa nhưng tiết kiệm thời gian (2h vs 30h). Tôi chỉ
   hỗ trợ đặt vé máy bay. Bạn có muốn tiếp tục không?"
```

**Code:**
```python
# actions/actions_llm.py
class ActionLLMFallback(Action):
    def name(self):
        return "action_llm_fallback"

    def run(self, dispatcher, tracker, domain):
        # Get user message
        user_message = tracker.latest_message.get("text")

        # Build context
        context = FALLBACK_HANDLER.build_context(tracker.events)

        # Generate LLM response
        llm_response = FALLBACK_HANDLER.generate_response(
            user_message=user_message,
            context=context
        )

        dispatcher.utter_message(text=llm_response)
        return []
```

---

### 2. Response Enhancement

**Khi dùng:**
- Rasa xử lý logic (DB query)
- LLM làm đẹp response

**Ví dụ:**

**BEFORE (Template):**
```
Bot: "Tìm thấy 2 chuyến bay:
     1. VN123 - 08:00 - 2.500.000đ
     2. VJ456 - 14:30 - 1.800.000đ"
```

**AFTER (LLM Enhanced):**
```
Bot: "Tuyệt! Tôi tìm thấy 2 chuyến bay phù hợp 🛫

     ✈️ Vietnam Airlines VN123
        ⏰ 08:00 → 10:15
        💰 2.500.000đ
        ✨ Bay sớm, tiện đi làm

     ✈️ VietJet VJ456
        ⏰ 14:30 → 16:45
        💰 1.800.000đ (rẻ hơn 700k!)
        ✨ Giá tốt, linh hoạt

     Bạn thích chuyến nào hơn?"
```

**Code:**
```python
class ActionShowFlights(Action):
    def run(self, dispatcher, tracker, domain):
        # Query database
        flights_data = search_flights(...)

        # Enhance with LLM
        enhanced_response = RESPONSE_ENHANCER.enhance_flight_results(
            flights_data=flights_data
        )

        dispatcher.utter_message(text=enhanced_response)
        return []
```

---

### 3. Intent Clarification

**Khi dùng:**
- Top 2 intents có confidence gần nhau
- Cần xác định intent chính xác

**Ví dụ:**

```
User: "Tôi không muốn đi nữa"

Rasa NLU:
  - cancel_booking: 0.48
  - deny: 0.42
  - goodbye: 0.10
  → Ambiguous!

LLM Clarification:
  Analyze: "User muốn hủy booking"
  → Return: "cancel_booking" ✅
```

**Code:**
```python
class ActionClarifyIntent(Action):
    def run(self, dispatcher, tracker, domain):
        intent_ranking = tracker.latest_message["intent_ranking"]

        if INTENT_CLARIFIER.is_ambiguous(intent_ranking):
            clarified = INTENT_CLARIFIER.clarify_intent(
                user_message=tracker.latest_message["text"],
                intent_ranking=intent_ranking
            )

            return [SlotSet("clarified_intent", clarified)]

        return []
```

---

## Cấu Hình Chi Tiết

### Chuyển Đổi Provider

**Dùng OpenAI:**
```python
# actions/actions_llm.py
LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-4"  # or "gpt-3.5-turbo"
```

**Dùng Claude:**
```python
# actions/actions_llm.py
LLM_PROVIDER = "claude"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
```

### Tuning Thresholds

```python
# Fallback threshold
CONFIDENCE_THRESHOLD = 0.7  # Lower = More LLM usage

# Ambiguity threshold
AMBIGUITY_THRESHOLD = 0.2  # Lower = More clarification
```

### Custom System Prompt

```python
# actions/llm/fallback.py
FALLBACK_HANDLER.set_system_prompt("""
Bạn là trợ lý của công ty XYZ.

Quy tắc:
1. Luôn nhắc tên công ty
2. Formal tone
3. ...
""")
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM rasa/rasa:3.6.21-full

# Install LLM dependencies
RUN pip install openai anthropic python-dotenv

# Copy actions
COPY actions /app/actions/
COPY .env /app/.env

USER 1001
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  rasa:
    image: rasa/rasa:3.6.21-full
    ports:
      - "5005:5005"
    volumes:
      - ./:/app
    command:
      - run
      - --enable-api
      - --cors
      - "*"

  action-server:
    build: .
    ports:
      - "5055:5055"
    volumes:
      - ./actions:/app/actions
      - ./.env:/app/.env
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    command:
      - start
      - --actions
      - actions
```

**Run:**
```bash
docker-compose up -d
```

---

## Monitoring & Costs

### Track Metrics

```python
# In actions_llm.py
logger.info(
    f"🤖 LLM used | "
    f"Provider: {LLM_PROVIDER} | "
    f"Model: {model} | "
    f"Tokens: {tokens}"
)
```

### Cost Estimation

**OpenAI GPT-4:**
- Input: $0.03 / 1K tokens
- Output: $0.06 / 1K tokens
- Average conversation: ~500 tokens = $0.03

**Claude Sonnet:**
- Input: $0.003 / 1K tokens
- Output: $0.015 / 1K tokens
- Average conversation: ~500 tokens = $0.009

**With 80/20 hybrid:**
- 80% requests → Rasa (free)
- 20% requests → LLM
- 1000 conversations/day = ~$6-18/month

---

## Troubleshooting

### Lỗi: API Key Invalid

```bash
# Check .env file
cat .env | grep API_KEY

# Test directly
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### Lỗi: LLM Not Available

```python
# Check provider initialization
if FALLBACK_HANDLER.provider.is_available():
    print("✅ LLM ready")
else:
    print("❌ LLM not available")
```

### Lỗi: Slow Response

```python
# Use faster model
LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-3.5-turbo"  # 10x faster

# Or reduce max_tokens
LLM_MAX_TOKENS = 200  # Instead of 500
```

---

## Best Practices

### 1. Cost Optimization

```python
# Cache common queries
from functools import lru_cache

@lru_cache(maxsize=100)
def get_llm_response(prompt):
    return provider.generate(...)
```

### 2. Error Handling

```python
try:
    llm_response = provider.generate(messages)
except Exception as e:
    logger.error(f"LLM failed: {e}")
    # Fallback to template
    llm_response = "Xin lỗi, tôi không hiểu..."
```

### 3. Testing

```python
# Test without LLM
LLM_PROVIDER = None  # Disable LLM

# Or use mock
from unittest.mock import Mock
FALLBACK_HANDLER.provider = Mock()
```

---

## Examples

Xem folder `examples/hybrid_bot/` để có:
- ✅ Full configuration
- ✅ Sample conversations
- ✅ Training data
- ✅ Custom actions

---

## Tóm Tắt

### Khi Nào Dùng LLM?

| Scenario | Use Rasa | Use LLM |
|----------|----------|---------|
| **Booking flow** | ✅ | ❌ |
| **FAQ đơn giản** | ✅ | ❌ |
| **Unexpected questions** | ❌ | ✅ |
| **Chitchat** | ❌ | ✅ |
| **Complex reasoning** | ❌ | ✅ |
| **Response enhancement** | Combo | ✅ |

### Architecture Tốt Nhất

```
┌────────────────────────────────┐
│  80% traffic → Rasa            │
│  (Fast, Free, Controlled)      │
└────────────────────────────────┘

┌────────────────────────────────┐
│  20% edge cases → LLM          │
│  (Smart, Flexible)             │
└────────────────────────────────┘

= Best user experience + Low cost
```

---

## Support

- 📧 Issues: [GitHub Issues](https://github.com/RasaHQ/rasa/issues)
- 💬 Community: [Rasa Forum](https://forum.rasa.com)
- 📖 Docs: [Rasa Docs](https://rasa.com/docs/)

**Module version:** 1.0.0
**Last updated:** 2024-11-18

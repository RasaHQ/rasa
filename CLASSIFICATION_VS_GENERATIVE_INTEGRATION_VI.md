# Classification Model vs Generative Model & Tích Hợp Rasa + LLM

## Mục Lục
1. [Classification Model Là Gì?](#classification-model-là-gì)
2. [Generative Model Là Gì?](#generative-model-là-gì)
3. [So Sánh Chi Tiết](#so-sánh-chi-tiết)
4. [Ví Dụ Thực Tế](#ví-dụ-thực-tế)
5. [Tích Hợp Rasa + LLM](#tích-hợp-rasa--llm)
6. [Best Practices](#best-practices)

---

## Classification Model Là Gì?

### Định Nghĩa Đơn Giản

**Classification Model** = Model **phân loại** - chọn 1 nhãn từ tập hợp có sẵn.

**Giống như:**
- Giáo viên chấm bài trắc nghiệm (A, B, C, D)
- Nhân viên phân loại thư (Hà Nội, TP.HCM, Đà Nẵng...)
- Bác sĩ chẩn đoán bệnh (Cúm, Covid, Viêm họng...)

**Đặc điểm chính:**
- ✅ Output là **nhãn có sẵn** (pre-defined labels)
- ✅ Không tạo ra nội dung mới
- ✅ Chỉ "chọn" từ danh sách

---

### Cách Hoạt Động

```
┌─────────────────────────────────────────────┐
│  INPUT (Text)                               │
│  "Tôi muốn đặt vé máy bay"                  │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  CLASSIFICATION MODEL                       │
│  (Rasa DIETClassifier)                      │
│                                             │
│  Training đã học:                           │
│  - Intent: greet                            │
│  - Intent: book_flight      ← Chọn cái này │
│  - Intent: ask_price                        │
│  - Intent: cancel_booking                   │
│  - Intent: goodbye                          │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  OUTPUT (Label)                             │
│  Intent: book_flight (confidence: 0.95)     │
└─────────────────────────────────────────────┘
```

### Ví Dụ Cụ Thể Với Rasa

#### Input:
```
"Tôi muốn đặt vé máy bay đi Hà Nội"
```

#### Process:
```python
# Model đã được train với các labels:
labels = [
    "greet",
    "goodbye",
    "book_flight",  # ← Đây!
    "ask_price",
    "cancel_booking"
]

# Model chỉ CHỌN 1 trong số labels này
# KHÔNG tạo ra label mới
```

#### Output:
```json
{
  "intent": "book_flight",
  "confidence": 0.95,
  "entities": [
    {
      "entity": "destination",
      "value": "Hà Nội"
    }
  ]
}
```

**Lưu ý quan trọng:**
- ❌ Model KHÔNG generate text
- ❌ Model KHÔNG tạo ra intent mới
- ✅ Model CHỈ chọn từ danh sách đã train

---

### Ưu Điểm Classification Model

| Ưu Điểm | Giải Thích |
|---------|------------|
| **Predictable** | Output luôn nằm trong tập xác định |
| **Fast** | Chỉ cần tính score cho mỗi label |
| **Accurate** | Với đủ training data, accuracy cao |
| **Controlled** | Không bao giờ output ngoài ý muốn |
| **Lightweight** | Model nhỏ (10-100MB) |
| **Debuggable** | Dễ debug (xem confusion matrix) |

### Nhược Điểm Classification Model

| Nhược Điểm | Giải Thích |
|------------|------------|
| **Rigid** | Chỉ nhận biết intents đã train |
| **Limited** | Không xử lý được intents mới |
| **Manual** | Cần định nghĩa trước tất cả labels |
| **No Generation** | Không thể tạo responses tự nhiên |

---

### Ví Dụ Thất Bại

**Khi user hỏi điều chưa train:**

```
User: "Cho tôi biết thời tiết ở Paris hôm nay"

Classification Model:
- Labels có sẵn: [greet, book_flight, ask_price, goodbye]
- KHÔNG có label "ask_weather"
→ Output: nlu_fallback (confidence: 0.3)
→ Bot: "Xin lỗi, tôi không hiểu" ❌
```

**Giải pháp:**
- Thêm intent "ask_weather" vào training data
- Hoặc dùng LLM làm fallback

---

## Generative Model Là Gì?

### Định Nghĩa Đơn Giản

**Generative Model** = Model **sáng tạo** - tạo ra nội dung mới.

**Giống như:**
- Nhà văn viết truyện (tạo câu chuyện mới)
- Họa sĩ vẽ tranh (tạo hình ảnh mới)
- Nhạc sĩ sáng tác (tạo giai điệu mới)

**Đặc điểm chính:**
- ✅ Output là **text hoàn toàn mới** (not pre-defined)
- ✅ Tạo ra nội dung chưa từng thấy
- ✅ Linh hoạt, sáng tạo

---

### Cách Hoạt Động

```
┌─────────────────────────────────────────────┐
│  INPUT (Text + Context)                     │
│  "Tôi muốn đặt vé máy bay"                  │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  GENERATIVE MODEL (LLM)                     │
│  (GPT-4, Claude)                            │
│                                             │
│  Không có labels định sẵn!                  │
│  Model GENERATE từng từ một:                │
│  "Tôi" → "sẽ" → "giúp" → "bạn" → ...       │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  OUTPUT (Generated Text)                    │
│  "Tôi sẽ giúp bạn đặt vé máy bay. Để tìm    │
│  chuyến bay phù hợp, bạn có thể cho tôi     │
│  biết điểm đi, điểm đến và ngày khởi hành   │
│  không ạ?"                                  │
└─────────────────────────────────────────────┘
```

### Quá Trình Generate

**Step-by-step generation:**

```
Input: "Tôi muốn đặt vé máy bay"

LLM generate từng token:
1. "Tôi"        (probability: 0.95)
2. "sẽ"         (probability: 0.89)
3. "giúp"       (probability: 0.92)
4. "bạn"        (probability: 0.94)
5. "đặt"        (probability: 0.87)
6. "vé"         (probability: 0.91)
7. "máy"        (probability: 0.88)
8. "bay"        (probability: 0.93)
9. "."          (probability: 0.85)
10. "\n"
11. "Để"
12. "tìm"
...

Final output: Hoàn chỉnh, tự nhiên, unique
```

### Ưu Điểm Generative Model

| Ưu Điểm | Giải Thích |
|---------|------------|
| **Flexible** | Xử lý được mọi input, kể cả chưa thấy |
| **Natural** | Responses tự nhiên như người |
| **Creative** | Tạo variations khác nhau |
| **No Training Data** | Zero-shot, few-shot learning |
| **Contextual** | Hiểu context sâu |
| **Multi-task** | 1 model làm nhiều tasks |

### Nhược Điểm Generative Model

| Nhược Điểm | Giải Thích |
|------------|------------|
| **Unpredictable** | Output có thể khác nhau mỗi lần |
| **Hallucination** | Có thể "bịa" thông tin sai |
| **Slow** | Generate từng từ mất thời gian |
| **Expensive** | API costs cao |
| **Large** | Model rất lớn (14-350GB) |
| **Hard to Control** | Khó control exact behavior |

---

### Ví Dụ Thành Công

**Khi user hỏi điều chưa train:**

```
User: "Cho tôi biết thời tiết ở Paris hôm nay"

Generative Model (LLM):
→ Generate: "Xin lỗi, tôi là chatbot hỗ trợ đặt vé máy bay
            và không có khả năng tra cứu thời tiết thời
            gian thực. Tôi có thể giúp bạn đặt vé bay đến
            Paris thay vì không ạ?" ✅
```

**Lợi ích:**
- ✅ Không cần train intent "ask_weather"
- ✅ Response tự nhiên, friendly
- ✅ Redirect về chức năng chính

---

## So Sánh Chi Tiết

### Table Tổng Hợp

| Aspect | Classification Model (Rasa) | Generative Model (LLM) |
|--------|----------------------------|------------------------|
| **Cơ chế** | Chọn label từ danh sách | Tạo text từng từ |
| **Output** | Fixed labels | Free-form text |
| **Training** | Supervised với labels | Pre-training + prompting |
| **Flexibility** | Thấp (chỉ nhận biết intents đã train) | Cao (zero-shot) |
| **Predictability** | Cao (luôn 1 trong N labels) | Thấp (vô số outputs) |
| **Speed** | Nhanh (< 100ms) | Chậm (1-5s) |
| **Size** | Nhỏ (10-100MB) | Rất lớn (14-350GB) |
| **Cost** | Free (self-host) | Expensive (API) |
| **Accuracy** | Cao với đủ data | Varies |
| **Control** | Cao | Thấp |
| **Use Case** | Task-specific | General-purpose |

---

### Ví Dụ So Sánh Trực Quan

#### Scenario: User đặt câu hỏi

**Input:** "Tôi muốn hủy chuyến bay đã đặt"

---

**CLASSIFICATION MODEL (Rasa):**

```
┌─────────────────────────────────────┐
│  Step 1: Intent Classification      │
├─────────────────────────────────────┤
│  Input: "Tôi muốn hủy chuyến bay"   │
│                                     │
│  Scores:                            │
│  - greet: 0.02                      │
│  - book_flight: 0.05                │
│  - cancel_booking: 0.93 ← WINNER   │
│  - ask_price: 0.03                  │
│                                     │
│  Output: cancel_booking             │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Step 2: Action Prediction          │
├─────────────────────────────────────┤
│  TEDPolicy nhìn story:              │
│  - intent: cancel_booking           │
│  - action: action_confirm_cancel    │
│                                     │
│  Output: action_confirm_cancel      │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Step 3: Response Template          │
├─────────────────────────────────────┤
│  Template: "Bạn có chắc muốn hủy   │
│             booking không?"         │
│                                     │
│  Output: Fixed text                 │
└─────────────────────────────────────┘
```

**Đặc điểm:**
- ⚡ Fast (< 100ms total)
- 🎯 Predictable
- 📦 Controlled
- ❌ Rigid templates

---

**GENERATIVE MODEL (LLM):**

```
┌─────────────────────────────────────┐
│  Step 1: Understanding + Generation │
├─────────────────────────────────────┤
│  Input: "Tôi muốn hủy chuyến bay"   │
│                                     │
│  LLM thinks:                        │
│  - User wants to cancel            │
│  - Need confirmation                │
│  - Be empathetic                    │
│  - Offer help                       │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Step 2: Generate Response          │
├─────────────────────────────────────┤
│  Generating token by token...       │
│  "Tôi" → "hiểu" → "rồi" → "."      │
│  "Bạn" → "muốn" → "hủy" → ...      │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Final Output                       │
├─────────────────────────────────────┤
│  "Tôi hiểu rồi. Bạn muốn hủy       │
│  chuyến bay đã đặt. Để tôi giúp    │
│  bạn, bạn có thể cung cấp mã đặt   │
│  chỗ hoặc thông tin chuyến bay     │
│  không? Tôi sẽ kiểm tra chính      │
│  sách hủy và hoàn tiền cho bạn."   │
└─────────────────────────────────────┘
```

**Đặc điểm:**
- 🐌 Slower (1-5s)
- 🎨 Natural, varied
- 💡 Contextual
- ⚠️ Unpredictable

---

### Analogy Dễ Hiểu

**Classification Model = Nhà hàng buffet:**
- ✅ Món ăn có sẵn (labels)
- ✅ Bạn chọn từ menu
- ✅ Fast, predictable
- ❌ Không có món ngoài menu

**Generative Model = Bếp trưởng Michelin:**
- ✅ Tạo món mới theo yêu cầu
- ✅ Creative, flexible
- ✅ Đáp ứng mọi request
- ❌ Slow, expensive

---

## Ví Dụ Thực Tế

### Case Study 1: Booking Chatbot

**User Journey:**

```
1. User: "Xin chào"
2. User: "Tôi muốn đặt vé"
3. User: "Từ Hà Nội đi Sài Gòn"
4. User: "Ngày 15 tháng 12"
5. User: "Giá bao nhiêu?"
6. User: "OK, đặt luôn"
```

---

**Approach A: Pure Classification (Rasa Only)**

```yaml
# domain.yml
intents:
  - greet
  - book_flight
  - provide_info
  - ask_price
  - confirm

responses:
  utter_greet:
    - text: "Xin chào! Tôi có thể giúp gì bạn?"
  utter_ask_route:
    - text: "Bạn muốn bay từ đâu đến đâu?"
  utter_ask_date:
    - text: "Bạn muốn bay vào ngày nào?"
  # ... more templates
```

**Conversation:**
```
1. User: "Xin chào"
   Bot: "Xin chào! Tôi có thể giúp gì bạn?" [TEMPLATE]

2. User: "Tôi muốn đặt vé"
   Bot: "Bạn muốn bay từ đâu đến đâu?" [TEMPLATE]

3. User: "Từ Hà Nội đi Sài Gòn"
   Bot: "Bạn muốn bay vào ngày nào?" [TEMPLATE]

4. User: "Ngày 15 tháng 12"
   Bot: [Search flights...]
   Bot: "Chuyến bay VN123: 2.500.000đ
        Chuyến bay VJ456: 1.800.000đ" [TEMPLATE]

5. User: "Giá bao nhiêu?"
   Bot: "Như đã hiển thị ở trên:
        - VN123: 2.500.000đ
        - VJ456: 1.800.000đ" [TEMPLATE]

6. User: "OK, đặt luôn"
   Bot: "Bạn muốn đặt chuyến nào?" [TEMPLATE]
```

**Đánh giá:**
- ✅ Fast, reliable
- ✅ Follows strict flow
- ❌ Repetitive, robotic
- ❌ Không flexible

---

**Approach B: Pure Generative (LLM Only)**

**Conversation:**
```
1. User: "Xin chào"
   Bot: "Xin chào bạn! Tôi là trợ lý đặt vé máy bay.
        Hôm nay tôi có thể giúp gì cho bạn?" [GENERATED]

2. User: "Tôi muốn đặt vé"
   Bot: "Tuyệt vời! Để tìm chuyến bay phù hợp nhất,
        bạn có thể cho tôi biết:
        - Điểm đi và điểm đến
        - Ngày khởi hành dự kiến
        - Số lượng hành khách" [GENERATED]

3. User: "Từ Hà Nội đi Sài Gòn"
   Bot: "Dạ, chuyến bay Hà Nội - TP. Hồ Chí Minh.
        Bạn dự định bay vào ngày nào ạ?" [GENERATED]

4. User: "Ngày 15 tháng 12"
   Bot: [Function call: search_flights()]
   Bot: "Tôi tìm thấy 2 chuyến bay phù hợp ngày 15/12:

        1. Vietnam Airlines VN123
           - Khởi hành: 08:00
           - Đến: 10:15
           - Giá: 2.500.000đ

        2. VietJet VJ456
           - Khởi hành: 14:30
           - Đến: 16:45
           - Giá: 1.800.000đ

        Bạn có muốn đặt chuyến nào không ạ?" [GENERATED]

5. User: "Giá bao nhiêu?"
   Bot: "Dạ, như tôi vừa thông tin ở trên:
        - Chuyến VN123 của Vietnam Airlines: 2.500.000đ
        - Chuyến VJ456 của VietJet: 1.800.000đ

        Bạn cần thông tin gì thêm không ạ?" [GENERATED]

6. User: "OK, đặt luôn"
   Bot: "Dạ vâng. Bạn muốn đặt chuyến nào trong 2 chuyến
        trên ạ? VN123 (2.5tr) hay VJ456 (1.8tr)?" [GENERATED]
```

**Đánh giá:**
- ✅ Natural, friendly
- ✅ Contextual
- ✅ Flexible
- ❌ Slow (5-10s total)
- ❌ Expensive ($0.05 per conversation)
- ❌ Có thể đi off-topic

---

**Approach C: Hybrid (Rasa + LLM) ⭐ RECOMMENDED**

**Conversation:**
```
1. User: "Xin chào"
   [Rasa: intent=greet, confidence=0.98]
   Bot: "Xin chào! Tôi có thể giúp gì bạn?" [RASA]

2. User: "Tôi muốn đặt vé"
   [Rasa: intent=book_flight, confidence=0.95]
   Bot: "Bạn muốn bay từ đâu đến đâu?" [RASA]

3. User: "Từ Hà Nội đi Sài Gòn"
   [Rasa: entities extracted]
   Bot: "Bạn muốn bay vào ngày nào?" [RASA]

4. User: "Ngày 15 tháng 12"
   [Rasa: date extracted]
   Bot: [Search + display results] [RASA]

5. User: "Ủa sao đắt thế? So với xe lửa thì sao?"
   [Rasa: confidence=0.25, FALLBACK TO LLM]
   Bot: "Tôi hiểu bạn quan tâm đến giá cả. Máy bay
        tuy đắt hơn tàu hỏa nhưng tiết kiệm thời gian
        (2 giờ vs 30 giờ). Tuy nhiên, tôi chỉ hỗ trợ
        đặt vé máy bay. Bạn có muốn tiếp tục đặt vé
        bay không ạ?" [LLM]

6. User: "OK, đặt VJ456"
   [Rasa: intent=confirm, confidence=0.92]
   Bot: [Book ticket] [RASA]
   Bot: "Đã đặt vé thành công!" [RASA]
```

**Đánh giá:**
- ✅ Fast (80% cases use Rasa)
- ✅ Cheap (LLM chỉ 20% cases)
- ✅ Controlled (Rasa handles main flow)
- ✅ Flexible (LLM handles edge cases)
- ⭐ **Best of both worlds!**

---

## Tích Hợp Rasa + LLM

### Tại Sao Nên Tích Hợp?

**Rasa alone:**
- ✅ Fast, cheap, controlled
- ❌ Không flexible với unexpected inputs

**LLM alone:**
- ✅ Flexible, natural
- ❌ Slow, expensive, unpredictable

**Rasa + LLM:**
- ✅ 80% traffic → Rasa (fast, cheap)
- ✅ 20% edge cases → LLM (smart)
- ✅ Best user experience

---

### Phương Pháp 1: LLM Fallback ⭐

**Concept:** Khi Rasa không tự tin → Gọi LLM

**Architecture:**
```
User Input
    ↓
Rasa NLU (DIETClassifier)
    ↓
Intent Confidence?
    ├─ High (>0.7) → Rasa Actions [Fast]
    └─ Low (<0.7)  → LLM Fallback [Smart]
```

**Cấu hình:**

```yaml
# config.yml
policies:
  - name: RulePolicy
    core_fallback_threshold: 0.3
    core_fallback_action_name: "action_llm_fallback"
```

**Custom Action:**

```python
# actions/actions.py
from rasa_sdk import Action
from openai import OpenAI

class ActionLLMFallback(Action):
    def name(self):
        return "action_llm_fallback"

    def run(self, dispatcher, tracker, domain):
        # Lấy message của user
        user_message = tracker.latest_message.get('text')

        # Build context từ conversation history
        context = self._build_context(tracker)

        # Gọi LLM
        client = OpenAI(api_key="your-key")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Bạn là trợ lý đặt vé máy bay.

                    QUY TẮC QUAN TRỌNG:
                    1. CHỈ giúp đặt vé máy bay
                    2. Nếu user hỏi ngoài topic, lịch sự chuyển hướng
                    3. Ngắn gọn (tối đa 3 câu)
                    4. Luôn hỏi rõ thông tin

                    Ví dụ:
                    User: "Thời tiết Paris thế nào?"
                    Bot: "Tôi không có thông tin thời tiết, nhưng
                         tôi có thể giúp bạn đặt vé bay đến Paris.
                         Bạn có muốn xem các chuyến bay không?"
                    """
                },
                {
                    "role": "user",
                    "content": f"Ngữ cảnh: {context}\n\nUser: {user_message}"
                }
            ],
            max_tokens=150,
            temperature=0.7
        )

        llm_text = response.choices[0].message.content
        dispatcher.utter_message(text=llm_text)

        return []

    def _build_context(self, tracker):
        """Lấy 5 messages gần nhất để làm context"""
        events = tracker.events[-10:]
        context = ""

        for event in events:
            if event.get('event') == 'user':
                context += f"User: {event.get('text')}\n"
            elif event.get('event') == 'bot':
                context += f"Bot: {event.get('text')}\n"

        return context
```

**Example Flow:**

```
User: "Đặt vé từ Hà Nội đi Sài Gòn"
→ Rasa confidence: 0.95
→ Use Rasa ✅ (< 100ms)
Bot: "Bạn muốn bay vào ngày nào?"

User: "Ủa mà sao thời tiết Paris hôm nay thế?"
→ Rasa confidence: 0.15
→ Use LLM ✅ (2s)
Bot: [LLM] "Xin lỗi, tôi không có thông tin thời tiết.
     Tôi chỉ hỗ trợ đặt vé máy bay. Bạn có muốn tiếp
     tục đặt vé đến Sài Gòn không?"
```

**Metrics:**
- ⚡ Rasa handles: 80% requests (< 100ms each)
- 🤖 LLM handles: 20% requests (1-2s each)
- 💰 Cost: ~$0.01 per conversation (vs $0.05 LLM-only)
- 📊 User satisfaction: Higher (gets answers for both)

---

### Phương Pháp 2: LLM Response Enhancement

**Concept:** Rasa làm logic, LLM làm đẹp responses

**Architecture:**
```
User Input
    ↓
Rasa NLU → Intent + Entities
    ↓
Rasa Core → Action to execute
    ↓
Action executes (DB query, API call)
    ↓
LLM → Generate natural response từ data
    ↓
User receives beautiful response
```

**Implementation:**

```python
# actions/actions.py
class ActionShowFlights(Action):
    def name(self):
        return "action_show_flights"

    def run(self, dispatcher, tracker, domain):
        # Lấy thông tin từ slots (Rasa extract)
        departure = tracker.get_slot('departure')
        destination = tracker.get_slot('destination')
        date = tracker.get_slot('date')

        # Query database (Rasa logic)
        flights = search_flights_db(departure, destination, date)

        # Prepare data
        flights_data = {
            "departure": departure,
            "destination": destination,
            "date": date,
            "results": flights
        }

        # LLM generate beautiful response
        response = self._generate_response(flights_data)

        dispatcher.utter_message(text=response)
        return []

    def _generate_response(self, data):
        """Dùng LLM để generate natural response"""
        client = OpenAI(api_key="your-key")

        prompt = f"""
        Tạo response thân thiện, chuyên nghiệp cho kết quả tìm vé.

        Thông tin tìm kiếm:
        - Từ: {data['departure']}
        - Đến: {data['destination']}
        - Ngày: {data['date']}

        Các chuyến bay tìm được:
        {json.dumps(data['results'], indent=2, ensure_ascii=False)}

        YÊU CẦU:
        - Thân thiện, chuyên nghiệp
        - Trình bày rõ ràng từng option
        - Highlight ưu điểm của mỗi chuyến
        - Hỏi user muốn chọn chuyến nào
        - Dùng emoji phù hợp (🛫, 💰, ⏰)
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        return response.choices[0].message.content
```

**So sánh kết quả:**

**BEFORE (Template):**
```
Bot: "Tìm thấy 2 chuyến bay:
     1. VN123 - 08:00 - 2.500.000đ
     2. VJ456 - 14:30 - 1.800.000đ
     Bạn chọn chuyến nào?"
```

**AFTER (LLM-enhanced):**
```
Bot: "Tuyệt! Tôi tìm thấy 2 chuyến bay phù hợp từ Hà Nội
     đến TP. Hồ Chí Minh vào ngày 15/12:

     🛫 Buổi sáng - Vietnam Airlines VN123
        ⏰ Khởi hành: 08:00 → Đến: 10:15
        💰 Giá vé: 2.500.000đ
        ✨ Ưu điểm: Bay sớm, tiện đi làm, hãng uy tín

     🛫 Buổi chiều - VietJet VJ456
        ⏰ Khởi hành: 14:30 → Đến: 16:45
        💰 Giá vé: 1.800.000đ (rẻ hơn 700k!)
        ✨ Ưu điểm: Giá tốt, thời gian linh hoạt

     Bạn thích chuyến nào hơn ạ? 😊"
```

**Lợi ích:**
- ✅ Response tự nhiên, friendly
- ✅ Dễ đọc, có cấu trúc
- ✅ Highlight thông tin quan trọng
- ✅ Vẫn có data accuracy (từ Rasa)

---

### Phương Pháp 3: LLM Intent Clarification

**Concept:** LLM giúp clarify khi Rasa không chắc

**Use Case:** Multiple intents có confidence gần nhau

**Implementation:**

```python
class ActionClarifyIntent(Action):
    def name(self):
        return "action_clarify_intent"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get('text')

        # Lấy top 3 intents từ Rasa
        top_intents = tracker.latest_message['intent_ranking'][:3]

        # Nếu confidence gần nhau → Unclear
        if top_intents[0]['confidence'] < 0.8 and \
           top_intents[1]['confidence'] > 0.4:

            # Ask LLM to clarify
            client = OpenAI(api_key="your-key")

            prompt = f"""
            User nói: "{user_message}"

            Possible intents (Rasa detected):
            1. {top_intents[0]['name']} (confidence: {top_intents[0]['confidence']:.2f})
            2. {top_intents[1]['name']} (confidence: {top_intents[1]['confidence']:.2f})
            3. {top_intents[2]['name']} (confidence: {top_intents[2]['confidence']:.2f})

            Mô tả intents:
            - book_flight: User muốn đặt vé máy bay
            - cancel_booking: User muốn hủy booking
            - reschedule_flight: User muốn đổi lịch bay
            - ask_price: User hỏi giá vé

            Intent nào phù hợp NHẤT với câu nói của user?

            Trả lời CHỈ TÊN INTENT, không giải thích.
            """

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1  # Low temp for consistency
            )

            clarified_intent = response.choices[0].message.content.strip()

            # Use clarified intent
            return [
                SlotSet("clarified_intent", clarified_intent),
                FollowupAction(f"utter_{clarified_intent}")
            ]

        else:
            # Confidence high enough, use Rasa
            return []
```

**Example:**

```
User: "Tôi không muốn đi nữa"

Rasa NLU:
- cancel_booking: 0.48
- deny: 0.42
- goodbye: 0.10
→ Unclear! (top 2 gần nhau)

LLM Clarification:
Prompt: "User said: 'Tôi không muốn đi nữa'
        Intents: cancel_booking(0.48), deny(0.42), goodbye(0.10)
        Which one?"

Response: "cancel_booking"

Action:
→ Trigger cancellation flow ✅
```

---

### Phương Pháp 4: Hybrid Router

**Concept:** Smart router quyết định Rasa hay LLM cho TOÀN BỘ request

**Architecture:**
```
┌─────────────────────────────────────┐
│  User Input                         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Smart Router (LLM-powered)         │
│  Quick classification:              │
│  - In-domain? → Rasa                │
│  - Out-of-domain? → LLM             │
│  - Chitchat? → LLM                  │
└─────────────────────────────────────┘
    ↓               ↓
┌─────────┐   ┌─────────┐
│  Rasa   │   │   LLM   │
│ Pipeline│   │  Direct │
└─────────┘   └─────────┘
```

**Implementation:**

```python
class SmartRouter:
    """Router thông minh chọn Rasa hoặc LLM"""

    def __init__(self):
        self.rasa_agent = Agent.load("models/")
        self.openai_client = OpenAI(api_key="your-key")
        self.cache = {}  # Cache domain checks

    async def route_message(self, user_message, sender_id):
        """Main routing logic"""

        # Check domain (cached)
        domain = self._check_domain(user_message)

        if domain == "in_domain":
            # Use Rasa (fast, controlled)
            return await self._rasa_response(user_message, sender_id)
        else:
            # Use LLM (flexible)
            return await self._llm_response(user_message, sender_id)

    def _check_domain(self, message):
        """Quick domain check với GPT-3.5 (cheap)"""

        # Check cache first
        if message in self.cache:
            return self.cache[message]

        # Quick LLM call
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Cheaper model for routing
            messages=[{
                "role": "user",
                "content": f"""
                Chatbot này CHỈ hỗ trợ đặt vé máy bay.

                User message: "{message}"

                Câu này có liên quan đến đặt vé máy bay không?
                - Đặt vé, hủy vé, đổi vé, hỏi giá → YES
                - Thời tiết, tin tức, chitchat → NO

                Trả lời: YES hoặc NO
                """
            }],
            max_tokens=5,
            temperature=0.1
        )

        answer = response.choices[0].message.content.lower()
        domain = "in_domain" if "yes" in answer else "out_domain"

        # Cache result
        self.cache[message] = domain

        return domain

    async def _rasa_response(self, message, sender_id):
        """Use Rasa pipeline"""
        responses = await self.rasa_agent.handle_text(message, sender_id)
        return responses[0]['text'] if responses else "Xin lỗi, tôi không hiểu."

    async def _llm_response(self, message, sender_id):
        """Use LLM directly"""
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": """Bạn là trợ lý đặt vé máy bay.
                Nếu user hỏi ngoài topic, lịch sự chuyển hướng."""
            }, {
                "role": "user",
                "content": message
            }],
            max_tokens=200
        )

        return response.choices[0].message.content
```

**Example Flow:**

```
User: "Đặt vé đi Đà Nẵng"
→ Router check domain: "YES (in-domain)"
→ Route to: Rasa ⚡
→ Response time: 80ms
→ Cost: $0

User: "Nói cho tôi nghe về lịch sử Paris"
→ Router check domain: "NO (out-domain)"
→ Route to: LLM 🤖
→ Response time: 1.5s
→ Cost: $0.02
```

**Metrics tracking:**

```python
# Track usage
metrics = {
    "total_requests": 0,
    "rasa_count": 0,
    "llm_count": 0,
    "rasa_avg_latency": [],
    "llm_avg_latency": [],
}

# After routing
if used_rasa:
    metrics["rasa_count"] += 1
else:
    metrics["llm_count"] += 1

# Alert if LLM > 30%
llm_percentage = metrics["llm_count"] / metrics["total_requests"]
if llm_percentage > 0.3:
    alert("⚠️ High LLM usage! Check Rasa training data.")
```

---

## Best Practices

### 1. Cost Optimization

**A. Caching Responses**

```python
# Simple cache
response_cache = {}

def get_llm_response(prompt):
    # Check cache
    cache_key = hashlib.md5(prompt.encode()).hexdigest()

    if cache_key in response_cache:
        return response_cache[cache_key]  # Instant, free!

    # Call LLM
    response = call_llm_api(prompt)

    # Cache it
    response_cache[cache_key] = response

    return response
```

**Lợi ích:**
- 💰 Giảm 40-60% API costs
- ⚡ Response nhanh hơn (0ms vs 2s)

**B. Model Selection**

```python
# Routing decision
if need_reasoning:
    model = "gpt-4"  # $0.03/1K tokens
elif simple_task:
    model = "gpt-3.5-turbo"  # $0.001/1K tokens (30x cheaper!)
```

### 2. Confidence-Based Fallback

**Tiered approach:**

```python
def handle_message(message, rasa_result):
    confidence = rasa_result['intent']['confidence']

    if confidence > 0.85:
        # Very confident → Use Rasa
        return rasa_response(rasa_result)

    elif confidence > 0.6:
        # Somewhat confident → Rasa with confirmation
        return rasa_with_confirmation(rasa_result)

    elif confidence > 0.3:
        # Low confidence → LLM clarify
        return llm_clarify(message, rasa_result)

    else:
        # Very low → Full LLM
        return llm_fallback(message)
```

### 3. Monitoring & Alerting

**Track key metrics:**

```python
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "rasa_handled": 0,
            "llm_fallback": 0,
            "rasa_confidence_avg": [],
            "llm_cost_total": 0.0,
            "response_time_avg": [],
        }

    def log_request(self, used_rasa, confidence, latency, cost=0):
        self.metrics["requests_total"] += 1

        if used_rasa:
            self.metrics["rasa_handled"] += 1
            self.metrics["rasa_confidence_avg"].append(confidence)
        else:
            self.metrics["llm_fallback"] += 1
            self.metrics["llm_cost_total"] += cost

        self.metrics["response_time_avg"].append(latency)

        # Check thresholds
        self._check_alerts()

    def _check_alerts(self):
        total = self.metrics["requests_total"]
        llm_pct = self.metrics["llm_fallback"] / total

        # Alert if LLM usage too high
        if llm_pct > 0.3:
            send_alert(f"⚠️ LLM usage: {llm_pct*100:.1f}% (target: <30%)")

        # Alert if costs too high
        if self.metrics["llm_cost_total"] > 10.0:  # $10/day
            send_alert(f"💰 LLM costs: ${self.metrics['llm_cost_total']:.2f}")
```

### 4. Prompt Engineering

**Effective system prompts:**

```python
SYSTEM_PROMPT = """
Bạn là trợ lý đặt vé máy bay thông minh.

QUY TẮC BẮT BUỘC:
1. CHỈ giúp về đặt vé máy bay (booking, hủy, đổi, giá)
2. Nếu user hỏi ngoài topic → Lịch sự từ chối + Redirect
3. Responses NGẮN GỌN (tối đa 3 câu)
4. Luôn hỏi thông tin CẦN THIẾT
5. KHÔNG BAO GIỜ bịa thông tin chuyến bay

PHONG CÁCH:
- Thân thiện, chuyên nghiệp
- Dùng emoji phù hợp (🛫, ✈️, 💰)
- Gọi user bằng "bạn"

VÍ DỤ TỐT:
User: "Thời tiết Paris thế nào?"
Bot: "Tôi không có thông tin thời tiết, nhưng tôi có thể
     giúp bạn đặt vé bay đến Paris! Bạn muốn xem các
     chuyến bay không? ✈️"

VÍ DỤ XẤU:
Bot: "Xin lỗi tôi không biết về thời tiết."  ← Không redirect!
"""
```

### 5. A/B Testing

**Test different strategies:**

```python
class ABTest:
    def __init__(self):
        self.strategies = {
            "pure_rasa": {"count": 0, "satisfaction": []},
            "hybrid": {"count": 0, "satisfaction": []},
        }

    def route_user(self, user_id):
        # 50/50 split
        strategy = "hybrid" if hash(user_id) % 2 == 0 else "pure_rasa"
        self.strategies[strategy]["count"] += 1
        return strategy

    def log_satisfaction(self, strategy, rating):
        self.strategies[strategy]["satisfaction"].append(rating)

    def get_results(self):
        for name, data in self.strategies.items():
            avg_satisfaction = sum(data["satisfaction"]) / len(data["satisfaction"])
            print(f"{name}: {avg_satisfaction:.2f}/5.0")
```

### 6. Error Handling

**Graceful degradation:**

```python
async def safe_llm_call(message):
    try:
        response = await call_llm_api(message, timeout=5)
        return response

    except TimeoutError:
        # LLM timeout → Fallback to default
        return "Xin lỗi, tôi đang xử lý hơi chậm. Bạn có thể thử lại không?"

    except APIError as e:
        # API error → Log and fallback
        log_error(f"LLM API error: {e}")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."

    except Exception as e:
        # Unknown error
        log_error(f"Unexpected error: {e}")
        return "Xin lỗi, có lỗi xảy ra. Tôi sẽ chuyển bạn đến hỗ trợ."
```

---

## Tóm Tắt

### Classification vs Generative - Tóm Gọn

| | Classification (Rasa) | Generative (LLM) |
|---|----------------------|------------------|
| **Làm gì** | Chọn label | Tạo text |
| **Output** | Fixed | Free-form |
| **Speed** | Fast (< 100ms) | Slow (1-5s) |
| **Cost** | Free | Expensive |
| **Control** | High | Low |
| **Use case** | Task-specific | General |

### Tích Hợp Rasa + LLM - 4 Phương Pháp

1. **LLM Fallback** ⭐ RECOMMENDED
   - Rasa 80%, LLM 20%
   - Best balance

2. **LLM Response Enhancement**
   - Rasa logic + LLM beauty
   - Natural responses

3. **LLM Intent Clarification**
   - LLM giúp khi unclear
   - Better accuracy

4. **Hybrid Router**
   - Smart routing
   - Optimal resource usage

### Decision Framework

```
Bạn cần build chatbot?
    ↓
┌─────────────────────────┐
│ Domain rõ ràng?         │
│ (booking, FAQ)          │
└─────────────────────────┘
    ├─ YES → Rasa + LLM Fallback ⭐
    │        (80% Rasa, 20% LLM)
    │
    └─ NO → LLM-first
            (Flexible, expensive)
```

### Key Takeaways

✅ **Classification**: Fast, predictable, cheap - Dùng cho main flow
✅ **Generative**: Flexible, natural, expensive - Dùng cho edge cases
✅ **Hybrid**: Best of both worlds - RECOMMENDED
✅ **Monitor**: Track metrics để optimize

---

**File đã tạo:** `CLASSIFICATION_VS_GENERATIVE_INTEGRATION_VI.md`

**Nội dung:**
- ✅ Giải thích chi tiết Classification vs Generative
- ✅ Ví dụ thực tế từng approach
- ✅ 4 phương pháp tích hợp Rasa + LLM
- ✅ Code examples (không đi sâu implementation)
- ✅ Best practices & monitoring
- ✅ Decision framework

Bạn có muốn tôi giải thích thêm phần nào không? 🤖

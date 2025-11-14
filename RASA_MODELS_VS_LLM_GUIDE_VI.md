# Rasa Có Dùng Mô Hình LLM Không?

## Câu Trả Lời Ngắn Gọn

**KHÔNG, Rasa KHÔNG sử dụng LLMs (Large Language Models) như GPT-4, Claude, hoặc Gemini.**

Tuy nhiên, Rasa **CÓ HỖ TRỢ** các **Pre-trained Language Models** (PLMs) như BERT, RoBERTa, GPT-2 - nhưng chúng là các models **nhỏ hơn NHIỀU** và hoạt động **KHÁC BIỆT** so với LLMs hiện đại.

---

## Mục Lục
1. [Rasa Dùng Mô Hình Gì?](#rasa-dùng-mô-hình-gì)
2. [Pre-trained Language Models vs LLMs](#pre-trained-language-models-vs-llms)
3. [Kiến Trúc Mô Hình Của Rasa](#kiến-trúc-mô-hình-của-rasa)
4. [So Sánh: Rasa vs LLM Chatbots](#so-sánh-rasa-vs-llm-chatbots)
5. [Khi Nào Dùng Rasa, Khi Nào Dùng LLM](#khi-nào-dùng-rasa-khi-nào-dùng-llm)
6. [Tích Hợp Rasa Với LLM](#tích-hợp-rasa-với-llm)

---

## Rasa Dùng Mô Hình Gì?

### 1. Custom Neural Networks (Chính)

Rasa chủ yếu dùng **custom neural networks tự build** với TensorFlow:

#### DIETClassifier (Dual Intent Entity Transformer)
- **Chức năng:** Phân loại intent + trích xuất entity
- **Kiến trúc:** Transformer-based
- **Kích thước:** ~10-50MB
- **Parameters:** ~5-20 million
- **Training:** Supervised learning trên YOUR data

```yaml
# config.yml
pipeline:
  - name: DIETClassifier
    epochs: 200
    transformer_size: 256
    number_of_transformer_layers: 2
```

**Đặc điểm:**
- ✅ Lightweight (chạy trên CPU)
- ✅ Fast inference (< 100ms)
- ✅ Task-specific (chỉ làm intent + entity)
- ❌ Không có khả năng reasoning phức tạp
- ❌ Không generate text tự do

#### TEDPolicy (Transformer Embedding Dialogue)
- **Chức năng:** Dự đoán action tiếp theo trong hội thoại
- **Kiến trúc:** Transformer + attention mechanism
- **Kích thước:** ~20-100MB
- **Training:** Học từ stories

```yaml
# config.yml
policies:
  - name: TEDPolicy
    epochs: 200
    transformer_size: 256
```

**Đặc điểm:**
- ✅ Hiểu context hội thoại
- ✅ Dự đoán action phù hợp
- ❌ Không generate responses
- ❌ Cần định nghĩa trước tất cả actions

### 2. Pre-trained Language Models (Tùy Chọn)

Rasa **HỖ TRỢ** (không bắt buộc) các pre-trained models từ HuggingFace:

#### LanguageModelFeaturizer

**Models hỗ trợ:**
- **BERT** (Bidirectional Encoder Representations from Transformers)
- **GPT-2** (Generative Pre-trained Transformer 2)
- **RoBERTa** (Robustly Optimized BERT)
- **DistilBERT** (Distilled BERT)
- **XLNet**
- **CamemBERT** (French)
- **LaBSE** (Language-agnostic BERT Sentence Embedding)

```yaml
# config.yml - Dùng BERT cho featurization
pipeline:
  - name: LanguageModelFeaturizer
    model_name: "bert"
    model_weights: "bert-base-multilingual-cased"
```

**Chức năng:**
- Chuyển text thành vector embeddings
- Cung cấp semantic understanding tốt hơn
- **KHÔNG** generate text
- **KHÔNG** làm reasoning

**Kích thước:**
| Model | Parameters | Size |
|-------|-----------|------|
| DistilBERT | 66M | ~250MB |
| BERT-base | 110M | ~440MB |
| RoBERTa-base | 125M | ~500MB |
| GPT-2 | 124M-1.5B | ~500MB-6GB |

**So với LLMs:**
| | Pre-trained LMs | LLMs |
|---|----------------|------|
| **Parameters** | 66M - 1.5B | 7B - 175B+ |
| **Size** | 250MB - 6GB | 14GB - 350GB+ |
| **Use case** | Feature extraction | Text generation + reasoning |

---

## Pre-trained Language Models vs LLMs

### Điểm Giống Nhau

✅ Đều dựa trên Transformer architecture
✅ Đều được pre-train trên large text corpora
✅ Đều hiểu ngữ cảnh và ngữ nghĩa

### Điểm Khác Biệt Quan Trọng

| Aspect | Pre-trained LMs (BERT, GPT-2) | LLMs (GPT-4, Claude, Gemini) |
|--------|------------------------------|------------------------------|
| **Kích thước** | 100M - 1.5B parameters | 7B - 175B+ parameters |
| **Capabilities** | Feature extraction, classification | Generation, reasoning, multi-step tasks |
| **Input** | Fixed length (512-1024 tokens) | Long context (8K-200K tokens) |
| **Output** | Embeddings, labels | Free-form text |
| **Training** | Pre-training + fine-tuning | Pre-training + RLHF + prompting |
| **Inference** | Fast (~50ms) | Slower (~1-5s) |
| **Cost** | Chạy local, free | API calls, expensive |
| **Use in Rasa** | Featurization only | Not used natively |

### Ví Dụ Minh Họa

**BERT (Pre-trained LM):**
```python
Input: "Đặt vé máy bay đi Hà Nội"
Output: [0.23, -0.45, 0.89, ...] # 768-dimensional vector
→ Vector này được dùng cho classification
```

**GPT-4 (LLM):**
```python
Input: "Đặt vé máy bay đi Hà Nội"
Output: "Tôi sẽ giúp bạn đặt vé máy bay đi Hà Nội.
         Bạn muốn khởi hành từ đâu và vào ngày nào?"
→ Generate text response trực tiếp
```

**Rasa với BERT:**
```
User: "Đặt vé máy bay đi Hà Nội"
1. BERT → vector [0.23, -0.45, ...]
2. DIETClassifier → intent: book_flight, entity: Hà Nội
3. TEDPolicy → action: utter_ask_departure
4. Response template → "Bạn muốn bay từ đâu?"
```

---

## Kiến Trúc Mô Hình Của Rasa

### Pipeline Hoàn Chỉnh

```
User Input: "Đặt vé từ Hà Nội đi Sài Gòn"
    ↓
┌─────────────────────────────────────────────┐
│  1. TOKENIZATION                            │
│  WhitespaceTokenizer                        │
│  → ["Đặt", "vé", "từ", "Hà", "Nội", ...]   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  2. FEATURIZATION (Optional: BERT/GPT-2)    │
│  LanguageModelFeaturizer (BERT)             │
│  → Dense vectors [768-dim per token]        │
│                                             │
│  CountVectorsFeaturizer                     │
│  → Sparse vectors [bag-of-words]            │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  3. INTENT CLASSIFICATION                   │
│  DIETClassifier (Custom Transformer)        │
│  → Intent: book_flight (confidence: 0.95)   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  4. ENTITY EXTRACTION                       │
│  DIETClassifier (same model)                │
│  → departure: "Hà Nội"                      │
│  → destination: "Sài Gòn"                   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  5. DIALOGUE MANAGEMENT                     │
│  TEDPolicy (Custom Transformer)             │
│  → Next action: utter_ask_date              │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  6. RESPONSE GENERATION                     │
│  Response Selector / Templates              │
│  → "Bạn muốn bay vào ngày nào?"             │
└─────────────────────────────────────────────┘
```

### Các Models Trong Rasa

**1. DIETClassifier**
- Type: Custom Transformer
- Size: 10-50MB
- Purpose: Intent + Entity
- Training: Supervised on labeled data

**2. TEDPolicy**
- Type: Custom Transformer
- Size: 20-100MB
- Purpose: Dialogue prediction
- Training: Supervised on stories

**3. ResponseSelector**
- Type: Custom Transformer
- Size: 10-30MB
- Purpose: Select FAQ responses
- Training: Supervised on Q&A pairs

**4. LanguageModelFeaturizer** (Optional)
- Type: Pre-trained BERT/GPT-2/RoBERTa
- Size: 250MB-6GB
- Purpose: Feature extraction only
- Training: Pre-trained, frozen weights

### Training Process

```
Training Data
├─ 500 NLU examples
├─ 20 stories
└─ 5 rules

    ↓

Rasa Train
├─ Train DIETClassifier (30 min)
├─ Train TEDPolicy (20 min)
└─ Optional: Load BERT weights (5 min)

    ↓

Trained Model (50-200MB)
└─ Ready for inference
```

**Không giống LLM:**
- ❌ Không có billions of parameters
- ❌ Không train trên internet-scale data
- ❌ Không có general knowledge
- ✅ Task-specific, lightweight, fast

---

## So Sánh: Rasa vs LLM Chatbots

### Architecture Comparison

**Rasa (Task-Specific Approach):**
```
┌─────────────────────────────────────────┐
│  User Input                             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  NLU Model (DIETClassifier)             │
│  - Intent classification                │
│  - Entity extraction                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Dialogue Manager (TEDPolicy)           │
│  - Action prediction                    │
│  - Context tracking                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Response Templates / Actions           │
│  - Pre-defined responses                │
│  - Custom actions (API calls, DB)       │
└─────────────────────────────────────────┘
    ↓
Bot Response
```

**LLM Chatbot (Generation Approach):**
```
┌─────────────────────────────────────────┐
│  User Input + Conversation History      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  LLM (GPT-4 / Claude)                   │
│  - Understand intent                    │
│  - Generate response                    │
│  - Reasoning & planning                 │
│  - All-in-one model                     │
└─────────────────────────────────────────┘
    ↓
Bot Response (generated)
```

### Feature Comparison

| Feature | Rasa | LLM Chatbot |
|---------|------|-------------|
| **Intent Recognition** | ✅ Excellent (trained) | ✅ Good (zero-shot) |
| **Entity Extraction** | ✅ Excellent (trained) | ⚠️ Moderate (prompting) |
| **Dialogue Control** | ✅ Excellent (stories) | ⚠️ Moderate (prompt eng.) |
| **Response Generation** | ⚠️ Templates only | ✅ Natural, varied |
| **Reasoning** | ❌ Limited | ✅ Strong |
| **Domain Knowledge** | ⚠️ Only trained data | ✅ Broad knowledge |
| **Consistency** | ✅ Very high | ⚠️ Can vary |
| **Latency** | ✅ Fast (<100ms) | ⚠️ Slower (1-5s) |
| **Cost** | ✅ Free (self-host) | ❌ Expensive (API) |
| **Privacy** | ✅ Fully controlled | ⚠️ Data sent to API |
| **Customization** | ✅ Full control | ⚠️ Limited to prompts |
| **Offline** | ✅ Yes | ❌ No |

### Example Conversations

**Scenario: Flight Booking**

**Rasa:**
```
User: Tôi muốn đặt vé máy bay
Bot: Bạn muốn bay từ đâu đến đâu?  [Template response]

User: Từ Hà Nội đi Sài Gòn
Bot: Bạn muốn bay vào ngày nào?     [Template response]

User: Ngày 15/12
Bot: [API call to search flights]
Bot: Đây là các chuyến bay khả dụng: [List from DB]

User: Tôi chọn chuyến 1
Bot: [Action: book_ticket]
Bot: Đã đặt vé thành công!          [Template response]
```

**Pros:**
✅ Predictable, controlled
✅ Always follows flow
✅ Reliable database integration
✅ Fast responses

**Cons:**
❌ Rigid templates
❌ Cannot handle unexpected questions
❌ Limited natural language variation

---

**LLM Chatbot:**
```
User: Tôi muốn đặt vé máy bay
Bot: Tôi sẽ giúp bạn đặt vé máy bay. Để tìm chuyến bay
     phù hợp nhất, bạn có thể cho tôi biết:
     - Bạn muốn bay từ đâu đến đâu?
     - Ngày khởi hành?
     - Số lượng hành khách?

User: Từ Hà Nội đi Sài Gòn, ngày 15/12, 2 người
Bot: Tuyệt vời! Tôi sẽ tìm các chuyến bay từ Hà Nội đến
     TP. Hồ Chí Minh cho 2 hành khách vào ngày 15/12.

     Bạn có muốn chuyến bay buổi sáng hay buổi chiều?
     Và bạn có ưu tiên hãng hàng không nào không?

User: Chuyến sáng, VietJet hoặc Bamboo đều được
Bot: Dạ, tôi hiểu rồi. Để tôi kiểm tra...
     [Function call: search_flights(...)]

     Tôi tìm thấy 3 chuyến bay phù hợp...
```

**Pros:**
✅ Natural, conversational
✅ Flexible, adaptive
✅ Can handle unexpected inputs
✅ Friendly, contextual

**Cons:**
❌ Unpredictable outputs
❌ Can go off-track
❌ API costs
❌ Slower responses

---

## Khi Nào Dùng Rasa, Khi Nào Dùng LLM

### Use Case Matrix

| Scenario | Recommend | Why? |
|----------|-----------|------|
| **Customer Support (Fixed FAQs)** | ✅ Rasa | Predictable, fast, cheap |
| **Booking/Reservation** | ✅ Rasa | Need strict flow control |
| **Banking/Finance** | ✅ Rasa | Security, compliance |
| **General Chatting** | ✅ LLM | Need conversational ability |
| **Complex Reasoning** | ✅ LLM | Rasa can't reason |
| **Content Creation** | ✅ LLM | Need generation |
| **Multi-language** | ⚠️ Both | Rasa with BERT / LLM with prompts |
| **Low Latency Required** | ✅ Rasa | <100ms vs 1-5s |
| **Privacy Critical** | ✅ Rasa | Self-hosted |
| **Budget Constrained** | ✅ Rasa | Free vs API costs |

### Decision Tree

```
┌─────────────────────────────────────────┐
│  Bạn cần chatbot làm gì?                │
└─────────────────────────────────────────┘
              ↓
    ┌─────────┴─────────┐
    │                   │
Task-specific        Open-ended
(booking, FAQ)       (chat, advice)
    │                   │
    ↓                   ↓
┌─────────┐       ┌─────────┐
│  RASA   │       │   LLM   │
└─────────┘       └─────────┘
```

### Detailed Scenarios

#### ✅ Nên Dùng Rasa Khi:

1. **Domain hẹp, rõ ràng**
   - Đặt vé
   - FAQ
   - Form filling
   - Appointment booking

2. **Cần control chặt chẽ**
   - Banking transactions
   - Healthcare advice
   - Legal information

3. **Ưu tiên tốc độ**
   - Real-time support
   - High traffic

4. **Ngân sách hạn chế**
   - Startup
   - Non-profit

5. **Privacy quan trọng**
   - Sensitive data
   - Compliance requirements

#### ✅ Nên Dùng LLM Khi:

1. **Cần conversational ability**
   - Casual chatting
   - Personal assistant

2. **Domain rộng, diverse**
   - General knowledge Q&A
   - Educational tutor

3. **Cần reasoning**
   - Problem solving
   - Decision support

4. **Cần generation**
   - Content creation
   - Email drafting
   - Summarization

5. **Có budget**
   - Enterprise
   - High-value use cases

#### ⚠️ Hybrid Approach (Best of Both)

**Kết hợp Rasa + LLM:**
```
┌─────────────────────────────────────────┐
│  User Input                             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Rasa NLU                               │
│  - Intent classification                │
│  - Entity extraction                    │
└─────────────────────────────────────────┘
    ↓
    ┌─────────┴─────────┐
    │                   │
Simple intents      Complex/Unknown
    │                   │
    ↓                   ↓
┌─────────┐       ┌─────────┐
│  Rasa   │       │   LLM   │
│ Actions │       │  Call   │
└─────────┘       └─────────┘
    │                   │
    └─────────┬─────────┘
              ↓
        Bot Response
```

**Ưu điểm:**
- ✅ Rasa xử lý 80% cases (fast, cheap)
- ✅ LLM xử lý 20% edge cases (flexible)
- ✅ Best of both worlds

---

## Tích Hợp Rasa Với LLM

### Phương Pháp 1: LLM Làm Fallback

**Khi nào:** Rasa không tự tin → Gọi LLM

```yaml
# domain.yml
actions:
  - action_llm_fallback

# actions/actions.py
class ActionLLMFallback(Action):
    def name(self) -> str:
        return "action_llm_fallback"

    async def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get('text')

        # Call GPT-4 / Claude API
        llm_response = call_llm_api(user_message, context=tracker.events)

        dispatcher.utter_message(text=llm_response)
        return []
```

**Config:**
```yaml
# config.yml
policies:
  - name: RulePolicy
    core_fallback_threshold: 0.3
    core_fallback_action_name: action_llm_fallback
```

### Phương Pháp 2: LLM Generate Responses

**Khi nào:** Muốn responses tự nhiên hơn templates

```python
# actions/actions.py
class ActionGenerateResponse(Action):
    def name(self) -> str:
        return "action_generate_response"

    async def run(self, dispatcher, tracker, domain):
        intent = tracker.latest_message['intent']['name']
        entities = tracker.latest_message['entities']

        # Build prompt
        prompt = f"""
        User intent: {intent}
        Entities: {entities}

        Generate a helpful response for the user.
        """

        # Call LLM
        response = call_llm_api(prompt)

        dispatcher.utter_message(text=response)
        return []
```

### Phương Pháp 3: LLM Cho Chitchat

**Khi nào:** Out-of-scope conversations

```yaml
# config.yml
pipeline:
  - name: DIETClassifier
  - name: FallbackClassifier
    threshold: 0.7

# domain.yml
intents:
  - chitchat

# rules.yml
rules:
- rule: Handle chitchat with LLM
  steps:
  - intent: chitchat
  - action: action_llm_chitchat
```

### Phương Pháp 4: LLM Data Augmentation

**Khi nào:** Tự động tạo training data

```python
# scripts/augment_data.py
from openai import OpenAI

def generate_nlu_examples(intent, num_examples=50):
    prompt = f"""
    Generate {num_examples} diverse examples for intent '{intent}'.
    Include variations in:
    - Phrasing
    - Formality
    - Typos
    - Vietnamese and English

    Intent: {intent}
    Examples:
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Generate data
book_flight_examples = generate_nlu_examples("book_flight")

# Add to data/nlu.yml
# Train Rasa model
```

---

## Tóm Tắt

### Câu Trả Lời Chi Tiết

**Rasa có dùng LLM không?**

**KHÔNG** - Rasa không dùng LLMs như GPT-4, Claude, Gemini.

**Rasa dùng gì?**

1. **Custom Transformers** (DIETClassifier, TEDPolicy)
   - Kích thước: 10-100MB
   - Parameters: 5-50M
   - Mục đích: Task-specific (intent, entity, dialogue)

2. **Pre-trained Language Models** (Optional)
   - BERT, GPT-2, RoBERTa
   - Kích thước: 250MB-6GB
   - Parameters: 66M-1.5B
   - Mục đích: Feature extraction only

**Khác biệt chính:**

| | Rasa Models | LLMs |
|---|-------------|------|
| **Size** | 10-100MB | 14-350GB |
| **Parameters** | 5-50M | 7-175B |
| **Capability** | Classification | Generation + Reasoning |
| **Speed** | <100ms | 1-5s |
| **Cost** | Free | Expensive |
| **Control** | High | Low |

**Kết luận:**

✅ **Dùng Rasa:** Task-specific chatbots (booking, FAQ, support)
✅ **Dùng LLM:** Open-ended conversations (general chat, advice)
✅ **Hybrid:** Rasa (main) + LLM (fallback) = Best approach

---

## File Đã Tạo

Tôi đã phân tích source code và tạo **hướng dẫn chi tiết** về models trong Rasa:

📄 **File mới:** `RASA_MODELS_VS_LLM_GUIDE_VI.md`

**Nội dung:**
- ✅ Rasa dùng models gì
- ✅ So sánh Pre-trained LMs vs LLMs
- ✅ Kiến trúc chi tiết
- ✅ Use case recommendations
- ✅ Cách tích hợp Rasa + LLM

**Kết quả chính:**
- Rasa = Task-specific models (lightweight, fast, controlled)
- LLMs = General-purpose models (heavy, flexible, expensive)
- Hybrid approach = Best of both worlds

Bạn có câu hỏi nào thêm về models hay architecture không? 🤖

# Hướng Dẫn Chi Tiết Module Training Trong Rasa

## Mục Lục
1. [Tổng Quan](#tổng-quan)
2. [Module Training Là Gì?](#module-training-là-gì)
3. [Các Loại Training](#các-loại-training)
4. [Quy Trình Training](#quy-trình-training)
5. [Các Tính Năng Chính](#các-tính-năng-chính)
6. [Ví Dụ Thực Tế](#ví-dụ-thực-tế)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Tổng Quan

**Module Training** là **trái tim** của Rasa - nơi chatbot **học** từ dữ liệu của bạn để có thể:
- Hiểu câu hỏi của người dùng (NLU)
- Quyết định hành động tiếp theo (Dialogue Management)
- Cải thiện dần theo thời gian (Incremental Training)

**Không có training = chatbot không thể hoạt động!**

---

## Module Training Là Gì?

### Định Nghĩa Đơn Giản

Training trong Rasa giống như **dạy học cho một đứa trẻ**:

```
Dữ liệu Đầu Vào → Module Training → Model Thông Minh → Chatbot Hoạt Động
```

**Tương tự:**
- **Đứa trẻ**: Rasa model (ban đầu chưa biết gì)
- **Sách giáo khoa**: Training data (NLU data, stories, rules)
- **Quá trình học**: Training process (algorithms, neural networks)
- **Kiểm tra**: Testing & validation
- **Tốt nghiệp**: Trained model (.tar.gz file)

### Các Thành Phần Chính

```
┌─────────────────────────────────────────────┐
│         MODULE TRAINING TRONG RASA          │
├─────────────────────────────────────────────┤
│                                             │
│  1. TRAINING DATA (Input)                   │
│     ├─ NLU data (nlu.yml)                   │
│     ├─ Stories (stories.yml)                │
│     ├─ Rules (rules.yml)                    │
│     └─ Domain (domain.yml)                  │
│                                             │
│  2. CONFIGURATION (Config)                  │
│     ├─ Pipeline (NLU components)            │
│     └─ Policies (Core components)           │
│                                             │
│  3. TRAINING ENGINE                         │
│     ├─ Graph Trainer                        │
│     ├─ NLU Trainer                          │
│     ├─ Core Trainer                         │
│     └─ Fingerprinting & Caching             │
│                                             │
│  4. OUTPUT                                  │
│     └─ Trained Model (.tar.gz)              │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Các Loại Training

Rasa có **4 loại training** chính:

### 1. Full Training (Train Toàn Bộ)

**Mục đích:** Train cả NLU và Dialogue Management từ đầu

**Khi nào dùng:**
- ✅ Lần đầu tiên train model
- ✅ Thay đổi lớn trong cấu hình
- ✅ Thêm/xóa intents, entities, actions

**Lệnh:**
```bash
rasa train
```

**Quy trình:**
```
Input:
  - data/nlu.yml         (câu hỏi mẫu)
  - data/stories.yml     (kịch bản hội thoại)
  - data/rules.yml       (quy tắc cố định)
  - domain.yml           (các intents, actions, responses)
  - config.yml           (cấu hình pipeline & policies)

Process:
  1. Load tất cả training data
  2. Validate data (kiểm tra lỗi)
  3. Train NLU model (hiểu ngôn ngữ)
  4. Train Core model (quản lý hội thoại)
  5. Package thành file .tar.gz

Output:
  - models/20231114-123456.tar.gz
```

**Thời gian:**
- Nhỏ (< 500 câu): 2-10 phút
- Trung bình (500-2000 câu): 10-30 phút
- Lớn (> 2000 câu): 30-120 phút

---

### 2. NLU-Only Training (Train Chỉ NLU)

**Mục đích:** Chỉ train phần hiểu ngôn ngữ tự nhiên

**Khi nào dùng:**
- ✅ Chỉ thay đổi dữ liệu NLU
- ✅ Thêm intent hoặc entity mới
- ✅ Không động đến stories/rules

**Lệnh:**
```bash
rasa train nlu
```

**Quy trình:**
```
Input:
  - data/nlu.yml
  - config.yml (chỉ phần pipeline)

Process:
  1. Load NLU training data
  2. Train các component trong pipeline:
     - Tokenizer (tách từ)
     - Featurizer (vector hóa)
     - Intent Classifier (phân loại ý định)
     - Entity Extractor (trích xuất entities)

Output:
  - models/nlu-20231114-123456.tar.gz
```

**Ví dụ thực tế:**
```yaml
# data/nlu.yml - Thêm intent mới
- intent: book_flight
  examples: |
    - đặt vé máy bay
    - tôi muốn bay đi Hà Nội
    - book flight to Saigon

# Chỉ cần: rasa train nlu
# Không cần train lại stories
```

**Ưu điểm:**
- ⚡ Nhanh hơn full training (30-50%)
- 💰 Tiết kiệm tài nguyên
- 🔄 Không ảnh hưởng Core model

---

### 3. Core-Only Training (Train Chỉ Core)

**Mục đích:** Chỉ train phần quản lý hội thoại

**Khi nào dùng:**
- ✅ Chỉ thay đổi stories/rules
- ✅ Thêm actions mới
- ✅ Không thay đổi NLU data

**Lệnh:**
```bash
rasa train core
```

**Quy trình:**
```
Input:
  - data/stories.yml
  - data/rules.yml
  - domain.yml
  - config.yml (chỉ phần policies)

Process:
  1. Load Core training data
  2. Train các policies:
     - MemoizationPolicy (ghi nhớ patterns)
     - TEDPolicy (machine learning)
     - RulePolicy (quy tắc cố định)

Output:
  - models/core-20231114-123456.tar.gz
```

**Ví dụ thực tế:**
```yaml
# data/stories.yml - Thêm story mới
- story: flight booking flow
  steps:
    - intent: book_flight
    - action: action_ask_destination
    - intent: inform
    - action: action_confirm_booking

# Chỉ cần: rasa train core
# Không cần train lại NLU
```

---

### 4. Incremental Training (Fine-tuning)

**Mục đích:** Cải thiện model có sẵn với dữ liệu mới

**Khi nào dùng:**
- ✅ Đã có model hoạt động tốt
- ✅ Chỉ muốn thêm ví dụ mới
- ✅ Không thay đổi labels (intents/entities/actions)

**Lệnh:**
```bash
# Finetune từ model mới nhất
rasa train --finetune

# Hoặc chỉ định model cụ thể
rasa train --finetune models/20231114-123456.tar.gz

# Dùng 50% epochs (nhanh hơn)
rasa train --finetune --epoch-fraction 0.5
```

**Quy trình:**
```
Input:
  - Existing trained model
  - New training examples (không có label mới)

Process:
  1. Load pretrained model weights
  2. Continue training với fewer epochs
  3. Update model với new data

Output:
  - models/finetuned-20231114-123456.tar.gz
```

**Ví dụ:**
```yaml
# TRƯỚC: Đã có 50 ví dụ cho intent "greet"
# SAU: Thêm 10 ví dụ mới

# data/nlu.yml (thêm)
- intent: greet
  examples: |
    - chào bác
    - hi anh
    - hello chị

# Fine-tune (nhanh hơn 50-70% so với train từ đầu)
rasa train --finetune --epoch-fraction 0.3
```

**Điều kiện:**
- ⚠️ Config phải giống y hệt (trừ epochs)
- ⚠️ Không thêm/xóa labels
- ⚠️ Model tương thích version

**Ưu điểm:**
- ⚡ Cực nhanh (dùng 30-50% epochs)
- 📈 Kết quả tốt hơn train from scratch với ít data
- 💾 Giữ lại knowledge từ model cũ

---

## Quy Trình Training

### Bước 1: Chuẩn Bị Dữ Liệu

#### NLU Data (data/nlu.yml)

**Mục đích:** Dạy chatbot hiểu người dùng nói gì

```yaml
version: "3.1"

nlu:
- intent: greet
  examples: |
    - xin chào
    - hello
    - hi

- intent: book_flight
  examples: |
    - đặt vé bay từ [Hà Nội](departure) đến [Sài Gòn](destination)
    - tôi muốn bay đi [Đà Nẵng](destination) vào [ngày mai](time)
    - book flight to [Tokyo](destination)
```

**Gồm:**
- **Intents:** Ý định của người dùng (greet, book_flight, ask_price...)
- **Examples:** Các câu mẫu
- **Entities:** Thông tin quan trọng (địa điểm, thời gian, số lượng...)

#### Stories (data/stories.yml)

**Mục đích:** Dạy chatbot cách dẫn dắt hội thoại

```yaml
version: "3.1"

stories:
- story: flight booking happy path
  steps:
  - intent: greet
  - action: utter_greet
  - intent: book_flight
    entities:
    - departure: "Hà Nội"
    - destination: "Sài Gòn"
  - action: action_search_flights
  - action: utter_show_results
  - intent: confirm
  - action: action_book_flight
  - action: utter_confirm_booking
```

**Giống như:** Kịch bản phim - user nói gì, bot trả lời gì

#### Rules (data/rules.yml)

**Mục đích:** Các quy tắc cố định, luôn đúng

```yaml
version: "3.1"

rules:
- rule: Say goodbye anytime
  steps:
  - intent: goodbye
  - action: utter_goodbye

- rule: Fallback when low confidence
  steps:
  - intent: nlu_fallback
  - action: utter_please_rephrase
```

**Khác với stories:** Rules luôn được thực thi, không phụ thuộc context

#### Domain (domain.yml)

**Mục đích:** Định nghĩa toàn bộ "vũ trụ" của chatbot

```yaml
version: "3.1"

intents:
  - greet
  - book_flight
  - goodbye

entities:
  - departure
  - destination
  - time

slots:
  departure:
    type: text
    mappings:
    - type: from_entity
      entity: departure

actions:
  - utter_greet
  - action_search_flights
  - action_book_flight

responses:
  utter_greet:
    - text: "Xin chào! Tôi có thể giúp gì bạn?"
```

### Bước 2: Cấu Hình (config.yml)

```yaml
recipe: default.v1
language: vi
assistant_id: flight_booking_bot

# Pipeline: Xử lý NLU
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: DIETClassifier
    epochs: 200

# Policies: Quản lý dialogue
policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
    epochs: 200
  - name: RulePolicy
```

### Bước 3: Validate Data (Kiểm Tra)

**Trước khi train, Rasa sẽ validate:**

```bash
rasa data validate
```

**Kiểm tra:**
- ✅ Syntax errors trong YAML
- ✅ Intents trong stories có trong domain không
- ✅ Actions được định nghĩa chưa
- ✅ Story conflicts (2 stories giống nhau nhưng khác outcome)
- ✅ Missing responses

**Ví dụ lỗi:**
```
ERROR: Action 'action_search_flights' in story 'booking' not found in domain
FIX: Thêm 'action_search_flights' vào domain.yml
```

### Bước 4: Training Process

**Chạy lệnh:**
```bash
rasa train
```

**Quá trình bên trong:**

#### Phase 1: Data Loading (5-10% thời gian)
```
[Loading] Reading training data files...
  ✓ data/nlu.yml (150 examples)
  ✓ data/stories.yml (20 stories)
  ✓ data/rules.yml (5 rules)
  ✓ domain.yml
```

#### Phase 2: Validation (5-10% thời gian)
```
[Validating] Checking for inconsistencies...
  ✓ All intents are valid
  ✓ All actions are defined
  ✓ No story conflicts detected
```

#### Phase 3: NLU Training (40-50% thời gian)
```
[Training NLU] Pipeline components:
  → WhitespaceTokenizer... Done
  → RegexFeaturizer... Done
  → CountVectorsFeaturizer... Processing (epoch 1/200)
  → DIETClassifier... Processing (epoch 50/200)
```

**Bên trong DIETClassifier:**
- Chuyển text → vectors
- Train neural network
- Học phân biệt intents
- Học extract entities

#### Phase 4: Core Training (30-40% thời gian)
```
[Training Core] Policies:
  → MemoizationPolicy... Done
  → TEDPolicy... Processing (epoch 100/200)
  → RulePolicy... Done
```

**Bên trong TEDPolicy:**
- Học patterns từ stories
- Dự đoán action tiếp theo
- Tối ưu với neural networks

#### Phase 5: Model Packaging (5% thời gian)
```
[Packaging] Creating model archive...
  ✓ Saved model to: models/20231114-123456.tar.gz
```

**Model file chứa:**
- Trained NLU model
- Trained Core model
- Configuration
- Fingerprints (để skip unchanged parts)

---

## Các Tính Năng Chính

### 1. Fingerprinting & Caching

**Mục đích:** Không train lại phần không thay đổi

**Cách hoạt động:**
```
Lần train đầu:
  - NLU data: hash = abc123
  - Stories: hash = def456
  → Train cả hai
  → Lưu fingerprints

Lần train thứ 2 (chỉ sửa NLU):
  - NLU data: hash = xyz789 (THAY ĐỔI)
  - Stories: hash = def456 (KHÔNG ĐỔI)
  → Chỉ train NLU
  → Reuse Core model từ cache
  → Tiết kiệm 40-60% thời gian!
```

**Xem logs:**
```
Training NLU model...
Core model has not changed. Using cached model.
```

**Force retrain (bỏ qua cache):**
```bash
rasa train --force
```

### 2. Data Augmentation

**Mục đích:** Tạo thêm training examples từ stories

**Ví dụ:**
```yaml
# Original story
- intent: greet
- action: utter_greet
- intent: book_flight
- action: utter_ask_destination

# Augmented variations (tự động tạo):
- intent: greet
- action: utter_greet
- intent: inform  # Khác story gốc
- action: utter_ask_destination

- intent: book_flight
- action: utter_ask_destination  # Bỏ qua utter_greet
```

**Cấu hình:**
```yaml
policies:
  - name: TEDPolicy
    epochs: 200
    # Augmentation multiplier
    # 50 stories → 50 * 20 = 1000 augmented stories
    augmentation_factor: 20
```

**Lợi ích:**
- ✅ Model robust hơn
- ✅ Xử lý tốt unexpected user behavior
- ✅ Không cần viết nhiều stories

### 3. Story Conflict Detection

**Mục đích:** Phát hiện stories mâu thuẫn

**Ví dụ conflict:**
```yaml
# Story 1
- story: booking path 1
  steps:
  - intent: book_flight
  - action: utter_greet      # Action A

# Story 2
- story: booking path 2
  steps:
  - intent: book_flight
  - action: action_search    # Action B (KHÁC!)
```

**Vấn đề:** Cùng input (intent) nhưng 2 output khác nhau → Model confused!

**Phát hiện:**
```bash
rasa data validate stories
```

```
Story Conflict found!
  Story 1: booking path 1
  Story 2: booking path 2
  Conflicting action: utter_greet vs action_search
```

**Giải pháp:**
- Thêm context (slots) để phân biệt
- Hoặc merge thành 1 story duy nhất

### 4. Interactive Learning

**Mục đích:** Train bằng cách chat thực tế với bot

**Lệnh:**
```bash
rasa interactive
```

**Quy trình:**
```
1. User: "xin chào"
2. Bot predict: utter_greet
3. You confirm: ✓ Correct
4. User: "đặt vé máy bay"
5. Bot predict: action_search_flights
6. You correct: ✗ Wrong, should be "utter_ask_destination"
7. Rasa learn từ correction
8. Export thành story mới
```

**Tính năng:**
- ✅ Realtime feedback
- ✅ Tự động tạo stories
- ✅ Visualize conversation flow
- ✅ Fix mistakes ngay lập tức

### 5. Comparison Training

**Mục đích:** So sánh nhiều configs để chọn tốt nhất

**Lệnh:**
```bash
rasa train core --compare \
  --config config1.yml config2.yml config3.yml \
  --runs 3 \
  --percentages 0 25 50
```

**Giải thích:**
- Train 3 configs khác nhau
- Mỗi config train 3 lần (để average)
- Test với 0%, 25%, 50% data removed

**Output:**
```
Config 1 (TEDPolicy only):
  - Accuracy: 85%
  - Training time: 10 min

Config 2 (MemoizationPolicy + TEDPolicy):
  - Accuracy: 92%
  - Training time: 15 min

Config 3 (All policies):
  - Accuracy: 94%
  - Training time: 25 min

→ Recommended: Config 2 (best accuracy/time trade-off)
```

### 6. Model Persistence & Versioning

**Tự động versioning:**
```bash
rasa train
# Output: models/20231114-123456.tar.gz

rasa train --fixed-model-name "production-v1"
# Output: models/production-v1.tar.gz
```

**Load specific model:**
```bash
rasa shell --model models/20231114-123456.tar.gz
```

**Model structure:**
```
20231114-123456.tar.gz
├── nlu/
│   ├── DIETClassifier.pkl
│   └── ...
├── core/
│   ├── TEDPolicy.pkl
│   └── ...
├── fingerprint.json
└── metadata.json
```

### 7. Dry Run (Test Training)

**Mục đích:** Kiểm tra training process mà không save model

```bash
rasa train --dry-run
```

**Dùng khi:**
- ✅ Test config mới
- ✅ Estimate training time
- ✅ Debug training errors
- ✅ CI/CD validation

---

## Ví Dụ Thực Tế

### Kịch Bản 1: Chatbot Đặt Vé Máy Bay

#### Bước 1: Setup Data

**NLU Data:**
```yaml
# data/nlu.yml
nlu:
- intent: greet
  examples: |
    - xin chào
    - hi
    - hello

- intent: book_flight
  examples: |
    - đặt vé từ [Hà Nội](departure) đến [Sài Gòn](destination)
    - bay đi [Đà Nẵng](destination)
    - book flight to [Tokyo](destination) from [Hanoi](departure)

- intent: provide_date
  examples: |
    - [ngày mai](date)
    - [15/12/2023](date)
    - [next monday](date)

- intent: confirm
  examples: |
    - yes
    - đồng ý
    - ok
```

**Stories:**
```yaml
# data/stories.yml
stories:
- story: successful booking
  steps:
  - intent: greet
  - action: utter_greet
  - intent: book_flight
    entities:
    - departure: "Hà Nội"
    - destination: "Sài Gòn"
  - action: utter_ask_date
  - intent: provide_date
    entities:
    - date: "15/12"
  - action: action_search_flights
  - action: utter_show_results
  - intent: confirm
  - action: action_book_ticket
  - action: utter_success
```

**Domain:**
```yaml
# domain.yml
intents:
  - greet
  - book_flight
  - provide_date
  - confirm

entities:
  - departure
  - destination
  - date

slots:
  departure:
    type: text
    mappings:
    - type: from_entity
      entity: departure
  destination:
    type: text
    mappings:
    - type: from_entity
      entity: destination

actions:
  - action_search_flights
  - action_book_ticket

responses:
  utter_greet:
    - text: "Xin chào! Tôi có thể giúp bạn đặt vé máy bay."
  utter_ask_date:
    - text: "Bạn muốn bay vào ngày nào?"
```

#### Bước 2: Train

```bash
# Lần đầu - Full training
rasa train

# Output:
# Training NLU model...
# Epochs: 100/200
# Training Core model...
# Model saved: models/20231114-120000.tar.gz
```

#### Bước 3: Test

```bash
rasa shell
```

```
User: xin chào
Bot: Xin chào! Tôi có thể giúp bạn đặt vé máy bay.

User: đặt vé từ Hà Nội đi Sài Gòn
Bot: Bạn muốn bay vào ngày nào?

User: ngày mai
Bot: [Searching flights...]
Bot: Đây là các chuyến bay khả dụng: ...
```

#### Bước 4: Thêm Data Mới

```yaml
# Thêm 10 ví dụ mới vào data/nlu.yml
- intent: book_flight
  examples: |
    - book vé bay
    - tôi cần đặt chuyến bay
    - muốn bay đi công tác
```

```bash
# Fine-tune model (nhanh hơn)
rasa train --finetune --epoch-fraction 0.3

# Chỉ mất 3-5 phút thay vì 10-15 phút!
```

---

### Kịch Bản 2: Cập Nhật Model Trong Production

**Tình huống:** Chatbot đang chạy production, cần thêm tính năng mới

#### Week 1: Production Model
```bash
# Train và deploy
rasa train --fixed-model-name production-v1.0
# Deploy: models/production-v1.0.tar.gz
```

#### Week 2: Add New Intent
```yaml
# Thêm intent mới
- intent: cancel_booking
  examples: |
    - hủy vé
    - cancel my flight
```

```bash
# Train NLU only (nhanh)
rasa train nlu --fixed-model-name production-v1.1-nlu

# Test riêng NLU
rasa shell nlu --model models/production-v1.1-nlu.tar.gz
```

#### Week 3: Add Stories
```yaml
# Thêm story mới
- story: cancellation flow
  steps:
  - intent: cancel_booking
  - action: action_cancel_booking
  - action: utter_cancellation_success
```

```bash
# Train Core only
rasa train core --fixed-model-name production-v1.1-core

# Test
rasa interactive
```

#### Week 4: Full Integration
```bash
# Train full model
rasa train --fixed-model-name production-v1.1

# A/B testing
# 50% traffic → production-v1.0
# 50% traffic → production-v1.1

# Monitor metrics, sau đó rollout 100%
```

---

## Best Practices

### 1. Training Data Quality

**Nguyên tắc vàng:**

**✅ DO:**
- Ít nhất 10-15 examples/intent
- Đa dạng cách diễn đạt
- Bao gồm typos phổ biến
- Real user messages (từ logs)

```yaml
# GOOD
- intent: greet
  examples: |
    - xin chào
    - chào bạn
    - hello
    - hi
    - chào buổi sáng
    - chao ban  # typo
    - alo
```

**❌ DON'T:**
```yaml
# BAD - Quá ít examples
- intent: greet
  examples: |
    - xin chào
    - hello

# BAD - Quá giống nhau
- intent: book_flight
  examples: |
    - đặt vé máy bay
    - đặt vé máy bay đi Hà Nội
    - đặt vé máy bay đi Sài Gòn
```

### 2. Training Frequency

**Khuyến nghị:**

| Giai Đoạn | Frequency | Lý Do |
|-----------|-----------|-------|
| **Development** | Mỗi khi code thay đổi | Test nhanh |
| **Staging** | 2-3 lần/tuần | Collect feedback |
| **Production** | 1-2 lần/tháng | Stability |

**Continuous Training Pipeline:**
```bash
# Nightly training job
0 2 * * * cd /app && rasa train --out models/nightly/
```

### 3. Version Control

**Luôn version control:**
```
git/
├── data/
│   ├── nlu.yml
│   ├── stories.yml
│   └── rules.yml
├── config.yml
├── domain.yml
└── models/  # Git ignore
    └── .gitignore
```

**Tag releases:**
```bash
git tag -a v1.0 -m "Production release 1.0"
git push origin v1.0
```

### 4. Monitor Training Metrics

**Track:**
- Training time (phát hiện regression)
- Model size (optimize deployment)
- Validation warnings (fix errors)

```bash
# Log training metrics
rasa train 2>&1 | tee logs/training-$(date +%Y%m%d).log
```

### 5. Incremental Training Strategy

**Khi nào dùng incremental:**
- ✅ Thêm < 20% examples mới
- ✅ Không thay đổi architecture
- ✅ Model hiện tại đã tốt (> 80% accuracy)

**Khi nào train from scratch:**
- ⚠️ Thêm > 50% data mới
- ⚠️ Thay đổi pipeline/policies
- ⚠️ Major refactor

---

## Troubleshooting

### Lỗi 1: Training Quá Lâu

**Triệu chứng:**
```
Training Core model...
Epochs: 50/200 (30 minutes passed...)
```

**Nguyên nhân:**
- Quá nhiều epochs
- Model quá phức tạp
- Dataset quá lớn

**Giải pháp:**
```yaml
# Giảm epochs
policies:
  - name: TEDPolicy
    epochs: 100  # Thay vì 200

# Hoặc dùng smaller model
  - name: TEDPolicy
    epochs: 200
    transformer_size: 128  # Thay vì 256
    number_of_transformer_layers: 1  # Thay vì 2
```

### Lỗi 2: Out of Memory

**Triệu chứng:**
```
Killed
(Process terminated)
```

**Giải pháp:**
```yaml
# Giảm batch size
policies:
  - name: TEDPolicy
    batch_size: [32, 64]  # Thay vì [64, 256]

pipeline:
  - name: DIETClassifier
    batch_size: [32, 64]
```

**Hoặc:**
```bash
# Train trên máy mạnh hơn
# Sử dụng cloud GPU
```

### Lỗi 3: Model Không Cải Thiện

**Triệu chứng:**
```
Epoch 100: loss=0.5, accuracy=75%
Epoch 150: loss=0.49, accuracy=76%
Epoch 200: loss=0.48, accuracy=76%
# Stuck!
```

**Nguyên nhân:**
- Training data kém chất lượng
- Config không phù hợp
- Overfitting

**Giải pháp:**

1. **Thêm data chất lượng:**
```yaml
# Phân tích confusion matrix
rasa test nlu --report confusion/

# Thêm examples cho intents bị confuse
```

2. **Tuning hyperparameters:**
```yaml
pipeline:
  - name: DIETClassifier
    epochs: 300  # Tăng
    learning_rate: 0.001  # Điều chỉnh
```

3. **Regularization:**
```yaml
pipeline:
  - name: DIETClassifier
    drop_rate: 0.2  # Thêm dropout
```

### Lỗi 4: Story Conflicts

**Triệu chứng:**
```
UserWarning: Story structure conflict found in stories:
  'story_1' and 'story_2'
```

**Giải pháp:**

```yaml
# BEFORE - Conflict
- story: ask price path 1
  steps:
  - intent: ask_price
  - action: utter_price_range  # Action A

- story: ask price path 2
  steps:
  - intent: ask_price
  - action: action_calculate_price  # Action B (conflict!)

# AFTER - Add context với slots
- story: ask price with destination
  steps:
  - intent: ask_price
  - slot_was_set:
    - destination: "Hanoi"
  - action: action_calculate_price  # Khi có destination

- story: ask price without destination
  steps:
  - intent: ask_price
  - action: utter_price_range  # Khi chưa có destination
```

---

## Tóm Tắt

### Module Training Trong 1 Phút

**Training là gì?**
- Quá trình dạy chatbot hiểu và phản hồi người dùng

**Input:**
- NLU data (câu mẫu)
- Stories (kịch bản)
- Rules (quy tắc)
- Config (cấu hình)

**Process:**
1. Load data
2. Validate
3. Train NLU (hiểu ngôn ngữ)
4. Train Core (quản lý hội thoại)
5. Package model

**Output:**
- File .tar.gz chứa trained model

**4 Loại Training:**
1. **Full** - Train toàn bộ
2. **NLU-only** - Chỉ train language understanding
3. **Core-only** - Chỉ train dialogue management
4. **Incremental** - Fine-tune model có sẵn (nhanh nhất)

**Tính Năng Nổi Bật:**
- ⚡ Caching - Skip phần không đổi
- 🔄 Augmentation - Tự tạo variations
- 🔍 Validation - Phát hiện lỗi
- 📊 Interactive - Train bằng chat
- 🎯 Comparison - So sánh configs

**Best Practices:**
- ✅ 10-15 examples/intent minimum
- ✅ Version control everything
- ✅ Incremental training khi có thể
- ✅ Monitor metrics
- ✅ Test before deploy

---

## Kết Luận

Module Training là **nền tảng** của Rasa. Hiểu rõ cách training hoạt động giúp bạn:

✅ Xây dựng chatbot chất lượng cao
✅ Optimize training time & resources
✅ Debug problems nhanh hơn
✅ Scale chatbot hiệu quả

**Next Steps:**
1. Đọc thêm: [Model Configuration](docs/docs/model-configuration.mdx)
2. Practice: Train chatbot mẫu
3. Advanced: Custom components & policies

**Happy Training!** 🚀🤖

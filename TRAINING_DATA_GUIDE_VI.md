# Hướng Dẫn Chi Tiết: Tạo Training Data Cho Rasa Chatbot

## 📚 Mục Lục

1. [Tổng Quan Training Data](#tổng-quan)
2. [NLU Training Data](#nlu-training-data)
3. [Stories Training Data](#stories-training-data)
4. [Rules Training Data](#rules-training-data)
5. [Best Practices](#best-practices)
6. [Tools & Tips](#tools--tips)
7. [Ví Dụ Thực Tế](#ví-dụ-thực-tế)
8. [Kiểm Tra & Testing](#kiểm-tra--testing)

---

## 📖 Tổng Quan

### Training Data Trong Rasa Gồm 3 Loại:

```
data/
├── nlu.yml        # Dạy bot hiểu ngôn ngữ (Intent & Entity)
├── stories.yml    # Dạy bot hội thoại (Conversation flows)
└── rules.yml      # Dạy bot quy tắc cứng (Fixed behaviors)
```

### Khi Nào Dùng Loại Nào?

| Loại | Khi Nào Dùng | Ví Dụ |
|------|--------------|-------|
| **NLU** | Dạy bot hiểu câu nói | "tìm laptop" → intent: search_product |
| **Stories** | Hội thoại nhiều bước, có ngữ cảnh | Đặt vé máy bay (hỏi điểm đi, điểm đến, ngày...) |
| **Rules** | Hành vi cố định, không phụ thuộc ngữ cảnh | "chào" → luôn chào lại |

---

## 🎯 NLU Training Data

### Cấu Trúc File NLU

```yaml
version: "3.1"

nlu:
  - intent: tên_intent
    examples: |
      - câu ví dụ 1
      - câu ví dụ 2
      - câu ví dụ 3 với [entity](entity_name)
```

### 1. Viết Intent Cơ Bản

**Ví dụ: Intent chào hỏi**

```yaml
nlu:
  - intent: greet
    examples: |
      - xin chào
      - chào bạn
      - hello
      - hi
      - chào buổi sáng
      - chào buổi chiều
      - hey
      - hế lô
```

**📌 Quy tắc vàng:**
- ✅ Mỗi intent cần **tối thiểu 10-15 examples**
- ✅ Examples phải **đa dạng** (ngắn, dài, có typo, slang...)
- ✅ Viết theo cách người dùng thật nói
- ❌ KHÔNG copy-paste giống nhau

### 2. Viết Intent Với Entities

**Ví dụ: Tìm sản phẩm**

```yaml
nlu:
  - intent: search_product
    examples: |
      - tìm [laptop](product_type)
      - tôi muốn mua [máy tính](product_type)
      - cho tôi xem [laptop](product_type) [gaming](category)
      - tìm [laptop](product_type) [ASUS](brand)
      - [laptop](product_type) giá [20 triệu](price_range)
      - tìm [laptop](product_type) cho [sinh viên](use_case)
      - cần [máy chủ](product_type) [Dell](brand) dưới [100 triệu](price_range)
      - [camera](product_type) [4MP](specification)
      - tìm [switch](product_type) [48 port](specification)
      - [phần mềm](product_type) [Windows](product_name)
```

**Cú pháp Entity Annotation:**
```
[giá trị entity](tên_entity)
```

**📌 Các loại Entity phổ biến:**
- `product_type`: laptop, máy chủ, camera...
- `brand`: Dell, HP, ASUS...
- `price_range`: dưới 20 triệu, từ 10-15 triệu...
- `specification`: 16GB RAM, 48 port, 4MP...
- `use_case`: gaming, văn phòng, doanh nghiệp...

### 3. Viết Intent Phức Tạp

**Ví dụ: So sánh sản phẩm**

```yaml
nlu:
  - intent: compare_products
    examples: |
      - so sánh [Dell R740](product_name) và [HP DL380](product_name)
      - so sánh [laptop](product_type) [Dell](brand) với [HP](brand)
      - khác nhau gì giữa [ASUS ROG](product_name) và [MSI Katana](product_name)
      - [Dell](brand) tốt hơn [HP](brand) không
      - sự khác biệt của [switch Cisco](product_type) và [HPE](brand)
      - so sánh giá [máy chủ](product_type)
      - [camera Hikvision](product_name) so với [Dahua](brand)
      - nên chọn [laptop](product_type) nào
      - sản phẩm nào tốt hơn
      - compare [server](product_type) [Dell](brand) vs [Lenovo](brand)
```

### 4. Synonyms (Từ Đồng Nghĩa)

Dùng khi nhiều cách nói khác nhau nhưng cùng ý nghĩa:

```yaml
nlu:
  - synonym: laptop
    examples: |
      - máy tính xách tay
      - laptop
      - máy tính
      - pc xách tay
      - notebook

  - synonym: máy chủ
    examples: |
      - server
      - máy chủ
      - may chu
      - máy chủ server

  - synonym: gaming
    examples: |
      - chơi game
      - gaming
      - game
      - chơi games
```

### 5. Regex Features

Dùng cho patterns cố định (số điện thoại, email, mã sản phẩm...):

```yaml
nlu:
  - regex: phone_number
    examples: |
      - \d{10}
      - \d{3}[-.\s]\d{3}[-.\s]\d{4}

  - regex: product_code
    examples: |
      - [A-Z]{2}\d{3,5}
      - LP\d{3}
      - SV\d{3}

  - regex: price_pattern
    examples: |
      - \d+\s?(triệu|tr|trieu)
      - \d+\s?(ngàn|k|ngan)
```

### 6. Lookup Tables

Dùng cho danh sách giá trị cố định (thành phố, sản phẩm...):

```yaml
nlu:
  - lookup: brands
    examples: |
      - Dell
      - HP
      - Lenovo
      - ASUS
      - MSI
      - Cisco
      - HPE
      - Hikvision
      - Dahua

  - lookup: product_types
    examples: |
      - laptop
      - máy chủ
      - server
      - switch
      - camera
      - phần mềm
```

---

## 📖 Stories Training Data

Stories dạy bot cách hội thoại nhiều bước.

### Cấu Trúc Story

```yaml
version: "3.1"

stories:
  - story: tên story (mô tả ngắn gọn)
    steps:
      - intent: ý định của user
      - action: hành động của bot
      - intent: user nói tiếp
      - action: bot phản hồi
```

### 1. Story Cơ Bản

**Ví dụ: Chào hỏi đơn giản**

```yaml
stories:
  - story: greet and goodbye
    steps:
      - intent: greet
      - action: utter_greet
      - intent: goodbye
      - action: utter_goodbye
```

**Diễn giải:**
1. User: "xin chào" (intent: greet)
2. Bot: "Xin chào! Tôi có thể giúp gì..." (action: utter_greet)
3. User: "tạm biệt" (intent: goodbye)
4. Bot: "Hẹn gặp lại!" (action: utter_goodbye)

### 2. Story Với Entities & Slots

**Ví dụ: Tìm sản phẩm**

```yaml
stories:
  - story: search laptop by price
    steps:
      - intent: search_product
        entities:
          - product_type: "laptop"
          - price_range: "25 triệu"
      - slot_was_set:
          - product_type: "laptop"
          - price_range: "25 triệu"
      - action: action_search_product
      - intent: affirm
      - action: action_show_details
```

**Diễn giải:**
1. User: "tìm laptop dưới 25 triệu"
2. Bot lưu: product_type="laptop", price_range="25 triệu"
3. Bot chạy: action_search_product (tìm kiếm)
4. User: "đúng rồi" (xác nhận)
5. Bot chạy: action_show_details (hiển thị chi tiết)

### 3. Story Với Nhiều Nhánh

**Ví dụ: Tư vấn sản phẩm**

```yaml
stories:
  - story: recommend laptop - gaming
    steps:
      - intent: recommend_product
        entities:
          - product_type: "laptop"
          - use_case: "gaming"
      - action: action_recommend_product
      - intent: ask_price
      - action: action_show_price
      - intent: affirm
      - action: action_add_to_cart

  - story: recommend laptop - office
    steps:
      - intent: recommend_product
        entities:
          - product_type: "laptop"
          - use_case: "văn phòng"
      - action: action_recommend_product
      - intent: ask_specifications
      - action: action_show_specifications
      - intent: compare_products
      - action: action_compare_products
```

### 4. Story Với Form (Thu Thập Thông Tin)

**Ví dụ: Đặt hàng**

```yaml
stories:
  - story: complete order form
    steps:
      - intent: place_order
      - action: order_form
      - active_loop: order_form
      - slot_was_set:
          - requested_slot: product_name
      - slot_was_set:
          - requested_slot: quantity
      - slot_was_set:
          - requested_slot: phone_number
      - slot_was_set:
          - requested_slot: null
      - active_loop: null
      - action: action_submit_order
```

### 5. Checkpoints (Tái Sử Dụng Story)

```yaml
stories:
  - story: greet user
    steps:
      - intent: greet
      - action: utter_greet
      - checkpoint: after_greet

  - story: greet then search
    steps:
      - checkpoint: after_greet
      - intent: search_product
      - action: action_search_product

  - story: greet then ask price
    steps:
      - checkpoint: after_greet
      - intent: ask_price
      - action: action_show_price
```

---

## ⚖️ Rules Training Data

Rules là quy tắc cứng, luôn thực thi giống nhau.

### Cấu Trúc Rule

```yaml
version: "3.1"

rules:
  - rule: tên rule
    steps:
      - intent: intent_name
      - action: action_name
```

### 1. Rule Cơ Bản

**Ví dụ: Chào hỏi**

```yaml
rules:
  - rule: Say goodbye anytime user says goodbye
    steps:
      - intent: goodbye
      - action: utter_goodbye

  - rule: Say thanks
    steps:
      - intent: thank
      - action: utter_thank
```

**📌 Đặc điểm:**
- Luôn chạy khi gặp intent
- Không phụ thuộc ngữ cảnh

### 2. Rule Với Conditions

**Ví dụ: Chỉ chào khi chưa chào**

```yaml
rules:
  - rule: Greet user only once
    condition:
      - slot_was_set:
          - user_greeted: false
    steps:
      - intent: greet
      - action: utter_greet
      - action: action_set_greeted_flag
```

### 3. Rule Kích Hoạt Form

```yaml
rules:
  - rule: Activate order form
    steps:
      - intent: place_order
      - action: order_form
      - active_loop: order_form

  - rule: Submit order form
    condition:
      - active_loop: order_form
    steps:
      - action: order_form
      - active_loop: null
      - slot_was_set:
          - requested_slot: null
      - action: action_submit_order
```

### 4. Rule Cho Fallback

```yaml
rules:
  - rule: Ask user to rephrase when confidence is low
    steps:
      - intent: nlu_fallback
      - action: utter_ask_rephrase

  - rule: Handle out of scope
    steps:
      - intent: out_of_scope
      - action: utter_out_of_scope
```

---

## ✅ Best Practices

### 1. Quy Tắc Viết NLU Data

#### ✅ ĐÚNG:
```yaml
nlu:
  - intent: search_product
    examples: |
      - tìm laptop
      - laptop gaming
      - cho tôi xem máy tính
      - tôi cần mua laptop ASUS
      - có laptop nào dưới 20 triệu
      - tìm máy tính cho sinh viên
      - laptop văn phòng giá rẻ
      - cho xem laptop chơi game
      - cần laptop thiết kế đồ họa
      - tìm laptop Dell có sẵn
```

**📌 Lý do tốt:**
- Đa dạng cách diễn đạt
- Có ngắn, có dài
- Có entity, không entity
- Tự nhiên như người nói

#### ❌ SAI:
```yaml
nlu:
  - intent: search_product
    examples: |
      - tìm laptop
      - tìm laptop gaming
      - tìm laptop ASUS
      - tìm laptop Dell
      - tìm laptop HP
```

**📌 Lý do sai:**
- Quá giống nhau (chỉ thay brand)
- Không đa dạng cấu trúc
- Bot sẽ overfit (chỉ học pattern "tìm laptop X")

### 2. Số Lượng Examples Cần Thiết

| Intent Type | Số Examples Tối Thiểu | Khuyến Nghị |
|-------------|------------------------|-------------|
| Intent đơn giản (greet, goodbye) | 10-15 | 20-30 |
| Intent trung bình (search, ask) | 20-30 | 50-100 |
| Intent phức tạp (với entities) | 30-50 | 100-200 |
| Intent quan trọng (core business) | 50-100 | 200-500 |

### 3. Viết Examples Đa Dạng

**Chiến lược "DIVERSE" (Đa Dạng Hóa):**

```yaml
nlu:
  - intent: ask_price
    examples: |
      # Ngắn
      - giá
      - bao nhiêu
      - giá bao nhiêu

      # Trung bình
      - laptop này giá bao nhiêu
      - giá Dell R740 là bao nhiêu
      - hỏi giá máy chủ

      # Dài
      - cho tôi hỏi laptop ASUS ROG này giá bao nhiêu
      - tôi muốn biết giá của máy chủ Dell PowerEdge R740

      # Slang/typo
      - giá bn
      - bao nhiu
      - gia bao nhieu

      # Formal/informal
      - Xin cho biết giá cả
      - Giá thành sản phẩm này là bao nhiêu ạ
      - giá mấy
      - giá bao nhiêu tiền

      # Với context
      - máy này giá bao nhiêu
      - cái đó giá bao nhiêu
      - sản phẩm vừa rồi giá thế nào
```

### 4. Xử Lý Intents Tương Tự

**Vấn đề:** 2 intents quá giống nhau → Bot nhầm lẫn

```yaml
# ❌ SAI: Quá tương tự
- intent: search_laptop
  examples: |
    - tìm laptop
    - laptop nào tốt

- intent: recommend_laptop
  examples: |
    - gợi ý laptop
    - laptop nào tốt
```

**✅ ĐÚNG: Tách biệt rõ ràng**

```yaml
- intent: search_product
  examples: |
    - tìm laptop gaming
    - có laptop Dell không
    - cho xem laptop ASUS ROG
    # Focus: Tìm sản phẩm CỤ THỂ

- intent: recommend_product
  examples: |
    - tư vấn laptop cho sinh viên
    - laptop nào phù hợp với tôi
    - gợi ý laptop trong tầm giá 20 triệu
    # Focus: Xin GỢI Ý dựa trên nhu cầu
```

### 5. Entities Best Practices

#### Cách Đặt Tên Entity

✅ **ĐÚNG:**
```
product_type, product_name, price_range, use_case
```

❌ **SAI:**
```
type, name, price, case  # Quá chung chung
ProductType, productName  # Không theo convention
```

#### Annotation Đầy Đủ

```yaml
nlu:
  - intent: search_product
    examples: |
      # ✅ Annotate TẤT CẢ entities quan trọng
      - tìm [laptop](product_type) [ASUS](brand) dưới [20 triệu](price_range)
      - [máy chủ](product_type) [Dell R740](product_name) cho [doanh nghiệp](use_case)

      # ❌ KHÔNG bỏ sót
      - tìm laptop ASUS dưới 20 triệu  # Thiếu annotation
```

### 6. Stories Best Practices

#### Viết Story Ngắn & Tập Trung

✅ **ĐÚNG:**
```yaml
stories:
  - story: search and show details
    steps:
      - intent: search_product
      - action: action_search_product
      - intent: ask_specifications
      - action: action_show_specifications
```

❌ **SAI:** Story quá dài (>10 steps)
```yaml
stories:
  - story: complete user journey  # Quá dài!
    steps:
      - intent: greet
      - action: utter_greet
      - intent: search_product
      - action: action_search_product
      - intent: ask_price
      - action: action_show_price
      - intent: compare_products
      - action: action_compare_products
      - intent: ask_specifications
      # ... 20 steps nữa
```

#### Sử Dụng OR Statement

```yaml
stories:
  - story: user can affirm or deny
    steps:
      - action: action_ask_confirmation
      - or:
        - intent: affirm
        - intent: deny
      - action: action_handle_response
```

### 7. Rules vs Stories

| Tình Huống | Dùng | Ví Dụ |
|------------|------|-------|
| Hành vi cố định, không đổi | **Rule** | Chào → Chào lại |
| Single-turn Q&A | **Rule** | Hỏi giờ → Trả lời giờ |
| Multi-turn conversation | **Story** | Đặt vé máy bay |
| Kích hoạt Form | **Rule** | Start form |
| Xử lý trong Form | **Story** | Form flow |

---

## 🛠️ Tools & Tips

### 1. Rasa Data Validator

Kiểm tra lỗi trong training data:

```bash
# Validate tất cả data
rasa data validate

# Validate stories
rasa data validate stories

# Check for conflicts
rasa data validate --max-history 5
```

**Common Errors:**
- Intent không có examples
- Story reference intent không tồn tại
- Entity không được define trong domain
- Conflicting stories

### 2. Rasa Interactive Learning

Tạo training data qua chat:

```bash
rasa interactive
```

**Workflow:**
1. Chat với bot
2. Sửa predictions nếu sai
3. Rasa tự động thêm vào stories
4. Export thành training data

### 3. Data Augmentation Tools

**a) Paraphrasing Tool (online):**
- QuillBot: https://quillbot.com
- Paraphrase Online

**b) Python Script:**
```python
# augment_data.py
import random

templates = [
    "tìm {product}",
    "cho tôi xem {product}",
    "có {product} nào không",
    "{product} giá bao nhiêu",
    "tôi cần {product}"
]

products = ["laptop", "máy chủ", "camera", "switch"]

for template in templates:
    for product in products:
        print(f"      - {template.format(product=product)}")
```

### 4. Markdown to YAML Converter

```python
# Convert từ format cũ sang mới
rasa data convert nlu --data data/ --out converted/ --format yaml
```

### 5. Annotate Tool - Rasa NLU Trainer

Web UI để annotate entities:

```bash
pip install rasa-nlu-trainer
rasa-nlu-trainer
```

### 6. Bulk Testing Intent

Tạo file test:

```yaml
# tests/test_nlu.yml
nlu:
  - intent: search_product
    examples: |
      - tìm laptop Dell
      - máy chủ HP
```

Run test:
```bash
rasa test nlu --nlu tests/test_nlu.yml
```

---

## 💡 Ví Dụ Thực Tế

### Use Case 1: Chatbot Đặt Đồ Ăn

**File: data/nlu.yml**

```yaml
version: "3.1"

nlu:
  # Intents
  - intent: order_food
    examples: |
      - tôi muốn đặt món
      - đặt đồ ăn
      - order
      - gọi món
      - cho tôi đặt [pizza](dish)
      - tôi muốn [phở](dish)
      - đặt [2](quantity) [bánh mì](dish)
      - [3](quantity) [cơm gà](dish) nhé

  - intent: ask_menu
    examples: |
      - menu có gì
      - danh sách món ăn
      - có món gì
      - xem thực đơn
      - món nào ngon

  - intent: ask_price
    examples: |
      - [pizza](dish) giá bao nhiêu
      - giá [phở](dish)
      - bao nhiêu tiền
      - [cơm gà](dish) bao nhiêu

  # Entities
  - synonym: pizza
    examples: |
      - pizza
      - piza
      - bánh pizza

  - lookup: dishes
    examples: |
      - pizza
      - phở
      - bánh mì
      - cơm gà
      - bún bò
```

**File: data/stories.yml**

```yaml
version: "3.1"

stories:
  - story: order food flow
    steps:
      - intent: order_food
        entities:
          - dish: "pizza"
          - quantity: "2"
      - action: action_add_to_cart
      - action: utter_ask_anything_else
      - intent: deny
      - action: utter_ask_delivery_info
      - action: delivery_form
      - active_loop: delivery_form
      - active_loop: null
      - action: action_confirm_order

  - story: check menu then order
    steps:
      - intent: ask_menu
      - action: action_show_menu
      - intent: order_food
      - action: action_add_to_cart
```

### Use Case 2: Chatbot Hỗ Trợ Kỹ Thuật

**File: data/nlu.yml**

```yaml
version: "3.1"

nlu:
  - intent: report_issue
    examples: |
      - [laptop](device) tôi bị lỗi
      - [máy tính](device) không khởi động được
      - [máy in](device) bị [kẹt giấy](issue)
      - [wifi](device) [không kết nối được](issue)
      - [màn hình](device) bị [nhấp nháy](issue)
      - [chuột](device) không hoạt động
      - [bàn phím](device) bị [liệt phím](issue)

  - intent: ask_solution
    examples: |
      - làm thế nào để sửa
      - cách khắc phục
      - hướng dẫn tôi
      - phải làm gì
      - giải quyết như thế nào

  - regex: ticket_number
    examples: |
      - TICKET-\d{6}
      - TK\d{4,6}
```

**File: data/rules.yml**

```yaml
version: "3.1"

rules:
  - rule: Create ticket for new issue
    steps:
      - intent: report_issue
      - action: action_create_ticket
      - action: utter_ticket_created

  - rule: Provide solution if known issue
    condition:
      - slot_was_set:
          - known_issue: true
    steps:
      - intent: ask_solution
      - action: action_provide_solution
```

---

## 🧪 Kiểm Tra & Testing

### 1. Test NLU Model

```bash
# Test với file test riêng
rasa test nlu --nlu tests/test_nlu.yml

# Cross-validation
rasa test nlu --nlu data/nlu.yml --cross-validation

# Test và xem confusion matrix
rasa test nlu --nlu data/nlu.yml --out results/
```

### 2. Test Stories

```bash
# Test stories
rasa test core --stories tests/test_stories.yml

# Test end-to-end
rasa test --stories tests/test_stories.yml
```

### 3. Interactive Testing

```bash
# Chat và kiểm tra predictions
rasa shell

# Debug mode
rasa shell --debug

# NLU only
rasa shell nlu
```

### 4. Metrics Quan Trọng

**NLU Metrics:**
- **Intent Accuracy**: >85% (tốt), >90% (xuất sắc)
- **Entity F1 Score**: >80% (tốt), >90% (xuất sắc)
- **Confidence**: >0.7 (chấp nhận được)

**Core Metrics:**
- **Action Accuracy**: >90%
- **Story Accuracy**: >85%

### 5. Phân Tích Lỗi

```bash
# Xem errors
rasa test nlu --nlu data/nlu.yml --out results/

# Mở file results/intent_errors.json
# Check intent nào bị nhầm nhiều nhất
```

**Common Issues:**
1. **Intent Confusion**: 2 intents quá giống → Gộp hoặc tách rõ hơn
2. **Low Confidence**: Thiếu training data → Thêm examples
3. **Entity Miss**: Không recognize → Thêm examples có entity

---

## 📋 Checklist Trước Khi Deploy

### Training Data Quality Check

- [ ] Mỗi intent có ít nhất 20 examples
- [ ] Examples đa dạng (ngắn, dài, formal, slang)
- [ ] Tất cả entities đều được annotate
- [ ] Không có intent trùng lặp/tương tự
- [ ] Synonyms và lookup tables đầy đủ
- [ ] Stories cover các flow chính
- [ ] Rules được dùng đúng chỗ

### Testing Check

- [ ] `rasa data validate` pass
- [ ] Intent accuracy >85%
- [ ] Entity F1 >80%
- [ ] Test thủ công 20+ câu
- [ ] Cross-validation results tốt

### Documentation

- [ ] Comment trong code
- [ ] Ghi chú các entities quan trọng
- [ ] Document conversation flows
- [ ] Training data versioning (git)

---

## 🎯 Tổng Kết

### Công Thức Thành Công

```
Quality Training Data =
    Diverse Examples (40%) +
    Sufficient Quantity (30%) +
    Proper Annotation (20%) +
    Good Testing (10%)
```

### Next Steps

1. **Bắt đầu nhỏ**: 5-10 intents, 20 examples mỗi intent
2. **Test sớm, test thường xuyên**: Mỗi lần thêm data → test ngay
3. **Lặp lại**: Deploy → Thu thập real user data → Cải thiện
4. **Tự động hóa**: Script để generate examples, validate data

### Resources

- 📚 Rasa Docs: https://rasa.com/docs/rasa/training-data-format
- 🎓 Rasa Masterclass: https://learning.rasa.com
- 💬 Rasa Forum: https://forum.rasa.com
- 🛠️ Rasa X: Tool để improve training data

---

**Chúc bạn tạo training data hiệu quả!** 🚀

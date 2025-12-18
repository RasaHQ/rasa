# Training Data Quick Reference

## 📋 Tóm Tắt Nhanh

### Cấu Trúc File

```yaml
version: "3.1"

nlu:  # Dạy bot hiểu ngôn ngữ
  - intent: tên_intent
    examples: |
      - câu ví dụ 1
      - câu với [entity](entity_type)

stories:  # Dạy bot hội thoại
  - story: tên story
    steps:
      - intent: user_intent
      - action: bot_action

rules:  # Quy tắc cố định
  - rule: tên rule
    steps:
      - intent: user_intent
      - action: bot_action
```

---

## 🎯 NLU Syntax

### Basic Intent
```yaml
- intent: greet
  examples: |
    - xin chào
    - hello
    - hi
```

### Intent with Entity
```yaml
- intent: search_product
  examples: |
    - tìm [laptop](product_type)
    - [laptop](product_type) [ASUS](brand) dưới [20 triệu](price)
```

### Synonym
```yaml
- synonym: laptop
  examples: |
    - laptop
    - máy tính
    - notebook
```

### Lookup Table
```yaml
- lookup: brands
  examples: |
    - Dell
    - HP
    - ASUS
```

### Regex
```yaml
- regex: phone_number
  examples: |
    - \d{10}
    - \d{3}[-.\s]\d{3}[-.\s]\d{4}
```

---

## 📖 Stories Syntax

### Basic Story
```yaml
- story: simple flow
  steps:
    - intent: greet
    - action: utter_greet
    - intent: goodbye
    - action: utter_goodbye
```

### Story with Entities
```yaml
- story: search flow
  steps:
    - intent: search_product
      entities:
        - product_type: "laptop"
    - slot_was_set:
        - product_type: "laptop"
    - action: action_search_product
```

### Story with OR
```yaml
- story: user confirms or denies
  steps:
    - action: utter_ask_confirm
    - or:
      - intent: affirm
      - intent: deny
    - action: action_handle_response
```

### Story with Checkpoint
```yaml
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
```

---

## ⚖️ Rules Syntax

### Simple Rule
```yaml
- rule: say goodbye
  steps:
    - intent: goodbye
    - action: utter_goodbye
```

### Rule with Condition
```yaml
- rule: greet only once
  condition:
    - slot_was_set:
        - user_greeted: false
  steps:
    - intent: greet
    - action: utter_greet
```

### Form Activation
```yaml
- rule: activate form
  steps:
    - intent: book_flight
    - action: booking_form
    - active_loop: booking_form
```

---

## 💡 Best Practices Cheat Sheet

### ✅ DO

```yaml
# ✅ Đa dạng, tự nhiên
- intent: search_product
  examples: |
    - tìm laptop
    - cho tôi xem máy tính
    - có laptop nào không
    - laptop gaming dưới 20 triệu
    - tôi cần mua laptop ASUS
```

### ❌ DON'T

```yaml
# ❌ Giống nhau quá
- intent: search_product
  examples: |
    - tìm laptop Dell
    - tìm laptop HP
    - tìm laptop ASUS
    - tìm laptop MSI
```

---

## 📊 Số Lượng Examples

| Intent Type | Min | Khuyến Nghị |
|-------------|-----|-------------|
| Đơn giản | 10-15 | 20-30 |
| Trung bình | 20-30 | 50-100 |
| Phức tạp | 30-50 | 100-200 |

---

## 🔧 Commands Thường Dùng

```bash
# Validate data
rasa data validate

# Train
rasa train

# Test NLU
rasa test nlu --nlu data/nlu.yml

# Interactive learning
rasa interactive

# Shell
rasa shell
rasa shell --debug
rasa shell nlu
```

---

## 🎨 Entity Annotation Format

```
[giá trị](tên_entity)

Examples:
- tìm [laptop](product_type)
- [Dell R740](product_name) giá bao nhiêu
- dưới [20 triệu](price_range)
```

---

## 🚀 Quick Start Template

```yaml
version: "3.1"

nlu:
  - intent: greet
    examples: |
      - xin chào
      - hello

  - intent: search_product
    examples: |
      - tìm [laptop](product_type)
      - [laptop](product_type) [Dell](brand)

stories:
  - story: basic flow
    steps:
      - intent: greet
      - action: utter_greet
      - intent: search_product
      - action: action_search_product

rules:
  - rule: say goodbye
    steps:
      - intent: goodbye
      - action: utter_goodbye
```

---

## 🔍 Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Intent not found | Intent trong story không có trong NLU | Thêm intent vào nlu.yml |
| Entity not recognized | Entity không được define | Thêm vào domain.yml |
| Low accuracy | Thiếu training data | Thêm examples |
| Overfitting | Examples quá giống nhau | Đa dạng hóa |

---

## 📝 Template Checklist

- [ ] Mỗi intent ≥ 20 examples
- [ ] Examples đa dạng (ngắn, dài, typo...)
- [ ] Entities được annotate đầy đủ
- [ ] Stories cover main flows
- [ ] Rules dùng đúng chỗ
- [ ] `rasa data validate` pass
- [ ] Test accuracy > 85%

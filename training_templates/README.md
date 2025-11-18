# Training Data Templates

Templates sẵn sàng để bắt đầu project Rasa chatbot nhanh chóng.

## 📁 Files

### 1. ecommerce_nlu_template.yml

Template NLU data cho chatbot e-commerce/bán hàng.

**Bao gồm 20+ intents:**
- Greetings (greet, goodbye, thank...)
- Product search (search_product, browse_category)
- Product info (ask_price, ask_specs, ask_stock, ask_warranty...)
- Comparison (compare_products, recommend_product)
- Shopping cart (add_to_cart, view_cart, checkout...)
- Order tracking (track_order, cancel_order...)
- Support (technical_support, contact_agent...)

**Entities:**
- product_type, product_name, brand
- price_range, quantity
- use_case, specification
- order_id

**Total:** 200+ training examples

### 2. ecommerce_stories_template.yml

Template stories cho e-commerce flows.

**Bao gồm 30+ stories:**
- Basic flows (greet → search → goodbye)
- Product discovery (search → ask specs → compare)
- Purchase flows (search → add to cart → checkout)
- Information queries (price, stock, warranty...)
- Order tracking
- Error handling

### 3. QUICK_REFERENCE.md

Cheat sheet tra cứu nhanh syntax Rasa.

**Nội dung:**
- NLU syntax (intent, entity, synonym, lookup, regex)
- Stories syntax (basic, with entities, OR, checkpoint)
- Rules syntax
- Best practices
- Common commands
- Common errors

## 🚀 Cách Sử Dụng

### Option 1: Copy Template

```bash
# Copy NLU template
cp training_templates/ecommerce_nlu_template.yml data/nlu.yml

# Copy stories template
cp training_templates/ecommerce_stories_template.yml data/stories.yml

# Edit theo nhu cầu của bạn
```

### Option 2: Tham Khảo

```bash
# Xem template để học cách viết
cat training_templates/ecommerce_nlu_template.yml

# Copy từng phần cần thiết
```

### Option 3: Customize

```bash
# Tạo file mới dựa trên template
cp training_templates/ecommerce_nlu_template.yml my_custom_nlu.yml

# Sửa đổi:
# 1. Xóa intents không cần
# 2. Thêm intents mới
# 3. Thêm examples
# 4. Thêm entities
```

## 🎯 Use Cases

### E-Commerce / Bán Hàng

**File:** `ecommerce_nlu_template.yml`, `ecommerce_stories_template.yml`

**Phù hợp cho:**
- Shop online (điện tử, thời trang, thực phẩm...)
- Showroom (ô tô, điện thoại, máy tính...)
- B2B sales (thiết bị công nghiệp, phần mềm...)

**Tính năng:**
- Tìm sản phẩm theo filter (giá, brand, specs...)
- So sánh sản phẩm
- Giỏ hàng & checkout
- Tracking đơn hàng

### Customer Support

**Tạo từ template:**

```yaml
nlu:
  - intent: report_issue
    examples: |
      - sản phẩm bị lỗi
      - [laptop](device) không khởi động
      - [máy in](device) bị [kẹt giấy](issue)

  - intent: ask_solution
    examples: |
      - làm thế nào để sửa
      - cách khắc phục
```

### Booking / Reservation

**Tạo từ template:**

```yaml
nlu:
  - intent: book_appointment
    examples: |
      - đặt lịch hẹn
      - book appointment [ngày mai](date)
      - đặt chỗ lúc [3 giờ chiều](time)
```

## 📝 Customization Guide

### 1. Thêm Intent Mới

```yaml
# Thêm vào file nlu.yml
nlu:
  - intent: your_new_intent
    examples: |
      - example 1
      - example 2
      - example với [entity](entity_type)
```

### 2. Thêm Entity

```yaml
# Thêm lookup table
nlu:
  - lookup: your_entity
    examples: |
      - value 1
      - value 2
```

### 3. Thêm Story

```yaml
# Thêm vào stories.yml
stories:
  - story: your story name
    steps:
      - intent: user_intent
      - action: bot_action
```

## 🔧 Validation

Sau khi customize, validate data:

```bash
# Validate bằng tool
python tools/validate_training_data.py data/nlu.yml

# Validate bằng Rasa
rasa data validate
```

## 💡 Tips

### Khi Bắt Đầu Project Mới

1. **Copy template phù hợp**
2. **Xóa intents không cần**
3. **Customize entities cho domain của bạn**
4. **Thêm examples thật từ users**
5. **Train và test**

### Khi Mở Rộng Project

1. **Giữ nguyên template làm reference**
2. **Thêm intents mới vào file riêng**
3. **Merge khi đã test kỹ**

### Best Practices

- ✅ Mỗi intent ≥ 20 examples
- ✅ Examples đa dạng
- ✅ Test thường xuyên
- ✅ Version control (git)

## 📚 Resources

- **Hướng dẫn chi tiết:** `TRAINING_DATA_GUIDE_VI.md`
- **Quick reference:** `QUICK_REFERENCE.md`
- **Tools:** `tools/` folder
- **Examples:** `examples/it_store_bot/`

## 🎓 Learning Path

1. Đọc `QUICK_REFERENCE.md` (15 phút)
2. Xem `ecommerce_nlu_template.yml` (30 phút)
3. Đọc `TRAINING_DATA_GUIDE_VI.md` (2 giờ)
4. Thực hành với template (4 giờ)
5. Build chatbot riêng (∞)

---

**Happy training!** 🚀

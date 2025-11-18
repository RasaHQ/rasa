# Training Data Tools

Công cụ hỗ trợ tạo và validate training data cho Rasa.

## 🛠️ Công Cụ

### 1. generate_training_data.py

Tự động generate training data từ templates.

**Sử dụng:**
```bash
python tools/generate_training_data.py
```

**Tính năng:**
- ✅ Generate examples từ templates
- ✅ Tự động annotate entities
- ✅ Augment data (synonyms, typos)
- ✅ Validate data ngay sau khi generate

**Output:**
- `generated_nlu.yml` - File NLU data mới

### 2. validate_training_data.py

Kiểm tra chất lượng training data.

**Sử dụng:**
```bash
# Validate single file
python tools/validate_training_data.py data/nlu.yml

# Validate IT store bot
python tools/validate_training_data.py examples/it_store_bot/data/nlu.yml
```

**Kiểm tra:**
- ❌ **Errors** (phải fix):
  - Intent có quá ít examples (< 10)
  - Duplicate examples quá nhiều (> 10%)
  - 2 intents có examples giống nhau

- ⚠️ **Warnings** (nên fix):
  - Intent có ít examples (< 20)
  - Examples quá ngắn (< 2 từ)
  - Examples quá giống nhau
  - Entity được dùng quá ít

**Output:**
```
📊 VALIDATION REPORT
===================
📈 STATISTICS:
   • Total Intents: 15
   • Total Examples: 320
   • Average Examples per Intent: 21.3

❌ ERRORS (2):
   ❌ Intent 'search_product': Only 8 examples (minimum: 10)
   ❌ Intents 'search' and 'find' have 3 identical examples!

⚠️ WARNINGS (1):
   ⚠️ Intent 'ask_price': Only 12 examples (recommended: 20)
```

## 📝 Examples

### Generate Data

```python
from generate_training_data import TrainingDataGenerator

generator = TrainingDataGenerator()

# Generate 50 examples per intent
examples = generator.generate_intent_examples('search_product', 50)

# Generate complete NLU file
generator.generate_nlu_file('output.yml')
```

### Annotate Entities

```python
from generate_training_data import EntityAnnotator

annotator = EntityAnnotator()

# Annotate single text
text = "tìm laptop Dell"
annotated = annotator.annotate(text)
# Output: "tìm [laptop](product_type) [Dell](brand)"

# Annotate entire file
annotator.annotate_file('input.yml', 'output.yml')
```

### Validate Data

```python
from validate_training_data import TrainingDataValidator

validator = TrainingDataValidator()

# Validate file
report = validator.validate_file('data/nlu.yml')

# Print report
validator.print_report(report)
```

## 🎯 Workflow Đề Xuất

```bash
# 1. Generate training data
python tools/generate_training_data.py

# 2. Validate data
python tools/validate_training_data.py generated_nlu.yml

# 3. Fix errors nếu có

# 4. Validate bằng Rasa
rasa data validate

# 5. Train
rasa train

# 6. Test
rasa test nlu --nlu generated_nlu.yml
```

## 🔧 Requirements

```bash
pip install pyyaml
```

## 📚 Resources

- Xem `TRAINING_DATA_GUIDE_VI.md` để học chi tiết
- Xem `training_templates/` để có templates
- Xem `QUICK_REFERENCE.md` để tra cứu nhanh

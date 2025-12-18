# Hướng Dẫn Demo Web Frontend - TechStore AI Chatbot

## 📋 Tổng Quan

Web frontend chuyên nghiệp để demo chatbot AI tư vấn sản phẩm IT cho khách hàng. Hệ thống bán các sản phẩm công nghệ thông tin như:

- 💻 **Máy tính & Laptop** (Gaming, Văn phòng, Workstation)
- 🖥️ **Máy chủ** (Dell, HP, Lenovo)
- 🔌 **Switch mạng** (Cisco, HPE)
- 📹 **Camera an ninh** (Hikvision, Dahua)
- 💿 **Phần mềm thương mại** (Windows, Office, Antivirus, CAD)

## 🎯 Tính Năng Chatbot AI

### 1. **Tìm Kiếm Thông Minh**
- Tìm sản phẩm theo loại, thương hiệu, giá cả
- Tìm theo thông số kỹ thuật (CPU, RAM, số port, độ phân giải...)
- Tìm theo nhu cầu sử dụng (gaming, văn phòng, doanh nghiệp...)

**Ví dụ:**
```
- "Tìm laptop gaming dưới 30 triệu"
- "Máy chủ Dell cho doanh nghiệp"
- "Camera 4MP có AI"
- "Switch 48 port managed"
```

### 2. **So Sánh Sản Phẩm**
- So sánh chi tiết 2-3 sản phẩm
- Hiển thị khác biệt về cấu hình, giá cả
- Phân tích ưu nhược điểm

**Ví dụ:**
```
- "So sánh Dell PowerEdge R740 và HP ProLiant DL380"
- "ASUS ROG khác MSI Katana gì?"
- "So sánh giá laptop gaming"
```

### 3. **Tư Vấn Chuyên Nghiệp**
- Gợi ý sản phẩm phù hợp với mục đích sử dụng
- Tư vấn dựa trên ngân sách
- Gợi ý giải pháp trọn gói

**Ví dụ:**
```
- "Tư vấn laptop cho sinh viên IT"
- "Giải pháp camera cho văn phòng 500m2"
- "Máy chủ phù hợp cho startup 20 người"
```

### 4. **Thông Số Kỹ Thuật**
- Giải thích chi tiết cấu hình
- Tư vấn về khả năng tương thích
- Hướng dẫn chọn sản phẩm

**Ví dụ:**
```
- "Dell R740 có cấu hình gì?"
- "Switch managed khác unmanaged như thế nào?"
- "Camera AcuSense là gì?"
```

### 5. **Báo Giá & Khuyến Mãi**
- Kiểm tra giá sản phẩm
- Thông tin khuyến mãi hiện hành
- Báo giá số lượng lớn

**Ví dụ:**
```
- "Giá máy chủ HP ProLiant"
- "Có khuyến mãi laptop không?"
- "Mua 10 camera giảm bao nhiêu?"
```

### 6. **Hỗ Trợ Kỹ Thuật**
- Hướng dẫn cài đặt
- Khắc phục sự cố
- Chính sách bảo hành

**Ví dụ:**
```
- "Hướng dẫn cài Windows Server"
- "Cấu hình switch Cisco"
- "Bảo hành máy chủ như thế nào?"
```

## 🚀 Cách Chạy Demo

### Phương Án 1: Demo Không Cần Rasa Server (Standalone)

Frontend có sẵn chế độ demo với câu trả lời mẫu, không cần chạy Rasa server.

```bash
# Mở file frontend/index.html bằng trình duyệt
cd frontend
# Trên macOS
open index.html

# Trên Linux
xdg-open index.html

# Hoặc Windows
start index.html
```

**Lưu ý:** Ở chế độ này, chatbot sẽ trả lời bằng logic JavaScript có sẵn (không kết nối Rasa).

### Phương Án 2: Demo Đầy Đủ với Rasa Server

Để chatbot hoạt động với AI thật sự:

#### Bước 1: Training Chatbot

```bash
# Di chuyển vào thư mục IT store bot
cd examples/it_store_bot

# Training model
rasa train

# Kết quả: Model được lưu trong thư mục models/
```

#### Bước 2: Chạy Rasa Action Server

```bash
# Terminal 1: Chạy action server
cd examples/it_store_bot
rasa run actions --port 5055
```

#### Bước 3: Chạy Rasa Server

```bash
# Terminal 2: Chạy Rasa server
cd examples/it_store_bot
rasa run --enable-api --cors "*" --port 5005
```

#### Bước 4: Mở Web Frontend

```bash
# Mở frontend/index.html bằng trình duyệt
# Hoặc sử dụng Python HTTP server
cd frontend
python3 -m http.server 8000

# Truy cập: http://localhost:8000
```

## 📱 Hướng Dẫn Demo Cho Khách Hàng

### Kịch Bản Demo 1: Tìm Laptop Gaming

1. **Mở chat widget** (click nút chat góc dưới phải)
2. **Nhập:** "Tôi cần laptop gaming dưới 30 triệu"
3. **Chatbot hiển thị:**
   - 2-3 sản phẩm phù hợp
   - Giá cả, cấu hình
   - Tình trạng còn hàng
   - Khuyến mãi (nếu có)
4. **Hỏi thêm:** "So sánh ASUS ROG và MSI Katana"
5. **Chatbot so sánh** chi tiết 2 sản phẩm

### Kịch Bản Demo 2: Tư Vấn Máy Chủ Doanh Nghiệp

1. **Nhập:** "Tư vấn máy chủ cho doanh nghiệp 50 nhân viên"
2. **Chatbot hỏi thêm về:**
   - Ngân sách
   - Mục đích sử dụng (file server, web server, database...)
   - Yêu cầu đặc biệt
3. **Chatbot đề xuất:**
   - 2-3 máy chủ phù hợp
   - Giải thích lý do chọn
   - Dịch vụ đi kèm

### Kịch Bản Demo 3: Giải Pháp Camera An Ninh

1. **Nhập:** "Tư vấn giải pháp camera cho văn phòng 500m2"
2. **Chatbot phân tích:**
   - Diện tích cần giám sát
   - Số lượng camera đề xuất
   - Thiết bị cần thiết (NVR, HDD, cáp...)
3. **Chatbot báo giá:**
   - Gói trọn bộ
   - Giá từng sản phẩm
   - Khuyến mãi bundle
   - Chi phí lắp đặt

### Kịch Bản Demo 4: So Sánh Switch Mạng

1. **Nhập:** "So sánh Cisco Catalyst và HPE OfficeConnect"
2. **Chatbot hiển thị:**
   - Bảng so sánh chi tiết
   - Số port, tốc độ, tính năng
   - Chênh lệch giá
   - Ưu nhược điểm
3. **Gợi ý:** Nên chọn sản phẩm nào dựa vào nhu cầu

### Kịch Bản Demo 5: Hỗ Trợ Kỹ Thuật

1. **Nhập:** "Hướng dẫn cài đặt Windows Server"
2. **Chatbot cung cấp:**
   - Link tài liệu hướng dẫn
   - Video tutorial
   - Dịch vụ hỗ trợ từ xa
   - Hotline kỹ thuật

## 🎨 Giao Diện Web

### Trang Chủ (Hero Section)
- **Tiêu đề nổi bật:** TechStore AI
- **Slogan:** "Trợ Lý AI Tư Vấn Sản Phẩm IT"
- **CTA Button:** "Bắt Đầu Tư Vấn Ngay"
- **Floating cards:** Hiển thị 4 danh mục chính

### Danh Mục Sản Phẩm
- **6 category cards:**
  1. Máy Tính & Laptop
  2. Máy Chủ (Server)
  3. Switch Mạng
  4. Camera An Ninh
  5. Phần Mềm
  6. Linh Kiện
- **Click vào card** → Tự động mở chat và hỏi về danh mục đó

### Tính Năng Chatbot (Features Section)
- **6 feature cards** với ví dụ thực tế:
  1. Tìm Kiếm Thông Minh
  2. So Sánh Sản Phẩm
  3. Tư Vấn Chuyên Nghiệp
  4. Thông Số Kỹ Thuật
  5. Báo Giá & Khuyến Mãi
  6. Hỗ Trợ Kỹ Thuật
- **"Xem ví dụ" button** → Demo câu hỏi mẫu

### Thống Kê (Stats Section)
- **5000+ Sản Phẩm**
- **95% Độ Chính Xác**
- **24/7 Hoạt Động**
- **<3s Thời Gian Phản Hồi**

### Chat Widget
- **Vị trí:** Góc dưới bên phải
- **Tính năng:**
  - Avatar bot với icon robot
  - Status: "Đang hoạt động"
  - Typing indicator
  - Quick replies (gợi ý nhanh)
  - Message formatting (bold, bullets)
  - Product cards

## 📊 Database Sản Phẩm Demo

### Laptop (4 sản phẩm)
1. **ASUS ROG Strix G15** - Gaming - 28,990,000đ
2. **MSI Katana GF66** - Gaming - 24,990,000đ
3. **Dell Latitude 5420** - Văn phòng - 15,990,000đ
4. **HP ProBook 450 G9** - Văn phòng - 21,490,000đ

### Máy Chủ (3 sản phẩm)
1. **Dell PowerEdge R740** - 145,000,000đ
2. **HP ProLiant DL380 Gen10** - 98,000,000đ
3. **Lenovo ThinkSystem SR650** - 112,000,000đ

### Switch (2 sản phẩm)
1. **Cisco Catalyst 2960-X 48 Port** - 42,500,000đ
2. **HPE OfficeConnect 1920S 48G** - 18,900,000đ

### Camera (2 sản phẩm)
1. **Hikvision DS-2CD2143G2-I** - 2,890,000đ
2. **Dahua IPC-HDW3441TM-AS** - 1,990,000đ

### Phần Mềm (3 sản phẩm)
1. **Windows 11 Pro** - 5,490,000đ
2. **Windows Server 2022 Standard** - 24,900,000đ
3. **Office 2021 Professional Plus** - 8,900,000đ

## 🔧 Tùy Chỉnh

### Thêm Sản Phẩm Mới

Chỉnh sửa file: `examples/it_store_bot/actions/actions_it_store.py`

```python
PRODUCT_DATABASE = {
    "laptop": [
        {
            "id": "LP005",
            "name": "Lenovo ThinkPad X1 Carbon",
            "brand": "Lenovo",
            "category": "laptop",
            "type": "office",
            "price": 35990000,
            "specs": {
                "cpu": "Intel i7-1260P",
                "ram": "16GB LPDDR5",
                "storage": "512GB NVMe SSD",
                "display": "14\" 2.8K OLED"
            },
            "stock": True,
            "warranty": "36 tháng",
            "promotion": None
        },
        # Thêm sản phẩm khác...
    ]
}
```

### Thay Đổi Màu Sắc & Branding

Chỉnh sửa file: `frontend/style.css`

```css
:root {
    --primary-color: #2563eb;      /* Màu chủ đạo */
    --secondary-color: #10b981;    /* Màu phụ */
    --accent-color: #f59e0b;       /* Màu nhấn */
}
```

### Thay Đổi Logo & Tên Công Ty

Chỉnh sửa file: `frontend/index.html`

```html
<div class="logo">
    <i class="fas fa-microchip"></i>
    <span>Tên Công Ty Bạn</span>
</div>
```

## 🌐 Triển Khai Production

### Bước 1: Deploy Rasa Server

```bash
# Sử dụng Docker
docker run -p 5005:5005 \
  -v $(pwd)/examples/it_store_bot:/app \
  rasa/rasa:3.6.21-full \
  run --enable-api --cors "*"
```

### Bước 2: Deploy Action Server

```bash
docker run -p 5055:5055 \
  -v $(pwd)/examples/it_store_bot/actions:/app/actions \
  rasa/rasa-sdk:3.6.0
```

### Bước 3: Deploy Frontend

Upload thư mục `frontend/` lên web hosting hoặc:

```bash
# Sử dụng Nginx
docker run -p 80:80 \
  -v $(pwd)/frontend:/usr/share/nginx/html \
  nginx:alpine
```

### Bước 4: Cập Nhật URL Server

Chỉnh sửa `frontend/script.js`:

```javascript
const RASA_SERVER_URL = 'https://your-rasa-server.com';
```

## 📝 Training Thêm Dữ Liệu

### Thêm Intent Mới

File: `examples/it_store_bot/data/nlu.yml`

```yaml
- intent: ask_delivery
  examples: |
    - giao hàng mất bao lâu
    - ship hàng thế nào
    - phí vận chuyển
    - có giao hàng miễn phí không
```

### Thêm Response

File: `examples/it_store_bot/domain.yml`

```yaml
responses:
  utter_delivery_info:
    - text: |
        🚚 Chính sách giao hàng:
        • Miễn phí trong nội thành (<50km)
        • 1-2 ngày cho sản phẩm thường
        • 3-5 ngày cho máy chủ (cần kiểm tra kỹ)
```

### Training Lại

```bash
cd examples/it_store_bot
rasa train
# Restart Rasa server để load model mới
```

## 🎯 Tips Demo Hiệu Quả

### 1. **Chuẩn Bị Trước**
- ✅ Kiểm tra Rasa server đang chạy
- ✅ Test 3-4 câu hỏi mẫu
- ✅ Chuẩn bị backup plan (demo standalone)

### 2. **Trong Lúc Demo**
- 🎤 Giải thích từng tính năng khi demo
- 💡 Nhấn mạnh lợi ích cho khách hàng:
  - Tiết kiệm thời gian tư vấn
  - Tăng tỷ lệ chuyển đổi
  - Hỗ trợ 24/7
  - Giảm chi phí nhân sự

### 3. **Xử Lý Lỗi**
- Nếu Rasa server down → Chuyển sang demo standalone
- Nếu chatbot không hiểu → Giải thích "đang training thêm data"
- Nếu trả lời sai → Ghi nhận feedback để cải thiện

### 4. **Kết Thúc Demo**
- 📊 Trình bày số liệu: accuracy, response time
- 💰 Báo giá triển khai
- 📅 Đề xuất lộ trình: POC → Pilot → Production

## 🔗 Tích Hợp Thêm

### Tích Hợp LLM (OpenAI/Claude)

Sử dụng module LLM đã tạo trước đó:

```bash
# Copy LLM module vào IT store bot
cp -r actions/llm examples/it_store_bot/actions/

# Cấu hình trong domain.yml
actions:
  - action_llm_fallback
  - action_search_product  # existing
```

### Tích Hợp Database Thật

Thay thế `PRODUCT_DATABASE` bằng:

```python
import pymongo
# hoặc
import mysql.connector

def search_products(product_type, ...):
    # Query database thật
    cursor = db.products.find({
        "category": product_type,
        "price": {"$lte": price_max}
    })
    return list(cursor)
```

### Tích Hợp CRM/ERP

```python
def notify_sales_team(customer_query, products_viewed):
    """Thông báo cho sales team khi có khách hỏi"""
    # Gửi webhook đến CRM
    requests.post("https://your-crm.com/webhook", json={
        "query": customer_query,
        "products": products_viewed,
        "timestamp": datetime.now()
    })
```

## 📞 Hỗ Trợ

Nếu cần hỗ trợ kỹ thuật:
- 📧 Email: support@techstore.vn
- 💬 Slack/Teams channel
- 📚 Docs: Xem các file guide khác trong repo

## 🎉 Kết Luận

Web frontend này cung cấp:
- ✅ Giao diện chuyên nghiệp, hiện đại
- ✅ 6 tính năng chatbot AI đầy đủ
- ✅ Database sản phẩm IT demo
- ✅ Standalone mode (không cần Rasa)
- ✅ Dễ dàng tùy chỉnh & mở rộng
- ✅ Kịch bản demo rõ ràng

**Sẵn sàng để demo cho khách hàng ngay!** 🚀

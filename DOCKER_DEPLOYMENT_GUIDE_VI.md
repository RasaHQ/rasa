# Hướng Dẫn Triển Khai Rasa Chatbot Trên Docker

## Mục Lục
1. [Giới Thiệu](#giới-thiệu)
2. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
3. [Cách 1: Sử Dụng Docker Image Có Sẵn](#cách-1-sử-dụng-docker-image-có-sẵn)
4. [Cách 2: Build Docker Image Từ Source](#cách-2-build-docker-image-từ-source)
5. [Cách 3: Triển Khai Với Docker Compose](#cách-3-triển-khai-với-docker-compose)
6. [Cấu Hình & Tùy Chỉnh](#cấu-hình--tùy-chỉnh)
7. [Các Lệnh Thường Dùng](#các-lệnh-thường-dùng)
8. [Troubleshooting](#troubleshooting)

---

## Giới Thiệu

Dự án này là **Rasa Open Source v3.6.21** - một framework mã nguồn mở để xây dựng chatbot thông minh với AI/ML.

Rasa cho phép bạn tạo các trợ lý ảo có thể:
- Hiểu ngôn ngữ tự nhiên (NLU)
- Quản lý hội thoại đa lớp
- Tích hợp với nhiều nền tảng: Facebook Messenger, Slack, Telegram, v.v.

---

## Yêu Cầu Hệ Thống

### Phần Cứng Tối Thiểu
- **RAM**: 4GB (khuyến nghị 8GB+)
- **CPU**: 2 cores (khuyến nghị 4+ cores)
- **Disk**: 10GB trống

### Phần Mềm
- **Docker**: phiên bản 20.10+
- **Docker Compose**: phiên bản 1.29+ (tùy chọn)
- **Git**: để clone repository

### Cài Đặt Docker

**Ubuntu/Debian:**
```bash
# Cập nhật packages
sudo apt-get update

# Cài đặt Docker
sudo apt-get install -y docker.io docker-compose

# Thêm user vào group docker (để không cần sudo)
sudo usermod -aG docker $USER

# Khởi động Docker
sudo systemctl start docker
sudo systemctl enable docker
```

**macOS:**
```bash
# Cài Docker Desktop từ: https://www.docker.com/products/docker-desktop
# Hoặc dùng Homebrew:
brew install --cask docker
```

**Windows:**
- Tải Docker Desktop: https://www.docker.com/products/docker-desktop

---

## Cách 1: Sử Dụng Docker Image Có Sẵn

Đây là cách nhanh nhất để chạy Rasa chatbot.

### Bước 1: Chuẩn Bị Dự Án Chatbot

```bash
# Tạo thư mục cho chatbot
mkdir my-chatbot
cd my-chatbot

# Khởi tạo dự án Rasa mới (sử dụng Docker)
docker run -v $(pwd):/app rasa/rasa:3.6.21-full init --no-prompt
```

### Bước 2: Train Model

```bash
# Train model với dữ liệu mẫu
docker run -v $(pwd):/app rasa/rasa:3.6.21-full train
```

### Bước 3: Chạy Chatbot

```bash
# Chạy Rasa server
docker run -v $(pwd):/app -p 5005:5005 rasa/rasa:3.6.21-full run --enable-api

# Hoặc chạy ở chế độ shell (tương tác)
docker run -it -v $(pwd):/app rasa/rasa:3.6.21-full shell
```

### Bước 4: Test Chatbot

Mở terminal khác và test API:
```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "sender": "test_user"}'
```

---

## Cách 2: Build Docker Image Từ Source

Nếu bạn muốn tùy chỉnh hoặc phát triển Rasa từ source code.

### Bước 1: Clone Repository

```bash
git clone https://github.com/RasaHQ/rasa.git
cd rasa
```

### Bước 2: Build Docker Image

```bash
# Build image với make (khuyến nghị)
make build-docker

# Image sẽ có tag: rasa:localdev
```

**Hoặc build thủ công:**
```bash
docker build -t my-rasa:latest \
  --build-arg IMAGE_BASE_NAME=rasa \
  --build-arg BASE_IMAGE_HASH=latest \
  --build-arg BASE_BUILDER_IMAGE_HASH=latest \
  -f Dockerfile .
```

### Bước 3: Chạy Image Đã Build

```bash
# Chạy với image vừa build
docker run -it -v $(pwd)/examples/moodbot:/app \
  -p 5005:5005 rasa:localdev shell

# Hoặc chạy server
docker run -v $(pwd)/examples/moodbot:/app \
  -p 5005:5005 rasa:localdev run --enable-api
```

---

## Cách 3: Triển Khai Với Docker Compose

Docker Compose giúp quản lý nhiều container (Rasa server, Action server, Database, v.v.)

### Bước 1: Tạo File docker-compose.yml

Tạo file `docker-compose.yml` trong thư mục dự án:

```yaml
version: '3.8'

services:
  # Rasa Server
  rasa:
    image: rasa/rasa:3.6.21-full
    container_name: rasa-server
    ports:
      - "5005:5005"
    volumes:
      - ./:/app
    command:
      - run
      - --enable-api
      - --cors
      - "*"
      - --debug
    networks:
      - rasa-network

  # Rasa Action Server (cho custom actions)
  action-server:
    image: rasa/rasa-sdk:3.6.2
    container_name: rasa-action-server
    ports:
      - "5055:5055"
    volumes:
      - ./actions:/app/actions
    networks:
      - rasa-network

  # Duckling (cho entity extraction - dates, numbers, etc.)
  duckling:
    image: rasa/duckling:latest
    container_name: rasa-duckling
    ports:
      - "8000:8000"
    networks:
      - rasa-network

networks:
  rasa-network:
    driver: bridge
```

### Bước 2: Chuẩn Bị Cấu Trúc Thư Mục

```bash
# Cấu trúc thư mục chatbot
my-chatbot/
├── docker-compose.yml
├── domain.yml
├── config.yml
├── credentials.yml
├── endpoints.yml
├── data/
│   ├── nlu.yml
│   └── stories.yml
├── actions/
│   ├── __init__.py
│   └── actions.py
└── models/
```

### Bước 3: Cấu Hình endpoints.yml

Tạo file `endpoints.yml` để kết nối với Action Server:

```yaml
action_endpoint:
  url: "http://action-server:5055/webhook"
```

### Bước 4: Train Model

```bash
# Train model trước khi start services
docker-compose run rasa train
```

### Bước 5: Khởi Động Services

```bash
# Start tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f rasa

# Kiểm tra status
docker-compose ps
```

### Bước 6: Test Chatbot

```bash
# Test qua REST API
curl -X POST http://localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Xin chào",
    "sender": "user123"
  }'

# Test interactive shell
docker-compose exec rasa rasa shell
```

---

## Cấu Hình & Tùy Chỉnh

### 1. Cấu Hình Pipeline NLU (config.yml)

```yaml
language: vi  # Ngôn ngữ tiếng Việt

pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100

policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
    max_history: 5
    epochs: 100
  - name: RulePolicy
```

### 2. Environment Variables

Thêm vào `docker-compose.yml`:

```yaml
services:
  rasa:
    environment:
      - RASA_TELEMETRY_ENABLED=false
      - RASA_LOG_LEVEL=DEBUG
      - RASA_MODEL=/app/models
```

### 3. Persistent Storage cho Models

```yaml
services:
  rasa:
    volumes:
      - ./:/app
      - ./models:/app/models
      - rasa-data:/tmp/rasa

volumes:
  rasa-data:
```

### 4. Sử Dụng Database (PostgreSQL)

Thêm PostgreSQL vào `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:13-alpine
    container_name: rasa-postgres
    environment:
      POSTGRES_DB: rasa
      POSTGRES_USER: rasa
      POSTGRES_PASSWORD: rasa123
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - rasa-network

  rasa:
    depends_on:
      - postgres
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_USER=rasa
      - DB_PASSWORD=rasa123
      - DB_DATABASE=rasa

volumes:
  postgres-data:
```

Cập nhật `endpoints.yml`:
```yaml
tracker_store:
  type: SQL
  dialect: postgresql
  url: postgres
  port: 5432
  db: rasa
  username: rasa
  password: rasa123
```

---

## Các Lệnh Thường Dùng

### Quản Lý Container

```bash
# Xem danh sách containers đang chạy
docker-compose ps

# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Xem logs
docker-compose logs -f rasa
docker-compose logs -f action-server

# Stop và xóa tất cả (bao gồm volumes)
docker-compose down -v
```

### Rasa Commands

```bash
# Train model
docker-compose run rasa train

# Train NLU only
docker-compose run rasa train nlu

# Test model
docker-compose run rasa test

# Interactive learning
docker-compose run rasa interactive

# Shell mode
docker-compose exec rasa rasa shell

# Validate domain và data
docker-compose run rasa data validate

# Visualize stories
docker-compose run rasa visualize
```

### Debug & Development

```bash
# Vào container shell
docker-compose exec rasa bash

# Xem Rasa version
docker-compose run rasa --version

# Test NLU model
docker-compose run rasa shell nlu

# Run với debug mode
docker-compose run rasa run --enable-api --debug --cors "*"
```

---

## Troubleshooting

### Lỗi: Port 5005 đã được sử dụng

```bash
# Tìm process đang dùng port
sudo lsof -i :5005

# Hoặc kill process
sudo kill -9 $(sudo lsof -t -i:5005)

# Hoặc đổi port trong docker-compose.yml
ports:
  - "5006:5005"  # Host:Container
```

### Lỗi: Model không load được

```bash
# Kiểm tra xem model có tồn tại không
docker-compose exec rasa ls -la /app/models/

# Train lại model
docker-compose run rasa train --force

# Xóa old models và train lại
rm -rf models/*
docker-compose run rasa train
```

### Lỗi: Action server không kết nối được

```bash
# Kiểm tra action server đang chạy
docker-compose ps action-server

# Check logs
docker-compose logs action-server

# Test connection từ rasa container
docker-compose exec rasa curl http://action-server:5055/health
```

### Lỗi: Out of Memory

Tăng memory limit trong `docker-compose.yml`:

```yaml
services:
  rasa:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### Lỗi: Permission denied với volumes

```bash
# Fix permissions
sudo chown -R $USER:$USER ./
chmod -R 755 ./

# Hoặc thêm user vào docker-compose.yml
services:
  rasa:
    user: "${UID}:${GID}"
```

### Debugging Tips

1. **Xem logs chi tiết:**
```bash
docker-compose logs -f --tail=100 rasa
```

2. **Test API endpoint:**
```bash
# Health check
curl http://localhost:5005/

# Model info
curl http://localhost:5005/status
```

3. **Rebuild khi có thay đổi code:**
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## Ví Dụ Chatbot Hoàn Chỉnh

### 1. Sử dụng Moodbot Example

```bash
# Copy moodbot example
cp -r examples/moodbot ./my-moodbot
cd my-moodbot

# Train
docker run -v $(pwd):/app rasa/rasa:3.6.21-full train

# Run
docker run -it -v $(pwd):/app -p 5005:5005 \
  rasa/rasa:3.6.21-full shell
```

### 2. Deploy Production với Nginx Reverse Proxy

Tạo `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rasa
    networks:
      - rasa-network

  rasa:
    image: rasa/rasa:3.6.21-full
    command:
      - run
      - --enable-api
      - --cors
      - "*"
    networks:
      - rasa-network

networks:
  rasa-network:
```

---

## Tài Liệu Tham Khảo

- **Rasa Documentation**: https://rasa.com/docs/rasa/
- **Docker Documentation**: https://docs.docker.com/
- **Rasa Docker Images**: https://hub.docker.com/r/rasa/rasa
- **Rasa Community Forum**: https://forum.rasa.com/

---

## Kết Luận

Bạn đã hoàn thành hướng dẫn triển khai Rasa Chatbot trên Docker!

**Các bước tiếp theo:**
1. Tùy chỉnh `domain.yml` và `data/` cho use case của bạn
2. Phát triển custom actions trong `actions/actions.py`
3. Train và test model thường xuyên
4. Deploy lên production server

**Cần trợ giúp?**
- GitHub Issues: https://github.com/RasaHQ/rasa/issues
- Rasa Community: https://forum.rasa.com/

Chúc bạn xây dựng chatbot thành công! 🚀

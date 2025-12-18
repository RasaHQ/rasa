// Configuration
const RASA_SERVER_URL = 'http://localhost:5005';
const USER_ID = 'user_' + Math.random().toString(36).substr(2, 9);

// Chat state
let chatOpen = false;
let conversationHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('TechStore AI initialized');
    // Auto-open chat after 3 seconds for demo
    setTimeout(() => {
        if (!chatOpen) {
            const badge = document.querySelector('.chat-badge');
            if (badge) {
                badge.style.animation = 'pulse 1s infinite';
            }
        }
    }, 3000);
});

// Toggle chat widget
function toggleChat() {
    const chatWidget = document.getElementById('chatWidget');
    const chatToggle = document.getElementById('chatToggle');
    const badge = document.querySelector('.chat-badge');

    chatOpen = !chatOpen;

    if (chatOpen) {
        chatWidget.classList.add('open');
        chatToggle.style.display = 'none';
        if (badge) badge.style.display = 'none';
        scrollToBottom();
    } else {
        chatWidget.classList.remove('open');
        chatToggle.style.display = 'flex';
    }
}

function openChat() {
    if (!chatOpen) {
        toggleChat();
    }
}

function closeChat() {
    if (chatOpen) {
        toggleChat();
    }
}

// Handle key press in input
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Send message to Rasa
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message) return;

    // Clear input
    input.value = '';

    // Add user message to chat
    addMessage(message, 'user');

    // Hide quick replies after first message
    hideQuickReplies();

    // Show typing indicator
    showTypingIndicator();

    try {
        // Send to Rasa
        const response = await fetch(`${RASA_SERVER_URL}/webhooks/rest/webhook`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sender: USER_ID,
                message: message
            })
        });

        // Hide typing indicator
        hideTypingIndicator();

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Add bot responses
        if (data && data.length > 0) {
            data.forEach((msg, index) => {
                setTimeout(() => {
                    if (msg.text) {
                        addMessage(msg.text, 'bot');
                    }
                    if (msg.image) {
                        addImage(msg.image, 'bot');
                    }
                    if (msg.custom) {
                        handleCustomMessage(msg.custom);
                    }
                }, index * 500);
            });
        } else {
            addMessage('Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể hỏi lại được không?', 'bot');
        }

    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();

        // Fallback response when Rasa server is not available
        addMessage(
            '⚠️ Hiện tại hệ thống đang trong chế độ demo. Để kết nối với Rasa server, vui lòng:\n\n' +
            '1. Chạy Rasa server: rasa run --enable-api --cors "*"\n' +
            '2. Chạy action server: rasa run actions\n\n' +
            'Tôi sẽ demo một số tính năng cho bạn:',
            'bot'
        );

        // Demo response based on keywords
        setTimeout(() => {
            const demoResponse = generateDemoResponse(message);
            addMessage(demoResponse, 'bot');
        }, 1000);
    }
}

// Generate demo response (fallback when Rasa is not running)
function generateDemoResponse(message) {
    const lowerMessage = message.toLowerCase();

    // Laptop queries
    if (lowerMessage.includes('laptop') || lowerMessage.includes('máy tính xách tay')) {
        if (lowerMessage.includes('gaming')) {
            return '🎮 **Laptop Gaming Đề Xuất:**\n\n' +
                   '**1. ASUS ROG Strix G15**\n' +
                   '• CPU: AMD Ryzen 7 6800H\n' +
                   '• GPU: RTX 3060 6GB\n' +
                   '• RAM: 16GB DDR5\n' +
                   '• Màn hình: 15.6" FHD 144Hz\n' +
                   '• Giá: 28,990,000đ\n\n' +
                   '**2. MSI Katana GF66**\n' +
                   '• CPU: Intel i7-12650H\n' +
                   '• GPU: RTX 3050 Ti 4GB\n' +
                   '• RAM: 16GB DDR4\n' +
                   '• Màn hình: 15.6" FHD 144Hz\n' +
                   '• Giá: 24,990,000đ\n\n' +
                   'Bạn muốn xem chi tiết sản phẩm nào?';
        } else {
            return '💼 **Laptop Văn Phòng Phổ Biến:**\n\n' +
                   '**1. Dell Latitude 5420**\n' +
                   '• CPU: Intel i5-1135G7\n' +
                   '• RAM: 8GB\n' +
                   '• SSD: 256GB NVMe\n' +
                   '• Giá: 15,990,000đ\n\n' +
                   '**2. HP ProBook 450 G9**\n' +
                   '• CPU: Intel i7-1255U\n' +
                   '• RAM: 16GB\n' +
                   '• SSD: 512GB\n' +
                   '• Giá: 21,490,000đ\n\n' +
                   'Bạn có ngân sách cụ thể không?';
        }
    }

    // Server queries
    if (lowerMessage.includes('máy chủ') || lowerMessage.includes('server')) {
        return '🖥️ **Máy Chủ Doanh Nghiệp:**\n\n' +
               '**1. Dell PowerEdge R740**\n' +
               '• CPU: 2x Intel Xeon Gold 6230 (20 cores/40 threads mỗi CPU)\n' +
               '• RAM: 128GB DDR4 ECC\n' +
               '• Storage: 4x 1.2TB SAS HDD\n' +
               '• Giá: 145,000,000đ\n\n' +
               '**2. HP ProLiant DL380 Gen10**\n' +
               '• CPU: 2x Intel Xeon Silver 4214 (12 cores/24 threads)\n' +
               '• RAM: 64GB DDR4 ECC\n' +
               '• Storage: 2x 960GB SSD\n' +
               '• Giá: 98,000,000đ\n\n' +
               'Doanh nghiệp bạn cần máy chủ cho mục đích gì?';
    }

    // Camera queries
    if (lowerMessage.includes('camera')) {
        return '📹 **Giải Pháp Camera An Ninh:**\n\n' +
               '**1. Hikvision DS-2CD2143G2-I**\n' +
               '• Độ phân giải: 4MP\n' +
               '• Công nghệ: AcuSense (nhận diện người/xe)\n' +
               '• Tầm nhìn ban đêm: 30m\n' +
               '• Giá: 2,890,000đ/camera\n\n' +
               '**2. Dahua IPC-HDW3441TM-AS**\n' +
               '• Độ phân giải: 4MP\n' +
               '• Tích hợp Mic\n' +
               '• AI SMD Plus\n' +
               '• Giá: 1,990,000đ/camera\n\n' +
               '**Gói Trọn Bộ Văn Phòng 500m2:**\n' +
               '• 12 camera IP 4MP\n' +
               '• 1 NVR 16 kênh Hikvision\n' +
               '• 1 HDD 4TB\n' +
               '• Phụ kiện lắp đặt\n' +
               '• Giá: 38,900,000đ (bao gồm setup)\n\n' +
               'Bạn cần bao nhiêu camera?';
    }

    // Switch queries
    if (lowerMessage.includes('switch')) {
        return '🔌 **Switch Mạng Doanh Nghiệp:**\n\n' +
               '**1. Cisco Catalyst 2960-X 48 Port**\n' +
               '• 48 port Gigabit + 2 port 10G SFP+\n' +
               '• Layer 2 Managed\n' +
               '• Lifetime warranty\n' +
               '• Giá: 42,500,000đ\n\n' +
               '**2. HPE OfficeConnect 1920S 48G**\n' +
               '• 48 port Gigabit + 4 port SFP\n' +
               '• Layer 2+ Smart Managed\n' +
               '• Giá: 18,900,000đ\n\n' +
               'Bạn cần switch cho văn phòng bao nhiêu nhân viên?';
    }

    // Software queries
    if (lowerMessage.includes('phần mềm') || lowerMessage.includes('software')) {
        return '💿 **Phần Mềm Thương Mại:**\n\n' +
               '**Microsoft:**\n' +
               '• Windows 11 Pro: 5,490,000đ\n' +
               '• Windows Server 2022 Standard: 24,900,000đ\n' +
               '• Office 2021 Professional Plus: 8,900,000đ\n' +
               '• Microsoft 365 Business: 189,000đ/tháng/user\n\n' +
               '**Antivirus:**\n' +
               '• Kaspersky Endpoint Security: 450,000đ/user/năm\n' +
               '• ESET Protect Advanced: 520,000đ/user/năm\n\n' +
               '**CAD/Design:**\n' +
               '• AutoCAD 2024: 45,500,000đ/năm\n' +
               '• SolidWorks 2024: 8,900 USD/năm\n\n' +
               'Bạn cần license cho bao nhiêu máy?';
    }

    // Comparison queries
    if (lowerMessage.includes('so sánh') || lowerMessage.includes('compare')) {
        return '⚖️ **So Sánh Sản Phẩm:**\n\n' +
               'Tôi có thể giúp bạn so sánh chi tiết về:\n' +
               '• Hiệu năng và cấu hình\n' +
               '• Giá cả và tính năng\n' +
               '• Ưu điểm và nhược điểm\n\n' +
               'Bạn muốn so sánh 2 sản phẩm nào? (VD: "So sánh Dell PowerEdge R740 và HP ProLiant DL380")';
    }

    // Price/promotion queries
    if (lowerMessage.includes('giá') || lowerMessage.includes('khuyến mãi') || lowerMessage.includes('promotion')) {
        return '🎁 **Chương Trình Khuyến Mãi Tháng Này:**\n\n' +
               '**1. Laptop Gaming:**\n' +
               '• Giảm 10-15% cho ASUS ROG series\n' +
               '• Tặng kèm chuột + balo gaming\n\n' +
               '**2. Máy Chủ:**\n' +
               '• Giảm 8% cho đơn hàng từ 100 triệu\n' +
               '• Hỗ trợ setup miễn phí\n\n' +
               '**3. Giải pháp Camera:**\n' +
               '• Mua 10 tặng 2 camera\n' +
               '• Miễn phí lắp đặt trong 50km\n\n' +
               'Bạn quan tâm sản phẩm nào?';
    }

    // Technical support
    if (lowerMessage.includes('cài đặt') || lowerMessage.includes('hướng dẫn') || lowerMessage.includes('support')) {
        return '🛠️ **Hỗ Trợ Kỹ Thuật:**\n\n' +
               'Chúng tôi cung cấp:\n' +
               '• Hướng dẫn cài đặt chi tiết\n' +
               '• Hỗ trợ cấu hình từ xa\n' +
               '• Bảo hành tận nơi\n' +
               '• Hotline 24/7: 1900-xxxx\n\n' +
               'Bạn cần hỗ trợ về vấn đề gì?';
    }

    // Default response
    return '👋 Tôi có thể giúp bạn:\n\n' +
           '🔍 **Tìm kiếm sản phẩm:** "Tìm laptop gaming dưới 30 triệu"\n' +
           '⚖️ **So sánh:** "So sánh Dell R740 và HP DL380"\n' +
           '💡 **Tư vấn:** "Tư vấn giải pháp camera cho văn phòng"\n' +
           '💰 **Báo giá:** "Giá máy chủ Dell PowerEdge"\n' +
           '🛠️ **Hỗ trợ:** "Hướng dẫn cài Windows Server"\n\n' +
           'Bạn cần tìm sản phẩm gì?';
}

// Add message to chat
function addMessage(text, sender) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';

    const content = document.createElement('div');
    content.className = 'message-content';

    // Parse text for better formatting
    const formattedText = formatMessage(text);
    content.innerHTML = formattedText;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    messagesContainer.appendChild(messageDiv);

    // Store in history
    conversationHistory.push({
        text: text,
        sender: sender,
        timestamp: new Date()
    });

    scrollToBottom();
}

// Format message text
function formatMessage(text) {
    // Convert markdown-style formatting
    let formatted = text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>') // Bold
        .replace(/\n/g, '<br>') // Line breaks
        .replace(/• /g, '&bull; '); // Bullets

    return formatted;
}

// Add image to chat
function addImage(imageUrl, sender) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<img src="${imageUrl}" style="max-width: 100%; border-radius: 8px;">`;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    messagesContainer.appendChild(messageDiv);

    scrollToBottom();
}

// Handle custom messages (e.g., product cards)
function handleCustomMessage(customData) {
    if (customData.type === 'product') {
        addProductCard(customData);
    } else if (customData.type === 'product_list') {
        customData.products.forEach(product => {
            addProductCard(product);
        });
    }
}

// Add product card
function addProductCard(product) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';

    const content = document.createElement('div');
    content.className = 'message-content';

    const card = document.createElement('div');
    card.className = 'product-card';
    card.innerHTML = `
        <h4>${product.name}</h4>
        <div class="price">${product.price}</div>
        <div class="specs">${product.specs || ''}</div>
        ${product.stock ? `<div style="color: var(--secondary-color); margin-top: 8px;">✓ Còn hàng</div>` : ''}
    `;

    content.appendChild(card);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    messagesContainer.appendChild(messageDiv);

    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    const messagesContainer = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typingIndicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    typingDiv.appendChild(avatar);
    typingDiv.appendChild(content);
    messagesContainer.appendChild(typingDiv);

    scrollToBottom();
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Scroll to bottom of chat
function scrollToBottom() {
    const messagesContainer = document.getElementById('chatMessages');
    setTimeout(() => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }, 100);
}

// Hide quick replies
function hideQuickReplies() {
    const quickReplies = document.getElementById('quickReplies');
    if (quickReplies) {
        quickReplies.style.display = 'none';
    }
}

// Send quick reply
function sendQuickReply(text) {
    document.getElementById('chatInput').value = text;
    sendMessage();
}

// Ask about category
function askAboutCategory(category) {
    openChat();
    setTimeout(() => {
        document.getElementById('chatInput').value = `Tôi muốn xem ${category}`;
        sendMessage();
    }, 300);
}

// Ask example question
function askExample(question) {
    openChat();
    setTimeout(() => {
        document.getElementById('chatInput').value = question;
        sendMessage();
    }, 300);
}

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

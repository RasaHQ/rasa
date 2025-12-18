"""
Custom Actions for IT Store Chatbot
Handles product search, comparison, pricing, and recommendations
"""

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import logging

logger = logging.getLogger(__name__)


# ===== PRODUCT DATABASE =====

PRODUCT_DATABASE = {
    "laptop": [
        {
            "id": "LP001",
            "name": "ASUS ROG Strix G15",
            "brand": "ASUS",
            "category": "laptop",
            "type": "gaming",
            "price": 28990000,
            "specs": {
                "cpu": "AMD Ryzen 7 6800H",
                "gpu": "NVIDIA RTX 3060 6GB",
                "ram": "16GB DDR5",
                "storage": "512GB NVMe SSD",
                "display": "15.6\" FHD 144Hz"
            },
            "stock": True,
            "warranty": "24 tháng",
            "promotion": "Giảm 10% + Tặng chuột gaming"
        },
        {
            "id": "LP002",
            "name": "MSI Katana GF66",
            "brand": "MSI",
            "category": "laptop",
            "type": "gaming",
            "price": 24990000,
            "specs": {
                "cpu": "Intel i7-12650H",
                "gpu": "NVIDIA RTX 3050 Ti 4GB",
                "ram": "16GB DDR4",
                "storage": "512GB NVMe SSD",
                "display": "15.6\" FHD 144Hz"
            },
            "stock": True,
            "warranty": "24 tháng",
            "promotion": None
        },
        {
            "id": "LP003",
            "name": "Dell Latitude 5420",
            "brand": "Dell",
            "category": "laptop",
            "type": "office",
            "price": 15990000,
            "specs": {
                "cpu": "Intel i5-1135G7",
                "ram": "8GB DDR4",
                "storage": "256GB NVMe SSD",
                "display": "14\" FHD"
            },
            "stock": True,
            "warranty": "36 tháng",
            "promotion": None
        },
        {
            "id": "LP004",
            "name": "HP ProBook 450 G9",
            "brand": "HP",
            "category": "laptop",
            "type": "office",
            "price": 21490000,
            "specs": {
                "cpu": "Intel i7-1255U",
                "ram": "16GB DDR4",
                "storage": "512GB NVMe SSD",
                "display": "15.6\" FHD"
            },
            "stock": True,
            "warranty": "24 tháng",
            "promotion": "Giảm 5%"
        }
    ],
    "server": [
        {
            "id": "SV001",
            "name": "Dell PowerEdge R740",
            "brand": "Dell",
            "category": "server",
            "price": 145000000,
            "specs": {
                "cpu": "2x Intel Xeon Gold 6230 (20 cores mỗi CPU)",
                "ram": "128GB DDR4 ECC",
                "storage": "4x 1.2TB SAS HDD",
                "raid": "RAID 10",
                "power": "2x 750W redundant"
            },
            "stock": True,
            "warranty": "36 tháng on-site",
            "promotion": "Giảm 8% cho đơn từ 100 triệu"
        },
        {
            "id": "SV002",
            "name": "HP ProLiant DL380 Gen10",
            "brand": "HP",
            "category": "server",
            "price": 98000000,
            "specs": {
                "cpu": "2x Intel Xeon Silver 4214 (12 cores mỗi CPU)",
                "ram": "64GB DDR4 ECC",
                "storage": "2x 960GB SSD",
                "raid": "RAID 1",
                "power": "2x 800W redundant"
            },
            "stock": True,
            "warranty": "36 tháng on-site",
            "promotion": "Hỗ trợ setup miễn phí"
        },
        {
            "id": "SV003",
            "name": "Lenovo ThinkSystem SR650",
            "brand": "Lenovo",
            "category": "server",
            "price": 112000000,
            "specs": {
                "cpu": "2x Intel Xeon Gold 5218 (16 cores mỗi CPU)",
                "ram": "96GB DDR4 ECC",
                "storage": "4x 2TB SATA HDD",
                "raid": "RAID 5",
                "power": "2x 750W redundant"
            },
            "stock": False,
            "warranty": "36 tháng",
            "promotion": None
        }
    ],
    "switch": [
        {
            "id": "SW001",
            "name": "Cisco Catalyst 2960-X 48 Port",
            "brand": "Cisco",
            "category": "switch",
            "price": 42500000,
            "specs": {
                "ports": "48x Gigabit Ethernet + 2x 10G SFP+",
                "type": "Layer 2 Managed",
                "power": "PoE+ 740W",
                "throughput": "216 Gbps"
            },
            "stock": True,
            "warranty": "Lifetime",
            "promotion": None
        },
        {
            "id": "SW002",
            "name": "HPE OfficeConnect 1920S 48G",
            "brand": "HPE",
            "category": "switch",
            "price": 18900000,
            "specs": {
                "ports": "48x Gigabit Ethernet + 4x SFP",
                "type": "Layer 2+ Smart Managed",
                "power": "Non-PoE",
                "throughput": "176 Gbps"
            },
            "stock": True,
            "warranty": "Lifetime",
            "promotion": "Giảm 5%"
        }
    ],
    "camera": [
        {
            "id": "CM001",
            "name": "Hikvision DS-2CD2143G2-I",
            "brand": "Hikvision",
            "category": "camera",
            "price": 2890000,
            "specs": {
                "resolution": "4MP",
                "technology": "AcuSense (AI nhận diện người/xe)",
                "night_vision": "30m",
                "type": "IP Dome"
            },
            "stock": True,
            "warranty": "24 tháng",
            "promotion": "Mua 10 tặng 2"
        },
        {
            "id": "CM002",
            "name": "Dahua IPC-HDW3441TM-AS",
            "brand": "Dahua",
            "category": "camera",
            "price": 1990000,
            "specs": {
                "resolution": "4MP",
                "technology": "AI SMD Plus",
                "night_vision": "30m",
                "type": "IP Dome với Mic tích hợp"
            },
            "stock": True,
            "warranty": "24 tháng",
            "promotion": "Bundle: 8 camera + NVR = 18 triệu"
        }
    ],
    "software": [
        {
            "id": "SW001",
            "name": "Windows 11 Pro",
            "brand": "Microsoft",
            "category": "software",
            "price": 5490000,
            "specs": {
                "type": "Operating System",
                "license": "Retail FPP",
                "devices": "1 PC"
            },
            "stock": True,
            "warranty": "Vĩnh viễn",
            "promotion": None
        },
        {
            "id": "SW002",
            "name": "Windows Server 2022 Standard",
            "brand": "Microsoft",
            "category": "software",
            "price": 24900000,
            "specs": {
                "type": "Server OS",
                "license": "16 Core",
                "support": "5 năm mainstream"
            },
            "stock": True,
            "warranty": "Vĩnh viễn",
            "promotion": None
        },
        {
            "id": "SW003",
            "name": "Office 2021 Professional Plus",
            "brand": "Microsoft",
            "category": "software",
            "price": 8900000,
            "specs": {
                "type": "Office Suite",
                "apps": "Word, Excel, PowerPoint, Outlook, Access, Publisher",
                "license": "1 PC"
            },
            "stock": True,
            "warranty": "Vĩnh viễn",
            "promotion": "Mua 5+ giảm 10%"
        }
    ]
}


def format_price(price):
    """Format price to Vietnamese currency"""
    return f"{price:,.0f}đ".replace(",", ".")


def format_product_card(product):
    """Format product information as a card"""
    specs_text = "\n".join([f"• {k.replace('_', ' ').title()}: {v}"
                            for k, v in product['specs'].items()])

    stock_text = "✅ Còn hàng" if product['stock'] else "❌ Hết hàng"
    promo_text = f"\n🎁 Khuyến mãi: {product['promotion']}" if product['promotion'] else ""

    return f"""**{product['name']}** ({product['brand']})
💰 Giá: {format_price(product['price'])}

**Thông số kỹ thuật:**
{specs_text}

🛡️ Bảo hành: {product['warranty']}
{stock_text}{promo_text}"""


def search_products(product_type=None, brand=None, price_max=None, use_case=None):
    """Search products based on filters"""
    results = []

    # Get products by type
    if product_type:
        type_map = {
            "máy tính": "laptop",
            "laptop": "laptop",
            "máy chủ": "server",
            "server": "server",
            "switch": "switch",
            "switch mạng": "switch",
            "camera": "camera",
            "phần mềm": "software",
            "software": "software"
        }
        category = type_map.get(product_type.lower())
        if category and category in PRODUCT_DATABASE:
            results = PRODUCT_DATABASE[category].copy()
    else:
        # Get all products
        for products in PRODUCT_DATABASE.values():
            results.extend(products)

    # Filter by brand
    if brand:
        results = [p for p in results if p['brand'].lower() == brand.lower()]

    # Filter by price
    if price_max:
        try:
            # Extract number from price string (e.g., "25 triệu" -> 25000000)
            if "triệu" in str(price_max):
                max_price = float(price_max.split()[0]) * 1000000
                results = [p for p in results if p['price'] <= max_price]
        except:
            pass

    # Filter by use case
    if use_case and 'type' in results[0]:
        use_case_map = {
            "gaming": "gaming",
            "chơi game": "gaming",
            "văn phòng": "office",
            "office": "office"
        }
        uc = use_case_map.get(use_case.lower())
        if uc:
            results = [p for p in results if p.get('type') == uc]

    return results


# ===== CUSTOM ACTIONS =====

class ActionSearchProduct(Action):
    """Search for products based on user criteria"""

    def name(self) -> Text:
        return "action_search_product"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        product_type = tracker.get_slot("product_type")
        brand = tracker.get_slot("brand")
        price_range = tracker.get_slot("price_range")
        use_case = tracker.get_slot("use_case")

        logger.info(f"Searching products: type={product_type}, brand={brand}, price={price_range}, use_case={use_case}")

        results = search_products(product_type, brand, price_range, use_case)

        if not results:
            dispatcher.utter_message(text="Xin lỗi, tôi không tìm thấy sản phẩm phù hợp. Bạn có thể thử tiêu chí khác?")
            return []

        # Limit to top 3 results
        results = results[:3]

        message = f"🔍 Tìm thấy {len(results)} sản phẩm phù hợp:\n\n"

        for i, product in enumerate(results, 1):
            message += f"**{i}. {product['name']}**\n"
            message += f"💰 {format_price(product['price'])}\n"
            message += f"{'✅ Còn hàng' if product['stock'] else '❌ Hết hàng'}\n"
            if product['promotion']:
                message += f"🎁 {product['promotion']}\n"
            message += "\n"

        message += "Bạn muốn xem chi tiết sản phẩm nào?"

        dispatcher.utter_message(text=message)

        return []


class ActionCompareProducts(Action):
    """Compare two or more products"""

    def name(self) -> Text:
        return "action_compare_products"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get recent entities or search results
        product_name = tracker.get_slot("product_name")
        product_type = tracker.get_slot("product_type")

        if product_type:
            results = search_products(product_type)[:2]
        else:
            # Default comparison
            results = PRODUCT_DATABASE['laptop'][:2]

        if len(results) < 2:
            dispatcher.utter_message(text="Vui lòng cho tôi biết 2 sản phẩm bạn muốn so sánh.")
            return []

        p1, p2 = results[0], results[1]

        message = f"""⚖️ **So Sánh: {p1['name']} vs {p2['name']}**

**💰 Giá:**
• {p1['name']}: {format_price(p1['price'])}
• {p2['name']}: {format_price(p2['price'])}

**🔧 Thông Số:**
{p1['name']}:
"""
        for k, v in p1['specs'].items():
            message += f"• {k.replace('_', ' ').title()}: {v}\n"

        message += f"\n{p2['name']}:\n"
        for k, v in p2['specs'].items():
            message += f"• {k.replace('_', ' ').title()}: {v}\n"

        message += f"""
**🛡️ Bảo Hành:**
• {p1['name']}: {p1['warranty']}
• {p2['name']}: {p2['warranty']}

**💡 Nhận Xét:**
"""

        if p1['price'] < p2['price']:
            message += f"• {p1['name']} rẻ hơn {format_price(p2['price'] - p1['price'])}\n"
        else:
            message += f"• {p2['name']} rẻ hơn {format_price(p1['price'] - p2['price'])}\n"

        message += "\nBạn cần thêm thông tin gì?"

        dispatcher.utter_message(text=message)

        return []


class ActionShowPrice(Action):
    """Show price and availability"""

    def name(self) -> Text:
        return "action_show_price"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        product_type = tracker.get_slot("product_type")
        product_name = tracker.get_slot("product_name")

        results = search_products(product_type)

        if not results:
            dispatcher.utter_message(text="Vui lòng cho tôi biết sản phẩm bạn muốn hỏi giá.")
            return []

        message = "💰 **Bảng Giá:**\n\n"

        for product in results[:5]:
            message += f"• {product['name']}: {format_price(product['price'])}"
            if product['promotion']:
                message += f" 🎁 {product['promotion']}"
            message += "\n"

        message += "\nGiá đã bao gồm VAT. Liên hệ để được báo giá chi tiết!"

        dispatcher.utter_message(text=message)

        return []


class ActionShowSpecifications(Action):
    """Show detailed product specifications"""

    def name(self) -> Text:
        return "action_show_specifications"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        product_type = tracker.get_slot("product_type")
        results = search_products(product_type)

        if not results:
            dispatcher.utter_message(text="Vui lòng cho tôi biết sản phẩm bạn muốn xem thông số.")
            return []

        product = results[0]
        message = format_product_card(product)

        dispatcher.utter_message(text=message)

        return []


class ActionCheckStock(Action):
    """Check product availability"""

    def name(self) -> Text:
        return "action_check_stock"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        product_type = tracker.get_slot("product_type")
        results = search_products(product_type)

        if not results:
            dispatcher.utter_message(text="Vui lòng cho tôi biết sản phẩm bạn muốn kiểm tra.")
            return []

        message = "📦 **Tình Trạng Kho:**\n\n"

        for product in results[:5]:
            status = "✅ Còn hàng" if product['stock'] else "❌ Hết hàng (liên hệ đặt trước)"
            message += f"• {product['name']}: {status}\n"

        dispatcher.utter_message(text=message)

        return []


class ActionShowWarranty(Action):
    """Show warranty information"""

    def name(self) -> Text:
        return "action_show_warranty"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        message = """🛡️ **Chính Sách Bảo Hành:**

**Laptop & PC:**
• Bảo hành 12-36 tháng tùy hãng
• Bảo hành tại trung tâm hoặc đổi mới trong 7 ngày
• Hỗ trợ kỹ thuật miễn phí

**Máy Chủ:**
• Bảo hành 36 tháng
• Bảo hành on-site (tại chỗ)
• Phản hồi trong 4-24h

**Camera & Switch:**
• Bảo hành 24 tháng - Lifetime
• Đổi mới trong 30 ngày
• Hỗ trợ kỹ thuật suốt đời

**Phần Mềm:**
• Hỗ trợ kỹ thuật miễn phí
• Cập nhật liên tục
• Đổi trả trong 7 ngày nếu lỗi

Liên hệ hotline: 1900-xxxx"""

        dispatcher.utter_message(text=message)

        return []


class ActionShowPromotions(Action):
    """Show current promotions"""

    def name(self) -> Text:
        return "action_show_promotions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        message = """🎁 **Chương Trình Khuyến Mãi Tháng Này:**

**💻 Laptop Gaming:**
• Giảm 10-15% ASUS ROG series
• Tặng chuột + balo gaming (trị giá 2 triệu)

**🖥️ Máy Chủ:**
• Giảm 8% đơn hàng từ 100 triệu
• Miễn phí setup và cài đặt
• Hỗ trợ vay 0% lãi suất

**📹 Camera:**
• Mua 10 tặng 2 camera cùng loại
• Miễn phí lắp đặt trong bán kính 50km
• Tặng HDD 1TB cho gói trọn bộ

**💿 Phần Mềm:**
• Giảm 10% khi mua từ 5 license trở lên
• Tặng 1 năm Microsoft 365 khi mua Windows Server

Liên hệ ngay để được tư vấn chi tiết!"""

        dispatcher.utter_message(text=message)

        return []


class ActionRecommendProduct(Action):
    """Recommend products based on use case"""

    def name(self) -> Text:
        return "action_recommend_product"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        use_case = tracker.get_slot("use_case")
        product_type = tracker.get_slot("product_type")

        # Default recommendations
        if "văn phòng" in str(use_case).lower() or "office" in str(use_case).lower():
            results = [p for p in PRODUCT_DATABASE['laptop'] if p.get('type') == 'office']
            message = "💼 **Gợi ý Laptop Văn Phòng:**\n\n"
        elif "gaming" in str(use_case).lower() or "chơi game" in str(use_case).lower():
            results = [p for p in PRODUCT_DATABASE['laptop'] if p.get('type') == 'gaming']
            message = "🎮 **Gợi ý Laptop Gaming:**\n\n"
        elif product_type == "máy chủ" or product_type == "server":
            results = PRODUCT_DATABASE['server'][:2]
            message = "🖥️ **Gợi ý Máy Chủ Doanh Nghiệp:**\n\n"
        else:
            results = PRODUCT_DATABASE['laptop'][:2]
            message = "💡 **Sản Phẩm Đề Xuất:**\n\n"

        for product in results:
            message += format_product_card(product) + "\n\n---\n\n"

        message += "Bạn muốn tìm hiểu thêm về sản phẩm nào?"

        dispatcher.utter_message(text=message)

        return []


class ActionProvideSupport(Action):
    """Provide technical support information"""

    def name(self) -> Text:
        return "action_provide_support"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        message = """🛠️ **Hỗ Trợ Kỹ Thuật:**

**📞 Liên Hệ:**
• Hotline: 1900-xxxx (24/7)
• Email: support@techstore.vn
• Chat: Qua website hoặc Zalo

**💻 Hỗ Trợ Từ Xa:**
• Cài đặt hệ điều hành
• Cấu hình server
• Setup camera & switch
• Khắc phục sự cố

**🏢 Hỗ Trợ Tại Chỗ:**
• Lắp đặt thiết bị
• Cấu hình hệ thống
• Đào tạo sử dụng
• Bảo trì định kỳ

**📚 Tài Liệu:**
• Hướng dẫn sử dụng chi tiết
• Video tutorial
• Knowledge base

Bạn cần hỗ trợ về vấn đề gì cụ thể?"""

        dispatcher.utter_message(text=message)

        return []


class ActionShowCategory(Action):
    """Show product categories"""

    def name(self) -> Text:
        return "action_show_category"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        message = """📁 **Danh Mục Sản Phẩm TechStore:**

**💻 Máy Tính & Laptop:**
• PC Desktop (Văn phòng, Gaming, Workstation)
• Laptop (Dell, HP, ASUS, MSI, Lenovo)
• Linh kiện máy tính

**🖥️ Máy Chủ (Server):**
• Dell PowerEdge
• HP ProLiant
• Lenovo ThinkSystem
• Tower & Rack Server

**🔌 Thiết Bị Mạng:**
• Switch (Cisco, HPE, D-Link)
• Router
• Firewall
• Access Point

**📹 Camera An Ninh:**
• IP Camera (Hikvision, Dahua)
• NVR & DVR
• Hệ thống giám sát

**💿 Phần Mềm:**
• Windows (11, Server)
• Microsoft Office
• Antivirus
• Software chuyên dụng

Bạn quan tâm danh mục nào?"""

        dispatcher.utter_message(text=message)

        return []

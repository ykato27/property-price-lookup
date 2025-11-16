"""
SUUMOスクレイパー
"""
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import hashlib

from src.scraper.base_scraper import BaseScraper


class SuumoScraper(BaseScraper):
    """SUUMOスクレイパー"""

    def get_site_name(self) -> str:
        """サイト名を取得"""
        return "SUUMO"

    def build_search_url(
        self, prefecture: str, city: Optional[str] = None, page: int = 1
    ) -> str:
        """
        検索URLを構築

        Note: これは簡易実装です。実際のSUUMOのURL構造に合わせて調整が必要です。
        """
        # 都道府県コードのマッピング（簡易版）
        prefecture_codes = {
            "東京都": "13",
            "神奈川県": "14",
            "埼玉県": "11",
            "千葉県": "12",
        }

        pref_code = prefecture_codes.get(prefecture, "13")

        # 中古マンションの検索URL（サンプル）
        base_url = "https://suumo.jp/jj/bukken/ichiran/JJ010FJ001/"

        # 実際のURLパラメータは要調整
        url = f"{base_url}?ar=030&bs=011&pc={pref_code}&pn={page}"

        return url

    def parse_property_list(self, html: str) -> List[str]:
        """
        物件一覧ページから物件詳細URLを抽出

        Note: 実際のSUUMOのHTML構造に合わせて調整が必要です。
        """
        soup = BeautifulSoup(html, "html.parser")
        urls = []

        # 物件リンクを抽出（実際のセレクタに調整が必要）
        property_links = soup.select("a[href*='/chuko/']")

        for link in property_links:
            href = link.get("href")
            if href and "bukken" in href:
                # 相対URLを絶対URLに変換
                if href.startswith("/"):
                    href = "https://suumo.jp" + href
                urls.append(href)

        # 重複を除去
        return list(set(urls))

    def parse_property_detail(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """
        物件詳細ページから情報を抽出

        Note: 実際のSUUMOのHTML構造に合わせて調整が必要です。
        """
        soup = BeautifulSoup(html, "html.parser")

        try:
            # property_idを生成（URLのハッシュ）
            property_id = hashlib.md5(url.encode()).hexdigest()

            # 基本情報を抽出（実際のセレクタに調整が必要）
            property_data = {
                "property_id": property_id,
                "url": url,
            }

            # 価格
            price_elem = soup.select_one(".price, .dottable-value")
            if price_elem:
                price_text = self.clean_text(price_elem.text)
                # 「万円」を含む場合は万円単位を円に変換
                if "万円" in price_text:
                    price = self.extract_number(price_text)
                    if price:
                        property_data["price"] = price * 10000
                else:
                    property_data["price"] = self.extract_number(price_text)

            # 住所
            address_elem = soup.select_one(".section_h1-header-title, address")
            if address_elem:
                address = self.clean_text(address_elem.text)
                property_data["address"] = address

                # 市区町村を抽出
                if address:
                    # 簡易的な抽出（要改善）
                    parts = address.split()
                    if len(parts) >= 2:
                        property_data["city"] = parts[1]

            # 専有面積
            area_elem = soup.select_one(
                'dt:contains("専有面積") + dd, th:contains("専有面積") + td'
            )
            if area_elem:
                area_text = self.clean_text(area_elem.text)
                property_data["floor_area"] = self.extract_float(area_text)

            # 築年数
            age_elem = soup.select_one(
                'dt:contains("築年") + dd, th:contains("築年") + td'
            )
            if age_elem:
                age_text = self.clean_text(age_elem.text)
                property_data["building_age"] = self.extract_number(age_text)

            # 間取り
            layout_elem = soup.select_one(
                'dt:contains("間取り") + dd, th:contains("間取り") + td'
            )
            if layout_elem:
                property_data["layout"] = self.clean_text(layout_elem.text)

            # 階数
            floor_elem = soup.select_one('dt:contains("階") + dd, th:contains("階") + td')
            if floor_elem:
                floor_text = self.clean_text(floor_elem.text)
                # 「3階/10階建」のような形式から抽出
                if "/" in floor_text:
                    parts = floor_text.split("/")
                    property_data["floor_number"] = self.extract_number(parts[0])
                    property_data["total_floors"] = self.extract_number(parts[1])
                else:
                    property_data["floor_number"] = self.extract_number(floor_text)

            # 構造
            structure_elem = soup.select_one(
                'dt:contains("構造") + dd, th:contains("構造") + td'
            )
            if structure_elem:
                property_data["structure"] = self.clean_text(structure_elem.text)

            # 最寄駅
            station_elem = soup.select_one(
                'dt:contains("交通") + dd, th:contains("交通") + td'
            )
            if station_elem:
                station_text = self.clean_text(station_elem.text)
                property_data["nearest_station"] = station_text

                # 徒歩分数を抽出
                property_data["station_distance"] = self.extract_number(station_text)

            # 向き
            direction_elem = soup.select_one(
                'dt:contains("向き") + dd, th:contains("バルコニー") + td'
            )
            if direction_elem:
                property_data["direction"] = self.clean_text(direction_elem.text)

            # 管理費
            mgmt_elem = soup.select_one(
                'dt:contains("管理費") + dd, th:contains("管理費") + td'
            )
            if mgmt_elem:
                mgmt_text = self.clean_text(mgmt_elem.text)
                property_data["management_fee"] = self.extract_number(mgmt_text)

            # 修繕積立金
            repair_elem = soup.select_one(
                'dt:contains("修繕積立金") + dd, th:contains("修繕積立金") + td'
            )
            if repair_elem:
                repair_text = self.clean_text(repair_elem.text)
                property_data["repair_reserve_fund"] = self.extract_number(repair_text)

            # 必須フィールドのチェック
            if "price" not in property_data:
                return None

            return property_data

        except Exception as e:
            print(f"パースエラー: {url} - {e}")
            return None


# デモ用のダミーデータ生成関数
def generate_dummy_properties(count: int = 10) -> List[Dict[str, Any]]:
    """
    テスト用のダミー物件データを生成

    Args:
        count: 生成する物件数

    Returns:
        ダミー物件データのリスト
    """
    import random
    import hashlib
    from datetime import datetime

    prefectures = ["東京都", "神奈川県", "埼玉県", "千葉県"]
    cities = {
        "東京都": ["渋谷区", "新宿区", "港区", "世田谷区", "目黒区"],
        "神奈川県": ["横浜市", "川崎市", "相模原市", "藤沢市"],
        "埼玉県": ["さいたま市", "川口市", "川越市", "所沢市"],
        "千葉県": ["千葉市", "市川市", "船橋市", "松戸市"],
    }
    layouts = ["1K", "1DK", "1LDK", "2DK", "2LDK", "3DK", "3LDK"]
    structures = ["RC造", "SRC造", "鉄骨造"]
    directions = ["南", "東", "西", "北", "南東", "南西"]

    properties = []

    for i in range(count):
        prefecture = random.choice(prefectures)
        city = random.choice(cities[prefecture])
        property_id = hashlib.md5(f"dummy_{i}_{datetime.now()}".encode()).hexdigest()

        property_data = {
            "property_id": property_id,
            "source_site": "SUUMO",
            "url": f"https://suumo.jp/dummy/{property_id}",
            "prefecture": prefecture,
            "city": city,
            "address": f"{prefecture}{city}{random.randint(1, 10)}-{random.randint(1, 20)}-{random.randint(1, 30)}",
            "price": random.randint(2000, 10000) * 10000,  # 2000万円〜1億円
            "building_age": random.randint(0, 40),
            "floor_area": round(random.uniform(30, 100), 2),
            "floor_number": random.randint(1, 15),
            "total_floors": random.randint(5, 20),
            "layout": random.choice(layouts),
            "structure": random.choice(structures),
            "nearest_station": f"{city}駅",
            "station_distance": random.randint(1, 15),
            "direction": random.choice(directions),
            "management_fee": random.randint(5000, 30000),
            "repair_reserve_fund": random.randint(3000, 20000),
        }

        properties.append(property_data)

    return properties

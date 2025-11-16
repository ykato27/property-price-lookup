"""
SUUMOスクレイパー
"""
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import hashlib
import json

from src.scraper.base_scraper import BaseScraper
from src.utils.image_downloader import ImageDownloader


class SuumoScraper(BaseScraper):
    """SUUMOスクレイパー（実際のサイト対応版）"""

    def __init__(self):
        super().__init__()
        self.image_downloader = ImageDownloader()

    def get_site_name(self) -> str:
        """サイト名を取得"""
        return "SUUMO"

    def build_search_url(
        self, prefecture: str, city: Optional[str] = None, page: int = 1
    ) -> str:
        """
        検索URLを構築（中古マンション）

        都道府県コード:
        - 東京都: 13
        - 神奈川県: 14
        - 埼玉県: 11
        - 千葉県: 12
        """
        prefecture_codes = {
            "東京都": "13",
            "神奈川県": "14",
            "埼玉県": "11",
            "千葉県": "12",
        }

        pref_code = prefecture_codes.get(prefecture, "13")

        # 中古マンションの検索URL
        # ar=030: 関東エリア
        # bs=011: 中古マンション
        # pc=XX: 都道府県コード
        # page=X: ページ番号
        base_url = "https://suumo.jp/jj/bukken/ichiran/JJ010FJ001/"
        url = f"{base_url}?ar=030&bs=011&pc={pref_code}&page={page}"

        return url

    def parse_property_list(self, html: str) -> List[str]:
        """
        物件一覧ページから物件詳細URLを抽出

        SUUMOの物件一覧ページの構造:
        - 物件カードは <div class="property_unit"> または類似のクラス
        - 詳細リンクは <a> タグで href に物件詳細URLが含まれる
        """
        soup = BeautifulSoup(html, "html.parser")
        urls = []

        # 物件カードを取得（複数のパターンに対応）
        property_cards = soup.select(
            ".property_unit, .property-unit, .cassetteitem, article.property"
        )

        if not property_cards:
            # より一般的なパターンで検索
            property_cards = soup.find_all("div", class_=lambda x: x and "property" in x.lower())

        for card in property_cards:
            # 詳細リンクを抽出
            link = card.find("a", href=lambda x: x and "/chuko/" in x and "/bukken/" in x)

            if not link:
                # より広範囲で検索
                link = card.find("a", href=lambda x: x and "bukken" in x)

            if link:
                href = link.get("href")
                if href:
                    # 相対URLを絶対URLに変換
                    if href.startswith("/"):
                        href = "https://suumo.jp" + href
                    elif not href.startswith("http"):
                        href = "https://suumo.jp/" + href

                    urls.append(href)

        # 重複を除去
        return list(set(urls))

    def parse_property_detail(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """
        物件詳細ページから情報を抽出

        SUUMOの物件詳細ページの構造:
        - 価格: <div class="price"> や <span class="price">
        - 住所: <table> 内の th:contains("所在地") の隣の td
        - その他の情報: <table> のkey-valueペア
        """
        soup = BeautifulSoup(html, "html.parser")

        try:
            # property_idを生成（URLのハッシュ）
            property_id = hashlib.md5(url.encode()).hexdigest()

            property_data = {
                "property_id": property_id,
                "url": url,
            }

            # ========== 価格 ==========
            price_elem = soup.select_one(
                ".price, .dettable .price, .bukken_price, "
                "span.price, div.price, .detail_price"
            )

            if not price_elem:
                # tableから抽出を試みる
                price_th = soup.find("th", string=lambda x: x and "価格" in x)
                if price_th:
                    price_elem = price_th.find_next_sibling("td")

            if price_elem:
                price_text = self.clean_text(price_elem.text)
                price = self._parse_price(price_text)
                if price:
                    property_data["price"] = price
                else:
                    return None  # 価格が取得できない場合はスキップ
            else:
                return None

            # ========== 住所・エリア情報 ==========
            address_elem = soup.select_one(".section_h1-header-title, address, .bukken_address")

            if not address_elem:
                # tableから抽出
                address_th = soup.find("th", string=lambda x: x and ("所在地" in x or "住所" in x))
                if address_th:
                    address_elem = address_th.find_next_sibling("td")

            if address_elem:
                address = self.clean_text(address_elem.text)
                property_data["address"] = address

                # 市区町村を抽出
                city = self._extract_city(address)
                if city:
                    property_data["city"] = city

            # ========== テーブル情報を一括抽出 ==========
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    th = row.find("th")
                    td = row.find("td")

                    if not th or not td:
                        continue

                    label = self.clean_text(th.text)
                    value = self.clean_text(td.text)

                    if not label or not value:
                        continue

                    # 専有面積
                    if "面積" in label or "専有面積" in label:
                        property_data["floor_area"] = self.extract_float(value)

                    # 築年数
                    elif "築年" in label:
                        property_data["building_age"] = self.extract_number(value)

                    # 間取り
                    elif "間取り" in label:
                        property_data["layout"] = value

                    # 階数
                    elif "階" in label and "総階数" not in label:
                        # 「3階/10階建」のような形式から抽出
                        if "/" in value or "階建" in value:
                            parts = value.replace("階建", "").split("/")
                            if len(parts) >= 1:
                                property_data["floor_number"] = self.extract_number(parts[0])
                            if len(parts) >= 2:
                                property_data["total_floors"] = self.extract_number(parts[1])
                        else:
                            property_data["floor_number"] = self.extract_number(value)

                    # 構造
                    elif "構造" in label or "建物構造" in label:
                        property_data["structure"] = value

                    # 交通・最寄駅
                    elif "交通" in label or "最寄駅" in label or "アクセス" in label:
                        property_data["nearest_station"] = value
                        property_data["station_distance"] = self.extract_number(value)

                    # 向き
                    elif "向き" in label or "バルコニー" in label:
                        property_data["direction"] = value

                    # 管理費
                    elif "管理費" in label:
                        property_data["management_fee"] = self.extract_number(value)

                    # 修繕積立金
                    elif "修繕積立金" in label:
                        property_data["repair_reserve_fund"] = self.extract_number(value)

            # ========== 画像URL抽出 ==========
            image_urls = self._extract_image_urls(soup)
            if image_urls:
                property_data["image_urls"] = json.dumps(image_urls)

                # 画像をダウンロード
                local_paths = self.image_downloader.download_images(
                    image_urls, property_id, max_images=5
                )
                if local_paths:
                    property_data["local_image_paths"] = json.dumps(local_paths)

            # 必須フィールドのチェック
            if "price" not in property_data:
                return None

            return property_data

        except Exception as e:
            print(f"パースエラー: {url} - {e}")
            return None

    def _parse_price(self, price_text: str) -> Optional[int]:
        """
        価格テキストを解析して数値に変換

        Args:
            price_text: 価格テキスト（例: "3,500万円"、"1億2000万円"）

        Returns:
            価格（円）
        """
        if not price_text:
            return None

        try:
            # 「億円」を含む場合
            if "億" in price_text:
                # 「1億2000万円」→ 120000000
                parts = price_text.replace("円", "").split("億")
                oku = self.extract_float(parts[0])
                man = 0
                if len(parts) > 1 and parts[1]:
                    man = self.extract_float(parts[1].replace("万", ""))

                if oku is not None:
                    return int(oku * 100000000 + (man or 0) * 10000)

            # 「万円」を含む場合
            elif "万円" in price_text or "万" in price_text:
                man = self.extract_float(price_text.replace("万円", "").replace("万", ""))
                if man is not None:
                    return int(man * 10000)

            # 数値のみの場合
            else:
                price = self.extract_number(price_text)
                if price is not None:
                    return price

        except Exception:
            pass

        return None

    def _extract_city(self, address: str) -> Optional[str]:
        """
        住所から市区町村を抽出

        Args:
            address: 住所

        Returns:
            市区町村
        """
        if not address:
            return None

        # 東京都の場合
        if "東京都" in address:
            # 「東京都渋谷区〜」→「渋谷区」
            parts = address.split("東京都")
            if len(parts) > 1:
                remaining = parts[1].strip()
                # 区を抽出
                if "区" in remaining:
                    city = remaining.split("区")[0] + "区"
                    return city
                # 市を抽出
                elif "市" in remaining:
                    city = remaining.split("市")[0] + "市"
                    return city

        # 神奈川県・埼玉県・千葉県の場合
        for pref in ["神奈川県", "埼玉県", "千葉県"]:
            if pref in address:
                parts = address.split(pref)
                if len(parts) > 1:
                    remaining = parts[1].strip()
                    # 市を抽出
                    if "市" in remaining:
                        city = remaining.split("市")[0] + "市"
                        return city

        return None

    def _extract_image_urls(self, soup: BeautifulSoup) -> List[str]:
        """
        物件画像URLを抽出

        Args:
            soup: BeautifulSoupオブジェクト

        Returns:
            画像URLのリスト
        """
        image_urls = []

        # 物件画像を含む要素を検索
        # パターン1: imgタグから直接
        img_tags = soup.select(
            ".carousel img, .gallery img, .photo img, "
            ".bukken_photo img, .detail_photo img, "
            "div[class*='image'] img, div[class*='photo'] img"
        )

        for img in img_tags:
            src = img.get("src") or img.get("data-src") or img.get("data-original")
            if src and self._is_valid_image_url(src):
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    src = "https://suumo.jp" + src
                image_urls.append(src)

        # 重複を除去
        return list(set(image_urls))[:10]  # 最大10枚

    def _is_valid_image_url(self, url: str) -> bool:
        """
        有効な画像URLかチェック

        Args:
            url: URL

        Returns:
            有効ならTrue
        """
        if not url:
            return False

        # サムネイルやアイコンは除外
        exclude_keywords = ["icon", "btn", "logo", "banner", "thumb_s", "thumb_xs"]
        for keyword in exclude_keywords:
            if keyword in url.lower():
                return False

        # 画像拡張子を含むかチェック
        valid_extensions = [".jpg", ".jpeg", ".png", ".gif"]
        return any(ext in url.lower() for ext in valid_extensions)


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

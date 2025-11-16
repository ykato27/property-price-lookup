"""
楽天不動産スクレイパー
"""
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import hashlib
import json

from src.scraper.base_scraper import BaseScraper
from src.utils.image_downloader import ImageDownloader


class RakutenScraper(BaseScraper):
    """楽天不動産スクレイパー"""

    def __init__(self):
        super().__init__()
        self.image_downloader = ImageDownloader()

    def get_site_name(self) -> str:
        """サイト名を取得"""
        return "楽天不動産"

    def build_search_url(
        self, prefecture: str, city: Optional[str] = None, page: int = 1
    ) -> str:
        """
        検索URLを構築

        楽天不動産のURL構造:
        https://realestate.rakuten.co.jp/used/mansion/search/
        """
        prefecture_codes = {
            "東京都": "13",
            "神奈川県": "14",
            "埼玉県": "11",
            "千葉県": "12",
        }

        pref_code = prefecture_codes.get(prefecture, "13")

        # 中古マンション検索URL
        base_url = "https://realestate.rakuten.co.jp/used/mansion/search/"

        if page > 1:
            url = f"{base_url}?pref={pref_code}&page={page}"
        else:
            url = f"{base_url}?pref={pref_code}"

        return url

    def parse_property_list(self, html: str) -> List[str]:
        """物件一覧ページから物件詳細URLを抽出"""
        soup = BeautifulSoup(html, "html.parser")
        urls = []

        # 物件カードを取得
        property_cards = soup.select(
            ".property-item, .bukken-item, .mansion-item"
        )

        if not property_cards:
            property_cards = soup.find_all("div", class_=lambda x: x and ("item" in x.lower() or "property" in x.lower()))

        for card in property_cards:
            # 詳細リンクを抽出
            link = card.find("a", href=lambda x: x and ("/used/mansion/" in x or "/detail/" in x))

            if link:
                href = link.get("href")
                if href:
                    # 相対URLを絶対URLに変換
                    if href.startswith("/"):
                        href = "https://realestate.rakuten.co.jp" + href
                    elif not href.startswith("http"):
                        href = "https://realestate.rakuten.co.jp/" + href

                    urls.append(href)

        return list(set(urls))

    def parse_property_detail(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """物件詳細ページから情報を抽出"""
        soup = BeautifulSoup(html, "html.parser")

        try:
            property_id = hashlib.md5(url.encode()).hexdigest()

            property_data = {
                "property_id": property_id,
                "url": url,
            }

            # 価格
            price_elem = soup.select_one(
                ".price, .bukken-price, .property-price, "
                "span.price, div.price"
            )

            if not price_elem:
                price_th = soup.find("th", string=lambda x: x and "価格" in x)
                if price_th:
                    price_elem = price_th.find_next_sibling("td")

            if price_elem:
                price_text = self.clean_text(price_elem.text)
                price = self._parse_price(price_text)
                if price:
                    property_data["price"] = price
                else:
                    return None
            else:
                return None

            # 住所
            address_elem = soup.select_one(".address, .bukken-address, address")

            if not address_elem:
                address_th = soup.find("th", string=lambda x: x and ("所在地" in x or "住所" in x))
                if address_th:
                    address_elem = address_th.find_next_sibling("td")

            if address_elem:
                address = self.clean_text(address_elem.text)
                property_data["address"] = address

                city = self._extract_city(address)
                if city:
                    property_data["city"] = city

            # テーブル情報を一括抽出
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

                    if "面積" in label:
                        property_data["floor_area"] = self.extract_float(value)
                    elif "築年" in label:
                        property_data["building_age"] = self.extract_number(value)
                    elif "間取り" in label:
                        property_data["layout"] = value
                    elif "階" in label:
                        if "/" in value:
                            parts = value.split("/")
                            property_data["floor_number"] = self.extract_number(parts[0])
                            if len(parts) > 1:
                                property_data["total_floors"] = self.extract_number(parts[1])
                        else:
                            property_data["floor_number"] = self.extract_number(value)
                    elif "構造" in label:
                        property_data["structure"] = value
                    elif "交通" in label or "最寄駅" in label or "アクセス" in label:
                        property_data["nearest_station"] = value
                        property_data["station_distance"] = self.extract_number(value)
                    elif "向き" in label or "バルコニー" in label:
                        property_data["direction"] = value
                    elif "管理費" in label:
                        property_data["management_fee"] = self.extract_number(value)
                    elif "修繕積立金" in label:
                        property_data["repair_reserve_fund"] = self.extract_number(value)

            # dl/dt/dd形式も抽出
            dls = soup.find_all("dl")

            for dl in dls:
                dts = dl.find_all("dt")
                dds = dl.find_all("dd")

                for dt, dd in zip(dts, dds):
                    label = self.clean_text(dt.text)
                    value = self.clean_text(dd.text)

                    if not label or not value:
                        continue

                    if "面積" in label and "floor_area" not in property_data:
                        property_data["floor_area"] = self.extract_float(value)
                    elif "築年" in label and "building_age" not in property_data:
                        property_data["building_age"] = self.extract_number(value)
                    elif "間取り" in label and "layout" not in property_data:
                        property_data["layout"] = value

            # 画像URL抽出
            image_urls = self._extract_image_urls(soup)
            if image_urls:
                property_data["image_urls"] = json.dumps(image_urls)
                local_paths = self.image_downloader.download_images(
                    image_urls, property_id, max_images=5
                )
                if local_paths:
                    property_data["local_image_paths"] = json.dumps(local_paths)

            if "price" not in property_data:
                return None

            return property_data

        except Exception as e:
            print(f"パースエラー: {url} - {e}")
            return None

    def _parse_price(self, price_text: str) -> Optional[int]:
        """価格テキストを解析"""
        if not price_text:
            return None

        try:
            if "億" in price_text:
                parts = price_text.replace("円", "").split("億")
                oku = self.extract_float(parts[0])
                man = 0
                if len(parts) > 1 and parts[1]:
                    man = self.extract_float(parts[1].replace("万", ""))
                if oku is not None:
                    return int(oku * 100000000 + (man or 0) * 10000)
            elif "万円" in price_text or "万" in price_text:
                man = self.extract_float(price_text.replace("万円", "").replace("万", ""))
                if man is not None:
                    return int(man * 10000)
            else:
                price = self.extract_number(price_text)
                if price is not None:
                    return price
        except Exception:
            pass

        return None

    def _extract_city(self, address: str) -> Optional[str]:
        """住所から市区町村を抽出"""
        if not address:
            return None

        if "東京都" in address:
            parts = address.split("東京都")
            if len(parts) > 1:
                remaining = parts[1].strip()
                if "区" in remaining:
                    return remaining.split("区")[0] + "区"
                elif "市" in remaining:
                    return remaining.split("市")[0] + "市"

        for pref in ["神奈川県", "埼玉県", "千葉県"]:
            if pref in address:
                parts = address.split(pref)
                if len(parts) > 1:
                    remaining = parts[1].strip()
                    if "市" in remaining:
                        return remaining.split("市")[0] + "市"

        return None

    def _extract_image_urls(self, soup: BeautifulSoup) -> List[str]:
        """物件画像URLを抽出"""
        image_urls = []

        img_tags = soup.select(
            ".photo img, .gallery img, .slider img, "
            ".bukken-photo img, .property-photo img, "
            "div[class*='photo'] img, div[class*='image'] img"
        )

        for img in img_tags:
            src = img.get("src") or img.get("data-src") or img.get("data-original")
            if src and self._is_valid_image_url(src):
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    src = "https://realestate.rakuten.co.jp" + src
                image_urls.append(src)

        return list(set(image_urls))[:10]

    def _is_valid_image_url(self, url: str) -> bool:
        """有効な画像URLかチェック"""
        if not url:
            return False

        exclude_keywords = ["icon", "btn", "logo", "banner", "thumb_s"]
        for keyword in exclude_keywords:
            if keyword in url.lower():
                return False

        valid_extensions = [".jpg", ".jpeg", ".png", ".gif"]
        return any(ext in url.lower() for ext in valid_extensions)

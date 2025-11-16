"""
スクレイピング基底クラス
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import requests
from bs4 import BeautifulSoup

from config.settings import SCRAPING_SETTINGS


class BaseScraper(ABC):
    """スクレイピング基底クラス"""

    def __init__(self):
        """初期化"""
        self.user_agent = SCRAPING_SETTINGS["user_agent"]
        self.request_interval = SCRAPING_SETTINGS["request_interval"]
        self.timeout = SCRAPING_SETTINGS["timeout"]
        self.max_retries = SCRAPING_SETTINGS["max_retries"]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    @abstractmethod
    def get_site_name(self) -> str:
        """
        サイト名を取得

        Returns:
            サイト名
        """
        pass

    @abstractmethod
    def build_search_url(
        self, prefecture: str, city: Optional[str] = None, page: int = 1
    ) -> str:
        """
        検索URLを構築

        Args:
            prefecture: 都道府県
            city: 市区町村（オプション）
            page: ページ番号

        Returns:
            検索URL
        """
        pass

    @abstractmethod
    def parse_property_list(self, html: str) -> List[str]:
        """
        物件一覧ページから物件詳細URLを抽出

        Args:
            html: HTML文字列

        Returns:
            物件詳細URLのリスト
        """
        pass

    @abstractmethod
    def parse_property_detail(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """
        物件詳細ページから情報を抽出

        Args:
            html: HTML文字列
            url: 物件詳細URL

        Returns:
            物件情報辞書
        """
        pass

    def fetch_html(self, url: str, retries: int = 0) -> Optional[str]:
        """
        HTMLを取得

        Args:
            url: URL
            retries: リトライ回数

        Returns:
            HTML文字列（失敗時はNone）
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            time.sleep(self.request_interval)  # リクエスト間隔を空ける
            return response.text
        except requests.RequestException as e:
            if retries < self.max_retries:
                print(f"リトライ {retries + 1}/{self.max_retries}: {url}")
                time.sleep(self.request_interval * 2)  # リトライ時は間隔を長く
                return self.fetch_html(url, retries + 1)
            else:
                print(f"取得失敗: {url} - {e}")
                return None

    def scrape_properties(
        self, prefecture: str, city: Optional[str] = None, max_pages: int = 5
    ) -> List[Dict[str, Any]]:
        """
        物件情報をスクレイピング

        Args:
            prefecture: 都道府県
            city: 市区町村（オプション）
            max_pages: 最大ページ数

        Returns:
            物件情報のリスト
        """
        properties = []

        for page in range(1, max_pages + 1):
            print(f"ページ {page}/{max_pages} を取得中...")

            # 検索ページのHTML取得
            search_url = self.build_search_url(prefecture, city, page)
            html = self.fetch_html(search_url)

            if not html:
                print(f"ページ {page} の取得に失敗しました")
                continue

            # 物件詳細URLリストを取得
            detail_urls = self.parse_property_list(html)

            if not detail_urls:
                print(f"ページ {page} に物件が見つかりませんでした")
                break

            # 各物件の詳細を取得
            for detail_url in detail_urls:
                detail_html = self.fetch_html(detail_url)

                if not detail_html:
                    continue

                property_data = self.parse_property_detail(detail_html, detail_url)

                if property_data:
                    # 共通フィールドを追加
                    property_data["source_site"] = self.get_site_name()
                    property_data["prefecture"] = prefecture
                    if city:
                        property_data["city"] = city

                    properties.append(property_data)

            print(f"ページ {page}: {len(detail_urls)} 件取得完了")

        return properties

    @staticmethod
    def clean_text(text: Optional[str]) -> Optional[str]:
        """
        テキストをクリーニング

        Args:
            text: テキスト

        Returns:
            クリーニング済みテキスト
        """
        if not text:
            return None
        return text.strip().replace("\n", "").replace("\t", "").replace("\u3000", "")

    @staticmethod
    def extract_number(text: Optional[str]) -> Optional[int]:
        """
        テキストから数値を抽出

        Args:
            text: テキスト

        Returns:
            数値（失敗時はNone）
        """
        if not text:
            return None

        import re

        numbers = re.findall(r"\d+", text.replace(",", ""))
        if numbers:
            return int(numbers[0])
        return None

    @staticmethod
    def extract_float(text: Optional[str]) -> Optional[float]:
        """
        テキストから浮動小数点数を抽出

        Args:
            text: テキスト

        Returns:
            浮動小数点数（失敗時はNone）
        """
        if not text:
            return None

        import re

        match = re.search(r"\d+\.?\d*", text.replace(",", ""))
        if match:
            return float(match.group())
        return None

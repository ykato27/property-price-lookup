"""
画像ダウンロード機能
"""
import os
import hashlib
import requests
from pathlib import Path
from typing import List, Optional
import time

from config.settings import DATA_DIR, SCRAPING_SETTINGS


class ImageDownloader:
    """画像ダウンロードクラス"""

    def __init__(self, image_dir: Optional[Path] = None):
        """
        初期化

        Args:
            image_dir: 画像保存ディレクトリ
        """
        self.image_dir = image_dir or DATA_DIR / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": SCRAPING_SETTINGS["user_agent"]}
        )

    def download_image(self, image_url: str, property_id: str) -> Optional[str]:
        """
        画像を1枚ダウンロード

        Args:
            image_url: 画像URL
            property_id: 物件ID

        Returns:
            ローカル保存パス（失敗時はNone）
        """
        try:
            # URLからファイル名を生成
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
            ext = self._get_extension(image_url)
            filename = f"{property_id}_{url_hash}{ext}"
            filepath = self.image_dir / filename

            # 既に存在する場合はスキップ
            if filepath.exists():
                return str(filepath)

            # 画像をダウンロード
            response = self.session.get(
                image_url, timeout=SCRAPING_SETTINGS["timeout"]
            )
            response.raise_for_status()

            # ファイルに保存
            with open(filepath, "wb") as f:
                f.write(response.content)

            # リクエスト間隔を空ける
            time.sleep(1)

            return str(filepath)

        except Exception as e:
            print(f"画像ダウンロード失敗: {image_url} - {e}")
            return None

    def download_images(
        self, image_urls: List[str], property_id: str, max_images: int = 5
    ) -> List[str]:
        """
        複数の画像をダウンロード

        Args:
            image_urls: 画像URLのリスト
            property_id: 物件ID
            max_images: 最大ダウンロード枚数

        Returns:
            ローカル保存パスのリスト
        """
        local_paths = []

        for i, image_url in enumerate(image_urls[:max_images]):
            local_path = self.download_image(image_url, property_id)
            if local_path:
                local_paths.append(local_path)

        return local_paths

    @staticmethod
    def _get_extension(url: str) -> str:
        """
        URLから拡張子を取得

        Args:
            url: URL

        Returns:
            拡張子（例: .jpg）
        """
        # URLから拡張子を抽出
        if ".jpg" in url.lower() or ".jpeg" in url.lower():
            return ".jpg"
        elif ".png" in url.lower():
            return ".png"
        elif ".gif" in url.lower():
            return ".gif"
        else:
            return ".jpg"  # デフォルト

    def get_image_path(self, property_id: str) -> Optional[str]:
        """
        物件IDから最初の画像パスを取得

        Args:
            property_id: 物件ID

        Returns:
            画像パス（存在しない場合はNone）
        """
        # 物件IDで始まるファイルを検索
        for filepath in self.image_dir.glob(f"{property_id}_*"):
            return str(filepath)
        return None

    def get_all_image_paths(self, property_id: str) -> List[str]:
        """
        物件IDから全ての画像パスを取得

        Args:
            property_id: 物件ID

        Returns:
            画像パスのリスト
        """
        return [str(f) for f in self.image_dir.glob(f"{property_id}_*")]

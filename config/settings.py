"""
システム設定ファイル
"""
import os
from pathlib import Path

# プロジェクトルートディレクトリ
BASE_DIR = Path(__file__).parent.parent

# データディレクトリ
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"

# データベースパス
DB_PATH = DATA_DIR / "properties.db"

# モデルパス
LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# スクレイピング設定
SCRAPING_SETTINGS = {
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "request_interval": 3,  # リクエスト間隔（秒）
    "timeout": 30,  # タイムアウト（秒）
    "max_retries": 3,  # 最大リトライ回数
}

# 対象エリア
TARGET_PREFECTURES = ["東京都", "神奈川県", "埼玉県", "千葉県"]

# 対象サイト
SUPPORTED_SITES = {
    "SUUMO": "https://suumo.jp/",
    "athome": "https://www.athome.co.jp/",
    "HOMES": "https://www.homes.co.jp/",
    "楽天不動産": "https://realestate.rakuten.co.jp/",
}

# 機械学習設定
ML_SETTINGS = {
    "test_size": 0.2,  # テストデータの割合
    "random_state": 42,  # 乱数シード
    "min_data_count": 100,  # 学習に必要な最小データ数
}

# データ保持期間（日数）
DATA_RETENTION_DAYS = 365

# ディレクトリ作成
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

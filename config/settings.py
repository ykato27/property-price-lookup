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

# 理論価格モデル（成約データベース）のパス
PRICE_MODEL_PATH = MODEL_DIR / "price_model.pkl"
PRICE_MODEL_METADATA_PATH = MODEL_DIR / "price_model_metadata.json"

# API等から取得した生データの保存先
RAW_DATA_DIR = DATA_DIR / "raw"

# 国土交通省「不動産情報ライブラリ」API（XIT001）設定
# 利用には申請・審査・APIキー発行が必要。キーは環境変数 REINFOLIB_API_KEY で渡す。
# 申請手順は docs/DATA_SOURCES.md を参照。
REINFOLIB_API_BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external"
REINFOLIB_API_KEY_ENV = "REINFOLIB_API_KEY"


def get_reinfolib_api_key() -> str | None:
    """環境変数から不動産情報ライブラリのAPIキーを取得する（未設定なら None）。"""
    key = os.environ.get(REINFOLIB_API_KEY_ENV, "").strip()
    return key or None

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
RAW_DATA_DIR.mkdir(exist_ok=True)

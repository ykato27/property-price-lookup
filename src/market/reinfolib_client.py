"""
国土交通省「不動産情報ライブラリ」API（XIT001）のクライアント。

- エンドポイント: https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001
- 認証: リクエストヘッダ `Ocp-Apim-Subscription-Key` にAPIキー。
  キーは利用申請・審査のうえ発行される（手順は docs/DATA_SOURCES.md）。
  本モジュールはキーを環境変数 REINFOLIB_API_KEY から読む。
- 応答は gzip 圧縮された JSON（requests が自動で伸長する）。

2026-10-23 に本環境から到達確認済み: キー無しリクエストには 401 が返る
（＝エンドポイントは実在し、キーさえあれば取得できる）。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from config.settings import RAW_DATA_DIR, REINFOLIB_API_BASE_URL, get_reinfolib_api_key

API_REQUEST_GUIDE = (
    "不動産情報ライブラリのAPIキーが未設定です。"
    "利用申請（https://www.reinfolib.mlit.go.jp/api/request/ ）でキーを取得し、"
    "環境変数 REINFOLIB_API_KEY に設定してください。手順は docs/DATA_SOURCES.md を参照。"
)


class ReinfolibAuthError(RuntimeError):
    """APIキーが未設定、またはキーで認証できなかった場合の例外。"""


class ReinfolibClient:
    """XIT001（不動産価格（取引価格・成約価格）情報取得API）のクライアント。"""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_key = api_key if api_key is not None else get_reinfolib_api_key()
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ReinfolibAuthError(API_REQUEST_GUIDE)
        return {"Ocp-Apim-Subscription-Key": self.api_key}

    def fetch_transactions(
        self,
        year: int,
        quarter: int,
        area: Optional[str] = None,
        city: Optional[str] = None,
        station: Optional[str] = None,
        price_classification: Optional[str] = None,
        language: str = "ja",
        save_raw: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        XIT001 で取引価格・成約価格情報を取得する。

        Args:
            year: 取引時期（年）。取引価格は2005年〜、成約価格は2021年〜
            quarter: 四半期（1〜4）
            area: 都道府県コード（2桁）。city / station のいずれか1つは必須
            city: 市区町村コード（5桁）
            station: 駅コード（6桁）
            price_classification: "01"=取引価格のみ, "02"=成約価格のみ, None=両方
            language: "ja" / "en"
            save_raw: 生JSONを data/raw/ に日付付きで保存するか

        Returns:
            生レコードのリスト（英語キー）
        """
        params: Dict[str, Any] = {"year": year, "quarter": quarter, "language": language}
        if area:
            params["area"] = area
        if city:
            params["city"] = city
        if station:
            params["station"] = station
        if price_classification:
            params["priceClassification"] = price_classification

        response = requests.get(
            f"{REINFOLIB_API_BASE_URL}/XIT001",
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if response.status_code in (401, 403):
            raise ReinfolibAuthError(
                f"不動産情報ライブラリAPIの認証に失敗しました（HTTP {response.status_code}）。"
                "APIキーの有効性を確認してください。手順は docs/DATA_SOURCES.md を参照。"
            )
        response.raise_for_status()

        data = response.json()
        records = data if isinstance(data, list) else data.get("data", [])

        if save_raw:
            self._save_raw(records, params)
        return records

    def _save_raw(self, records: List[Dict[str, Any]], params: Dict[str, Any]) -> Path:
        """取得した生データを日付付きファイルで保存する（再現性のため）。"""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scope = params.get("city") or params.get("area") or params.get("station") or "all"
        path = RAW_DATA_DIR / f"xit001_{params['year']}q{params['quarter']}_{scope}_{stamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        return path

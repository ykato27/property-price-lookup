"""
成約事例の取り込み（API / ファイル → transactions テーブル）。

ファイル形式:
- .json : XIT001 応答そのもの（レコードの配列）または {"data": [...]}
- .csv  : 公式ダウンロードファイル（日本語ヘッダ。UTF-8 / Shift_JIS 両対応）
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.database.db_manager import DatabaseManager
from src.market.normalize import is_condo, normalize_records
from src.market.reinfolib_client import ReinfolibClient


def load_records_from_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """ファイルから生レコードを読み込む。"""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", [])
        return list(data)
    if path.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "cp932"):
            try:
                df = pd.read_csv(path, encoding=encoding, dtype=str)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"CSVの文字コードを判定できませんでした: {path}")
        return df.fillna("").to_dict("records")
    raise ValueError(f"未対応のファイル形式です: {path.suffix}（.json / .csv のみ）")


def import_records(
    db: DatabaseManager,
    raw_records: List[Dict[str, Any]],
    source_hint: Optional[str] = None,
    condo_only: bool = True,
) -> Dict[str, int]:
    """
    生レコードを正規化して transactions へ取り込む。

    Args:
        condo_only: True なら「中古マンション等」のみ取り込む

    Returns:
        {"received": 受領件数, "normalized": 正規化成功, "inserted": 実挿入（重複除く）}
    """
    normalized = normalize_records(raw_records, source_hint=source_hint)
    if condo_only:
        normalized = [r for r in normalized if is_condo(r)]
    inserted = db.bulk_insert_transactions(normalized)
    return {
        "received": len(raw_records),
        "normalized": len(normalized),
        "inserted": inserted,
    }


def import_file(
    db: DatabaseManager,
    path: Union[str, Path],
    source_hint: Optional[str] = None,
    condo_only: bool = True,
) -> Dict[str, int]:
    """ファイルから成約事例を取り込む。"""
    return import_records(
        db, load_records_from_file(path), source_hint=source_hint, condo_only=condo_only
    )


def import_from_api(
    db: DatabaseManager,
    year: int,
    quarter: int,
    area: Optional[str] = None,
    city: Optional[str] = None,
    price_classification: Optional[str] = None,
    client: Optional[ReinfolibClient] = None,
    condo_only: bool = True,
) -> Dict[str, int]:
    """API（XIT001）から取得してそのまま取り込む。キーは環境変数から読む。"""
    client = client or ReinfolibClient()
    records = client.fetch_transactions(
        year=year,
        quarter=quarter,
        area=area,
        city=city,
        price_classification=price_classification,
    )
    return import_records(db, records, condo_only=condo_only)

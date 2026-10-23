"""
募集物件（listing）の投入: CSV / 手入力フォーム共通の取り込み処理。

ポータルサイトに個人向け公式APIは無く、スクレイピングを中核に置く設計は
規約・運用面で脆い（調査レポート §2.4）。そのため listing は
「利用者が手元で入手した情報」を CSV またはフォームで投入するのを正規の
入力とする。取得日（listing_date）と出所（source_site）は必須:
古い listing は陳腐化するため、鮮度を追跡できないデータは受け付けない。
"""
import hashlib
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from src.database.db_manager import DatabaseManager

# CSVヘッダ（日本語/英語）→ properties カラム
COLUMN_ALIASES = {
    "募集価格": "price", "価格": "price", "販売価格": "price", "price": "price",
    "都道府県": "prefecture", "prefecture": "prefecture",
    "市区町村": "city", "city": "city",
    "住所": "address", "所在地": "address", "address": "address",
    "面積": "floor_area", "専有面積": "floor_area", "floor_area": "floor_area",
    "間取り": "layout", "layout": "layout",
    "建築年": "building_year", "building_year": "building_year",
    "築年": "building_age", "築年数": "building_age", "building_age": "building_age",
    "所在階": "floor_number", "floor_number": "floor_number",
    "総階数": "total_floors", "建物階数": "total_floors", "total_floors": "total_floors",
    "総戸数": "total_units", "total_units": "total_units",
    "構造": "structure", "structure": "structure",
    "最寄駅": "nearest_station", "nearest_station": "nearest_station",
    "駅距離": "station_distance", "駅徒歩": "station_distance",
    "駅距離（分）": "station_distance", "station_distance": "station_distance",
    "向き": "direction", "direction": "direction",
    "管理費": "management_fee", "management_fee": "management_fee",
    "修繕積立金": "repair_reserve_fund", "repair_reserve_fund": "repair_reserve_fund",
    "権利形態": "rights_type", "rights_type": "rights_type",
    "賃貸中": "is_tenanted", "オーナーチェンジ": "is_tenanted", "is_tenanted": "is_tenanted",
    "月額賃料": "monthly_rent", "現賃料": "monthly_rent", "monthly_rent": "monthly_rent",
    "取得日": "listing_date", "掲載日": "listing_date", "listing_date": "listing_date",
    "出所": "source_site", "掲載元": "source_site", "source": "source_site",
    "source_site": "source_site",
    "URL": "url", "url": "url",
}

REQUIRED_FIELDS = ["price", "prefecture", "city", "floor_area", "listing_date", "source_site"]

_INT_FIELDS = [
    "price", "building_year", "building_age", "floor_number", "total_floors",
    "total_units", "station_distance", "management_fee", "repair_reserve_fund",
    "monthly_rent",
]


def _to_int(value: Any) -> int | None:
    """数値化する。価格は円単位が前提（万円単位のCSVは事前に円へ直すこと）。"""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        return int(float(str(value).replace(",", "").replace("円", "")))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_tenanted(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip() in ("1", "有", "はい", "賃貸中", "オーナーチェンジ", "true", "True") else 0
    return 1 if value else 0


def normalize_listing_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, List[str]]:
    """
    1行の listing 入力を properties レコードに正規化する。

    Returns:
        (レコード or None, エラーメッセージのリスト)
    """
    rec: Dict[str, Any] = {}
    for key, value in row.items():
        col = COLUMN_ALIASES.get(str(key).strip())
        if col:
            rec[col] = value

    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        v = rec.get(field)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            errors.append(f"必須項目がありません: {field}")
    if errors:
        return None, errors

    out: Dict[str, Any] = {
        "source_site": str(rec["source_site"]).strip(),
        "url": str(rec.get("url") or "").strip(),
        "prefecture": str(rec["prefecture"]).strip(),
        "city": str(rec["city"]).strip(),
        "address": (str(rec["address"]).strip() if rec.get("address") else None),
        "price": _to_int(rec["price"]),
        "floor_area": _to_float(rec["floor_area"]),
        "layout": (str(rec["layout"]).strip() if rec.get("layout") else None),
        "structure": (str(rec["structure"]).strip() if rec.get("structure") else None),
        "nearest_station": (str(rec["nearest_station"]).strip() if rec.get("nearest_station") else None),
        "direction": (str(rec["direction"]).strip() if rec.get("direction") else None),
        "rights_type": (str(rec["rights_type"]).strip() if rec.get("rights_type") else None),
        "listing_date": str(rec["listing_date"]).strip()[:10],
        "is_tenanted": _parse_tenanted(rec.get("is_tenanted")),
    }
    for field in _INT_FIELDS:
        if field not in ("price",):
            out[field] = _to_int(rec.get(field))
    if out["price"] is None or out["floor_area"] is None:
        return None, ["価格または面積を数値として解釈できませんでした"]

    # 建築年と築年の相互補完（築年は listing_date の年からの近似）
    try:
        listing_year = int(out["listing_date"][:4])
    except (TypeError, ValueError):
        listing_year = date.today().year
    if out.get("building_year") and not out.get("building_age"):
        out["building_age"] = max(0, listing_year - out["building_year"])
    if out.get("building_age") is not None and not out.get("building_year"):
        out["building_year"] = listing_year - out["building_age"]

    # 再投入しても同一物件として扱えるよう、内容から決定的にIDを作る
    key = "|".join(
        str(out.get(k)) for k in
        ("source_site", "url", "prefecture", "city", "address", "price",
         "floor_area", "building_year", "layout", "listing_date")
    )
    out["property_id"] = "lst_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return out, []


def import_listings(
    db: DatabaseManager, rows: List[Dict[str, Any]]
) -> Tuple[int, List[str]]:
    """
    listing 行のリストを取り込む。

    Returns:
        (取り込み成功件数, 行番号つきエラーのリスト)
    """
    success = 0
    errors: List[str] = []
    for i, row in enumerate(rows, start=1):
        record, row_errors = normalize_listing_row(row)
        if row_errors:
            errors.extend(f"{i}行目: {e}" for e in row_errors)
            continue
        if db.insert_property(record):
            success += 1
        else:
            errors.append(f"{i}行目: DBへの保存に失敗しました")
    return success, errors


def import_listings_csv(
    db: DatabaseManager, path: Union[str, Path]
) -> Tuple[int, List[str]]:
    """CSVファイルから listing を取り込む。"""
    df = pd.read_csv(path, dtype=str).fillna("")
    return import_listings(db, df.to_dict("records"))

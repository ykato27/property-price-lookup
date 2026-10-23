"""
不動産情報ライブラリ API（XIT001）応答・公式ダウンロードファイルの正規化。

入力形式:
- XIT001 の JSON（英語キー。値は language パラメータ依存で日本語/英語）
- 旧・不動産取引価格情報ダウンロード CSV（日本語ヘッダ、Shift_JIS のことが多い）

出力は transactions テーブルにそのまま入る辞書のリスト。
最寄駅・駅距離は旧形式にはあるが、現行 XIT001 の操作説明の出力一覧には
明記がないため、存在すれば使う任意項目として扱う（docs/DATA_SOURCES.md）。
"""
import hashlib
import re
from typing import Any, Dict, List, Optional

# 価格情報区分 → source コード
_PRICE_CATEGORY_TO_SOURCE = {
    "不動産取引価格情報": "trade",
    "成約価格情報": "contract",
}

# 日本語CSVヘッダ → API英語キーへの対応（公式ダウンロードファイル用）
_JA_HEADER_MAP = {
    "種類": "Type",
    "地域": "Region",
    "市区町村コード": "MunicipalityCode",
    "都道府県名": "Prefecture",
    "市区町村名": "Municipality",
    "地区名": "DistrictName",
    "最寄駅：名称": "NearestStation",
    "最寄駅:名称": "NearestStation",
    "最寄駅名称": "NearestStation",
    "最寄駅：距離（分）": "TimeToNearestStation",
    "最寄駅:距離（分）": "TimeToNearestStation",
    "最寄駅：距離(分)": "TimeToNearestStation",
    "最寄駅距離": "TimeToNearestStation",
    "取引価格（総額）": "TradePrice",
    "取引価格(総額)": "TradePrice",
    "間取り": "FloorPlan",
    "面積（㎡）": "Area",
    "面積(㎡)": "Area",
    "面積": "Area",
    "取引価格（㎡単価）": "UnitPrice",
    "取引価格(㎡単価)": "UnitPrice",
    "㎡単価": "UnitPrice",
    "建築年": "BuildingYear",
    "建物の構造": "Structure",
    "構造": "Structure",
    "用途": "Use",
    "今後の利用目的": "Purpose",
    "建ぺい率（％）": "CoverageRatio",
    "建ぺい率(％)": "CoverageRatio",
    "建蔽率（％）": "CoverageRatio",
    "容積率（％）": "FloorAreaRatio",
    "容積率(％)": "FloorAreaRatio",
    "都市計画": "CityPlanning",
    "用途地域": "CityPlanning",
    "取引時点": "Period",
    "取引時期": "Period",
    "改装": "Renovation",
    "改装の有無": "Renovation",
    "取引の事情等": "Remarks",
    "価格情報区分": "PriceCategory",
}


def parse_building_year(value: Any) -> Optional[int]:
    """建築年表記（"1972年" 等）から西暦年を取り出す。取れなければ None。"""
    if value is None:
        return None
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(value))
    return int(m.group(1)) if m else None


def parse_period(value: Any) -> Optional[tuple]:
    """取引時期表記（"2015年第2四半期"）から (年, 四半期) を取り出す。"""
    if value is None:
        return None
    m = re.search(r"(20\d{2}|19\d{2})\D+([1-4])", str(value))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_station_distance(value: Any) -> Optional[float]:
    """
    駅距離表記を「分」の代表値に変換する。

    旧形式の範囲表記（"30分～60分", "1時間～1時間30分", "2時間以上"）は
    中央値（"2時間以上" は下限の120分）で代表させる。数値ならそのまま。
    変換できない場合は None。
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()

    def to_minutes(token: str) -> Optional[float]:
        token = token.strip()
        m = re.fullmatch(r"(\d+(?:\.\d+)?)時間(\d+(?:\.\d+)?)?分?", token)
        if m:
            return float(m.group(1)) * 60 + float(m.group(2) or 0)
        m = re.fullmatch(r"(\d+(?:\.\d+)?)分", token)
        if m:
            return float(m.group(1))
        m = re.fullmatch(r"(\d+(?:\.\d+)?)", token)
        if m:
            return float(m.group(1))
        return None

    if "～" in s or "~" in s:
        parts = re.split(r"[～~]", s)
        if len(parts) == 2:
            lo, hi = to_minutes(parts[0]), to_minutes(parts[1])
            if lo is not None and hi is not None:
                return (lo + hi) / 2
    if s.endswith("以上"):
        return to_minutes(s[:-2])
    return to_minutes(s)


def _to_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    f = _to_float(value)
    return int(f) if f is not None else None


def translate_japanese_headers(record: Dict[str, Any]) -> Dict[str, Any]:
    """日本語ヘッダのレコードを API 英語キーへ寄せる（既知のキーのみ）。"""
    return {_JA_HEADER_MAP.get(k, k): v for k, v in record.items()}


def normalize_record(record: Dict[str, Any], source_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    1件の生レコードを transactions 用に正規化する。

    必須項目（市区町村コード・取引価格・面積・取引時期）が欠ける場合は None。
    """
    rec = translate_japanese_headers(record)

    period = parse_period(rec.get("Period"))
    trade_price = _to_int(rec.get("TradePrice"))
    area = _to_float(rec.get("Area"))
    municipality_code = str(rec.get("MunicipalityCode") or "").strip()
    if period is None or trade_price is None or not area or not municipality_code:
        return None

    unit_price = _to_float(rec.get("UnitPrice"))
    if unit_price is None:
        unit_price = trade_price / area

    price_category = str(rec.get("PriceCategory") or "").strip() or None
    source = (
        _PRICE_CATEGORY_TO_SOURCE.get(price_category or "")
        or source_hint
        or "trade"
    )

    normalized = {
        "source": source,
        "price_category": price_category,
        "type": (str(rec.get("Type")).strip() if rec.get("Type") else None),
        "region": (str(rec.get("Region")).strip() if rec.get("Region") else None),
        "municipality_code": municipality_code,
        "prefecture": str(rec.get("Prefecture") or "").strip(),
        "municipality": str(rec.get("Municipality") or "").strip(),
        "district_name": (str(rec.get("DistrictName")).strip() if rec.get("DistrictName") else None),
        "nearest_station": (str(rec.get("NearestStation")).strip() if rec.get("NearestStation") else None),
        "station_distance_min": parse_station_distance(rec.get("TimeToNearestStation")),
        "trade_price": trade_price,
        "floor_plan": (str(rec.get("FloorPlan")).strip() if rec.get("FloorPlan") else None),
        "area": area,
        "unit_price": unit_price,
        "building_year": parse_building_year(rec.get("BuildingYear")),
        "structure": (str(rec.get("Structure")).strip() if rec.get("Structure") else None),
        "use": (str(rec.get("Use")).strip() if rec.get("Use") else None),
        "purpose": (str(rec.get("Purpose")).strip() if rec.get("Purpose") else None),
        "coverage_ratio": _to_float(rec.get("CoverageRatio")),
        "floor_area_ratio": _to_float(rec.get("FloorAreaRatio")),
        "city_planning": (str(rec.get("CityPlanning")).strip() if rec.get("CityPlanning") else None),
        "renovation": (str(rec.get("Renovation")).strip() if rec.get("Renovation") else None),
        "remarks": (str(rec.get("Remarks")).strip() if rec.get("Remarks") else None),
        "period_year": period[0],
        "period_quarter": period[1],
    }
    normalized["record_hash"] = _record_hash(normalized)
    return normalized


def _record_hash(rec: Dict[str, Any]) -> str:
    """重複取り込み防止用のハッシュ。全項目の組合せから決定的に作る。"""
    keys = [
        "source", "municipality_code", "district_name", "trade_price",
        "floor_plan", "area", "building_year", "period_year", "period_quarter",
        "nearest_station", "structure", "renovation",
    ]
    joined = "|".join(str(rec.get(k)) for k in keys)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def normalize_records(
    records: List[Dict[str, Any]], source_hint: Optional[str] = None
) -> List[Dict[str, Any]]:
    """生レコードのリストを正規化する（正規化できない行は落とす）。"""
    out = []
    for r in records:
        n = normalize_record(r, source_hint=source_hint)
        if n is not None:
            out.append(n)
    return out


def is_condo(rec: Dict[str, Any]) -> bool:
    """中古マンション等のレコードかどうか。"""
    t = rec.get("type") or ""
    return "マンション" in t

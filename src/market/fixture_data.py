"""
テスト・デモ専用の「成約事例フィクスチャ」生成。

不動産情報ライブラリ API は利用申請・APIキーが必要で、キー未発行の環境では
実データを取得できない。その間の代替として、XIT001 の実項目形式に沿った
構造化ダミー成約データを決定的に生成する。

注意:
- これは実データではない。価格水準は東京23区の公開相場観に寄せてあるが、
  検証値（MAPE等）を実データの性能と誤解させないため、生成物を使う箇所では
  必ず「フィクスチャ」と明示すること。
- 実データが取り込めたら、この生成器はテスト専用に戻す。
"""
from typing import Any, Dict, List

import numpy as np

# 区ごとの基準㎡単価（万円/㎡、2021年時点のおおまかな水準）
_WARD_BASE_UNIT_PRICE = {
    "千代田区": 165.0, "中央区": 145.0, "港区": 175.0, "新宿区": 125.0,
    "文京区": 115.0, "台東区": 92.0, "墨田区": 78.0, "江東区": 92.0,
    "品川区": 118.0, "目黒区": 122.0, "大田区": 82.0, "世田谷区": 92.0,
    "渋谷区": 148.0, "中野区": 92.0, "杉並区": 88.0, "豊島区": 98.0,
    "北区": 68.0, "荒川区": 72.0, "板橋区": 63.0, "練馬区": 68.0,
    "足立区": 53.0, "葛飾区": 48.0, "江戸川区": 58.0,
}

_WARD_CODES = {
    "千代田区": "13101", "中央区": "13102", "港区": "13103", "新宿区": "13104",
    "文京区": "13105", "台東区": "13106", "墨田区": "13107", "江東区": "13108",
    "品川区": "13109", "目黒区": "13110", "大田区": "13111", "世田谷区": "13112",
    "渋谷区": "13113", "中野区": "13114", "杉並区": "13115", "豊島区": "13116",
    "北区": "13117", "荒川区": "13118", "板橋区": "13119", "練馬区": "13120",
    "足立区": "13121", "葛飾区": "13122", "江戸川区": "13123",
}

_LAYOUT_BY_AREA = [
    (30, "1K"), (40, "1LDK"), (55, "2LDK"), (70, "3LDK"), (1e9, "4LDK"),
]


def _layout_for(area: float, rng: np.random.Generator) -> str:
    for threshold, layout in _LAYOUT_BY_AREA:
        if area < threshold:
            return layout
    return "3LDK"


def generate_fixture_transactions(
    n: int = 1500,
    seed: int = 42,
    start: tuple = (2021, 1),
    end: tuple = (2025, 4),
) -> List[Dict[str, Any]]:
    """
    XIT001 形式（日本語の値）の成約事例フィクスチャを生成する。

    価格生成: 区の基準単価 × 築年減価 × 駅距離係数 × 時点トレンド × 対数正規ノイズ。
    ノイズ σ=0.13 は「公開データのモデル誤差は約±10%」という先行研究の水準に合わせ、
    楽観的すぎる検証値が出ないようにしている。
    """
    rng = np.random.default_rng(seed)
    wards = list(_WARD_BASE_UNIT_PRICE.keys())

    quarters = []
    y, q = start
    while (y, q) <= end:
        quarters.append((y, q))
        q += 1
        if q > 4:
            y, q = y + 1, 1

    records: List[Dict[str, Any]] = []
    for i in range(n):
        ward = wards[rng.integers(0, len(wards))]
        base = _WARD_BASE_UNIT_PRICE[ward]
        year, quarter = quarters[rng.integers(0, len(quarters))]
        building_year = int(rng.integers(1970, 2025))
        age = max(0, year - building_year)
        area = float(np.clip(rng.normal(62, 18), 22, 120))
        station_min = float(np.clip(rng.gamma(shape=2.2, scale=3.6), 1, 30))

        age_factor = max(0.45, 1.12 - 0.016 * age)
        station_factor = max(0.72, 1.0 - 0.009 * station_min)
        trend_factor = 1.02 ** (year - 2021 + (quarter - 1) / 4)
        noise = float(rng.lognormal(mean=0.0, sigma=0.13))
        unit_price_man = base * age_factor * station_factor * trend_factor * noise
        trade_price = int(round(unit_price_man * 10000 * area / 10000.0)) * 10000

        records.append(
            {
                "Type": "中古マンション等",
                "Region": "住宅地",
                "MunicipalityCode": _WARD_CODES[ward],
                "Prefecture": "東京都",
                "Municipality": ward,
                "DistrictName": f"{ward[:-1]}中央",
                "NearestStation": f"{ward[:-1]}駅",
                "TimeToNearestStation": f"{int(station_min)}分",
                "TradePrice": str(trade_price),
                "FloorPlan": _layout_for(area, rng),
                "Area": f"{area:.2f}",
                "UnitPrice": str(int(unit_price_man * 10000)),
                "BuildingYear": f"{building_year}年",
                "Structure": "ＲＣ",
                "CityPlanning": "第一種住居地域",
                "Period": f"{year}年第{quarter}四半期",
                "Renovation": "改装済" if rng.random() < 0.18 else "",
                "PriceCategory": "成約価格情報",
            }
        )
    return records

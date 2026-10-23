"""
第1層: 類似成約事例ベースの理論価格（取引事例比較法の自動化）。

同じ市区町村の中で、築年・面積（・駅距離）が近い成約事例をプールし、
時点補正した㎡単価の中央値と四分位を返す。事例数 n と最新事例の時期を
必ず併記し、後段のゲート・信頼度判定に使う。

公開データにはマンション名・所在階が無いため「同一マンション比較」は
できず、地区＋駅圏＋築年＋面積の類似比較が精度の天井になる
（調査レポート §1.2 / §2.1）。
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.market.time_adjust import UnitPriceIndex, period_ordinal


@dataclass
class ComparableStats:
    n: int
    median_unit_price: float  # 円/㎡（時点補正後）
    q25_unit_price: float
    q75_unit_price: float
    target_area: float
    latest_period: Optional[Tuple[int, int]]
    has_recent: bool  # 評価時点から8四半期以内（≒直近2年）の事例を含むか
    pool_level: str  # どの段階のプールで確定したか

    @property
    def median_price(self) -> float:
        return self.median_unit_price * self.target_area

    @property
    def q25_price(self) -> float:
        return self.q25_unit_price * self.target_area

    @property
    def q75_price(self) -> float:
        return self.q75_unit_price * self.target_area


def _passes_age(row_year, target_year, tol) -> bool:
    if target_year is None or row_year is None or (isinstance(row_year, float) and np.isnan(row_year)):
        return True
    return abs(int(row_year) - int(target_year)) <= tol


def find_comparables(
    transactions: pd.DataFrame,
    *,
    municipality_code: Optional[str] = None,
    municipality: Optional[str] = None,
    prefecture: Optional[str] = None,
    area: float,
    building_year: Optional[int] = None,
    valuation_period: Tuple[int, int] = (2025, 4),
    index: Optional[UnitPriceIndex] = None,
    min_pool: int = 10,
) -> Optional[ComparableStats]:
    """
    類似成約プールを段階的に広げながら探し、四分位統計を返す。

    プール拡大の順序（各段で min_pool 件以上あれば確定）:
      1. 同一市区町村 × 築年±7年 × 面積±15㎡
      2. 同一市区町村 × 面積±15㎡
      3. 同一市区町村 全件
      4. 同一都道府県 全件（信頼度は下がる）
    どこにも無ければ None。
    """
    if transactions is None or transactions.empty:
        return None
    index = index or UnitPriceIndex.from_transactions(transactions)
    df = transactions

    if municipality_code:
        same_city = df[df["municipality_code"].astype(str) == str(municipality_code)]
    elif municipality:
        same_city = df[df["municipality"] == municipality]
    else:
        same_city = df.iloc[0:0]
    stages = []
    if building_year is not None and not same_city.empty:
        age_ok = same_city["building_year"].apply(
            lambda y: _passes_age(y, building_year, 7)
        )
        area_ok = (same_city["area"] - area).abs() <= 15
        stages.append(("市区町村×築年±7年×面積±15㎡", same_city[age_ok & area_ok]))
        stages.append(("市区町村×面積±15㎡", same_city[area_ok]))
    stages.append(("市区町村全件", same_city))
    if prefecture:
        stages.append(
            ("都道府県全件", df[df["prefecture"] == prefecture])
        )

    chosen = None
    chosen_level = ""
    for level, pool in stages:
        if len(pool) >= min_pool:
            chosen, chosen_level = pool, level
            break
    if chosen is None:
        # 最も広いプールでも件数が足りない場合は、あるだけ使う（n は正直に返す）
        level, pool = stages[-1]
        if len(pool) == 0:
            return None
        chosen, chosen_level = pool, level

    ordinals = [
        period_ordinal(y, q)
        for y, q in zip(chosen["period_year"], chosen["period_quarter"])
    ]
    unit_prices = chosen["unit_price"].astype(float).to_numpy()
    adjusted = np.array(
        [
            index.adjust(
                up,
                (int(y), int(q)),
                valuation_period,
                group=prefecture,
            )
            for up, (y, q) in zip(
                unit_prices, zip(chosen["period_year"], chosen["period_quarter"])
            )
        ]
    )
    latest_ord = max(ordinals)
    latest_period = (latest_ord // 4, latest_ord % 4 + 1)
    has_recent = period_ordinal(*valuation_period) - latest_ord <= 8

    return ComparableStats(
        n=int(len(chosen)),
        median_unit_price=float(np.median(adjusted)),
        q25_unit_price=float(np.quantile(adjusted, 0.25)),
        q75_unit_price=float(np.quantile(adjusted, 0.75)),
        target_area=float(area),
        latest_period=latest_period,
        has_recent=bool(has_recent),
        pool_level=chosen_level,
    )

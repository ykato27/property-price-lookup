"""
時点補正: 古い成約事例の価格を評価時点の水準に引き直す。

本来は国交省「不動産価格指数」を使うのが正統だが、指数そのものの自動取得は
別途手当が必要なため、v1 では蓄積した成約データ自体から作る簡易指数
（都道府県×四半期の㎡単価中央値の比）を使う。簡易指数は標本の構成変化を
含むため厳密な価格指数ではない。その限界は docs/DESIGN.md に明記した。
外部指数CSVが用意できれば差し替え可能な構造にしている。
"""
from typing import Dict, Optional, Tuple

import pandas as pd


def period_ordinal(year: int, quarter: int) -> int:
    """(年, 四半期) を単調な序数に変換する。"""
    return int(year) * 4 + (int(quarter) - 1)


def period_from_ordinal(ordinal: int) -> Tuple[int, int]:
    return ordinal // 4, ordinal % 4 + 1


class UnitPriceIndex:
    """四半期ごとの㎡単価中央値から作る簡易価格指数。"""

    def __init__(self, by_group: Dict[str, Dict[int, float]], global_series: Dict[int, float]):
        self.by_group = by_group
        self.global_series = global_series

    @classmethod
    def from_transactions(
        cls, df: pd.DataFrame, group_col: str = "prefecture", min_count: int = 5
    ) -> "UnitPriceIndex":
        """成約データから指数を構築する。件数が少ない四半期はグループ指数から省く。"""
        if df.empty:
            return cls({}, {})
        d = df.copy()
        d["ordinal"] = [
            period_ordinal(y, q) for y, q in zip(d["period_year"], d["period_quarter"])
        ]
        global_series = (
            d.groupby("ordinal")["unit_price"].median().dropna().to_dict()
        )
        by_group: Dict[str, Dict[int, float]] = {}
        for group, sub in d.groupby(group_col):
            med = sub.groupby("ordinal")["unit_price"].agg(["median", "count"])
            med = med[med["count"] >= min_count]["median"].dropna()
            if len(med) >= 2:
                by_group[str(group)] = med.to_dict()
        return cls(by_group, {int(k): float(v) for k, v in global_series.items()})

    def _nearest(self, series: Dict[int, float], ordinal: int) -> Optional[float]:
        if not series:
            return None
        if ordinal in series:
            return series[ordinal]
        nearest = min(series.keys(), key=lambda k: abs(k - ordinal))
        return series[nearest]

    def factor(
        self,
        from_period: Tuple[int, int],
        to_period: Tuple[int, int],
        group: Optional[str] = None,
    ) -> float:
        """
        from_period の価格を to_period 水準へ引き直す倍率を返す。
        指数が作れない場合は 1.0（補正なし）。
        """
        series = None
        if group is not None:
            series = self.by_group.get(str(group))
        if not series:
            series = self.global_series
        if not series:
            return 1.0
        base_from = self._nearest(series, period_ordinal(*from_period))
        base_to = self._nearest(series, period_ordinal(*to_period))
        if not base_from or not base_to:
            return 1.0
        return base_to / base_from

    def adjust(
        self,
        unit_price: float,
        from_period: Tuple[int, int],
        to_period: Tuple[int, int],
        group: Optional[str] = None,
    ) -> float:
        """㎡単価を時点補正する。"""
        return unit_price * self.factor(from_period, to_period, group)

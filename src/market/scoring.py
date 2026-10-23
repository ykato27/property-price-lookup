"""
割安スコア・ゲート条件・リスクフラグ。

定義（調査レポート「システム設計への提言」に準拠）:
- 割安スコア = (理論価格の保守値 − 募集価格) / 理論価格の保守値
  保守値は「第1層（類似成約）の25%点」と「第2層（ML）の25%点」を
  事例数 n に応じてブレンドしたもの。中央値ではなく25%点を使うのは、
  モデルの誤差（MAPE約9〜12%）を割り引いて偽陽性を抑えるため。
- 「割安候補」と表示してよいのはゲート条件を全て満たす場合のみ:
    * 類似成約 n >= 10
    * 直近2年以内の成約事例を含む
    * スコア >= 15%
    * 予測区間の幅（75%点/25%点）<= 1.6
    * ハードフィルタ（旧耐震の近似判定・定期借地権）に抵触しない
- 信頼度（n・区間幅・事例の新しさ）は必ず併記する。スコア単独で
  購入判断に使わせない。
"""
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.market.comparables import ComparableStats, find_comparables
from src.market.price_model import PriceModel
from src.market.time_adjust import UnitPriceIndex

GATE_MIN_SIMILAR = 10
GATE_MIN_DEVIATION = 0.15
GATE_MAX_INTERVAL_RATIO = 1.6

# 修繕積立金のガイドライン目安（国交省ガイドラインの例示水準、円/㎡・月）
RESERVE_GUIDELINE_PER_SQM = 335  # 平均水準の例
RESERVE_SHORTAGE_PER_SQM = 150   # これを下回れば「目安を大幅に下回る」としてフラグ

SMALL_SCALE_UNITS = 30
STALE_LISTING_DAYS = 90


@dataclass
class RiskFlag:
    code: str
    label: str
    severity: str  # "hard"（ゲート除外） / "warn"（要注意表示）
    detail: str


@dataclass
class Valuation:
    property_id: str
    listing_price: int
    theoretical_price: Optional[float]  # ブレンド中央値ベースの総額
    conservative_price: Optional[float]  # ブレンド25%点ベースの総額（スコアの物差し）
    interval_low: Optional[float]
    interval_high: Optional[float]
    score: Optional[float]  # 乖離率（保守値基準、0-1）
    similar_count: int
    has_recent: bool
    pool_level: str
    model_used: bool
    gate_passed: bool
    gate_reasons: List[str] = field(default_factory=list)
    flags: List[RiskFlag] = field(default_factory=list)
    confidence: str = "低"
    confidence_detail: str = ""

    def to_prediction_record(self, model_version: str) -> Dict[str, Any]:
        """predictions テーブルへの保存形式に変換する。"""
        theoretical = int(self.theoretical_price) if self.theoretical_price else 0
        return {
            "property_id": self.property_id,
            "predicted_price": theoretical,
            "actual_price": int(self.listing_price),
            "price_difference": theoretical - int(self.listing_price),
            # 互換のため discount_rate には保守値基準のスコア（%）を入れる
            "discount_rate": round(self.score * 100, 2) if self.score is not None else 0.0,
            "model_version": model_version,
            "theoretical_price": theoretical or None,
            "conservative_price": int(self.conservative_price) if self.conservative_price else None,
            "interval_low": int(self.interval_low) if self.interval_low else None,
            "interval_high": int(self.interval_high) if self.interval_high else None,
            "similar_count": self.similar_count,
            "confidence": self.confidence,
            "risk_flags": json.dumps(
                [
                    {"code": f.code, "label": f.label, "severity": f.severity, "detail": f.detail}
                    for f in self.flags
                ],
                ensure_ascii=False,
            ),
            "gate_passed": 1 if self.gate_passed else 0,
            "score": self.score,
        }


def assess_risk_flags(prop: Dict[str, Any], valuation_date: date) -> List[RiskFlag]:
    """物件属性からリスクフラグを立てる。判定できない項目はフラグにしない。"""
    flags: List[RiskFlag] = []

    building_year = prop.get("building_year")
    if building_year:
        if int(building_year) <= 1980:
            flags.append(RiskFlag(
                code="old_seismic",
                label="旧耐震の可能性",
                severity="hard",
                detail=(
                    f"建築年 {int(building_year)} 年。1981年6月1日の新耐震基準の"
                    "建築確認より前の可能性が高い（築年からの近似判定）。"
                    "耐震診断・改修の有無を必ず確認すること。"
                ),
            ))
        elif int(building_year) == 1981:
            flags.append(RiskFlag(
                code="seismic_boundary",
                label="旧耐震の境界年",
                severity="warn",
                detail="建築年が1981年。建築確認日が6月1日より前かで新旧が分かれるため要確認。",
            ))

    rights = str(prop.get("rights_type") or "")
    if "借地" in rights:
        flags.append(RiskFlag(
            code="leasehold",
            label="借地権（定期の可能性）",
            severity="hard",
            detail=(
                f"権利形態: {rights}。定期借地権は期間満了で更地返還・建物解体が前提で、"
                "資産価値がゼロに向かう。地代・残存期間・解体積立金を確認すること。"
            ),
        ))

    if prop.get("is_tenanted"):
        flags.append(RiskFlag(
            code="tenanted",
            label="賃貸中（オーナーチェンジ）",
            severity="warn",
            detail=(
                "賃貸中のため内見で室内状態を確認できない。現賃料が相場より高い場合、"
                "退去・賃料改定で利回りが崩れる。賃貸借契約書の確認が前提。"
            ),
        ))

    reserve = prop.get("repair_reserve_fund")
    area = prop.get("floor_area")
    if reserve and area:
        per_sqm = float(reserve) / float(area)
        if per_sqm < RESERVE_SHORTAGE_PER_SQM:
            flags.append(RiskFlag(
                code="reserve_shortage",
                label="修繕積立金が目安を大幅に下回る",
                severity="warn",
                detail=(
                    f"修繕積立金 {per_sqm:,.0f}円/㎡・月。国交省ガイドラインの目安"
                    f"（平均約{RESERVE_GUIDELINE_PER_SQM}円/㎡・月）を大幅に下回る。"
                    "将来の増額・一時金の先送りの可能性がある。長期修繕計画と残高を確認すること。"
                ),
            ))

    total_units = prop.get("total_units")
    if total_units and int(total_units) <= SMALL_SCALE_UNITS:
        flags.append(RiskFlag(
            code="small_scale",
            label="小規模マンション",
            severity="warn",
            detail=(
                f"総戸数 {int(total_units)} 戸。小規模棟は1戸あたりの修繕負担が重く、"
                "長期修繕計画の整備率も低い傾向がある。"
            ),
        ))

    listing_date = prop.get("listing_date")
    if listing_date:
        try:
            days = (valuation_date - date.fromisoformat(str(listing_date)[:10])).days
            if days > STALE_LISTING_DAYS:
                flags.append(RiskFlag(
                    code="stale_listing",
                    label="listingが古い",
                    severity="warn",
                    detail=f"取得日から {days} 日経過。既に成約・価格改定されている可能性がある。",
                ))
        except ValueError:
            pass
    return flags


def resolve_municipality_code(
    transactions: pd.DataFrame, prefecture: Optional[str], city: Optional[str]
) -> Optional[str]:
    """成約データ側の (都道府県, 市区町村名) から市区町村コードを引く。"""
    if transactions is None or transactions.empty or not city:
        return None
    sub = transactions[transactions["municipality"] == city]
    if prefecture:
        sub_pref = sub[sub["prefecture"] == prefecture]
        if not sub_pref.empty:
            sub = sub_pref
    if sub.empty:
        return None
    return str(sub["municipality_code"].iloc[0])


def valuate_property(
    prop: Dict[str, Any],
    transactions: pd.DataFrame,
    model: Optional[PriceModel] = None,
    index: Optional[UnitPriceIndex] = None,
    valuation_period: Tuple[int, int] = (2025, 4),
    valuation_date: Optional[date] = None,
) -> Valuation:
    """1物件を評価する（第1層＋第2層のブレンド、ゲート判定、フラグ付け）。"""
    valuation_date = valuation_date or date.today()
    area = float(prop["floor_area"])
    listing_price = int(prop["price"])
    building_year = prop.get("building_year")
    if building_year is None and prop.get("building_age") is not None:
        building_year = valuation_period[0] - int(prop["building_age"])

    code = resolve_municipality_code(transactions, prop.get("prefecture"), prop.get("city"))
    stats: Optional[ComparableStats] = find_comparables(
        transactions,
        municipality_code=code,
        municipality=prop.get("city"),
        prefecture=prop.get("prefecture"),
        area=area,
        building_year=int(building_year) if building_year else None,
        valuation_period=valuation_period,
        index=index,
    )

    # 第2層（ML）: 物件を行形式にして分位点を取る
    ml_q = None
    if model is not None:
        row = pd.DataFrame(
            [
                {
                    "prefecture": prop.get("prefecture"),
                    "municipality": prop.get("city"),
                    "district_name": None,
                    "structure": prop.get("structure"),
                    "city_planning": None,
                    "region": None,
                    "floor_plan": prop.get("layout"),
                    "area": area,
                    "building_year": building_year,
                    "period_year": valuation_period[0],
                    "period_quarter": valuation_period[1],
                    "station_distance_min": prop.get("station_distance"),
                }
            ]
        )
        ml_q = model.predict_unit_quantiles(row)[0] * area

    n = stats.n if stats else 0
    # ブレンド重み: 事例が多いほど事例側（第1層）を信じる
    w = n / (n + 20) if stats else 0.0
    if ml_q is None:
        w = 1.0 if stats else 0.0

    def blend(l1_value, ml_value):
        if stats is None:
            return float(ml_value)
        if ml_value is None:
            return float(l1_value)
        return w * float(l1_value) + (1 - w) * float(ml_value)

    if stats is None and ml_q is None:
        theoretical = conservative = low = high = None
        score = None
    else:
        theoretical = blend(stats.median_price if stats else None, ml_q[1] if ml_q is not None else None)
        conservative = blend(stats.q25_price if stats else None, ml_q[0] if ml_q is not None else None)
        low = blend(stats.q25_price if stats else None, ml_q[0] if ml_q is not None else None)
        high = blend(stats.q75_price if stats else None, ml_q[2] if ml_q is not None else None)
        score = (conservative - listing_price) / conservative if conservative else None

    flags = assess_risk_flags(prop, valuation_date)
    hard_flags = [f for f in flags if f.severity == "hard"]

    reasons: List[str] = []
    if n < GATE_MIN_SIMILAR:
        reasons.append(f"類似成約が不足（n={n} < {GATE_MIN_SIMILAR}）")
    if not (stats and stats.has_recent):
        reasons.append("直近2年以内の成約事例がない")
    if score is None or score < GATE_MIN_DEVIATION:
        reasons.append(
            f"乖離が閾値未満（スコア={score:.1%} < {GATE_MIN_DEVIATION:.0%}）"
            if score is not None else "理論価格を算出できない"
        )
    if low and high and high / max(low, 1) > GATE_MAX_INTERVAL_RATIO:
        reasons.append(f"予測区間が広すぎる（75%点/25%点={high / max(low, 1):.2f} > {GATE_MAX_INTERVAL_RATIO}）")
    if hard_flags:
        reasons.append("ハードフィルタ抵触: " + "、".join(f.label for f in hard_flags))

    gate_passed = not reasons

    # 信頼度: n・区間幅・事例の新しさから決める
    width = (high / max(low, 1)) if (low and high) else None
    if n >= 30 and width is not None and width <= 1.35 and stats and stats.has_recent:
        confidence = "高"
    elif n >= GATE_MIN_SIMILAR and width is not None and width <= GATE_MAX_INTERVAL_RATIO:
        confidence = "中"
    else:
        confidence = "低"
    detail = (
        f"類似成約 n={n}（プール: {stats.pool_level if stats else 'なし'}）、"
        f"区間幅 75%点/25%点={width:.2f}、" if width else f"類似成約 n={n}、"
    )
    detail += f"最新事例 {stats.latest_period[0]}年Q{stats.latest_period[1]}" if stats and stats.latest_period else "事例なし"

    return Valuation(
        property_id=str(prop["property_id"]),
        listing_price=listing_price,
        theoretical_price=theoretical,
        conservative_price=conservative,
        interval_low=low,
        interval_high=high,
        score=score,
        similar_count=n,
        has_recent=bool(stats and stats.has_recent),
        pool_level=stats.pool_level if stats else "なし",
        model_used=ml_q is not None,
        gate_passed=gate_passed,
        gate_reasons=reasons,
        flags=flags,
        confidence=confidence,
        confidence_detail=detail,
    )


def latest_period_of(transactions: pd.DataFrame) -> Tuple[int, int]:
    """成約データの最新四半期を評価時点とする。データが無ければ (2025, 4)。"""
    if transactions is None or transactions.empty:
        return (2025, 4)
    ordinals = [
        int(y) * 4 + (int(q) - 1)
        for y, q in zip(transactions["period_year"], transactions["period_quarter"])
    ]
    latest = max(ordinals)
    return latest // 4, latest % 4 + 1


def _clean_prop(prop: Dict[str, Any]) -> Dict[str, Any]:
    """DB由来の NaN を None にそろえる（判定ロジックを単純に保つため）。"""
    out = {}
    for k, v in prop.items():
        if isinstance(v, float) and np.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


def valuate_all_properties(
    db,
    model: Optional[PriceModel] = None,
    valuation_period: Optional[Tuple[int, int]] = None,
    model_version: str = "market-v1",
) -> List[Valuation]:
    """
    DB内の全 listing を評価し、predictions へ「最新」として保存する。
    再実行しても predictions 側で旧評価が is_latest=0 になるため、
    割安一覧は重複しない。
    """
    transactions = db.get_transactions()
    if transactions.empty:
        return []
    valuation_period = valuation_period or latest_period_of(transactions)
    index = UnitPriceIndex.from_transactions(transactions)
    properties = db.get_properties(limit=100000)

    results: List[Valuation] = []
    for _, row in properties.iterrows():
        prop = _clean_prop(row.to_dict())
        if prop.get("price") is None or prop.get("floor_area") is None:
            continue
        valuation = valuate_property(
            prop,
            transactions,
            model=model,
            index=index,
            valuation_period=valuation_period,
        )
        db.insert_prediction(valuation.to_prediction_record(model_version))
        results.append(valuation)
    return results

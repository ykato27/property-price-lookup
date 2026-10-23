"""
投資判断レイヤー（v1）: 表面利回りと NOI 試算。

原則:
- 賃料（月額賃料）が入力されている物件にのみ計算する。賃料が無い物件に
  推定利回りは出さない（公的賃料統計は市区町村集計で粗く、個別物件の
  利回りに使うと誤解を招くため。調査レポート §5.3）。
- 試算の仮定（空室率・賃貸管理手数料・固定資産税概算・購入諸経費率）は
  すべて明示し、呼び出し側で変更できるようにする。
- 割安スコアとは分離して表示する（価格の割安さと収益性は別の主張）。
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class InvestmentAssumptions:
    """NOI試算の仮定。値はすべて「仮定」であり、実績ではない。"""

    vacancy_rate: float = 0.05            # 空室率 5%
    management_commission_rate: float = 0.05  # 賃貸管理手数料 = 賃料の5%（実務目安、要確認の仮定）
    property_tax_rate: float = 0.004      # 固定資産税・都市計画税の概算 = 価格の0.4%/年
    acquisition_cost_rate: float = 0.07   # 購入時諸経費（仲介手数料・登記・税）= 価格の7%

    def describe(self) -> Dict[str, str]:
        return {
            "空室率": f"{self.vacancy_rate:.0%}",
            "賃貸管理手数料": f"賃料の{self.management_commission_rate:.0%}",
            "固定資産税・都市計画税（概算）": f"価格の{self.property_tax_rate:.1%}/年",
            "購入時諸経費": f"価格の{self.acquisition_cost_rate:.0%}",
        }


def analyze_investment(
    price: int,
    monthly_rent: Optional[int],
    management_fee: Optional[int] = None,
    repair_reserve_fund: Optional[int] = None,
    assumptions: Optional[InvestmentAssumptions] = None,
) -> Optional[Dict[str, Any]]:
    """
    利回り試算を行う。賃料が無ければ None を返す（推定利回りは出さない）。

    Returns:
        {
          "gross_yield": 表面利回り（年率, 0-1）,
          "annual_rent": 年間賃料（満室想定）,
          "noi": 年間NOI（空室・経費控除後）,
          "noi_yield": NOI利回り（取得総額ベース）,
          "annual_costs": {...内訳},
          "assumptions": {...仮定の説明},
        }
    """
    if not monthly_rent or not price:
        return None
    a = assumptions or InvestmentAssumptions()

    annual_rent = int(monthly_rent) * 12
    vacancy_loss = annual_rent * a.vacancy_rate
    commission = annual_rent * a.management_commission_rate
    running = (int(management_fee or 0) + int(repair_reserve_fund or 0)) * 12
    property_tax = price * a.property_tax_rate

    noi = annual_rent - vacancy_loss - commission - running - property_tax
    total_acquisition = price * (1 + a.acquisition_cost_rate)

    return {
        "gross_yield": annual_rent / price,
        "annual_rent": annual_rent,
        "noi": noi,
        "noi_yield": noi / total_acquisition,
        "total_acquisition": total_acquisition,
        "annual_costs": {
            "空室損": vacancy_loss,
            "賃貸管理手数料": commission,
            "管理費・修繕積立金": running,
            "固定資産税・都市計画税（概算）": property_tax,
        },
        "assumptions": a.describe(),
    }

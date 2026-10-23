"""
精査チェックリストの出力。

本システムの出力は「購入推奨」ではなく「精査の優先順位付け」である。
割安候補は、重要事項調査報告書・長期修繕計画・総会議事録・賃貸借契約書等の
一次資料で確認すべき項目とセットで初めて意味を持つため、候補ごとに
チェックリスト付き Markdown を書き出せるようにする。
"""
from typing import Any, Dict

from src.market.investment import analyze_investment
from src.market.scoring import Valuation

BASE_CHECKLIST = [
    "重要事項調査報告書を取得し、滞納・修繕積立金残高・管理組合の借入を確認した",
    "長期修繕計画書（計画期間・大規模修繕の時期と費用・積立金の増額予定）を確認した",
    "総会議事録（直近2〜3年分）で値上げ否決・紛争・大規模修繕の議論を確認した",
    "管理規約・使用細則（賃貸可否、民泊、ペット、事務所利用）を確認した",
    "売主の売却理由と価格改定履歴（値下げの有無・時期）を確認した",
    "室内状態（内見）: 水回り・給排水・雨漏り・結露・騒音を確認した",
    "ハザードマップ（洪水・土砂・津波）と用途地域・再建築可否を確認した",
    "登記簿で権利関係（抵当権・差押・借地権の条件）を確認した",
    "出口の想定: 同じ条件で自分が売る側になったときの買い手層を言語化した",
]

FLAG_CHECKLIST = {
    "old_seismic": "耐震診断の実施有無・耐震改修の実績と費用負担を確認した",
    "seismic_boundary": "建築確認日（1981年6月1日の前後）を建築確認済証で確認した",
    "leasehold": "借地権の残存期間・地代・更新料・解体積立金・譲渡承諾料を確認した",
    "tenanted": "賃貸借契約書（期間・賃料・敷金・特約・保証会社）と賃料の相場適合性を確認した",
    "reserve_shortage": "積立金残高と長期修繕計画の必要額との差額、一時金の可能性を確認した",
    "small_scale": "総戸数が少ない分、1戸あたり修繕負担と管理費水準を確認した",
    "stale_listing": "掲載が古いため、現在の販売状況・価格を掲載元に再確認した",
}


def build_checklist_markdown(prop: Dict[str, Any], valuation: Valuation) -> str:
    """割安候補1件分の精査シート（Markdown）を生成する。"""
    lines = [
        f"# 精査シート: {prop.get('prefecture', '')} {prop.get('city', '')} "
        f"{prop.get('address') or ''}",
        "",
        "> 本システムの出力は購入推奨ではありません。成約データに基づく"
        "「精査の優先順位付け」です。購入判断は一次資料の確認後に行ってください。",
        "",
        "## 物件概要",
        f"- 募集価格: {int(prop['price']):,} 円",
        f"- 専有面積: {prop.get('floor_area')} ㎡ / 間取り: {prop.get('layout') or '不明'}",
        f"- 建築年: {prop.get('building_year') or '不明'} / 所在階: {prop.get('floor_number') or '不明'}",
        f"- 最寄駅: {prop.get('nearest_station') or '不明'} 徒歩{prop.get('station_distance') or '不明'}分",
        f"- 管理費: {prop.get('management_fee') or '不明'} 円/月、"
        f"修繕積立金: {prop.get('repair_reserve_fund') or '不明'} 円/月",
        f"- 出所: {prop.get('source_site')} / 取得日: {prop.get('listing_date') or '不明'}",
        f"- URL: {prop.get('url') or 'なし'}",
        "",
        "## 理論価格と割安スコア",
    ]
    if valuation.theoretical_price:
        lines += [
            f"- 理論価格（中央値）: {valuation.theoretical_price:,.0f} 円",
            f"- 理論価格レンジ（25%点〜75%点）: "
            f"{valuation.interval_low:,.0f} 〜 {valuation.interval_high:,.0f} 円",
            f"- 保守値（25%点）: {valuation.conservative_price:,.0f} 円",
            f"- 割安スコア（保守値基準の乖離）: {valuation.score:.1%}",
        ]
    lines += [
        f"- 類似成約: n={valuation.similar_count}（プール: {valuation.pool_level}）",
        f"- 信頼度: {valuation.confidence}（{valuation.confidence_detail}）",
        f"- ゲート: {'通過（割安候補）' if valuation.gate_passed else '未通過'}"
        + (f" — {'; '.join(valuation.gate_reasons)}" if valuation.gate_reasons else ""),
        "",
        "## リスクフラグ",
    ]
    if valuation.flags:
        for f in valuation.flags:
            mark = "⛔ ハード" if f.severity == "hard" else "⚠️ 要注意"
            lines.append(f"- {mark} **{f.label}**: {f.detail}")
    else:
        lines.append("- （検出されたフラグはありません。一次資料での確認は依然として必要です）")

    inv = analyze_investment(
        int(prop["price"]),
        prop.get("monthly_rent"),
        prop.get("management_fee"),
        prop.get("repair_reserve_fund"),
    )
    lines += ["", "## 投資指標（賃料が入力されている場合のみ）"]
    if inv:
        lines += [
            f"- 表面利回り: {inv['gross_yield']:.2%}",
            f"- NOI（年間）: {inv['noi']:,.0f} 円",
            f"- NOI利回り（取得総額ベース）: {inv['noi_yield']:.2%}",
            "- 試算の仮定: " + "、".join(f"{k}={v}" for k, v in inv["assumptions"].items()),
        ]
    else:
        lines.append("- 賃料が未入力のため利回りは表示しません（推定利回りは出しません）。")

    lines += ["", "## 精査チェックリスト"]
    for item in BASE_CHECKLIST:
        lines.append(f"- [ ] {item}")
    for f in valuation.flags:
        extra = FLAG_CHECKLIST.get(f.code)
        if extra:
            lines.append(f"- [ ] {extra}")
    lines.append("")
    return "\n".join(lines)

"""
割安マンション検出＋投資判断支援アプリ

設計の要点（docs/DESIGN.md 参照）:
- 理論価格は国交省「不動産情報ライブラリ」の成約/取引データから作る
  （ポータルサイトのスクレイピングを中核に置かない）。
- 募集物件（listing）は CSV 投入＋手入力で取り込む。取得日・出所が必須。
- 「割安候補」はゲート条件（類似成約 n>=10、直近2年以内の事例、保守値への
  乖離15%以上、ハードフィルタ非抵触）を満たしたものだけ。
- 本アプリの出力は購入推奨ではなく、精査の優先順位付けである。
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import (
    PRICE_MODEL_METADATA_PATH,
    PRICE_MODEL_PATH,
    get_reinfolib_api_key,
)
from src.database.db_manager import DatabaseManager
from src.market.checklist import build_checklist_markdown
from src.market.importer import import_file, import_records
from src.market.investment import analyze_investment
from src.market.listings import import_listings, import_listings_csv
from src.market.price_model import PriceModel, train_price_model
from src.market.reinfolib_client import ReinfolibAuthError, ReinfolibClient
from src.market.scoring import RiskFlag, Valuation, valuate_all_properties

st.set_page_config(page_title="割安マンション検出", page_icon="🏢", layout="wide")

DISCLAIMER = (
    "⚠️ 本システムの出力は**購入推奨ではありません**。成約データに基づく理論価格との"
    "比較による「精査の優先順位付け」です。理論価格にはモデル誤差（期間分割評価で "
    "MAPE 約10%前後）があり、割安に見える物件の相当部分は旧耐震・積立不足・借地権などの"
    "リスクの裏返しです。購入判断は重要事項調査報告書・長期修繕計画・総会議事録などの"
    "一次資料を確認した上で行ってください。"
)


@st.cache_resource
def get_db() -> DatabaseManager:
    db = DatabaseManager()
    db.connect()
    db.initialize_database()
    return db


def load_price_model():
    if PRICE_MODEL_PATH.exists():
        try:
            return PriceModel.load(PRICE_MODEL_PATH)
        except Exception:
            return None
    return None


def format_yen(value) -> str:
    try:
        return f"{int(value):,} 円"
    except (TypeError, ValueError):
        return "-"


def main():
    st.title("🏢 割安マンション検出＋投資判断支援")
    st.info(DISCLAIMER)

    db = get_db()

    # サイドバー: データの状態
    with st.sidebar:
        st.header("データの状態")
        tx_count = db.get_transaction_count()
        props = db.get_properties(limit=100000)
        st.metric("成約データ（理論価格の土台）", f"{tx_count:,} 件")
        st.metric("募集物件（listing）", f"{len(props):,} 件")
        if tx_count == 0:
            st.warning(
                "成約データがまだありません。「データ取込」タブから不動産情報ライブラリの"
                "データを投入してください（APIキーの申請手順は docs/DATA_SOURCES.md）。"
            )
        key = get_reinfolib_api_key()
        st.write("不動産情報ライブラリ APIキー: " + ("設定済み ✅" if key else "未設定"))
        st.write("価格モデル（第2層）: " + ("学習済み ✅" if PRICE_MODEL_PATH.exists() else "未学習（第1層のみで評価）"))

    tab_candidates, tab_import, tab_model, tab_stats = st.tabs(
        ["🎯 割安候補", "📥 データ取込", "🧮 理論価格モデル", "📊 統計・設定"]
    )

    # ================= 割安候補 =================
    with tab_candidates:
        st.header("割安候補（ゲート条件を満たした物件のみ）")
        st.caption(
            "ゲート条件: 類似成約 n≥10 / 直近2年以内の事例あり / 保守値（25%点）への乖離≥15% / "
            "予測区間幅≤1.6 / ハードフィルタ（旧耐震の近似判定・定期借地権）非抵触。"
            "信頼度は n・区間幅・事例の新しさから付けています。"
        )
        if st.button("🔄 全 listing を再評価する", key="valuate_btn"):
            model = load_price_model()
            with st.spinner("理論価格を計算しています..."):
                results = valuate_all_properties(db, model=model)
            passed = sum(1 for v in results if v.gate_passed)
            st.success(f"{len(results)} 件を評価しました。割安候補は {passed} 件です。")

        latest = db.get_latest_valuations()
        if latest.empty:
            st.info("まだ評価が実行されていません。「全 listing を再評価する」を押してください。")
        else:
            prop_map = {row["property_id"]: row for _, row in props.iterrows()}

            def valuation_table(df: pd.DataFrame) -> pd.DataFrame:
                rows = []
                for _, v in df.iterrows():
                    p = prop_map.get(v["property_id"])
                    if p is None:
                        continue
                    rows.append(
                        {
                            "所在地": f"{p['prefecture']} {p['city']}",
                            "募集価格": format_yen(p["price"]),
                            "理論価格（中央値）": format_yen(v["theoretical_price"]),
                            "保守値（25%点）": format_yen(v["conservative_price"]),
                            "予測区間": f"{format_yen(v['interval_low'])} 〜 {format_yen(v['interval_high'])}",
                            "割安スコア": f"{v['score']:.1%}" if pd.notna(v["score"]) else "-",
                            "類似成約n": int(v["similar_count"]),
                            "信頼度": v["confidence"],
                            "フラグ数": len(json.loads(v["risk_flags"] or "[]")),
                        }
                    )
                return pd.DataFrame(rows)

            candidates = latest[latest["gate_passed"] == 1]
            if candidates.empty:
                st.info("現在、ゲート条件を満たす割安候補はありません。")
            else:
                st.dataframe(valuation_table(candidates), width="stretch", hide_index=True)
                st.subheader("候補の詳細と精査チェックリスト")
                for _, v in candidates.iterrows():
                    p = prop_map.get(v["property_id"])
                    if p is None:
                        continue
                    title = f"{p['prefecture']} {p['city']} — {format_yen(p['price'])}（スコア {v['score']:.1%}）"
                    with st.expander(title):
                        show_property_detail(p, v)

            st.subheader("全物件の評価（候補外を含む）")
            st.dataframe(valuation_table(latest), width="stretch", hide_index=True)

    # ================= データ取込 =================
    with tab_import:
        st.header("成約データの取込（理論価格の土台）")
        st.write(
            "国土交通省「不動産情報ライブラリ」の取引価格・成約価格データを取り込みます。"
            "APIキーの申請手順とデータ仕様は docs/DATA_SOURCES.md を参照してください。"
        )
        uploaded = st.file_uploader(
            "公式サイトからダウンロードしたファイル（JSON / CSV）",
            type=["json", "csv"],
            key="tx_upload",
        )
        if uploaded is not None and st.button("成約ファイルを取り込む"):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            result = import_file(db, tmp_path)
            st.success(f"取り込み結果: {result}（成約データ総件数: {db.get_transaction_count():,}）")

        with st.expander("APIから直接取得する（REINFOLIB_API_KEY が必要）"):
            if not get_reinfolib_api_key():
                st.warning(
                    "APIキーが未設定です。https://www.reinfolib.mlit.go.jp/api/request/ から"
                    "利用申請を行い、発行されたキーを環境変数 REINFOLIB_API_KEY に設定してください。"
                )
            col1, col2, col3 = st.columns(3)
            year = col1.number_input("年", min_value=2005, max_value=2030, value=2024)
            quarter = col2.selectbox("四半期", [1, 2, 3, 4], index=0)
            area = col3.text_input("都道府県コード（例: 13=東京都）", value="13")
            if st.button("APIから取得して取り込む", disabled=not get_reinfolib_api_key()):
                try:
                    client = ReinfolibClient()
                    records = client.fetch_transactions(year=int(year), quarter=int(quarter), area=area or None)
                    result = import_records(db, records)
                    st.success(f"取り込み結果: {result}（総件数: {db.get_transaction_count():,}）")
                except ReinfolibAuthError as e:
                    st.error(str(e))

        st.divider()
        st.header("募集物件（listing）の投入")
        st.write(
            "ポータルサイトのスクレイピングは行いません。手元で入手した募集情報を "
            "CSV またはフォームで投入してください。**取得日と出所は必須**です"
            "（古い listing は陳腐化するため）。"
        )
        listing_csv = st.file_uploader("listing CSV（日本語ヘッダ可）", type=["csv"], key="listing_upload")
        if listing_csv is not None and st.button("listing CSVを取り込む"):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(listing_csv.getbuffer())
                tmp_path = tmp.name
            success, errors = import_listings_csv(db, tmp_path)
            st.success(f"{success} 件を取り込みました")
            for e in errors:
                st.error(e)

        with st.form("listing_form"):
            st.subheader("手入力フォーム")
            c1, c2, c3 = st.columns(3)
            price = c1.number_input("募集価格（円）*", min_value=0, step=100_000)
            prefecture = c2.text_input("都道府県*", value="東京都")
            city = c3.text_input("市区町村*", value="")
            c4, c5, c6 = st.columns(3)
            floor_area = c4.number_input("専有面積（㎡）*", min_value=0.0, step=1.0)
            building_year = c5.number_input("建築年（西暦）", min_value=0, max_value=2100, value=0)
            layout = c6.text_input("間取り", value="")
            c7, c8, c9 = st.columns(3)
            station = c7.text_input("最寄駅", value="")
            station_min = c8.number_input("駅徒歩（分）", min_value=0, value=0)
            total_units = c9.number_input("総戸数", min_value=0, value=0)
            c10, c11, c12 = st.columns(3)
            mgmt_fee = c10.number_input("管理費（円/月）", min_value=0, value=0)
            reserve = c11.number_input("修繕積立金（円/月）", min_value=0, value=0)
            monthly_rent = c12.number_input("月額賃料（円、賃貸中・根拠が確かな場合のみ）", min_value=0, value=0)
            c13, c14 = st.columns(2)
            rights_type = c13.selectbox("権利形態", ["所有権", "所有権（借地権付）", "定期借地権", "普通借地権", "不明"])
            is_tenanted = c14.checkbox("賃貸中（オーナーチェンジ）")
            c15, c16 = st.columns(2)
            listing_date = c15.date_input("取得日*")
            source_site = c16.text_input("出所*（例: 掲載サイト名、不動産会社名）", value="")
            submitted = st.form_submit_button("listing を登録する")
            if submitted:
                row = {
                    "price": price or None,
                    "prefecture": prefecture,
                    "city": city,
                    "floor_area": floor_area or None,
                    "building_year": building_year or None,
                    "layout": layout or None,
                    "nearest_station": station or None,
                    "station_distance": station_min or None,
                    "total_units": total_units or None,
                    "management_fee": mgmt_fee or None,
                    "repair_reserve_fund": reserve or None,
                    "monthly_rent": monthly_rent or None,
                    "rights_type": rights_type,
                    "is_tenanted": 1 if is_tenanted else 0,
                    "listing_date": listing_date.isoformat(),
                    "source_site": source_site,
                }
                success, errors = import_listings(db, [row])
                if success:
                    st.success("登録しました。「割安候補」タブで再評価してください。")
                for e in errors:
                    st.error(e)

        with st.expander("🧪 テスト用ダミーデータ（通常フローでは使いません）"):
            st.caption(
                "旧バージョン互換のダミー生成です。生成される価格は乱数のため、"
                "割安判定の根拠にはなりません。動作確認にのみ使ってください。"
            )
            if st.button("ダミー物件を10件生成する"):
                from src.scraper.suumo_scraper import SuumoScraper

                scraper = SuumoScraper(db)
                dummy = scraper.generate_dummy_properties(10)
                saved = scraper.save_properties(dummy)
                st.success(f"ダミー物件を {saved} 件保存しました（出所: dummy）")

    # ================= 理論価格モデル =================
    with tab_model:
        st.header("理論価格モデル（2層）")
        st.markdown(
            """
            - **第1層（類似成約）**: 同一市区町村×築年帯×面積帯の成約プールから、
              時点補正後の中央値・25%点・75%点を取る。事例数 n を必ず併記。
            - **第2層（ML）**: LightGBM で log(㎡単価) の分位点回帰（25%/中央値/75%）を行い予測区間を出す。
            - 評価は**期間分割**（古い期間で学習→新しい期間で検証）で MAPE と区間被覆率を測る。
              ランダム分割は同一地区の将来情報が学習に混入するため使わない。
            """
        )
        if PRICE_MODEL_METADATA_PATH.exists():
            meta = json.loads(PRICE_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
            m = meta.get("metrics") or {}
            st.subheader("現在のモデルの期間分割評価（学習時の実測）")
            if m:
                c1, c2, c3 = st.columns(3)
                c1.metric("MAPE", f"{m.get('mape_percent', 0):.1f} %")
                c2.metric("50%区間の被覆率", f"{m.get('coverage_50_percent', 0):.1f} %（理想 50%）")
                c3.metric("学習件数", f"{meta.get('trained_rows', 0):,} 件")
                st.caption(
                    "被覆率が50%を大きく下回る場合、予測区間は過信気味であることを意味します。"
                )
        if st.button("価格モデルを学習する（成約データから）"):
            tx = db.get_transactions()
            if len(tx) < 200:
                st.error(f"学習には最低200件の成約データが必要です（現在 {len(tx)} 件）。")
            else:
                with st.spinner("学習と期間分割評価を実行しています..."):
                    model, metrics = train_price_model(tx)
                    model.save(metrics=metrics)
                st.success(
                    f"学習完了: MAPE {metrics['mape_percent']:.1f}%、"
                    f"50%区間被覆率 {metrics['coverage_50_percent']:.1f}%"
                )
                st.json(metrics)

    # ================= 統計・設定 =================
    with tab_stats:
        st.header("統計")
        c1, c2, c3 = st.columns(3)
        c1.metric("成約データ", f"{db.get_transaction_count():,} 件")
        c2.metric("募集物件", f"{len(props):,} 件")
        latest_stats = db.get_latest_valuations()
        c3.metric(
            "割安候補（最新評価）",
            f"{int((latest_stats['gate_passed'] == 1).sum()) if not latest_stats.empty else 0} 件",
        )
        if not props.empty:
            st.subheader("市区町村別の listing 件数")
            st.bar_chart(props.groupby("city").size().sort_values(ascending=False).head(20))
        st.subheader("データソースについて")
        st.write(
            "- 成約/取引データ: 国土交通省「不動産情報ライブラリ」（APIキー申請制）。詳細は docs/DATA_SOURCES.md\n"
            "- 募集物件: 利用者が CSV / 手入力で投入したもの（スクレイピングは行わない）\n"
            "- 設計判断の根拠: docs/DESIGN.md"
        )


def show_property_detail(p: pd.Series, v: pd.Series):
    """割安候補1件の詳細（フラグ・投資指標・チェックリスト）。"""
    flags = json.loads(v["risk_flags"] or "[]")
    st.write(f"**理論価格レンジ**: {format_yen(v['interval_low'])} 〜 {format_yen(v['interval_high'])}")
    st.write(f"**信頼度の根拠**: 類似成約 n={int(v['similar_count'])}、信頼度 {v['confidence']}")
    if flags:
        st.write("**リスクフラグ**")
        for f in flags:
            icon = "⛔" if f["severity"] == "hard" else "⚠️"
            st.write(f"- {icon} {f['label']}: {f['detail']}")
    else:
        st.write("リスクフラグ: なし（一次資料での確認は依然として必要です）")

    inv = analyze_investment(
        int(p["price"]),
        int(p["monthly_rent"]) if pd.notna(p.get("monthly_rent")) else None,
        int(p["management_fee"]) if pd.notna(p.get("management_fee")) else None,
        int(p["repair_reserve_fund"]) if pd.notna(p.get("repair_reserve_fund")) else None,
    )
    st.write("**投資指標**（割安スコアとは別軸の情報です）")
    if inv:
        c1, c2, c3 = st.columns(3)
        c1.metric("表面利回り", f"{inv['gross_yield']:.2%}")
        c2.metric("NOI（年間）", format_yen(inv["noi"]))
        c3.metric("NOI利回り", f"{inv['noi_yield']:.2%}")
        st.caption("試算の仮定: " + "、".join(f"{k}={val}" for k, val in inv["assumptions"].items()))
    else:
        st.write("賃料が未入力のため、利回りは表示しません（推定利回りは出しません）。")

    valuation = Valuation(
        property_id=str(p["property_id"]),
        listing_price=int(p["price"]),
        theoretical_price=float(v["theoretical_price"]) if pd.notna(v["theoretical_price"]) else None,
        conservative_price=float(v["conservative_price"]) if pd.notna(v["conservative_price"]) else None,
        interval_low=float(v["interval_low"]) if pd.notna(v["interval_low"]) else None,
        interval_high=float(v["interval_high"]) if pd.notna(v["interval_high"]) else None,
        score=float(v["score"]) if pd.notna(v["score"]) else None,
        similar_count=int(v["similar_count"]),
        has_recent=True,
        pool_level="-",
        model_used=True,
        gate_passed=bool(v["gate_passed"]),
        flags=[RiskFlag(**f) for f in flags],
        confidence=str(v["confidence"]),
    )
    prop_dict = {
        k: (None if isinstance(val, float) and pd.isna(val) else val)
        for k, val in p.items()
    }
    checklist_md = build_checklist_markdown(prop_dict, valuation)
    st.download_button(
        "📋 精査チェックリスト（Markdown）をダウンロード",
        data=checklist_md.encode("utf-8"),
        file_name=f"checklist_{p['property_id']}.md",
        mime="text/markdown",
        key=f"dl_{p['property_id']}",
    )


if __name__ == "__main__":
    main()

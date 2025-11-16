"""
不動産割安物件発見システム - Streamlitアプリ
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from PIL import Image

from config.settings import (
    TARGET_PREFECTURES,
    SUPPORTED_SITES,
    LATEST_MODEL_PATH,
    MODEL_METADATA_PATH,
)
from src.database.db_manager import DatabaseManager
from src.scraper.suumo_scraper import generate_dummy_properties, SuumoScraper
from src.scraper.athome_scraper import AthomeScraper
from src.scraper.homes_scraper import HomesScraper
from src.scraper.rakuten_scraper import RakutenScraper
from src.ml.model_trainer import ModelTrainer
from src.ml.predictor import PricePredictor
from src.utils.helpers import (
    format_price,
    format_area,
    format_age,
    format_station_distance,
    get_discount_color,
)

# ページ設定
st.set_page_config(
    page_title="不動産割安物件発見システム",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 日本語フォント設定（matplotlib）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
sns.set_style("whitegrid")


def initialize_database():
    """データベースを初期化"""
    with DatabaseManager() as db:
        db.initialize_database()


def main():
    """メイン関数"""
    # データベース初期化
    initialize_database()

    # タイトル
    st.title("🏠 不動産割安物件発見システム")
    st.markdown("機械学習で市場価格より安い物件を見つけます")

    # サイドバー: 検索条件
    with st.sidebar:
        st.header("🔍 検索条件")

        # 都道府県選択
        selected_prefectures = st.multiselect(
            "都道府県",
            options=TARGET_PREFECTURES,
            default=TARGET_PREFECTURES,
        )

        # 割引率設定
        min_discount_rate = st.slider(
            "最低割引率 (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
        )

        # 最低価格設定
        min_price = st.slider(
            "最低価格 (万円)",
            min_value=1000,
            max_value=10000,
            value=3000,
            step=500,
        )

        # 最大表示件数
        max_display = st.selectbox(
            "最大表示件数",
            options=[10, 50, 100],
            index=1,
        )

    # タブ作成
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 割安物件一覧", "📥 データ取得", "🤖 モデル学習", "📊 統計情報"]
    )

    # タブ1: 割安物件一覧
    with tab1:
        show_bargain_properties(
            selected_prefectures, min_discount_rate, min_price, max_display
        )

    # タブ2: データ取得
    with tab2:
        show_data_acquisition()

    # タブ3: モデル学習
    with tab3:
        show_model_training()

    # タブ4: 統計情報
    with tab4:
        show_statistics()


def show_bargain_properties(
    selected_prefectures, min_discount_rate, min_price, max_display
):
    """割安物件一覧を表示"""
    st.header("📋 割安物件一覧")

    # モデルが存在するかチェック
    if not LATEST_MODEL_PATH.exists():
        st.warning("⚠️ 学習済みモデルが見つかりません。先に「モデル学習」タブでモデルを学習してください。")
        return

    with DatabaseManager() as db:
        # 割安物件を取得
        try:
            bargain_df = db.get_bargain_properties(
                min_discount_rate=min_discount_rate, limit=max_display
            )

            if len(bargain_df) == 0:
                st.info("条件に一致する割安物件が見つかりませんでした。")
                return

            # 都道府県でフィルタ
            if selected_prefectures:
                bargain_df = bargain_df[
                    bargain_df["prefecture"].isin(selected_prefectures)
                ]

            # 最低価格でフィルタ
            bargain_df = bargain_df[bargain_df["price"] >= min_price * 10000]

            st.success(f"🎯 {len(bargain_df)} 件の割安物件が見つかりました！")

            # 物件カード表示
            for idx, row in bargain_df.iterrows():
                with st.container():
                    # 画像を表示する場合は4列に変更
                    has_images = row.get('local_image_paths') and row.get('local_image_paths') != 'null'

                    if has_images:
                        img_col, col1, col2, col3 = st.columns([1, 2, 2, 1])

                        # 画像表示
                        with img_col:
                            try:
                                image_paths = json.loads(row['local_image_paths'])
                                if image_paths and len(image_paths) > 0:
                                    # 最初の画像を表示
                                    img_path = Path(image_paths[0])
                                    if img_path.exists():
                                        image = Image.open(img_path)
                                        st.image(image, use_column_width=True)
                            except Exception:
                                st.write("🏠")
                    else:
                        col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        st.subheader(f"{row['prefecture']} {row['city']}")
                        st.write(f"**住所:** {row.get('address', '不明')}")
                        st.write(f"**間取り:** {row.get('layout', '不明')}")
                        st.write(
                            f"**駅:** {row.get('nearest_station', '不明')} {format_station_distance(row.get('station_distance'))}"
                        )

                    with col2:
                        st.metric(
                            "販売価格",
                            format_price(row["price"]),
                        )
                        st.metric(
                            "予測価格",
                            format_price(row["predicted_price"]),
                        )

                    with col3:
                        discount_color = get_discount_color(row["discount_rate"])
                        st.markdown(
                            f"<h2 style='color: {discount_color}; text-align: center;'>{row['discount_rate']:.1f}%</h2>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            "<p style='text-align: center;'>割引率</p>",
                            unsafe_allow_html=True,
                        )

                    # 詳細情報
                    with st.expander("詳細情報"):
                        detail_col1, detail_col2 = st.columns(2)

                        with detail_col1:
                            st.write(f"**専有面積:** {format_area(row.get('floor_area'))}")
                            st.write(f"**築年数:** {format_age(row.get('building_age'))}")
                            st.write(f"**階数:** {row.get('floor_number', '不明')}階 / {row.get('total_floors', '不明')}階建")
                            st.write(f"**構造:** {row.get('structure', '不明')}")

                        with detail_col2:
                            st.write(f"**向き:** {row.get('direction', '不明')}")
                            st.write(f"**管理費:** {format_price(row.get('management_fee', 0))}/月")
                            st.write(f"**修繕積立金:** {format_price(row.get('repair_reserve_fund', 0))}/月")
                            st.write(f"**取得元:** {row.get('source_site', '不明')}")

                        st.write(f"**URL:** {row.get('url', 'なし')}")

                    st.divider()

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")


def show_data_acquisition():
    """データ取得画面を表示"""
    st.header("📥 データ取得")

    # スクレイピングモード選択
    scrape_mode = st.radio(
        "データ取得モード",
        options=["ダミーデータ", "実際のスクレイピング"],
        help="ダミーデータ: テスト用のランダムデータを生成\n実際のスクレイピング: 各サイトから実際のデータを取得",
    )

    if scrape_mode == "実際のスクレイピング":
        st.warning(
            "⚠️ 実際のスクレイピングを実行します。\n"
            "各サイトの利用規約を確認してから実行してください。\n"
            "過度なアクセスはサーバーに負荷をかけるため、適切な間隔を空けてください。"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        # 都道府県選択
        prefecture = st.selectbox("都道府県", TARGET_PREFECTURES)

    with col2:
        if scrape_mode == "ダミーデータ":
            # 取得件数
            data_count = st.number_input("取得件数", min_value=10, max_value=1000, value=100, step=10)
        else:
            # ページ数
            max_pages = st.number_input("最大ページ数", min_value=1, max_value=10, value=2, step=1)

    with col3:
        if scrape_mode == "実際のスクレイピング":
            # サイト選択
            site_name = st.selectbox(
                "取得元サイト",
                options=list(SUPPORTED_SITES.keys()),
            )

    # データ取得ボタン
    button_label = "🚀 ダミーデータ取得開始" if scrape_mode == "ダミーデータ" else "🚀 スクレイピング開始"

    if st.button(button_label, type="primary"):
        with st.spinner("データ取得中..."):
            try:
                if scrape_mode == "ダミーデータ":
                    # ダミーデータ生成
                    properties = generate_dummy_properties(count=data_count)
                    source_site = "SUUMO(ダミー)"
                else:
                    # 実際のスクレイピング
                    scrapers = {
                        "SUUMO": SuumoScraper(),
                        "athome": AthomeScraper(),
                        "HOMES": HomesScraper(),
                        "楽天不動産": RakutenScraper(),
                    }

                    scraper = scrapers.get(site_name)
                    if not scraper:
                        st.error(f"サイト {site_name} のスクレイパーが見つかりません")
                        return

                    properties = scraper.scrape_properties(
                        prefecture=prefecture,
                        max_pages=max_pages
                    )
                    source_site = site_name

                # データベースに保存
                with DatabaseManager() as db:
                    success_count = db.bulk_insert_properties(properties)

                    # ログ記録
                    log_data = {
                        "source_site": source_site,
                        "prefecture": prefecture,
                        "records_count": success_count,
                        "success": True,
                        "error_message": None,
                    }
                    db.insert_scraping_log(log_data)

                st.success(f"✅ {success_count} 件のデータを取得しました！")

                if properties:
                    # 取得データのサマリ表示
                    df = pd.DataFrame(properties)
                    st.subheader("取得データサマリ")

                    summary_col1, summary_col2, summary_col3 = st.columns(3)

                    with summary_col1:
                        st.metric("平均価格", format_price(df["price"].mean()))

                    with summary_col2:
                        if "floor_area" in df.columns:
                            st.metric("平均専有面積", format_area(df["floor_area"].mean()))

                    with summary_col3:
                        if "building_age" in df.columns:
                            st.metric("平均築年数", format_age(df["building_age"].mean()))

            except Exception as e:
                st.error(f"❌ エラーが発生しました: {e}")
                import traceback
                st.code(traceback.format_exc())

    # 取得履歴表示
    st.subheader("📜 取得履歴")
    with DatabaseManager() as db:
        logs_df = db.get_scraping_logs(limit=20)

        if len(logs_df) > 0:
            st.dataframe(
                logs_df[
                    ["executed_at", "source_site", "prefecture", "records_count", "success"]
                ],
                use_container_width=True,
            )
        else:
            st.info("まだデータ取得履歴がありません")


def show_model_training():
    """モデル学習画面を表示"""
    st.header("🤖 モデル学習")

    # データ件数確認
    with DatabaseManager() as db:
        property_count = db.get_property_count()

    st.info(f"現在のデータ件数: **{property_count}** 件")

    if property_count < 100:
        st.warning("⚠️ データが不足しています。最低100件のデータが必要です。")
        return

    # 学習実行ボタン
    if st.button("🚀 学習開始", type="primary"):
        with st.spinner("モデル学習中..."):
            try:
                # データ取得
                with DatabaseManager() as db:
                    df = db.get_all_properties_for_training()

                st.write(f"学習データ: {len(df)} 件")

                # モデル学習
                trainer = ModelTrainer()
                progress_bar = st.progress(0)

                progress_bar.progress(30)
                metrics = trainer.train(df, remove_outliers_flag=True)

                progress_bar.progress(70)
                trainer.save_model()

                progress_bar.progress(100)

                st.success("✅ モデル学習完了！")

                # 評価指標表示
                st.subheader("📈 評価指標")

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    st.metric("RMSE (検証)", f"{metrics['val_rmse']:,.0f}")

                with metric_col2:
                    st.metric("MAE (検証)", f"{metrics['val_mae']:,.0f}")

                with metric_col3:
                    st.metric("R² (検証)", f"{metrics['val_r2']:.4f}")

                # 特徴量重要度
                st.subheader("🔍 特徴量重要度 (Top 10)")
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = trainer.feature_importance.head(10)
                ax.barh(top_features["feature"], top_features["importance"])
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance")
                st.pyplot(fig)

                # 予測実行
                st.subheader("🎯 予測実行")
                if st.button("全物件の価格を予測"):
                    predictor = PricePredictor()
                    predictor.load_model()

                    predictions_df = predictor.predict(df)
                    prediction_records = predictor.create_prediction_records(
                        predictions_df
                    )

                    # 予測結果をデータベースに保存
                    with DatabaseManager() as db:
                        success_count = db.bulk_insert_predictions(prediction_records)

                    st.success(f"✅ {success_count} 件の予測を保存しました！")

            except Exception as e:
                st.error(f"❌ エラーが発生しました: {e}")
                import traceback

                st.code(traceback.format_exc())

    # モデル情報表示
    st.subheader("ℹ️ モデル情報")
    metadata = ModelTrainer.load_metadata()

    if metadata:
        st.write(f"**学習日時:** {metadata.get('trained_at', '不明')}")
        st.write(f"**モデル種別:** {metadata.get('model_type', '不明')}")

        if "metrics" in metadata:
            st.json(metadata["metrics"])
    else:
        st.info("まだモデルが学習されていません")


def show_statistics():
    """統計情報を表示"""
    st.header("📊 統計情報")

    with DatabaseManager() as db:
        stats = db.get_statistics()

        # 基本統計
        st.subheader("📈 基本統計")
        stat_col1, stat_col2, stat_col3 = st.columns(3)

        with stat_col1:
            st.metric("物件総数", f"{stats['total_properties']:,} 件")

        with stat_col2:
            st.metric("予測総数", f"{stats['total_predictions']:,} 件")

        with stat_col3:
            st.metric("平均価格", format_price(stats['avg_price']))

        # 都道府県別件数
        st.subheader("🗾 都道府県別件数")
        if stats["prefecture_counts"]:
            pref_df = pd.DataFrame(
                list(stats["prefecture_counts"].items()),
                columns=["都道府県", "件数"],
            )
            st.bar_chart(pref_df.set_index("都道府県"))
        else:
            st.info("データがありません")

        # サイト別件数
        st.subheader("🌐 サイト別件数")
        if stats["site_counts"]:
            site_df = pd.DataFrame(
                list(stats["site_counts"].items()),
                columns=["サイト", "件数"],
            )
            st.bar_chart(site_df.set_index("サイト"))
        else:
            st.info("データがありません")

        # 物件データ取得
        if stats["total_properties"] > 0:
            df = db.get_properties(limit=10000)

            # 価格分布
            st.subheader("💰 価格分布")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df["price"] / 10000, bins=50, edgecolor="black")
            ax.set_xlabel("Price (万円)")
            ax.set_ylabel("Frequency")
            ax.set_title("Price Distribution")
            st.pyplot(fig)

            # 築年数分布
            st.subheader("🏗️ 築年数分布")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df["building_age"].dropna(), bins=40, edgecolor="black")
            ax.set_xlabel("Building Age (years)")
            ax.set_ylabel("Frequency")
            ax.set_title("Building Age Distribution")
            st.pyplot(fig)


if __name__ == "__main__":
    main()

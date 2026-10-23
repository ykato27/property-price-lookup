"""
デモ用シード: フィクスチャ成約＋サンプル listing を投入し、評価まで実行する。

注意: ここで使う成約データは XIT001 形式に似せた構造化ダミー（フィクスチャ）
であり、実データではない。UI の動作確認専用。実データを入れる場合は
APIキー取得後に scripts/import_transactions.py を使うこと（docs/DATA_SOURCES.md）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager
from src.market.fixture_data import generate_fixture_transactions
from src.market.importer import import_records
from src.market.listings import import_listings
from src.market.price_model import train_price_model
from src.market.scoring import valuate_all_properties

# フィクスチャ市場の水準（港区・築浅）に合わせて作ったサンプル listing。
# 一部は意図的に安く（割安候補）、一部はフラグが立つ設定にしてある。
SAMPLE_LISTINGS = [
    {
        "募集価格": "62000000", "都道府県": "東京都", "市区町村": "港区",
        "専有面積": "65", "間取り": "3LDK", "建築年": "2010", "所在階": "8",
        "総戸数": "64", "構造": "ＲＣ", "最寄駅": "港駅", "駅距離": "6",
        "管理費": "12000", "修繕積立金": "11000", "権利形態": "所有権",
        "月額賃料": "280000", "取得日": "2025-11-15", "出所": "デモ（サンプル）",
    },
    {
        "募集価格": "78000000", "都道府県": "東京都", "市区町村": "港区",
        "専有面積": "65", "間取り": "3LDK", "建築年": "2010", "所在階": "3",
        "総戸数": "64", "構造": "ＲＣ", "最寄駅": "港駅", "駅距離": "6",
        "管理費": "12000", "修繕積立金": "11000", "権利形態": "所有権",
        "取得日": "2025-11-15", "出所": "デモ（サンプル）",
    },
    {
        "募集価格": "29000000", "都道府県": "東京都", "市区町村": "世田谷区",
        "専有面積": "55", "間取り": "2LDK", "建築年": "1978", "所在階": "2",
        "総戸数": "24", "構造": "ＲＣ", "最寄駅": "世田谷駅", "駅距離": "9",
        "管理費": "9000", "修繕積立金": "4000", "権利形態": "所有権",
        "賃貸中": "オーナーチェンジ", "月額賃料": "130000",
        "取得日": "2025-11-10", "出所": "デモ（サンプル）",
    },
    {
        "募集価格": "34000000", "都道府県": "東京都", "市区町村": "葛飾区",
        "専有面積": "60", "間取り": "3LDK", "建築年": "2008", "所在階": "5",
        "総戸数": "40", "構造": "ＲＣ", "最寄駅": "葛飾駅", "駅距離": "11",
        "管理費": "10000", "修繕積立金": "9000", "権利形態": "定期借地権",
        "取得日": "2025-11-12", "出所": "デモ（サンプル）",
    },
]


def main() -> int:
    with DatabaseManager() as db:
        db.initialize_database()
        result = import_records(db, generate_fixture_transactions(n=2000, seed=42))
        print(f"フィクスチャ成約を取り込み: {result}（実データではありません）")

        success, errors = import_listings(db, SAMPLE_LISTINGS)
        print(f"サンプル listing: {success} 件取り込み, エラー: {errors}")

        df = db.get_transactions()
        model, metrics = train_price_model(df)
        model.save(metrics=metrics)
        print(f"価格モデル学習完了。期間分割評価: MAPE {metrics['mape_percent']:.1f}%, "
              f"50%区間被覆率 {metrics['coverage_50_percent']:.1f}%（フィクスチャでの値）")

        valuations = valuate_all_properties(db, model=model)
        for v in valuations:
            status = "割安候補" if v.gate_passed else "候補外"
            score = f"{v.score:.1%}" if v.score is not None else "-"
            print(f"  [{status}] {v.property_id}: スコア {score}, n={v.similar_count}, "
                  f"信頼度 {v.confidence}, フラグ {[f.label for f in v.flags]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

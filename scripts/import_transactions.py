"""
成約データ（不動産情報ライブラリ）の取り込み CLI。

使い方:
    # ダウンロード済みファイル（JSON / CSV）から取り込む
    uv run python scripts/import_transactions.py --file data/raw/xit001.json

    # APIから取得して取り込む（環境変数 REINFOLIB_API_KEY が必要）
    uv run python scripts/import_transactions.py --api --year 2024 --quarter 1 --area 13

    # 動作確認用フィクスチャ（実データではない）を取り込む
    uv run python scripts/import_transactions.py --fixture 1200
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager
from src.market.importer import import_file
from src.market.reinfolib_client import ReinfolibAuthError, ReinfolibClient


def main() -> int:
    parser = argparse.ArgumentParser(description="不動産情報ライブラリの成約データを取り込む")
    parser.add_argument("--file", help="ダウンロード済みファイル（JSON/CSV）のパス")
    parser.add_argument("--api", action="store_true", help="APIから取得する")
    parser.add_argument("--fixture", type=int, help="フィクスチャ件数（実データではない動作確認用）")
    parser.add_argument("--year", type=int, help="API: 対象年")
    parser.add_argument("--quarter", type=int, help="API: 四半期（1-4）")
    parser.add_argument("--area", help="API: 都道府県コード（例: 13）")
    parser.add_argument("--city", help="API: 市区町村コード（5桁）")
    parser.add_argument("--station", help="API: 駅コード（6桁）")
    parser.add_argument("--include-land", action="store_true", help="土地・戸建も含める（既定は中古マンションのみ）")
    args = parser.parse_args()

    with DatabaseManager() as db:
        db.initialize_database()
        if args.file:
            result = import_file(db, args.file, condo_only=not args.include_land)
        elif args.fixture:
            from src.market.fixture_data import generate_fixture_transactions
            from src.market.importer import import_records

            result = import_records(db, generate_fixture_transactions(n=args.fixture))
            print("※ フィクスチャ（XIT001形式の構造化ダミー）を取り込みました。実データではありません。")
        elif args.api:
            try:
                client = ReinfolibClient()
                records = client.fetch_transactions(
                    year=args.year,
                    quarter=args.quarter,
                    area=args.area,
                    city=args.city,
                    station=args.station,
                )
            except ReinfolibAuthError as e:
                print(str(e))
                return 2
            from src.market.importer import import_records

            result = import_records(db, records, condo_only=not args.include_land)
        else:
            parser.print_help()
            return 1
        print(f"結果: {result}")
        print(f"成約データ総件数: {db.get_transaction_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

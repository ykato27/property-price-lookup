"""
精度実験ハーネスの CLI: 「どの粒度・どの方法が精度を上げるか」を固定プロトコルで回す。

使い方:
    # DB（data/properties.db の transactions, source='trade'）で実行
    uv run python scripts/run_granularity_experiments.py

    # フィクスチャ＋DBの実データ（港区）を混ぜて動作確認
    uv run python scripts/run_granularity_experiments.py --source fixture+db

    # フィクスチャのみ
    uv run python scripts/run_granularity_experiments.py --source fixture --fixture-n 1500

結果は experiments/granularity_results.md（実行ごとに日時つきで追記）と
experiments/granularity_results.csv に出力する。

評価プロトコルとリーク防止ルールは src/market/experiments.py の冒頭を参照。
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DB_PATH
from src.database.db_manager import DatabaseManager
from src.market.experiments import (
    render_results_markdown,
    results_to_frame,
    run_experiments,
)
from src.market.fixture_data import generate_fixture_transactions
from src.market.normalize import normalize_records

RESULTS_MD_HEADER = """# 精度実験 結果（粒度 × 手法）

「どの粒度・どの方法が精度を上げるか」を、固定の評価プロトコルで検証した記録。
実験設計とリーク防止ルールは `src/market/experiments.py` の冒頭に定義してある。

- 固定プロトコル: 学習 ≤2024Q4 / 検証 2025Q1–2026Q1、対象区の行で評価。
  データが満たせない場合は最後の5四半期を検証にするフォールバック（結果に明記）。
- 指標: MAPE（中央値・取引総額ベース）、50%区間被覆率、区間幅（q75/q25平均）。
- ターゲットエンコーディング・時点補正・調整・ブレンド重み・区間較正は、
  すべて学習期間のデータだけで決める（検証期間の情報は使わない）。

> **注意**: 本番データ（23区・複数年）での本実験はデータ取り込み後に再実行が必要。
> それ以前の実行は、フィクスチャや小規模な実データによる予備結果である。

"""


def load_db_transactions(db_path: Path) -> pd.DataFrame:
    with DatabaseManager(db_path=db_path) as db:
        db.initialize_database()
        return db.get_transactions(source="trade")


def load_fixture_transactions(n: int, seed: int = 42) -> pd.DataFrame:
    records = normalize_records(generate_fixture_transactions(n=n, seed=seed))
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="粒度×手法の精度実験を回す")
    parser.add_argument(
        "--source", choices=("db", "fixture", "fixture+db"), default="db",
        help="データ源（既定: db = transactions の source='trade'）",
    )
    parser.add_argument("--fixture-n", type=int, default=1500, help="フィクスチャ件数")
    parser.add_argument("--target-ward", default="港区", help="評価対象区（既定: 港区）")
    parser.add_argument("--db-path", default=str(DB_PATH), help="DBファイルのパス")
    parser.add_argument(
        "--out-dir", default="experiments",
        help="結果の出力先ディレクトリ（既定: experiments）",
    )
    parser.add_argument(
        "--no-combined", action="store_true",
        help="二段目（最良水準同士の組合せ確認）を省略する",
    )
    args = parser.parse_args()

    frames = []
    labels = []
    if args.source in ("fixture", "fixture+db"):
        frames.append(load_fixture_transactions(args.fixture_n))
        labels.append(f"フィクスチャ{args.fixture_n}件")
    if args.source in ("db", "fixture+db"):
        db_df = load_db_transactions(Path(args.db_path))
        frames.append(db_df)
        labels.append(f"DB実データ(trade){len(db_df)}件")
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    dataset_label = " + ".join(labels)
    print(f"データ: {dataset_label}（計{len(df)}件）")

    results, meta = run_experiments(
        df,
        target_ward=args.target_ward,
        dataset_label=dataset_label,
        include_combined=not args.no_combined,
    )

    run_at = datetime.now().isoformat(timespec="seconds")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "granularity_results.md"
    if not md_path.exists():
        md_path.write_text(RESULTS_MD_HEADER, encoding="utf-8")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(render_results_markdown(results, meta, run_at))

    csv_path = out_dir / "granularity_results.csv"
    frame = results_to_frame(results, run_at)
    if csv_path.exists():
        frame.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(csv_path, index=False)

    print(f"プロトコル: {meta['protocol']}（学習 {meta['train_period']} / 検証 {meta['test_period']}）")
    if meta.get("split_note"):
        print(f"注意: {meta['split_note']}")
    for r in results:
        mape = r.get("mape_percent")
        mape_s = f"{mape:.2f}%" if isinstance(mape, (int, float)) else "-"
        print(f"  {r['config']}: MAPE={mape_s} 被覆率={r.get('coverage_50_percent')} 状態={r['status']}")
    print(f"結果を追記しました: {md_path} / {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
理論価格モデル2層（時点補正・類似成約統計・LightGBM分位点）のテスト。

データはフィクスチャ生成器による XIT001 形式の構造化ダミー（実データではない）。
"""
import numpy as np
import pandas as pd
import pytest

from src.database.db_manager import DatabaseManager
from src.market.comparables import find_comparables
from src.market.fixture_data import generate_fixture_transactions
from src.market.importer import import_records
from src.market.price_model import (
    PriceModel,
    evaluate_period_split,
    train_price_model,
)
from src.market.time_adjust import UnitPriceIndex, period_ordinal


@pytest.fixture(scope="module")
def transactions_df(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tx")
    with DatabaseManager(db_path=tmp / "t.db") as db:
        db.initialize_database()
        import_records(db, generate_fixture_transactions(n=1600, seed=42))
        df = db.get_transactions()
    return df


def test_period_ordinal_monotonic():
    assert period_ordinal(2024, 4) < period_ordinal(2025, 1)
    assert period_ordinal(2025, 1) - period_ordinal(2024, 1) == 4


def test_time_adjust_index_moves_with_trend(transactions_df):
    index = UnitPriceIndex.from_transactions(transactions_df)
    # フィクスチャは年+2%トレンドなので、2021→2025 の補正率は 1 を上回るはず
    factor = index.factor((2021, 1), (2025, 4), group="東京都")
    assert factor > 1.0


def test_comparables_returns_stats_with_n(transactions_df):
    stats = find_comparables(
        transactions_df,
        municipality_code="13103",  # 港区
        prefecture="東京都",
        area=65.0,
        building_year=2000,
        valuation_period=(2025, 4),
    )
    assert stats is not None
    assert stats.n >= 10
    assert stats.q25_price < stats.median_price < stats.q75_price
    assert stats.has_recent  # 2025年の事例を含むプール
    assert stats.latest_period is not None


def test_comparables_unknown_city_returns_none_or_prefecture_pool(transactions_df):
    stats = find_comparables(
        transactions_df,
        municipality_code="99999",
        municipality="存在しない区",
        prefecture="東京都",
        area=60.0,
        valuation_period=(2025, 4),
    )
    # 市区町村プールは空でも、都道府県プールで拾える
    assert stats is not None
    assert stats.pool_level == "都道府県全件"


def test_price_model_quantiles_ordered(transactions_df):
    model, metrics = train_price_model(transactions_df, test_quarters=4)
    rows = pd.DataFrame(
        [
            {
                "prefecture": "東京都",
                "municipality": "港区",
                "district_name": "港中央",
                "structure": "ＲＣ",
                "city_planning": "第一種住居地域",
                "region": "住宅地",
                "floor_plan": "3LDK",
                "area": 70.0,
                "building_year": 2005,
                "period_year": 2025,
                "period_quarter": 4,
                "station_distance_min": 6.0,
            }
        ]
    )
    preds = model.predict_unit_quantiles(rows)
    assert preds.shape == (1, 3)
    assert preds[0, 0] < preds[0, 1] < preds[0, 2]
    # 港区・築浅・駅近の㎡単価として非常識な水準でないこと（50万〜400万円/㎡）
    assert 500_000 < preds[0, 1] < 4_000_000


def test_period_split_metrics_are_honest(transactions_df):
    metrics = evaluate_period_split(transactions_df, test_quarters=4)
    assert metrics["method"] == "period_split"
    assert metrics["train_rows"] > 0 and metrics["test_rows"] > 0
    assert metrics["train_period_ordinal"][1] < metrics["test_period_ordinal"][0]
    # フィクスチャのノイズ水準（σ=0.13）から、MAPE は 0 でも 50%超でもない範囲
    assert 3.0 < metrics["mape_percent"] < 30.0
    # 50%区間の被覆率は理論値50%の周辺（大きく外れていれば較正不良）
    assert 25.0 < metrics["coverage_50_percent"] < 75.0


def test_price_model_save_load_roundtrip(transactions_df, tmp_path):
    model = PriceModel().fit(transactions_df)
    model_path = tmp_path / "pm.pkl"
    meta_path = tmp_path / "pm.json"
    model.save(model_path=model_path, metadata_path=meta_path, metrics={"mape_percent": 9.9})
    loaded = PriceModel.load(model_path)
    rows = transactions_df.head(3).copy()
    a = model.predict_unit_quantiles(rows)
    b = loaded.predict_unit_quantiles(rows)
    assert np.allclose(a, b)

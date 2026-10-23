"""
精度実験ハーネス（src/market/experiments.py）のテスト。

検証するのは主に「評価プロトコルの分割」と「リーク防止」の不変条件:
- 固定プロトコル（学習≤2024Q4／検証2025Q1–2026Q1）が守られること
- データ不足時はフォールバックが明示されること
- ターゲットエンコーディングが学習側の統計だけで決まること
- 第1層が学習プールだけから予測すること
- 区間較正係数が負にならず、OOF被覆率を50%以上に引き上げること
"""
import numpy as np
import pandas as pd
import pytest

from src.market.experiments import (
    ExperimentConfig,
    SmoothedTargetEncoder,
    calibration_delta,
    predict_layer1,
    prepare_transactions,
    resolve_split,
    run_experiments,
)
from src.market.fixture_data import generate_fixture_transactions
from src.market.normalize import normalize_records
from src.market.time_adjust import period_ordinal


def _synthetic_df(quarters, rows_per_quarter=60, ward="港区", code="13103"):
    """分割テスト用の最小データ（価格は一定）。"""
    records = []
    for (y, q) in quarters:
        for _ in range(rows_per_quarter):
            records.append({
                "trade_price": 30_000_000, "area": 60.0,
                "period_year": y, "period_quarter": q,
                "municipality": ward, "municipality_code": code,
                "prefecture": "東京都", "district_name": "赤坂",
                "nearest_station": "赤坂駅", "station_distance_min": 5.0,
                "floor_plan": "3LDK", "building_year": 2000,
                "structure": "ＲＣ", "city_planning": "商業", "region": "住宅地",
                "renovation": "", "remarks": "",
            })
    return prepare_transactions(pd.DataFrame(records))


def test_resolve_split_fixed_protocol():
    quarters = [(y, q) for y in range(2021, 2027) for q in (1, 2, 3, 4) if (y, q) <= (2026, 1)]
    df = _synthetic_df(quarters, rows_per_quarter=30)
    split = resolve_split(df, target_ward="港区")
    assert split.protocol == "fixed"
    assert int(split.train["ordinal"].max()) == period_ordinal(2024, 4)
    assert int(split.test["ordinal"].min()) >= period_ordinal(2025, 1)
    assert int(split.test["ordinal"].max()) <= period_ordinal(2026, 1)
    assert (split.test["municipality"] == "港区").all()


def test_resolve_split_fallback_uses_last_quarters():
    # 5四半期しかないデータ: 固定プロトコルは成立せず、最後の3四半期が検証になる
    quarters = [(2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1)]
    df = _synthetic_df(quarters, rows_per_quarter=60)
    split = resolve_split(df, target_ward="港区")
    assert split.protocol == "fallback_last3q"
    assert split.note  # フォールバック使用が明記される
    train_ords = set(split.train["ordinal"].unique())
    test_ords = set(split.test["ordinal"].unique())
    assert not (train_ords & test_ords)
    assert max(test_ords) == period_ordinal(2026, 1)
    assert len(split.train) == 120 and len(split.test) == 180


def test_target_encoder_uses_train_statistics_only():
    values = pd.Series(["A", "A", "B"])
    y = np.array([1.0, 3.0, 10.0])
    enc = SmoothedTargetEncoder(smoothing=20.0).fit(values, y)
    global_mean = 14.0 / 3.0
    assert enc.global_mean == pytest.approx(global_mean)
    # 平滑化式どおり: (sum + m*global) / (count + m)
    assert enc.transform(pd.Series(["A"]))[0] == pytest.approx((4.0 + 20 * global_mean) / 22)
    assert enc.transform(pd.Series(["B"]))[0] == pytest.approx((10.0 + 20 * global_mean) / 21)
    # 学習に無いカテゴリは全体平均（検証側の値を混ぜない）
    assert enc.transform(pd.Series(["C"]))[0] == pytest.approx(global_mean)


def test_layer1_predicts_from_train_pool_only():
    train = _synthetic_df([(2024, 1)], rows_per_quarter=6)
    train["unit_price"] = 1_000_000.0
    train["trade_price"] = 50_000_000
    row = train.iloc[[0]].copy()
    row["period_year"] = 2024
    row["period_quarter"] = 2
    row["building_year"] = 2002  # 築年帯は学習行（2000年築・2024年取引=築24年→帯20）と一致
    row["area"] = 55.0          # 面積帯40で一致
    preds, info = predict_layer1(train, row, granularity="city")
    assert info["pred_rate"] == 1.0
    # 単価が一様なプールなので中央値 = 1,000,000 × 55㎡（時点補正係数も1）
    assert preds[0, 1] == pytest.approx(55_000_000.0)
    assert preds[0, 0] <= preds[0, 1] <= preds[0, 2]


def test_calibration_delta_expands_narrow_intervals():
    rng = np.random.default_rng(0)
    n = 200
    actual = rng.uniform(20e6, 80e6, n)
    # 区間が狭すぎる（被覆率が大きく50%未満になる）OOF予測
    oof = pd.DataFrame({
        "actual": actual,
        "q25": actual * 0.98,
        "median": actual.copy(),
        "q75": actual * 1.02,
    })
    oof.loc[oof.index[:150], "actual"] = oof.loc[oof.index[:150], "actual"] * 1.2
    delta = calibration_delta(oof)
    assert delta > 0
    lo = np.log(oof["q25"]) - delta
    hi = np.log(oof["q75"]) + delta
    covered = np.mean((np.log(oof["actual"]) >= lo) & (np.log(oof["actual"]) <= hi))
    assert covered >= 0.5

    # 既に被覆率50%以上なら拡大しない
    wide = pd.DataFrame({
        "actual": actual, "q25": actual * 0.5, "median": actual.copy(), "q75": actual * 1.5,
    })
    assert calibration_delta(wide) == 0.0


def test_run_experiments_smoke_on_fixture():
    records = normalize_records(generate_fixture_transactions(n=700, seed=7))
    df = pd.DataFrame(records)
    results, meta = run_experiments(
        df,
        target_ward="港区",
        dataset_label="fixture-smoke",
        include_combined=False,
        configs=[ExperimentConfig(), ExperimentConfig(method="layer1", pool="city")],
    )
    assert meta["protocol"] == "fixed"
    by_method = {r["method"]: r for r in results}
    assert by_method["gbm"]["status"] == "ok"
    assert by_method["gbm"]["mape_percent"] is not None
    assert by_method["gbm"]["test_n"] > 0
    assert by_method["layer1"]["status"] == "ok"

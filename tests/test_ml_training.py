"""
property-price-lookup ML修正の回帰テスト

- 学習リーク: エンコーダーのfitは訓練分割のみに適用されること
- 未見カテゴリ: 予測時に ValueError でバッチ全体が死なず、専用値 -1 になること
- 中央値補完: 検証/予測時は訓練データの中央値を使うこと
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from config.settings import ML_SETTINGS
from src.ml.feature_engineering import FeatureEngineer
from src.ml.model_trainer import ModelTrainer

VAL_ONLY_ROW = 9  # random_state=42, test_size=0.2 の分割で検証側に入る行


def make_df(n: int = 150, val_only_city: str = "Z市") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cities = rng.choice(["A市", "B市", "C市"], n).tolist()
    cities[VAL_ONLY_ROW] = val_only_city  # この行は検証分割にのみ現れる
    return pd.DataFrame(
        {
            "price": rng.integers(2_000_000, 80_000_000, n).astype(float),
            "building_age": rng.integers(0, 40, n).astype(float),
            "floor_area": rng.uniform(25, 120, n),
            "floor_number": rng.integers(1, 15, n).astype(float),
            "total_floors": rng.integers(3, 20, n).astype(float),
            "station_distance": rng.integers(1, 20, n).astype(float),
            "management_fee": rng.integers(5000, 30000, n).astype(float),
            "repair_reserve_fund": rng.integers(5000, 25000, n).astype(float),
            "prefecture": "東京都",
            "city": cities,
            "layout": rng.choice(["1K", "1LDK", "2LDK", "3LDK"], n),
            "structure": rng.choice(["RC", "SRC", "木造"], n),
            "direction": rng.choice(["南", "北", "東", "西"], n),
        }
    )


def test_train_fits_encoders_on_train_split_only():
    """train() 内部の分割後にfitされること。検証側だけの市は知らないはず。"""
    df = make_df()
    trainer = ModelTrainer()
    trainer.train(df, remove_outliers_flag=False)

    # train() と同じ条件で素データを分割し直して期待値を求める
    df_train_expected, _ = train_test_split(
        df,
        test_size=ML_SETTINGS["test_size"],
        random_state=ML_SETTINGS["random_state"],
    )
    train_cities = set(df_train_expected["city"].astype(str).unique())
    assert "Z市" not in train_cities  # 前提: Z市は検証側のみ

    encoded_cities = set(trainer.feature_engineer.label_encoders["city"].classes_)
    assert encoded_cities <= train_cities, (
        f"検証データのカテゴリが学習時に混入している: {encoded_cities - train_cities}"
    )


def test_unseen_category_maps_to_minus_one_without_crash():
    """学習時に無い市・間取りが来ても落ちず、-1 になること。"""
    df = make_df()
    fe = FeatureEngineer()
    fe.prepare_training_data(df)

    df_new = make_df(n=20, val_only_city="Q市").head(5).copy()
    df_new["city"] = "Q市"  # 完全な未見
    df_new["layout"] = "5LDK"  # 完全な未見

    X, y = fe.prepare_validation_data(df_new)  # 例外が出たらバグ
    assert (X["city_encoded"] == -1).all()
    assert (X["layout_encoded"] == -1).all()
    # 既知カテゴリは正常にエンコードされる
    assert (X["prefecture_encoded"] != -1).all()


def test_median_imputation_uses_train_statistics():
    """欠損補完の中央値は訓練データのものを使うこと（検証バッチの中央値ではない）。"""
    df_train = make_df()
    df_train["building_age"] = 10.0  # 訓練側の中央値は 10
    fe = FeatureEngineer()
    fe.prepare_training_data(df_train)

    df_new = make_df(n=10)
    df_new["building_age"] = 30.0
    df_new.loc[0, "building_age"] = np.nan  # 欠損1件。バッチ中央値は30、訓練中央値は10

    X, _ = fe.prepare_validation_data(df_new)
    assert X.loc[0, "building_age"] == pytest.approx(10.0)

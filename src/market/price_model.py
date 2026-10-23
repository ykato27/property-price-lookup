"""
第2層: LightGBM による理論価格モデル（log(㎡単価) の分位点回帰）。

- 25%点・中央値・75%点を別モデルで学習し、予測区間として使う。
- 評価はランダム分割ではなく期間分割（古い期間で学習→新しい期間で検証）。
  ランダム分割は同一地区の将来情報が学習に混入し、精度を過大評価する
  （調査レポート「検証方法」）。現行 ModelTrainer のランダム分割評価は
  この期間分割評価に置き換える。
- カテゴリエンコーダは学習期間のデータでのみ fit する（未見は -1）。
"""
import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.settings import PRICE_MODEL_METADATA_PATH, PRICE_MODEL_PATH
from src.market.time_adjust import period_ordinal

QUANTILES = (0.25, 0.5, 0.75)

_NUMERIC_FEATURES = [
    "area",
    "log_area",
    "age_at_trade",
    "station_distance_min",
    "rooms",
    "period_ordinal",
]

_CATEGORICAL_FEATURES = [
    "prefecture",
    "municipality",
    "district_name",
    "structure",
    "city_planning",
    "region",
    "floor_plan",
]

FEATURE_COLUMNS = _NUMERIC_FEATURES + [f"{c}_encoded" for c in _CATEGORICAL_FEATURES]

MIN_TRAIN_ROWS = 200


def extract_rooms(floor_plan: Any) -> float:
    """間取り（"3LDK" 等）から部屋数を取る。取れなければ NaN。"""
    if not isinstance(floor_plan, str):
        return np.nan
    m = re.search(r"(\d+)", floor_plan)
    return float(m.group(1)) if m else np.nan


def build_features(df: pd.DataFrame, encoders: Optional[Dict[str, Dict[str, int]]] = None,
                   fit_encoders: bool = False) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    成約/物件データから特徴量を作る。

    fit_encoders=True のときだけカテゴリマッピングを学習する（学習データ専用）。
    予測時は fit 済みマッピングを使い、未見カテゴリは -1。
    """
    d = pd.DataFrame(index=df.index)
    area = pd.to_numeric(df["area"], errors="coerce") if "area" in df else np.nan
    d["area"] = area
    d["log_area"] = np.log(area.clip(lower=1))
    if "building_year" in df and "period_year" in df:
        d["age_at_trade"] = (
            pd.to_numeric(df["period_year"], errors="coerce")
            - pd.to_numeric(df["building_year"], errors="coerce")
        ).clip(0, 90)
    else:
        d["age_at_trade"] = np.nan
    d["station_distance_min"] = (
        pd.to_numeric(df["station_distance_min"], errors="coerce")
        if "station_distance_min" in df
        else np.nan
    )
    d["rooms"] = df["floor_plan"].apply(extract_rooms) if "floor_plan" in df else np.nan
    if "period_year" in df and "period_quarter" in df:
        d["period_ordinal"] = [
            period_ordinal(int(y), int(q))
            for y, q in zip(df["period_year"], df["period_quarter"])
        ]
    else:
        d["period_ordinal"] = np.nan

    encoders = encoders if encoders is not None else {}
    for col in _CATEGORICAL_FEATURES:
        if col in df:
            raw = df[col]
            # 欠損（NaN/NA）を含む実データでも壊れないよう、欠損は専用カテゴリに寄せてから文字列化する
            # （pandas 3 では astype(str) が欠損を文字列化しないため、混在型のままソートすると落ちる）
            values = raw.astype(object).where(raw.notna(), "__missing__").astype(str)
        else:
            values = pd.Series("__missing__", index=df.index)
        if fit_encoders:
            mapping = {v: i for i, v in enumerate(sorted(values.unique()))}
            encoders[col] = mapping
        mapping = encoders.get(col, {})
        d[f"{col}_encoded"] = values.map(mapping).fillna(-1).astype(int)
    return d[FEATURE_COLUMNS], encoders


class PriceModel:
    """log(㎡単価) の分位点回帰モデル群（25%点・中央値・75%点）。"""

    def __init__(self):
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.encoders: Dict[str, Dict[str, int]] = {}
        self.train_period_min: Optional[int] = None
        self.train_period_max: Optional[int] = None
        self.n_train: int = 0

    def fit(self, transactions: pd.DataFrame) -> "PriceModel":
        df = transactions.copy()
        df = df.dropna(subset=["trade_price", "area"])
        df = df[df["area"] > 0]
        if len(df) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"学習データが不足しています（{len(df)}件）。最低{MIN_TRAIN_ROWS}件の成約事例が必要です。"
            )
        unit_price = df["trade_price"] / df["area"]
        y = np.log(unit_price.to_numpy(dtype=float))

        X, self.encoders = build_features(df, fit_encoders=True)
        ordinals = X["period_ordinal"].to_numpy()
        self.train_period_min = int(np.nanmin(ordinals))
        self.train_period_max = int(np.nanmax(ordinals))
        self.n_train = len(df)

        for alpha in QUANTILES:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbose=-1,
            )
            model.fit(X, y)
            self.models[alpha] = model
        return self

    def predict_unit_quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        """
        物件行（transactions と同名カラムを持つ DataFrame）について
        ㎡単価の [25%点, 中央値, 75%点]（円/㎡）を返す。分位の単調性は保証する。
        """
        if not self.models:
            raise ValueError("モデルが学習されていません")
        X, _ = build_features(rows, encoders=self.encoders, fit_encoders=False)
        preds = np.column_stack(
            [np.exp(self.models[a].predict(X)) for a in QUANTILES]
        )
        return np.sort(preds, axis=1)

    # ---------- 保存・読込 ----------

    def save(self, model_path: Path = PRICE_MODEL_PATH,
             metadata_path: Path = PRICE_MODEL_METADATA_PATH,
             metrics: Optional[Dict[str, Any]] = None):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "models": self.models,
                    "encoders": self.encoders,
                    "train_period_min": self.train_period_min,
                    "train_period_max": self.train_period_max,
                    "n_train": self.n_train,
                },
                f,
            )
        metadata = {
            "trained_at": datetime.now().isoformat(),
            "model_type": "LightGBM quantile regression on log(unit price)",
            "quantiles": list(QUANTILES),
            "n_train": self.n_train,
            "train_period_ordinal": [self.train_period_min, self.train_period_max],
            "evaluation": metrics or {},
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_path: Path = PRICE_MODEL_PATH) -> "PriceModel":
        if not Path(model_path).exists():
            raise FileNotFoundError(f"理論価格モデルが見つかりません: {model_path}")
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.models = data["models"]
        obj.encoders = data["encoders"]
        obj.train_period_min = data["train_period_min"]
        obj.train_period_max = data["train_period_max"]
        obj.n_train = data["n_train"]
        return obj


def evaluate_period_split(
    transactions: pd.DataFrame, test_quarters: int = 4
) -> Dict[str, Any]:
    """
    期間分割で評価する: 最後の test_quarters 四半期を検証期間とし、
    それ以前だけで学習したモデルで MAPE と区間被覆率を測る。

    Returns:
        指標の辞書（mape, coverage_50, mean_interval_ratio, 件数, 期間情報）
    """
    df = transactions.copy()
    df = df.dropna(subset=["trade_price", "area"])
    df = df[df["area"] > 0]
    df["ordinal"] = [
        period_ordinal(int(y), int(q))
        for y, q in zip(df["period_year"], df["period_quarter"])
    ]
    periods = sorted(df["ordinal"].unique())
    if len(periods) <= test_quarters:
        raise ValueError(
            f"期間が不足しています（{len(periods)}四半期）。期間分割評価には"
            f"{test_quarters + 1}四半期以上が必要です。"
        )
    cutoff = periods[-test_quarters]
    train_df = df[df["ordinal"] < cutoff]
    test_df = df[df["ordinal"] >= cutoff]

    model = PriceModel().fit(train_df)
    preds = model.predict_unit_quantiles(test_df)
    actual = (test_df["trade_price"] / test_df["area"]).to_numpy(dtype=float)

    mape = float(np.mean(np.abs(actual - preds[:, 1]) / actual) * 100)
    coverage = float(np.mean((actual >= preds[:, 0]) & (actual <= preds[:, 2])) * 100)
    width_ratio = float(np.mean(preds[:, 2] / np.maximum(preds[:, 0], 1)))

    return {
        "method": "period_split",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_period_ordinal": [int(train_df["ordinal"].min()), int(train_df["ordinal"].max())],
        "test_period_ordinal": [int(test_df["ordinal"].min()), int(test_df["ordinal"].max())],
        "mape_percent": round(mape, 2),
        "coverage_50_percent": round(coverage, 2),
        "coverage_50_target": 50.0,
        "mean_interval_ratio_q75_over_q25": round(width_ratio, 3),
    }


def train_price_model(
    transactions: pd.DataFrame, test_quarters: int = 4
) -> Tuple[PriceModel, Dict[str, Any]]:
    """
    期間分割で評価したうえで、全期間データで最終モデルを学習して返す。
    評価指標は「評価専用モデル」のもので、最終モデルの自己評価ではない。
    """
    metrics = evaluate_period_split(transactions, test_quarters=test_quarters)
    model = PriceModel().fit(transactions)
    return model, metrics


def render_evaluation_report(metrics: Dict[str, Any], data_note: str = "") -> str:
    """評価指標を Markdown レポートに整形する。"""
    lines = [
        "# 理論価格モデル 評価レポート（期間分割）",
        "",
        f"- 評価方法: 期間分割（古い期間で学習 → 新しい期間で検証）。ランダム分割は使わない",
        f"- 学習件数（評価用モデル）: {metrics.get('train_rows')} 件",
        f"- 検証件数: {metrics.get('test_rows')} 件",
        f"- MAPE（中央値モデル）: {metrics.get('mape_percent')} %",
        f"- 区間被覆率（25%点〜75%点に入った割合）: {metrics.get('coverage_50_percent')} %（理論値 50%）",
        f"- 平均区間幅（75%点/25%点）: {metrics.get('mean_interval_ratio_q75_over_q25')}",
        "",
        "解釈: MAPE が約9〜10%あるモデルでは、乖離15%未満の「割安」は"
        "誤差と区別できない。本システムのゲート閾値（15%）はこの前提で置いている。",
    ]
    if data_note:
        lines += ["", f"データについての注意: {data_note}"]
    return "\n".join(lines) + "\n"

"""
第2層: LightGBM による理論価格モデル（log(㎡単価) の分位点回帰）。

- 25%点・中央値・75%点を別モデルで学習し、予測区間として使う。
- 評価はランダム分割ではなく期間分割（古い期間で学習→新しい期間で検証）。
  ランダム分割は同一地区の将来情報が学習に混入し、精度を過大評価する
  （調査レポート「検証方法」）。現行 ModelTrainer のランダム分割評価は
  この期間分割評価に置き換える。
- カテゴリエンコーダは学習期間のデータでのみ fit する（未見は -1）。

構成は粒度×手法の本実験（experiments/granularity_results.md、固定プロトコル:
学習 ≤2024Q4 / 検証 2025Q1–2026Q1、対象 港区・世田谷区）で選んだ最良構成:
- 特徴量は te_extra 相当: 基本特徴量＋地区名・最寄駅名のターゲット
  エンコーディング（平滑化つき）＋建ぺい率・容積率・改装・用途・
  今後の利用目的・間取りタイプ・取引の事情等の有無。
  （全域・基本特徴量のみの旧構成は 港区 MAPE 26.18% / 世田谷区 20.74%、
   本構成は 23.21% / 16.23%。docs/EVALUATION.md 参照）
- ハイパーパラメータは学習期間内時系列CV（OOF）で選んだ値（GBM_PARAMS）。
- 予測区間には、学習期間内OOFで決めた較正係数 δ（log空間の対称拡大）を
  適用する。未較正の区間は被覆率が 33〜37% と過信気味だったため。
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

# --- 本実験で採用した構成（te_extra + 調整済みパラメータ + 区間較正） ---

# 学習期間内時系列CV（OOF MAPE）で選んだパラメータ。
# 実験では対象区によらず同一の値が選ばれた（学習プールが共通のため）。
GBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.06,
    num_leaves=63,
    min_child_samples=10,
    subsample=0.9,
    colsample_bytree=0.9,
)

TE_COLUMNS = ("district_name", "nearest_station")
TE_SMOOTHING = 20.0

_EXTRA_NUMERIC = ["coverage_ratio", "floor_area_ratio"]
_EXTRA_CATEGORICAL = ["renovation", "use", "purpose", "layout_type"]

EXTENDED_FEATURE_COLUMNS = (
    FEATURE_COLUMNS
    + [f"te_{c}" for c in TE_COLUMNS]
    + _EXTRA_NUMERIC
    + [f"{c}_encoded" for c in _EXTRA_CATEGORICAL]
    + ["has_remarks"]
)

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


def _clean_category(series: pd.Series) -> pd.Series:
    """欠損を専用カテゴリ "__missing__" に寄せて文字列化する。"""
    raw = series if series is not None else pd.Series(dtype=object)
    return raw.astype(object).where(raw.notna(), "__missing__").astype(str)


class SmoothedTargetEncoder:
    """平滑化つきターゲットエンコーディング。fit したデータの統計だけで変換する。

    実験ハーネス（src/market/experiments.py）の同名クラスと同一の方式。
    未見カテゴリは全体平均にフォールバックする。
    """

    def __init__(self, smoothing: float = TE_SMOOTHING):
        self.smoothing = smoothing
        self.global_mean: float = 0.0
        self.mapping: Dict[str, float] = {}

    def fit(self, values: pd.Series, y: np.ndarray) -> "SmoothedTargetEncoder":
        v = _clean_category(values)
        self.global_mean = float(np.mean(y))
        stats = (
            pd.DataFrame({"v": v.to_numpy(), "y": y})
            .groupby("v")["y"]
            .agg(["sum", "count"])
        )
        m = self.smoothing
        self.mapping = {
            key: (row["sum"] + m * self.global_mean) / (row["count"] + m)
            for key, row in stats.iterrows()
        }
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        v = _clean_category(values)
        return v.map(self.mapping).fillna(self.global_mean).to_numpy(dtype=float)

    def to_state(self) -> Dict[str, Any]:
        return {"global_mean": self.global_mean, "mapping": dict(self.mapping)}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "SmoothedTargetEncoder":
        enc = cls()
        enc.global_mean = float(state.get("global_mean", 0.0))
        enc.mapping = {str(k): float(v) for k, v in state.get("mapping", {}).items()}
        return enc


def _layout_type(floor_plan: Any) -> str:
    """間取り詳細: "3LDK"→"LDK" 等。実データの間取りは全角表記（"３ＬＤＫ"）の
    ため現状はほぼ "__missing__" になり、特徴量としては実質不活性
    （実験でも同一実装で検証済み。本実験の数字はこの前提で出ている）。"""
    if not isinstance(floor_plan, str):
        return "__missing__"
    m = re.search(r"([A-Za-z]+)\s*$", floor_plan.strip())
    return m.group(1).upper() if m else "__missing__"


def _calibration_delta_from_oof(oof: pd.DataFrame) -> float:
    """OOF予測（取引総額）の被覆率が50%以上になる最小の log 空間対称拡大 δ（≥0）。

    実験ハーネスの calibration_delta と同一の方式。oof は
    actual / q25 / median / q75（いずれも取引総額・円）の列を持つ DataFrame。
    """
    if oof is None or oof.empty:
        return 0.0
    actual = np.log(oof["actual"].to_numpy(dtype=float))
    lo = np.log(np.maximum(oof["q25"].to_numpy(dtype=float), 1.0))
    hi = np.log(np.maximum(oof["q75"].to_numpy(dtype=float), 1.0))

    def coverage(delta: float) -> float:
        return float(np.mean((actual >= lo - delta) & (actual <= hi + delta)))

    if coverage(0.0) >= 0.5:
        return 0.0
    lo_d, hi_d = 0.0, 1.0
    for _ in range(40):
        mid = (lo_d + hi_d) / 2
        if coverage(mid) >= 0.5:
            hi_d = mid
        else:
            lo_d = mid
    return float(hi_d)


class PriceModel:
    """log(㎡単価) の分位点回帰モデル群（25%点・中央値・75%点）。

    特徴量は te_extra 構成（基本＋TE＋未使用列の活用）、パラメータは
    GBM_PARAMS、予測区間には学習期間内OOFで決めた較正 δ を適用する。
    feature_version=1 は旧構成（基本特徴量のみ・較正なし）で保存された
    モデルの読み込み互換用。
    """

    def __init__(self):
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.encoders: Dict[str, Dict[str, int]] = {}
        self.te_encoders: Dict[str, SmoothedTargetEncoder] = {}
        self.extra_encoders: Dict[str, Dict[str, int]] = {}
        self.extra_medians: Dict[str, float] = {}
        self.calibration_delta: float = 0.0
        self.feature_version: int = 2
        self.train_period_min: Optional[int] = None
        self.train_period_max: Optional[int] = None
        self.n_train: int = 0

    # ---------- 特徴量 ----------

    def _build_all_features(
        self, df: pd.DataFrame, y: Optional[np.ndarray] = None, fit: bool = False
    ) -> pd.DataFrame:
        """基本特徴量＋拡張特徴量（te_extra 構成）を作る。

        fit=True のときだけ TE・中央値・エンコーダを学習する（学習データ専用）。
        予測時は fit 済みの統計を使い、列が無い入力（listing 行など）は
        欠損として扱う（TE は全体平均、数値は学習時中央値、カテゴリは -1）。
        """
        if fit:
            X, self.encoders = build_features(df, fit_encoders=True)
        else:
            X, _ = build_features(df, encoders=self.encoders, fit_encoders=False)
        X = X.copy()

        for col in TE_COLUMNS:
            values = df[col] if col in df else pd.Series([None] * len(df), index=df.index)
            if fit:
                self.te_encoders[col] = SmoothedTargetEncoder().fit(values, y)
            enc = self.te_encoders.get(col)
            X[f"te_{col}"] = enc.transform(values) if enc is not None else 0.0

        for col in _EXTRA_NUMERIC:
            raw = (
                pd.to_numeric(df[col], errors="coerce")
                if col in df
                else pd.Series(np.nan, index=df.index)
            )
            if fit:
                self.extra_medians[col] = float(raw.median()) if raw.notna().any() else 0.0
            X[col] = raw.fillna(self.extra_medians.get(col, 0.0)).to_numpy()

        extra_series: Dict[str, pd.Series] = {}
        for col in ("renovation", "use", "purpose"):
            extra_series[col] = (
                df[col] if col in df else pd.Series([None] * len(df), index=df.index)
            )
        extra_series["layout_type"] = (
            df["floor_plan"].apply(_layout_type)
            if "floor_plan" in df
            else pd.Series(["__missing__"] * len(df), index=df.index)
        )
        for name, series in extra_series.items():
            cleaned = _clean_category(series)
            if fit:
                self.extra_encoders[name] = {
                    v: i for i, v in enumerate(sorted(cleaned.unique()))
                }
            mapping = self.extra_encoders.get(name, {})
            X[f"{name}_encoded"] = cleaned.map(mapping).fillna(-1).astype(int)

        remarks = df["remarks"] if "remarks" in df else pd.Series([None] * len(df), index=df.index)
        X["has_remarks"] = (
            remarks.astype(object)
            .where(remarks.notna(), "")
            .astype(str)
            .str.strip()
            .ne("")
            .astype(float)
            .to_numpy()
        )
        return X[EXTENDED_FEATURE_COLUMNS]

    # ---------- 学習 ----------

    def _fit_core(self, df: pd.DataFrame) -> "PriceModel":
        """較正なしでモデル本体だけを学習する（OOF のフォールド学習にも使う）。"""
        unit_price = df["trade_price"] / df["area"]
        y = np.log(unit_price.to_numpy(dtype=float))

        X = self._build_all_features(df, y=y, fit=True)
        ordinals = X["period_ordinal"].to_numpy()
        self.train_period_min = int(np.nanmin(ordinals))
        self.train_period_max = int(np.nanmax(ordinals))
        self.n_train = len(df)

        for alpha in QUANTILES:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                random_state=42,
                verbose=-1,
                **GBM_PARAMS,
            )
            model.fit(X, y)
            self.models[alpha] = model
        return self

    def _compute_calibration_delta(self, df: pd.DataFrame) -> float:
        """学習期間内の時系列OOF予測から区間較正係数 δ を決める。

        学習期間を時系列4ブロックに切り、先頭を除く各ブロックを
        「それ以前だけで学習したモデル」で予測して OOF を作る。
        特徴量・TE もフォールドごとに再計算する（リーク防止）。
        ブロックが作れない・OOF が少ない場合は 0（較正なし）。
        """
        d = df.copy()
        d["ordinal"] = [
            period_ordinal(int(y), int(q))
            for y, q in zip(d["period_year"], d["period_quarter"])
        ]
        ordinals = np.array(sorted(d["ordinal"].unique()))
        if len(ordinals) < 4:
            return 0.0
        groups = np.array_split(ordinals, 4)[1:]
        parts = []
        for g in groups:
            fold_train = d[d["ordinal"] < int(g.min())]
            block = d[d["ordinal"].isin(set(g.tolist()))]
            if len(fold_train) < MIN_TRAIN_ROWS or block.empty:
                continue
            temp = PriceModel()._fit_core(fold_train)
            unit_q = temp._predict_unit_raw(block)
            area = block["area"].to_numpy(dtype=float)
            parts.append(
                pd.DataFrame(
                    {
                        "actual": block["trade_price"].to_numpy(dtype=float),
                        "q25": unit_q[:, 0] * area,
                        "median": unit_q[:, 1] * area,
                        "q75": unit_q[:, 2] * area,
                    }
                )
            )
        if not parts:
            return 0.0
        oof = pd.concat(parts)
        if len(oof) < 30:
            return 0.0
        return _calibration_delta_from_oof(oof)

    def fit(self, transactions: pd.DataFrame) -> "PriceModel":
        df = transactions.copy()
        df = df.dropna(subset=["trade_price", "area"])
        df = df[df["area"] > 0]
        if len(df) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"学習データが不足しています（{len(df)}件）。最低{MIN_TRAIN_ROWS}件の成約事例が必要です。"
            )
        self._fit_core(df)
        self.calibration_delta = self._compute_calibration_delta(df)
        return self

    # ---------- 予測 ----------

    def _predict_unit_raw(self, rows: pd.DataFrame) -> np.ndarray:
        """較正を適用しない ㎡単価の分位点予測 [q25, median, q75]。"""
        if self.feature_version >= 2:
            X = self._build_all_features(rows, fit=False)
        else:
            X, _ = build_features(rows, encoders=self.encoders, fit_encoders=False)
        preds = np.column_stack(
            [np.exp(self.models[a].predict(X)) for a in QUANTILES]
        )
        return np.sort(preds, axis=1)

    def predict_unit_quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        """
        物件行（transactions と同名カラムを持つ DataFrame）について
        ㎡単価の [25%点, 中央値, 75%点]（円/㎡）を返す。分位の単調性は保証する。
        区間には fit 時に決めた較正 δ（log空間の対称拡大）を適用する。
        """
        if not self.models:
            raise ValueError("モデルが学習されていません")
        preds = self._predict_unit_raw(rows)
        if self.calibration_delta > 0:
            preds = preds.copy()
            preds[:, 0] *= np.exp(-self.calibration_delta)
            preds[:, 2] *= np.exp(self.calibration_delta)
            preds = np.sort(preds, axis=1)
        return preds

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
                    "te_encoders": {
                        col: enc.to_state() for col, enc in self.te_encoders.items()
                    },
                    "extra_encoders": self.extra_encoders,
                    "extra_medians": self.extra_medians,
                    "calibration_delta": self.calibration_delta,
                    "feature_version": self.feature_version,
                    "train_period_min": self.train_period_min,
                    "train_period_max": self.train_period_max,
                    "n_train": self.n_train,
                },
                f,
            )
        metadata = {
            "trained_at": datetime.now().isoformat(),
            "model_type": "LightGBM quantile regression on log(unit price)",
            "feature_set": "te_extra (base + district/station TE + coverage/FAR/renovation/use/purpose/layout/remarks)",
            "gbm_params": GBM_PARAMS,
            "calibration_delta": self.calibration_delta,
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
        # 旧形式（feature_version 無し）は基本特徴量のみのモデルとして読み込む
        obj.feature_version = int(data.get("feature_version", 1))
        obj.te_encoders = {
            col: SmoothedTargetEncoder.from_state(state)
            for col, state in data.get("te_encoders", {}).items()
        }
        obj.extra_encoders = data.get("extra_encoders", {})
        obj.extra_medians = data.get("extra_medians", {})
        obj.calibration_delta = float(data.get("calibration_delta", 0.0))
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
        "解釈: 実データ（東京23区＋神奈川 82,575件、固定プロトコルの本実験）では"
        "MAPE は区によって 16〜23% 程度であり、先行研究の約9%（所在階・"
        "マンション名等を含む大規模データ）には届かない。乖離15%のゲートは"
        "誤差帯の中にあるため、候補表示では事例数・区間幅・リスクフラグの"
        "併記が前提になる（docs/EVALUATION.md）。",
    ]
    if data_note:
        lines += ["", f"データについての注意: {data_note}"]
    return "\n".join(lines) + "\n"

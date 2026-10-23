"""
精度改善の実験ハーネス: 「どの粒度・どの方法が効くか」を固定プロトコルで検証する。

本番モデル（price_model.py）とは分離した実験専用コード。ここでの結論を
本番へ移植するかどうかは、結果を見て別途判断する。

固定の評価プロトコル（全バリアント共通・変更禁止）:
- 学習期間: 取引時期 ≤ 2024Q4、検証期間: 2025Q1〜2026Q1。
- データがその期間を満たさない（学習行が不足する）場合は、利用可能な
  最後の5四半期を検証期間にするフォールバックを試し、それでも学習行が
  足りなければ検証四半期数を 4→3→2→1 と縮める。使ったプロトコルは
  結果に必ず明記する。
- 検証行は「対象区（既定: 港区）」の行に限る。対象区の検証行が不足する
  場合は全区の検証行で評価し、その旨を結果に明記する。
- 指標: MAPE（中央値モデル、取引総額ベース）、50%区間被覆率、
  区間幅（q75/q25 の平均）、検証件数。

実験軸:
- A. 学習データの範囲: ward=対象区のみ / all=全域・全期間 /
  all_3y=全域・学習期間末から直近3年 / all_2y=同2年
- B. 特徴量: base=現行（price_model.build_features） /
  te=base＋地区名・最寄駅名のターゲットエンコーディング /
  te_extra=te＋未使用列（建ぺい率・容積率・改装・用途・今後の利用目的・
  間取り詳細・取引の事情の有無）
- C. 手法: gbm=LightGBM分位点回帰 / layer1=類似成約の中央値（プール粒度
  city/district/station） / blend=gbm と layer1 の幾何ブレンド
  （gbm は tuned=True で学習期間内時系列CVによるパラメータ調整版になる）
- D. 目的変数: unit=log(㎡単価) / total=log(取引総額)

リーク防止（厳守）:
- ターゲットエンコーディング・時点補正指数・カテゴリエンコーダは、
  常に「その予測に使う学習側データ」だけで計算する。学習期間内CVでも
  フォールドごとに再計算する。
- ハイパーパラメータ選択・ブレンド重み・区間較正係数は、すべて学習期間内の
  時系列CV（out-of-fold予測）だけで決める。検証期間の情報は一切使わない。
- 区間較正は OOF 予測に対して log 空間で対称拡大 δ を求め（被覆率50%を
  達成する最小の δ≥0）、検証区間へそのまま適用する。
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.market.price_model import QUANTILES, build_features
from src.market.time_adjust import UnitPriceIndex, period_from_ordinal, period_ordinal

# ---- 固定プロトコル ----
FIXED_TRAIN_END = period_ordinal(2024, 4)
FIXED_TEST_START = period_ordinal(2025, 1)
FIXED_TEST_END = period_ordinal(2026, 1)
MIN_SPLIT_TRAIN = 100
MIN_SPLIT_TEST = 10
MIN_GBM_TRAIN = 100

DEFAULT_GBM_PARAMS: Dict[str, Any] = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9,
)

# 調整版の探索範囲（控えめ）。中央値モデルの学習期間内CVで選ぶ。
PARAM_GRID: List[Dict[str, Any]] = [
    dict(n_estimators=400, learning_rate=lr, num_leaves=leaves,
         min_child_samples=mcs, subsample=0.9, colsample_bytree=0.9)
    for lr in (0.03, 0.06)
    for leaves in (15, 31, 63)
    for mcs in (10, 30)
]

TE_COLUMNS = ("district_name", "nearest_station")
TE_SMOOTHING = 20.0

LAYER1_MIN_POOL = 5
LAYER1_AGE_BAND = 10   # 築年帯（年）
LAYER1_AREA_BAND = 20.0  # 面積帯（㎡）

BLEND_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


# ============================================================
# データ準備と分割
# ============================================================

def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """実験用に整える: 必須項目の欠損除去、ordinal と ㎡単価の付与。"""
    d = df.copy()
    d = d.dropna(subset=["trade_price", "area"])
    d = d[d["area"] > 0]
    d["unit_price"] = d["trade_price"].astype(float) / d["area"].astype(float)
    d["ordinal"] = [
        period_ordinal(int(y), int(q))
        for y, q in zip(d["period_year"], d["period_quarter"])
    ]
    return d.sort_values("ordinal").reset_index(drop=True)


@dataclass
class Split:
    train: pd.DataFrame          # 学習期間の全行（スコープ適用前）
    test: pd.DataFrame           # 検証行（対象区で絞り済み）
    protocol: str                # "fixed" / "fallback_last{k}q"
    note: str = ""

    @property
    def train_period(self) -> str:
        return _period_range(self.train)

    @property
    def test_period(self) -> str:
        return _period_range(self.test)


def _period_range(df: pd.DataFrame) -> str:
    if df.empty:
        return "-"
    lo, hi = int(df["ordinal"].min()), int(df["ordinal"].max())
    y1, q1 = period_from_ordinal(lo)
    y2, q2 = period_from_ordinal(hi)
    return f"{y1}Q{q1}–{y2}Q{q2}"


def resolve_split(df: pd.DataFrame, target_ward: str = "港区") -> Split:
    """
    固定プロトコルで学習/検証に分割する。満たせない場合はフォールバック。

    フォールバック順: 最後の5四半期を検証 → 4 → 3 → 2 → 1四半期。
    いずれも「学習行 ≥ MIN_SPLIT_TRAIN かつ 検証行 ≥ MIN_SPLIT_TEST」を満たす
    最初のものを採用する。
    """
    train = df[df["ordinal"] <= FIXED_TRAIN_END]
    test_all = df[(df["ordinal"] >= FIXED_TEST_START) & (df["ordinal"] <= FIXED_TEST_END)]
    protocol = "fixed"
    note = ""
    if len(train) < MIN_SPLIT_TRAIN or len(test_all) < MIN_SPLIT_TEST:
        adopted = None
        ordinals = sorted(df["ordinal"].unique())
        for k in (5, 4, 3, 2, 1):
            if len(ordinals) <= k:
                continue
            test_ords = set(ordinals[-k:])
            cand_train = df[~df["ordinal"].isin(test_ords)]
            cand_test = df[df["ordinal"].isin(test_ords)]
            if len(cand_train) >= MIN_SPLIT_TRAIN and len(cand_test) >= MIN_SPLIT_TEST:
                adopted = (cand_train, cand_test, k)
                break
        if adopted is None:
            raise ValueError(
                f"分割できません（全{len(df)}件）。学習≥{MIN_SPLIT_TRAIN}件・"
                f"検証≥{MIN_SPLIT_TEST}件を満たす期間分割が存在しません。"
            )
        train, test_all, k = adopted
        protocol = f"fallback_last{k}q"
        note = (
            f"固定プロトコル（学習≤2024Q4／検証2025Q1–2026Q1）を満たす行数が"
            f"不足したため、最後の{k}四半期を検証期間とするフォールバックを使用。"
        )

    test = test_all[test_all["municipality"] == target_ward]
    if len(test) < MIN_SPLIT_TEST:
        test = test_all
        note = (note + " " if note else "") + (
            f"対象区（{target_ward}）の検証行が{MIN_SPLIT_TEST}件未満のため、"
            "検証は全区の行で実施。"
        )
    return Split(train=train, test=test, protocol=protocol, note=note.strip())


def apply_scope(train: pd.DataFrame, scope: str, target_ward: str) -> pd.DataFrame:
    """軸A: 学習プールをスコープで絞る。"""
    if scope == "ward":
        return train[train["municipality"] == target_ward]
    if scope == "all":
        return train
    max_ord = int(train["ordinal"].max())
    if scope == "all_3y":
        return train[train["ordinal"] > max_ord - 12]
    if scope == "all_2y":
        return train[train["ordinal"] > max_ord - 8]
    raise ValueError(f"未知のスコープです: {scope}")


# ============================================================
# 特徴量
# ============================================================

def _target_values(df: pd.DataFrame, target: str) -> np.ndarray:
    """目的変数（log）。unit=㎡単価 / total=取引総額。"""
    if target == "unit":
        return np.log(df["unit_price"].to_numpy(dtype=float))
    if target == "total":
        return np.log(df["trade_price"].to_numpy(dtype=float))
    raise ValueError(f"未知の目的変数です: {target}")


def _clean_category(series: pd.Series) -> pd.Series:
    raw = series if series is not None else pd.Series(dtype=object)
    return raw.astype(object).where(raw.notna(), "__missing__").astype(str)


class SmoothedTargetEncoder:
    """平滑化つきターゲットエンコーディング。fit したデータの統計だけで変換する。"""

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


def _label_encode_pair(
    train_values: pd.Series, test_values: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """学習側だけでマッピングを作るラベルエンコード（未見は -1）。"""
    tr = _clean_category(train_values)
    te = _clean_category(test_values)
    mapping = {v: i for i, v in enumerate(sorted(tr.unique()))}
    return (
        tr.map(mapping).fillna(-1).to_numpy(dtype=float),
        te.map(mapping).fillna(-1).to_numpy(dtype=float),
    )


def _layout_type(floor_plan: Any) -> str:
    """間取り詳細: "3LDK"→"LDK"、"2DK"→"DK"、"1K"→"K"、"1R"→"R"。"""
    if not isinstance(floor_plan, str):
        return "__missing__"
    import re

    m = re.search(r"([A-Za-z]+)\s*$", floor_plan.strip())
    return m.group(1).upper() if m else "__missing__"


def build_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: str,
    target: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    学習/検証の特徴量を作る。エンコーダ・TE・中央値はすべて train_df だけで決める。

    Returns: (X_train, X_test, columns, notes)
    """
    notes: List[str] = []
    y_train = _target_values(train_df, target)

    X_train, encoders = build_features(train_df, fit_encoders=True)
    X_test, _ = build_features(test_df, encoders=encoders, fit_encoders=False)
    X_train = X_train.copy()
    X_test = X_test.copy()

    if feature_set in ("te", "te_extra"):
        for col in TE_COLUMNS:
            if col in train_df:
                enc = SmoothedTargetEncoder().fit(train_df[col], y_train)
                X_train[f"te_{col}"] = enc.transform(train_df[col])
                X_test[f"te_{col}"] = enc.transform(test_df[col])

    if feature_set == "te_extra":
        # 数値の未使用列（DBに列が無い／学習側でほぼ欠損なら無効化して記録）
        for col in ("coverage_ratio", "floor_area_ratio"):
            if col in train_df:
                tr = pd.to_numeric(train_df[col], errors="coerce")
                if tr.notna().sum() >= 10:
                    med = float(tr.median())
                    X_train[col] = tr.fillna(med).to_numpy()
                    X_test[col] = (
                        pd.to_numeric(test_df[col], errors="coerce")
                        .fillna(med)
                        .to_numpy()
                    )
                else:
                    notes.append(f"{col} は学習側の有効行が少なく特徴量から除外")
            else:
                notes.append(f"{col} 列がデータに無いため特徴量から除外")
        # カテゴリの未使用列（学習側だけでマッピング）
        extra_cats = []
        if "renovation" in train_df:
            extra_cats.append(("renovation", train_df["renovation"], test_df["renovation"]))
        if "use" in train_df:
            extra_cats.append(("use", train_df["use"], test_df["use"]))
        if "purpose" in train_df:
            extra_cats.append(("purpose", train_df["purpose"], test_df["purpose"]))
        extra_cats.append((
            "layout_type",
            train_df["floor_plan"].apply(_layout_type) if "floor_plan" in train_df else pd.Series(dtype=object, index=train_df.index),
            test_df["floor_plan"].apply(_layout_type) if "floor_plan" in test_df else pd.Series(dtype=object, index=test_df.index),
        ))
        for name, tr_values, te_values in extra_cats:
            tr_codes, te_codes = _label_encode_pair(tr_values, te_values)
            X_train[f"{name}_encoded"] = tr_codes
            X_test[f"{name}_encoded"] = te_codes
        # 取引の事情等の有無（自由記述の有無だけを使う）
        if "remarks" in train_df:
            def _has(s: pd.Series) -> np.ndarray:
                return s.astype(object).where(s.notna(), "").astype(str).str.strip().ne("").astype(float).to_numpy()

            X_train["has_remarks"] = _has(train_df["remarks"])
            X_test["has_remarks"] = _has(test_df["remarks"])

    columns = list(X_train.columns)
    return X_train[columns], X_test[columns], columns, notes


# ============================================================
# GBM（分位点回帰）
# ============================================================

def fit_quantile_gbms(
    X: pd.DataFrame, y: np.ndarray, params: Dict[str, Any], alphas: Sequence[float] = QUANTILES
) -> Dict[float, lgb.LGBMRegressor]:
    models: Dict[float, lgb.LGBMRegressor] = {}
    for alpha in alphas:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha, random_state=42, verbose=-1, **params
        )
        model.fit(X, y)
        models[alpha] = model
    return models


def predict_price_quantiles(
    models: Dict[float, lgb.LGBMRegressor],
    X: pd.DataFrame,
    rows: pd.DataFrame,
    target: str,
) -> np.ndarray:
    """分位点予測を取引総額（円）に直して返す。列は [q25, median, q75]。"""
    cols = []
    for alpha in QUANTILES:
        pred_log = models[alpha].predict(X)
        price = np.exp(pred_log)
        if target == "unit":
            price = price * rows["area"].to_numpy(dtype=float)
        cols.append(price)
    return np.sort(np.column_stack(cols), axis=1)


# ============================================================
# 第1層: 帯（バンド）ベースの類似成約
# ============================================================

def _age_band(row_year: Any, period_year: Any) -> int:
    try:
        age = int(period_year) - int(row_year)
    except (TypeError, ValueError):
        return -1
    if age < 0:
        return -1
    return (age // LAYER1_AGE_BAND) * LAYER1_AGE_BAND


def _area_band(area: Any) -> float:
    try:
        a = float(area)
    except (TypeError, ValueError):
        return -1.0
    return float(np.floor(a / LAYER1_AREA_BAND) * LAYER1_AREA_BAND)


def _pool_keys(df: pd.DataFrame, granularity: str) -> List[Tuple]:
    age_b = [_age_band(y, py) for y, py in zip(df["building_year"], df["period_year"])]
    area_b = [_area_band(a) for a in df["area"]]
    muni = df["municipality_code"].astype(str).tolist()
    district = _clean_category(df["district_name"]).tolist() if "district_name" in df else ["__missing__"] * len(df)
    station = _clean_category(df["nearest_station"]).tolist() if "nearest_station" in df else ["__missing__"] * len(df)
    keys = []
    for i in range(len(df)):
        if granularity == "city":
            keys.append((muni[i], age_b[i], area_b[i]))
        elif granularity == "district":
            keys.append((muni[i], district[i], age_b[i], area_b[i]))
        elif granularity == "station":
            keys.append((station[i], age_b[i], area_b[i]))
        else:
            raise ValueError(f"未知のプール粒度です: {granularity}")
    return keys


def _relaxed_keys(key: Tuple, granularity: str) -> List[Tuple]:
    """プールが痩せたときの緩和段階（具体→広域）。"""
    if granularity == "city":
        muni, ab, arb = key
        return [key, (muni, ab), (muni,)]
    if granularity == "district":
        muni, dist, ab, arb = key
        return [key, (muni, dist), (muni, ab, arb), (muni,)]
    station, ab, arb = key
    return [key, (station,)]


def predict_layer1(
    train_df: pd.DataFrame,
    rows: pd.DataFrame,
    granularity: str = "city",
    index: Optional[UnitPriceIndex] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    学習プールだけから類似成約の四分位（取引総額）を予測する。

    予測できない行は NaN。時点補正指数も学習プールだけで作る。
    Returns: (pred[n,3], info) info には予測可能率とフル帯一致率を含む。
    """
    index = index or UnitPriceIndex.from_transactions(train_df)
    train_keys = _pool_keys(train_df, granularity)
    pools: Dict[Tuple, List[int]] = {}
    for i, k in enumerate(train_keys):
        pools.setdefault(k, []).append(i)

    # 緩和キー用のプールも先に作る（キーの射影ごとに集約）
    relaxed_pools: Dict[Tuple, List[int]] = {}

    def pool_for(key: Tuple) -> List[int]:
        if key in pools:
            return pools[key]
        if key not in relaxed_pools:
            members = [i for i, k in enumerate(train_keys) if _project(k, granularity, key) == key]
            relaxed_pools[key] = members
        return relaxed_pools[key]

    train_unit = train_df["unit_price"].to_numpy(dtype=float)
    train_periods = list(zip(train_df["period_year"], train_df["period_quarter"]))
    train_muni = train_df["municipality_code"].astype(str).tolist()

    row_keys = _pool_keys(rows, granularity)
    row_muni = rows["municipality_code"].astype(str).tolist()
    preds = np.full((len(rows), 3), np.nan)
    full_band_hits = 0
    for r in range(len(rows)):
        key = row_keys[r]
        chosen: Optional[List[int]] = None
        for cand in _relaxed_keys(key, granularity):
            members = pool_for(cand)
            if len(members) >= LAYER1_MIN_POOL:
                chosen = members
                if cand == key:
                    full_band_hits += 1
                break
        if chosen is None:
            # 最も広い段階でも件数不足なら、市区町村プールで拾えるだけ拾う
            members = [i for i, m in enumerate(train_muni) if m == row_muni[r]]
            if members:
                chosen = members
        if not chosen:
            continue
        target_period = (int(rows["period_year"].iloc[r]), int(rows["period_quarter"].iloc[r]))
        pref = rows["prefecture"].iloc[r] if "prefecture" in rows else None
        adjusted = np.array([
            index.adjust(train_unit[i], train_periods[i], target_period, group=pref)
            for i in chosen
        ])
        area = float(rows["area"].iloc[r])
        qs = np.quantile(adjusted, [0.25, 0.5, 0.75]) * area
        preds[r] = qs
    info = {
        "pred_rate": float(np.mean(np.isfinite(preds[:, 1]))) if len(rows) else 0.0,
        "full_band_rate": float(full_band_hits / len(rows)) if len(rows) else 0.0,
    }
    return preds, info


def _project(full_key: Tuple, granularity: str, relaxed_key: Tuple) -> Tuple:
    """フルキーを緩和キーの形に射影する（緩和プールの照合用）。"""
    if granularity == "city":
        muni, ab, arb = full_key
        if len(relaxed_key) == 3:
            return full_key
        if len(relaxed_key) == 2:
            return (muni, ab)
        return (muni,)
    if granularity == "district":
        muni, dist, ab, arb = full_key
        if len(relaxed_key) == 4:
            return full_key
        if len(relaxed_key) == 2:
            return (muni, dist)
        if len(relaxed_key) == 3:
            return (muni, ab, arb)
        return (muni,)
    station, ab, arb = full_key
    if len(relaxed_key) == 3:
        return full_key
    return (station,)


# ============================================================
# 学習期間内CV（時系列OOF）: 調整・ブレンド重み・較正に使う
# ============================================================

def _oof_blocks(train_df: pd.DataFrame, n_blocks: int = 3) -> List[pd.DataFrame]:
    """学習期間の後半を時系列の検証ブロックに切る（先頭ブロックは常時学習側）。"""
    ordinals = np.array(sorted(train_df["ordinal"].unique()))
    if len(ordinals) < n_blocks + 1:
        return []
    groups = np.array_split(ordinals, n_blocks + 1)[1:]
    blocks = []
    for g in groups:
        blocks.append(train_df[train_df["ordinal"].isin(set(g.tolist()))])
    return blocks


def oof_gbm_predictions(
    train_df: pd.DataFrame,
    feature_set: str,
    target: str,
    params: Dict[str, Any],
    alphas: Sequence[float] = QUANTILES,
) -> Optional[pd.DataFrame]:
    """
    学習期間内の時系列OOF予測。各フォールドで特徴量・TEも再計算する（リーク防止）。
    返り値には元の index と実績・予測四分位が入る。作れなければ None。
    """
    blocks = _oof_blocks(train_df)
    if not blocks:
        return None
    out = []
    for block in blocks:
        fold_train = train_df[train_df["ordinal"] < int(block["ordinal"].min())]
        if len(fold_train) < MIN_GBM_TRAIN:
            continue
        X_tr, X_va, _, _ = build_xy(fold_train, block, feature_set, target)
        y_tr = _target_values(fold_train, target)
        models = fit_quantile_gbms(X_tr, y_tr, params, alphas=alphas)
        if len(alphas) == len(QUANTILES):
            preds = predict_price_quantiles(models, X_va, block, target)
        else:
            med = np.exp(models[alphas[0]].predict(X_va))
            if target == "unit":
                med = med * block["area"].to_numpy(dtype=float)
            preds = np.column_stack([med, med, med])
        part = pd.DataFrame({
            "actual": block["trade_price"].to_numpy(dtype=float),
            "q25": preds[:, 0], "median": preds[:, 1], "q75": preds[:, 2],
        }, index=block.index)
        out.append(part)
    if not out:
        return None
    return pd.concat(out)


def tune_params_oof(
    train_df: pd.DataFrame, feature_set: str, target: str
) -> Tuple[Dict[str, Any], Optional[float]]:
    """中央値モデルのOOF MAPEでパラメータを選ぶ（学習期間内のみ）。"""
    best_params: Optional[Dict[str, Any]] = None
    best_mape: Optional[float] = None
    for params in PARAM_GRID:
        oof = oof_gbm_predictions(train_df, feature_set, target, params, alphas=(0.5,))
        if oof is None or oof.empty:
            continue
        mape = float(np.mean(np.abs(oof["actual"] - oof["median"]) / oof["actual"]) * 100)
        if best_mape is None or mape < best_mape:
            best_mape, best_params = mape, params
    if best_params is None:
        return dict(DEFAULT_GBM_PARAMS), None
    return dict(best_params), best_mape


def calibration_delta(oof: pd.DataFrame) -> float:
    """OOF予測の被覆率が50%以上になる最小の log 空間対称拡大 δ（≥0）。"""
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


def apply_calibration(preds: np.ndarray, delta: float) -> np.ndarray:
    if delta <= 0:
        return preds
    out = preds.copy()
    out[:, 0] = out[:, 0] * np.exp(-delta)
    out[:, 2] = out[:, 2] * np.exp(delta)
    return out


# ============================================================
# 指標
# ============================================================

def compute_metrics(actual: np.ndarray, preds: np.ndarray) -> Dict[str, Any]:
    ok = np.isfinite(preds[:, 1]) & np.isfinite(actual) & (actual > 0)
    n_total = len(actual)
    if ok.sum() == 0:
        return {"n": 0, "pred_rate": 0.0, "mape": None, "coverage": None, "width": None}
    a = actual[ok]
    p = preds[ok]
    mape = float(np.mean(np.abs(a - p[:, 1]) / a) * 100)
    coverage = float(np.mean((a >= p[:, 0]) & (a <= p[:, 2])) * 100)
    width = float(np.mean(p[:, 2] / np.maximum(p[:, 0], 1.0)))
    return {
        "n": int(ok.sum()),
        "pred_rate": float(ok.sum() / n_total) if n_total else 0.0,
        "mape": round(mape, 2),
        "coverage": round(coverage, 2),
        "width": round(width, 3),
    }


# ============================================================
# 実験の実行
# ============================================================

@dataclass
class ExperimentConfig:
    scope: str = "all"          # A
    feature_set: str = "base"   # B
    method: str = "gbm"         # C: gbm / layer1 / blend
    target: str = "unit"        # D
    pool: str = "city"          # layer1/blend のプール粒度
    tuned: bool = False         # C(ii): パラメータ調整版
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        method = self.method + ("+tuned" if self.tuned and self.method in ("gbm", "blend") else "")
        if self.method in ("layer1", "blend"):
            method += f"[{self.pool}]"
        return f"A:{self.scope}/B:{self.feature_set}/C:{method}/D:{self.target}"

    def key(self) -> Tuple:
        return (self.scope, self.feature_set, self.method, self.target, self.pool, self.tuned)


def _run_config(
    split: Split, cfg: ExperimentConfig, target_ward: str
) -> Dict[str, Any]:
    """1構成を実行して結果 dict を返す。失敗は status=error で記録する。"""
    result: Dict[str, Any] = {
        "config": cfg.label(),
        "scope": cfg.scope, "feature_set": cfg.feature_set,
        "method": cfg.method, "tuned": cfg.tuned, "target": cfg.target,
        "pool": cfg.pool if cfg.method in ("layer1", "blend") else "",
        "status": "ok", "notes": "",
    }
    train_pool = apply_scope(split.train, cfg.scope, target_ward)
    test = split.test
    result["train_rows"] = int(len(train_pool))
    result["test_rows"] = int(len(test))
    notes: List[str] = []

    try:
        actual = test["trade_price"].to_numpy(dtype=float)
        preds: Optional[np.ndarray] = None
        params = dict(DEFAULT_GBM_PARAMS)
        oof: Optional[pd.DataFrame] = None

        if cfg.method in ("gbm", "blend"):
            if len(train_pool) < MIN_GBM_TRAIN:
                result["status"] = "skipped"
                result["notes"] = f"学習プールが{MIN_GBM_TRAIN}件未満のためGBMをスキップ"
                return result
            if cfg.tuned:
                params, oof_mape = tune_params_oof(train_pool, cfg.feature_set, cfg.target)
                result["chosen_params"] = str(params)
                if oof_mape is not None:
                    notes.append(f"調整時OOF MAPE={oof_mape:.2f}%")
            X_tr, X_te, _, feat_notes = build_xy(train_pool, test, cfg.feature_set, cfg.target)
            notes.extend(feat_notes)
            y_tr = _target_values(train_pool, cfg.target)
            models = fit_quantile_gbms(X_tr, y_tr, params)
            preds = predict_price_quantiles(models, X_te, test, cfg.target)
            oof = oof_gbm_predictions(train_pool, cfg.feature_set, cfg.target, params)

        if cfg.method == "layer1":
            preds, info = predict_layer1(train_pool, test, cfg.pool)
            result["pred_rate"] = round(info["pred_rate"], 3)
            notes.append(f"フル帯一致率={info['full_band_rate']:.2f}")

        if cfg.method == "blend":
            l1_preds, info = predict_layer1(train_pool, test, cfg.pool)
            result["pred_rate"] = round(info["pred_rate"], 3)
            weight = _select_blend_weight(train_pool, cfg, params)
            result["blend_weight_gbm"] = weight
            assert preds is not None
            blended = preds.copy()
            both = np.isfinite(l1_preds[:, 1])
            for j in range(3):
                blended[both, j] = np.exp(
                    weight * np.log(np.maximum(preds[both, j], 1.0))
                    + (1 - weight) * np.log(np.maximum(l1_preds[both, j], 1.0))
                )
            preds = np.sort(blended, axis=1)
            if oof is not None:
                oof = _blend_oof(oof, train_pool, cfg, weight)

        assert preds is not None
        metrics = compute_metrics(actual, preds)
        result.update({
            "test_n": metrics["n"],
            "mape_percent": metrics["mape"],
            "coverage_50_percent": metrics["coverage"],
            "mean_interval_ratio": metrics["width"],
        })
        if "pred_rate" not in result:
            result["pred_rate"] = round(metrics["pred_rate"], 3)

        # 区間較正（GBM系のみ。較正係数は学習期間内OOFだけで決める）
        if cfg.method in ("gbm", "blend") and oof is not None and len(oof) >= 30:
            delta = calibration_delta(oof)
            result["calibration_delta"] = round(delta, 4)
            cal_metrics = compute_metrics(actual, apply_calibration(preds, delta))
            result["coverage_cal_percent"] = cal_metrics["coverage"]
            result["width_cal"] = cal_metrics["width"]
        else:
            result["calibration_delta"] = None
            result["coverage_cal_percent"] = None
            result["width_cal"] = None
    except Exception as e:  # 実験は落とさず記録する
        result["status"] = "error"
        result["notes"] = f"{type(e).__name__}: {e}"
        return result

    result["notes"] = " / ".join(n for n in notes if n)
    return result


def _select_blend_weight(
    train_pool: pd.DataFrame, cfg: ExperimentConfig, params: Dict[str, Any]
) -> float:
    """学習期間内OOFでブレンド重み（GBM側）を選ぶ。OOFが作れなければ 0.5。"""
    oof = oof_gbm_predictions(train_pool, cfg.feature_set, cfg.target, params)
    blocks = _oof_blocks(train_pool)
    if oof is None or not blocks:
        return 0.5
    l1_rows: List[pd.DataFrame] = []
    for block in blocks:
        fold_train = train_pool[train_pool["ordinal"] < int(block["ordinal"].min())]
        if len(fold_train) < LAYER1_MIN_POOL:
            continue
        preds, _ = predict_layer1(fold_train, block, cfg.pool)
        l1_rows.append(pd.DataFrame({
            "l1_q25": preds[:, 0], "l1_median": preds[:, 1], "l1_q75": preds[:, 2],
        }, index=block.index))
    if not l1_rows:
        return 0.5
    l1_oof = pd.concat(l1_rows)
    joined = oof.join(l1_oof, how="inner")
    if joined.empty:
        return 0.5
    actual = joined["actual"].to_numpy(dtype=float)
    gbm_med = joined["median"].to_numpy(dtype=float)
    l1_med = joined["l1_median"].to_numpy(dtype=float)
    best_w, best_mape = 0.5, None
    for w in BLEND_WEIGHT_GRID:
        blended = np.where(
            np.isfinite(l1_med),
            np.exp(w * np.log(np.maximum(gbm_med, 1.0)) + (1 - w) * np.log(np.maximum(l1_med, 1.0))),
            gbm_med,
        )
        mape = float(np.mean(np.abs(actual - blended) / actual) * 100)
        if best_mape is None or mape < best_mape:
            best_mape, best_w = mape, w
    return float(best_w)


def _blend_oof(oof: pd.DataFrame, train_pool: pd.DataFrame, cfg: ExperimentConfig, weight: float) -> pd.DataFrame:
    """較正用に、OOFのGBM予測へ同じ重みで layer1 をブレンドしたものを返す。"""
    blocks = _oof_blocks(train_pool)
    l1_rows: List[pd.DataFrame] = []
    for block in blocks:
        fold_train = train_pool[train_pool["ordinal"] < int(block["ordinal"].min())]
        if len(fold_train) < LAYER1_MIN_POOL:
            continue
        preds, _ = predict_layer1(fold_train, block, cfg.pool)
        l1_rows.append(pd.DataFrame({
            "l1_q25": preds[:, 0], "l1_median": preds[:, 1], "l1_q75": preds[:, 2],
        }, index=block.index))
    if not l1_rows:
        return oof
    joined = oof.join(pd.concat(l1_rows), how="left")
    out = joined.copy()
    for gbm_col, l1_col in (("q25", "l1_q25"), ("median", "l1_median"), ("q75", "l1_q75")):
        both = np.isfinite(joined[l1_col].to_numpy(dtype=float))
        vals = joined[gbm_col].to_numpy(dtype=float)
        blended = np.exp(
            weight * np.log(np.maximum(vals, 1.0))
            + (1 - weight) * np.log(np.maximum(joined[l1_col].to_numpy(dtype=float), 1.0))
        )
        out[gbm_col] = np.where(both, blended, vals)
    return out[["actual", "q25", "median", "q75"]]


def ofat_configs() -> List[ExperimentConfig]:
    """一段目: 基準構成＋1軸ずつ変えた構成。"""
    base = ExperimentConfig()
    configs = [base]
    for scope in ("ward", "all_3y", "all_2y"):
        configs.append(ExperimentConfig(scope=scope))
    for fs in ("te", "te_extra"):
        configs.append(ExperimentConfig(feature_set=fs))
    configs.append(ExperimentConfig(tuned=True))
    for pool in ("city", "district", "station"):
        configs.append(ExperimentConfig(method="layer1", pool=pool))
    configs.append(ExperimentConfig(target="total"))
    return configs


def _best_level(results: List[Dict[str, Any]], field_name: str, candidates: Sequence[Any], default: Any) -> Any:
    """OFAT結果から、指定軸の最良水準（GBM系・status=ok の MAPE 最小）を返す。"""
    best, best_mape = default, None
    for r in results:
        if r.get("status") != "ok" or r.get("mape_percent") is None:
            continue
        if r.get(field_name) in candidates:
            if best_mape is None or r["mape_percent"] < best_mape:
                best_mape = r["mape_percent"]
                best = r[field_name]
    return best


def combined_configs(results: List[Dict[str, Any]]) -> List[ExperimentConfig]:
    """二段目: OFATの最良水準同士の組合せ（GBM版とブレンド版）。"""
    gbm_rows = [r for r in results if r.get("method") == "gbm" and r.get("status") == "ok"]
    if not gbm_rows:
        return []
    best_scope = _best_level(gbm_rows, "scope", ("ward", "all", "all_3y", "all_2y"), "all")
    best_fs = _best_level(gbm_rows, "feature_set", ("base", "te", "te_extra"), "base")
    best_target = _best_level(gbm_rows, "target", ("unit", "total"), "unit")
    tuned_rows = [r for r in gbm_rows if r.get("tuned")]
    plain_rows = [r for r in gbm_rows if not r.get("tuned")]
    tuned_best = min((r["mape_percent"] for r in tuned_rows if r["mape_percent"] is not None), default=None)
    plain_best = min((r["mape_percent"] for r in plain_rows if r["mape_percent"] is not None), default=None)
    use_tuned = tuned_best is not None and (plain_best is None or tuned_best <= plain_best)

    l1_rows = [r for r in results if r.get("method") == "layer1" and r.get("status") == "ok" and r.get("mape_percent") is not None]
    best_pool = "city"
    if l1_rows:
        best_pool = min(l1_rows, key=lambda r: r["mape_percent"])["pool"]

    return [
        ExperimentConfig(scope=best_scope, feature_set=best_fs, target=best_target,
                         tuned=use_tuned, name="combined_gbm"),
        ExperimentConfig(scope=best_scope, feature_set=best_fs, target=best_target,
                         method="blend", pool=best_pool, tuned=use_tuned,
                         name="combined_blend"),
    ]


def run_experiments(
    df: pd.DataFrame,
    target_ward: str = "港区",
    dataset_label: str = "",
    include_combined: bool = True,
    configs: Optional[List[ExperimentConfig]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    実験一式を実行する。Returns: (結果リスト, メタ情報)
    """
    prepared = prepare_transactions(df)
    split = resolve_split(prepared, target_ward=target_ward)
    meta: Dict[str, Any] = {
        "dataset": dataset_label,
        "target_ward": target_ward,
        "protocol": split.protocol,
        "split_note": split.note,
        "train_period": split.train_period,
        "test_period": split.test_period,
        "total_rows": int(len(prepared)),
    }
    run_configs = configs if configs is not None else ofat_configs()
    results: List[Dict[str, Any]] = []
    seen = set()
    for cfg in run_configs:
        if cfg.key() in seen:
            continue
        seen.add(cfg.key())
        results.append(_run_config(split, cfg, target_ward))
    if include_combined and configs is None:
        for cfg in combined_configs(results):
            if cfg.key() in seen:
                continue
            seen.add(cfg.key())
            results.append(_run_config(split, cfg, target_ward))
    for r in results:
        r["dataset"] = dataset_label
        r["protocol"] = split.protocol
        r["target_ward"] = target_ward
        r["train_period"] = split.train_period
        r["test_period"] = split.test_period
    return results, meta


# ============================================================
# 結果の出力
# ============================================================

RESULT_COLUMNS = [
    "run_at", "dataset", "protocol", "target_ward", "config",
    "scope", "feature_set", "method", "tuned", "target", "pool",
    "train_rows", "test_rows", "train_period", "test_period", "test_n",
    "mape_percent", "coverage_50_percent", "mean_interval_ratio",
    "calibration_delta", "coverage_cal_percent", "width_cal",
    "pred_rate", "blend_weight_gbm", "chosen_params", "status", "notes",
]


def results_to_frame(results: List[Dict[str, Any]], run_at: str) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {c: r.get(c) for c in RESULT_COLUMNS}
        row["run_at"] = run_at
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_results_markdown(results: List[Dict[str, Any]], meta: Dict[str, Any], run_at: str) -> str:
    """1回分の実行結果を Markdown セクションに整形する（追記用）。"""
    lines = [
        f"## 実行: {run_at}",
        "",
        f"- データセット: {meta.get('dataset', '')}（全{meta.get('total_rows', '-')}件）",
        f"- 対象区: {meta.get('target_ward', '')}",
        f"- プロトコル: {meta.get('protocol', '')}（学習 {meta.get('train_period', '')} / 検証 {meta.get('test_period', '')}）",
    ]
    if meta.get("split_note"):
        lines.append(f"- 分割に関する注意: {meta['split_note']}")
    lines += [
        "",
        "| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in results:
        lines.append(
            "| {config} | {scope} | {fs} | {method} | {tuned} | {target} | {pool} | {tr} | {te} | {mape} | {cov} | {width} | {covc} | {widthc} | {status} | {notes} |".format(
                config=r.get("config", ""),
                scope=r.get("scope", ""), fs=r.get("feature_set", ""),
                method=r.get("method", ""), tuned="有" if r.get("tuned") else "-",
                target=r.get("target", ""), pool=r.get("pool") or "-",
                tr=r.get("train_rows", ""), te=r.get("test_n", r.get("test_rows", "")),
                mape=_fmt(r.get("mape_percent")), cov=_fmt(r.get("coverage_50_percent")),
                width=_fmt(r.get("mean_interval_ratio"), 3),
                covc=_fmt(r.get("coverage_cal_percent")), widthc=_fmt(r.get("width_cal"), 3),
                status=r.get("status", ""), notes=(r.get("notes") or "").replace("|", "／"),
            )
        )
    lines += ["", ""]
    return "\n".join(lines)

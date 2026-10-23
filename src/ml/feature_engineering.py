"""
特徴量エンジニアリング

fit/transform を分離している。fit は訓練データにのみ適用し、
検証・予測データには transform のみを使う（学習リーク防止）。
未見のカテゴリは専用値 -1 にマッピングし、バッチ全体を失敗させない。
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder

# 欠損値を中央値で埋める数値カラム
_NUMERIC_COLUMNS = [
    "building_age",
    "floor_area",
    "floor_number",
    "total_floors",
    "station_distance",
    "management_fee",
    "repair_reserve_fund",
]

# 欠損値を "不明" で埋めるカテゴリカラム
_CATEGORICAL_FILLNA_COLUMNS = [
    "prefecture",
    "city",
    "layout",
    "structure",
    "direction",
]

# ラベルエンコーディング対象のカラム
_LABEL_ENCODE_COLUMNS = [
    "prefecture",
    "city",
    "layout",
    "structure",
    "direction",
    "station_distance_category",
    "age_category",
    "area_category",
]

# 未見カテゴリに割り当てる専用値
UNKNOWN_CATEGORY_VALUE = -1


class FeatureEngineer:
    """特徴量エンジニアリングクラス"""

    def __init__(self):
        """初期化"""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.numeric_medians: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        訓練データのみで変換に必要な統計量を学習する。

        検証・予測データには fit を適用してはならない（学習リークになる）。
        そちらには transform_features / prepare_validation_data を使う。

        Args:
            df: 訓練データのDataFrame

        Returns:
            self
        """
        df = df.copy()
        self.numeric_medians = {
            col: df[col].median()
            for col in _NUMERIC_COLUMNS
            if col in df.columns
        }
        df = self._fill_missing(df)
        df = self._create_derived_features(df)
        for col in _LABEL_ENCODE_COLUMNS:
            if col in df.columns:
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                self.label_encoders[col] = encoder
        return self

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        学習済みの変換のみを適用する（fit はしない）。

        未見のカテゴリは専用値 -1 にマッピングされるため、
        予測バッチ全体が ValueError で失敗することはない。

        Args:
            df: 変換対象のDataFrame

        Returns:
            特徴量追加後のDataFrame
        """
        df = df.copy()
        df = self._fill_missing(df)
        df = self._create_derived_features(df)
        for col in _LABEL_ENCODE_COLUMNS:
            if col in df.columns and col in self.label_encoders:
                encoder = self.label_encoders[col]
                values = df[col].astype(str)
                known_mask = values.isin(set(encoder.classes_))
                encoded = pd.Series(
                    UNKNOWN_CATEGORY_VALUE, index=df.index, dtype="int64"
                )
                if known_mask.any():
                    encoded.loc[known_mask] = encoder.transform(values[known_mask])
                df[f"{col}_encoded"] = encoded
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量を生成（後方互換のための fit + transform）。

        新規コードでは fit / transform_features / prepare_*_data を使うこと。
        """
        self.fit(df)
        return self.transform_features(df)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値を補完する。数値は fit 時に学習した中央値を使う。

        fit 前（numeric_medians が空）の場合のみ対象データの中央値を使う。
        """
        medians = getattr(self, "numeric_medians", {})
        for col in _NUMERIC_COLUMNS:
            if col in df.columns:
                median = medians.get(col, df[col].median())
                df[col] = df[col].fillna(median)

        for col in _CATEGORICAL_FILLNA_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("不明")

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        派生特徴量の生成

        Args:
            df: DataFrame

        Returns:
            派生特徴量追加後のDataFrame
        """
        # 築年数の2乗（非線形関係の捕捉）
        if "building_age" in df.columns:
            df["building_age_squared"] = df["building_age"] ** 2

        # 専有面積あたりの管理費
        if "floor_area" in df.columns and "management_fee" in df.columns:
            df["mgmt_fee_per_sqm"] = df["management_fee"] / (df["floor_area"] + 1)

        # 相対的な階数（階数/総階数）
        if "floor_number" in df.columns and "total_floors" in df.columns:
            df["floor_ratio"] = df["floor_number"] / (df["total_floors"] + 1)

        # 駅徒歩距離カテゴリ
        if "station_distance" in df.columns:
            df["station_distance_category"] = pd.cut(
                df["station_distance"],
                bins=[0, 5, 10, 15, float("inf")],
                labels=["very_close", "close", "medium", "far"],
            )

        # 間取りから部屋数を抽出
        if "layout" in df.columns:
            df["room_count"] = df["layout"].apply(self._extract_room_count)

        # 築年数カテゴリ
        if "building_age" in df.columns:
            df["age_category"] = pd.cut(
                df["building_age"],
                bins=[0, 5, 10, 20, 30, float("inf")],
                labels=["new", "relatively_new", "medium", "old", "very_old"],
            )

        # 専有面積カテゴリ
        if "floor_area" in df.columns:
            df["area_category"] = pd.cut(
                df["floor_area"],
                bins=[0, 40, 60, 80, float("inf")],
                labels=["small", "medium", "large", "very_large"],
            )

        return df

    @staticmethod
    def _extract_room_count(layout: str) -> int:
        """
        間取りから部屋数を抽出

        Args:
            layout: 間取り（例: 2LDK）

        Returns:
            部屋数
        """
        if not isinstance(layout, str):
            return 1

        import re

        match = re.search(r"(\d+)", layout)
        if match:
            return int(match.group(1))
        return 1

    def get_feature_columns(self) -> List[str]:
        """
        学習に使用する特徴量カラムのリストを取得

        Returns:
            特徴量カラムのリスト
        """
        return [
            # 基本特徴量
            "floor_area",
            "building_age",
            "floor_number",
            "total_floors",
            "station_distance",
            "management_fee",
            "repair_reserve_fund",
            # 派生特徴量
            "building_age_squared",
            "mgmt_fee_per_sqm",
            "floor_ratio",
            "room_count",
            # エンコード済みカテゴリ特徴量
            "prefecture_encoded",
            "city_encoded",
            "layout_encoded",
            "structure_encoded",
            "direction_encoded",
            "station_distance_category_encoded",
            "age_category_encoded",
            "area_category_encoded",
        ]

    def _select_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """特徴量カラムとターゲットを切り出し、欠損行を除去する。"""
        feature_cols = self.get_feature_columns()
        available_cols = [col for col in feature_cols if col in df.columns]

        X = df[available_cols]
        y = df["price"]

        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        return X, y

    def prepare_training_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        学習用データを準備（fit + transform）。

        訓練分割にのみ使うこと。検証データには prepare_validation_data を使う。

        Args:
            df: 物件データのDataFrame（訓練分割）

        Returns:
            (特徴量DataFrame, ターゲットSeries)
        """
        self.fit(df)
        return self._select_xy(self.transform_features(df))

    def prepare_validation_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        検証・評価用データを準備（transform のみ）。

        事前に prepare_training_data() か fit() を呼んでおくこと。
        未見カテゴリは -1 になる。

        Args:
            df: 物件データのDataFrame（検証分割など）

        Returns:
            (特徴量DataFrame, ターゲットSeries)
        """
        if not self.label_encoders:
            raise ValueError(
                "エンコーダーが学習されていません。"
                "先に prepare_training_data() か fit() を呼んでください。"
            )
        return self._select_xy(self.transform_features(df))


def remove_outliers(df: pd.DataFrame, column: str, n_std: float = 3.0) -> pd.DataFrame:
    """
    外れ値を除去

    Args:
        df: DataFrame
        column: 対象カラム
        n_std: 標準偏差の倍数

    Returns:
        外れ値除去後のDataFrame
    """
    mean = df[column].mean()
    std = df[column].std()

    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

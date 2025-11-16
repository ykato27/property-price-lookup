"""
特徴量エンジニアリング
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """特徴量エンジニアリングクラス"""

    def __init__(self):
        """初期化"""
        self.label_encoders = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量を生成

        Args:
            df: 元のDataFrame

        Returns:
            特徴量追加後のDataFrame
        """
        df = df.copy()

        # 基本特徴量のクリーニング
        df = self._clean_basic_features(df)

        # 派生特徴量の生成
        df = self._create_derived_features(df)

        # カテゴリ特徴量のエンコーディング
        df = self._encode_categorical_features(df)

        return df

    def _clean_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基本特徴量のクリーニング

        Args:
            df: DataFrame

        Returns:
            クリーニング済みDataFrame
        """
        # 欠損値の処理
        numeric_columns = [
            "building_age",
            "floor_area",
            "floor_number",
            "total_floors",
            "station_distance",
            "management_fee",
            "repair_reserve_fund",
        ]

        for col in numeric_columns:
            if col in df.columns:
                # 中央値で埋める
                df[col] = df[col].fillna(df[col].median())

        # カテゴリ変数の欠損値
        categorical_columns = [
            "prefecture",
            "city",
            "layout",
            "structure",
            "direction",
        ]

        for col in categorical_columns:
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

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カテゴリ特徴量のエンコーディング

        Args:
            df: DataFrame

        Returns:
            エンコーディング済みDataFrame
        """
        # ラベルエンコーディング対象のカラム
        label_encode_cols = [
            "prefecture",
            "city",
            "layout",
            "structure",
            "direction",
            "station_distance_category",
            "age_category",
            "area_category",
        ]

        for col in label_encode_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    # 学習済みエンコーダーを使用
                    df[f"{col}_encoded"] = self.label_encoders[col].transform(
                        df[col].astype(str)
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

    def prepare_training_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        学習用データを準備

        Args:
            df: 物件データのDataFrame

        Returns:
            (特徴量DataFrame, ターゲットSeries)
        """
        # 特徴量生成
        df = self.create_features(df)

        # 学習用カラムを取得
        feature_cols = self.get_feature_columns()

        # 存在するカラムのみ使用
        available_cols = [col for col in feature_cols if col in df.columns]

        X = df[available_cols]
        y = df["price"]

        # 欠損値を含む行を削除
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        return X, y


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

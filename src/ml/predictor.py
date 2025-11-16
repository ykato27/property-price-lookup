"""
価格予測モジュール
"""
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd

from config.settings import LATEST_MODEL_PATH
from src.ml.model_trainer import ModelTrainer
from src.ml.feature_engineering import FeatureEngineer


class PricePredictor:
    """価格予測クラス"""

    def __init__(self):
        """初期化"""
        self.model = None
        self.feature_engineer = None
        self.model_version = None

    def load_model(self):
        """モデルを読み込み"""
        try:
            self.model, self.feature_engineer = ModelTrainer.load_model(
                LATEST_MODEL_PATH
            )
            metadata = ModelTrainer.load_metadata()
            self.model_version = metadata.get("trained_at", "unknown")
            print("モデル読み込み完了")
        except FileNotFoundError:
            raise FileNotFoundError(
                "学習済みモデルが見つかりません。先にモデルを学習してください。"
            )

    def predict(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """
        物件価格を予測

        Args:
            properties_df: 物件データのDataFrame

        Returns:
            予測結果を含むDataFrame
        """
        if self.model is None:
            self.load_model()

        # 特徴量準備
        df = properties_df.copy()
        df = self.feature_engineer.create_features(df)

        # 学習時と同じ特徴量カラムを使用
        feature_cols = self.feature_engineer.get_feature_columns()
        available_cols = [col for col in feature_cols if col in df.columns]

        X = df[available_cols]

        # 欠損値を0で埋める（予測時）
        X = X.fillna(0)

        # 予測実行
        predictions = self.model.predict(X)

        # 結果をDataFrameに追加
        df["predicted_price"] = predictions.astype(int)
        df["actual_price"] = df["price"]
        df["price_difference"] = df["predicted_price"] - df["actual_price"]
        df["discount_rate"] = (
            (df["predicted_price"] - df["actual_price"]) / df["predicted_price"] * 100
        )

        return df

    def create_prediction_records(
        self, predictions_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        予測結果をデータベース挿入用のレコードに変換

        Args:
            predictions_df: 予測結果のDataFrame

        Returns:
            予測レコードのリスト
        """
        records = []

        for _, row in predictions_df.iterrows():
            record = {
                "property_id": row["property_id"],
                "predicted_price": int(row["predicted_price"]),
                "actual_price": int(row["actual_price"]),
                "price_difference": int(row["price_difference"]),
                "discount_rate": float(row["discount_rate"]),
                "model_version": self.model_version or datetime.now().isoformat(),
            }
            records.append(record)

        return records

    def find_bargain_properties(
        self, properties_df: pd.DataFrame, min_discount_rate: float = 20.0
    ) -> pd.DataFrame:
        """
        割安物件を検出

        Args:
            properties_df: 物件データのDataFrame
            min_discount_rate: 最低割引率（%）

        Returns:
            割安物件のDataFrame
        """
        # 予測実行
        predictions_df = self.predict(properties_df)

        # 割引率でフィルタリング
        bargain_df = predictions_df[
            predictions_df["discount_rate"] >= min_discount_rate
        ].copy()

        # 割引率で降順ソート
        bargain_df = bargain_df.sort_values("discount_rate", ascending=False)

        return bargain_df

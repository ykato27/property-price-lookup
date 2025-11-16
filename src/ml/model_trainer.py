"""
機械学習モデルトレーナー
"""
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

from config.settings import LATEST_MODEL_PATH, MODEL_METADATA_PATH, ML_SETTINGS
from src.ml.feature_engineering import FeatureEngineer, remove_outliers


class ModelTrainer:
    """機械学習モデルトレーナー"""

    def __init__(self):
        """初期化"""
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.feature_importance = None
        self.metrics = {}

    def train(
        self, df: pd.DataFrame, remove_outliers_flag: bool = True
    ) -> Dict[str, Any]:
        """
        モデルを学習

        Args:
            df: 学習データのDataFrame
            remove_outliers_flag: 外れ値除去フラグ

        Returns:
            学習結果のメトリクス
        """
        print(f"学習データ件数: {len(df)}")

        # 外れ値除去
        if remove_outliers_flag:
            original_count = len(df)
            df = remove_outliers(df, "price", n_std=3.0)
            print(f"外れ値除去: {original_count} -> {len(df)} 件")

        # 特徴量準備
        X, y = self.feature_engineer.prepare_training_data(df)

        print(f"特徴量数: {X.shape[1]}")
        print(f"有効データ件数: {len(X)}")

        # 最小データ数チェック
        if len(X) < ML_SETTINGS["min_data_count"]:
            raise ValueError(
                f"データ数が不足しています。最低{ML_SETTINGS['min_data_count']}件必要です。"
            )

        # 訓練・検証データ分割
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=ML_SETTINGS["test_size"],
            random_state=ML_SETTINGS["random_state"],
        )

        print(f"訓練データ: {len(X_train)} 件")
        print(f"検証データ: {len(X_val)} 件")

        # LightGBMモデル学習
        print("\nモデル学習中...")
        self.model = self._train_lightgbm(X_train, y_train, X_val, y_val)

        # 評価
        self.metrics = self._evaluate(X_train, y_train, X_val, y_val)

        # 特徴量重要度
        self.feature_importance = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\n学習完了!")
        print(f"訓練データ RMSE: {self.metrics['train_rmse']:,.0f}")
        print(f"検証データ RMSE: {self.metrics['val_rmse']:,.0f}")
        print(f"検証データ R²: {self.metrics['val_r2']:.4f}")

        return self.metrics

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> lgb.LGBMRegressor:
        """
        LightGBMモデルを学習

        Args:
            X_train: 訓練データ特徴量
            y_train: 訓練データターゲット
            X_val: 検証データ特徴量
            y_val: 検証データターゲット

        Returns:
            学習済みモデル
        """
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": ML_SETTINGS["random_state"],
        }

        model = lgb.LGBMRegressor(**params, n_estimators=1000)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        return model

    def _evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """
        モデルを評価

        Args:
            X_train: 訓練データ特徴量
            y_train: 訓練データターゲット
            X_val: 検証データ特徴量
            y_val: 検証データターゲット

        Returns:
            評価指標の辞書
        """
        # 訓練データの予測
        y_train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

        # 検証データの予測
        y_val_pred = self.model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100

        return {
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "train_mape": train_mape,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "val_mape": val_mape,
        }

    def save_model(
        self, model_path: Path = LATEST_MODEL_PATH, metadata_path: Path = MODEL_METADATA_PATH
    ):
        """
        モデルを保存

        Args:
            model_path: モデル保存パス
            metadata_path: メタデータ保存パス
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません")

        # モデル保存
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_engineer": self.feature_engineer,
                },
                f,
            )

        # メタデータ保存
        metadata = {
            "trained_at": datetime.now().isoformat(),
            "model_type": "LightGBM",
            "metrics": self.metrics,
            "feature_importance": self.feature_importance.to_dict("records")[:10],
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"モデル保存完了: {model_path}")
        print(f"メタデータ保存完了: {metadata_path}")

    @staticmethod
    def load_model(
        model_path: Path = LATEST_MODEL_PATH,
    ) -> Tuple[Any, FeatureEngineer]:
        """
        モデルを読み込み

        Args:
            model_path: モデルパス

        Returns:
            (モデル, 特徴量エンジニア)
        """
        if not model_path.exists():
            raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        return data["model"], data["feature_engineer"]

    @staticmethod
    def load_metadata(metadata_path: Path = MODEL_METADATA_PATH) -> Dict[str, Any]:
        """
        メタデータを読み込み

        Args:
            metadata_path: メタデータパス

        Returns:
            メタデータ辞書
        """
        if not metadata_path.exists():
            return {}

        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

"""
データベース操作マネージャー
"""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd

from config.settings import DB_PATH, DATA_RETENTION_DAYS
from src.database.schema import (
    ALL_TABLES,
    CREATE_INDEXES,
    PREDICTION_ADDITIONAL_COLUMNS,
    PROPERTY_ADDITIONAL_COLUMNS,
)


class DatabaseManager:
    """データベース操作を管理するクラス"""

    def __init__(self, db_path: Path = DB_PATH):
        """
        初期化

        Args:
            db_path: データベースファイルパス
        """
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        """コンテキストマネージャー: 接続開始"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー: 接続終了"""
        self.close()

    def connect(self):
        """データベースに接続"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """データベース接続を閉じる"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def initialize_database(self):
        """データベースを初期化（テーブル作成＋既存DBへのカラム追加）"""
        cursor = self.conn.cursor()

        # テーブル作成
        for table_sql in ALL_TABLES:
            cursor.execute(table_sql)

        # インデックス作成
        for index_sql in CREATE_INDEXES:
            cursor.execute(index_sql)

        # 既存DBへの追加カラム（冪等なマイグレーション）
        self._add_missing_columns(cursor, "properties", PROPERTY_ADDITIONAL_COLUMNS)
        self._add_missing_columns(cursor, "predictions", PREDICTION_ADDITIONAL_COLUMNS)

        self.conn.commit()

    @staticmethod
    def _add_missing_columns(cursor, table: str, columns: Dict[str, str]):
        """テーブルに不足しているカラムを追加する（既にあれば何もしない）。"""
        existing = {
            row[1] for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, ddl in columns.items():
            if name not in existing:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    # ========== Properties テーブル操作 ==========

    def insert_property(self, property_data: Dict[str, Any]) -> bool:
        """
        物件データを挿入

        Args:
            property_data: 物件データ辞書

        Returns:
            成功: True, 失敗: False
        """
        try:
            cursor = self.conn.cursor()
            columns = ", ".join(property_data.keys())
            placeholders = ", ".join(["?" for _ in property_data])
            sql = f"INSERT OR REPLACE INTO properties ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, list(property_data.values()))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"物件データ挿入エラー: {e}")
            return False

    def bulk_insert_properties(self, properties: List[Dict[str, Any]]) -> int:
        """
        物件データを一括挿入

        Args:
            properties: 物件データのリスト

        Returns:
            挿入成功件数
        """
        success_count = 0
        for prop in properties:
            if self.insert_property(prop):
                success_count += 1
        return success_count

    def get_properties(
        self,
        prefecture: Optional[str] = None,
        city: Optional[str] = None,
        source_site: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        物件データを取得

        Args:
            prefecture: 都道府県でフィルタ
            city: 市区町村でフィルタ
            source_site: 取得元サイトでフィルタ
            limit: 取得件数制限

        Returns:
            物件データのDataFrame
        """
        sql = "SELECT * FROM properties WHERE 1=1"
        params = []

        if prefecture:
            sql += " AND prefecture = ?"
            params.append(prefecture)

        if city:
            sql += " AND city = ?"
            params.append(city)

        if source_site:
            sql += " AND source_site = ?"
            params.append(source_site)

        sql += " ORDER BY scraped_at DESC"

        if limit:
            sql += f" LIMIT {limit}"

        return pd.read_sql_query(sql, self.conn, params=params)

    def get_all_properties_for_training(self) -> pd.DataFrame:
        """
        学習用に全物件データを取得

        Returns:
            物件データのDataFrame
        """
        sql = """
        SELECT * FROM properties
        WHERE price IS NOT NULL
          AND floor_area IS NOT NULL
          AND building_age IS NOT NULL
        ORDER BY scraped_at DESC
        """
        return pd.read_sql_query(sql, self.conn)

    def get_property_count(self) -> int:
        """
        物件データの総件数を取得

        Returns:
            物件データ件数
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM properties")
        return cursor.fetchone()[0]

    # ========== Predictions テーブル操作 ==========

    def insert_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """
        予測データを挿入

        同一物件の過去の予測は「最新のみ有効」の運用として is_latest=0 にし、
        再予測しても割安一覧が重複しないようにする。

        Args:
            prediction_data: 予測データ辞書

        Returns:
            成功: True, 失敗: False
        """
        try:
            cursor = self.conn.cursor()
            if "property_id" in prediction_data:
                cursor.execute(
                    "UPDATE predictions SET is_latest = 0 WHERE property_id = ?",
                    (prediction_data["property_id"],),
                )
            data = dict(prediction_data)
            data.setdefault("is_latest", 1)
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            sql = f"INSERT INTO predictions ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, list(data.values()))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"予測データ挿入エラー: {e}")
            return False

    def bulk_insert_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """
        予測データを一括挿入

        Args:
            predictions: 予測データのリスト

        Returns:
            挿入成功件数
        """
        success_count = 0
        for pred in predictions:
            if self.insert_prediction(pred):
                success_count += 1
        return success_count

    def get_bargain_properties(
        self, min_discount_rate: float = 20.0, limit: int = 100
    ) -> pd.DataFrame:
        """
        割安物件を取得

        Args:
            min_discount_rate: 最低割引率（%）
            limit: 取得件数制限

        Returns:
            割安物件のDataFrame
        """
        sql = """
        SELECT
            p.*,
            pr.predicted_price,
            pr.price_difference,
            pr.discount_rate,
            pr.predicted_at
        FROM properties p
        INNER JOIN predictions pr ON p.property_id = pr.property_id
        WHERE pr.discount_rate >= ?
          AND COALESCE(pr.is_latest, 1) = 1
        ORDER BY pr.discount_rate DESC
        LIMIT ?
        """
        return pd.read_sql_query(sql, self.conn, params=[min_discount_rate, limit])

    def get_latest_valuations(self, gate_only: bool = False) -> pd.DataFrame:
        """
        最新の評価（理論価格・割安スコア・ゲート・フラグ）付き物件一覧を取得

        Args:
            gate_only: True ならゲート通過（割安候補）のみ

        Returns:
            物件＋最新評価のDataFrame
        """
        sql = """
        SELECT
            p.*,
            pr.predicted_price,
            pr.theoretical_price,
            pr.conservative_price,
            pr.interval_low,
            pr.interval_high,
            pr.discount_rate,
            pr.score,
            pr.similar_count,
            pr.confidence,
            pr.risk_flags,
            pr.gate_passed,
            pr.predicted_at
        FROM properties p
        INNER JOIN predictions pr ON p.property_id = pr.property_id
        WHERE COALESCE(pr.is_latest, 1) = 1
        """
        if gate_only:
            sql += " AND pr.gate_passed = 1"
        sql += " ORDER BY pr.score DESC"
        return pd.read_sql_query(sql, self.conn)

    # ========== Transactions テーブル操作 ==========

    def bulk_insert_transactions(self, records: List[Dict[str, Any]]) -> int:
        """
        成約事例を一括挿入（record_hash で重複を除く）

        Args:
            records: 正規化済み成約レコードのリスト

        Returns:
            実際に挿入された件数
        """
        if not records:
            return 0
        cursor = self.conn.cursor()
        columns = list(records[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        sql = (
            f"INSERT OR IGNORE INTO transactions ({', '.join(columns)}) "
            f"VALUES ({placeholders})"
        )
        before = cursor.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        cursor.executemany(sql, [tuple(r[c] for c in columns) for r in records])
        self.conn.commit()
        after = cursor.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        return after - before

    def get_transactions(
        self,
        prefecture: Optional[str] = None,
        municipality_code: Optional[str] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        成約事例を取得

        Returns:
            成約事例のDataFrame
        """
        sql = "SELECT * FROM transactions WHERE 1=1"
        params: List[Any] = []
        if prefecture:
            sql += " AND prefecture = ?"
            params.append(prefecture)
        if municipality_code:
            sql += " AND municipality_code = ?"
            params.append(municipality_code)
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY period_year, period_quarter"
        return pd.read_sql_query(sql, self.conn, params=params)

    def get_transaction_count(self) -> int:
        """成約事例の総件数を取得"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions")
        return cursor.fetchone()[0]

    # ========== Scraping Logs テーブル操作 ==========

    def insert_scraping_log(self, log_data: Dict[str, Any]) -> bool:
        """
        スクレイピングログを挿入

        Args:
            log_data: ログデータ辞書

        Returns:
            成功: True, 失敗: False
        """
        try:
            cursor = self.conn.cursor()
            columns = ", ".join(log_data.keys())
            placeholders = ", ".join(["?" for _ in log_data])
            sql = f"INSERT INTO scraping_logs ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, list(log_data.values()))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"ログ挿入エラー: {e}")
            return False

    def get_scraping_logs(self, limit: int = 50) -> pd.DataFrame:
        """
        スクレイピングログを取得

        Args:
            limit: 取得件数制限

        Returns:
            ログのDataFrame
        """
        sql = f"SELECT * FROM scraping_logs ORDER BY executed_at DESC LIMIT {limit}"
        return pd.read_sql_query(sql, self.conn)

    # ========== データメンテナンス ==========

    def delete_old_data(self, days: int = DATA_RETENTION_DAYS) -> int:
        """
        古いデータを削除

        Args:
            days: 保持日数

        Returns:
            削除件数
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cursor = self.conn.cursor()

        # 古い物件データを削除
        cursor.execute(
            "DELETE FROM properties WHERE scraped_at < ?",
            (cutoff_date.isoformat(),),
        )
        deleted_count = cursor.rowcount

        # 孤立した予測データを削除
        cursor.execute(
            """
            DELETE FROM predictions
            WHERE property_id NOT IN (SELECT property_id FROM properties)
        """
        )

        self.conn.commit()
        return deleted_count

    # ========== 統計情報 ==========

    def get_statistics(self) -> Dict[str, Any]:
        """
        データベース統計情報を取得

        Returns:
            統計情報辞書
        """
        cursor = self.conn.cursor()

        # 物件総数
        cursor.execute("SELECT COUNT(*) FROM properties")
        total_properties = cursor.fetchone()[0]

        # 予測総数
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        # 都道府県別件数
        cursor.execute(
            """
            SELECT prefecture, COUNT(*) as count
            FROM properties
            GROUP BY prefecture
            ORDER BY count DESC
        """
        )
        prefecture_counts = dict(cursor.fetchall())

        # サイト別件数
        cursor.execute(
            """
            SELECT source_site, COUNT(*) as count
            FROM properties
            GROUP BY source_site
            ORDER BY count DESC
        """
        )
        site_counts = dict(cursor.fetchall())

        # 平均価格
        cursor.execute("SELECT AVG(price) FROM properties")
        avg_price = cursor.fetchone()[0] or 0

        return {
            "total_properties": total_properties,
            "total_predictions": total_predictions,
            "prefecture_counts": prefecture_counts,
            "site_counts": site_counts,
            "avg_price": avg_price,
        }

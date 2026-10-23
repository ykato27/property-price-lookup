"""
データベーススキーマ定義
"""

# propertiesテーブル作成SQL
CREATE_PROPERTIES_TABLE = """
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id TEXT UNIQUE NOT NULL,
    source_site TEXT NOT NULL,
    url TEXT NOT NULL,
    prefecture TEXT NOT NULL,
    city TEXT NOT NULL,
    address TEXT,
    price INTEGER NOT NULL,
    building_age INTEGER,
    floor_area REAL,
    floor_number INTEGER,
    total_floors INTEGER,
    layout TEXT,
    structure TEXT,
    nearest_station TEXT,
    station_distance INTEGER,
    direction TEXT,
    management_fee INTEGER,
    repair_reserve_fund INTEGER,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# predictionsテーブル作成SQL
CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id TEXT NOT NULL,
    predicted_price INTEGER NOT NULL,
    actual_price INTEGER NOT NULL,
    price_difference INTEGER NOT NULL,
    discount_rate REAL NOT NULL,
    model_version TEXT NOT NULL,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (property_id) REFERENCES properties(property_id)
);
"""

# transactionsテーブル作成SQL
# 国土交通省「不動産情報ライブラリ」の取引価格情報・成約価格情報の成約事例。
# 募集物件（properties）とは物差しが違うため別テーブルに分ける。
# 公開データにはマンション名・所在階・総戸数・管理費等は含まれない。
CREATE_TRANSACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    price_category TEXT,
    type TEXT,
    region TEXT,
    municipality_code TEXT NOT NULL,
    prefecture TEXT NOT NULL,
    municipality TEXT NOT NULL,
    district_name TEXT,
    nearest_station TEXT,
    station_distance_min REAL,
    trade_price INTEGER NOT NULL,
    floor_plan TEXT,
    area REAL NOT NULL,
    unit_price REAL,
    building_year INTEGER,
    structure TEXT,
    city_planning TEXT,
    renovation TEXT,
    remarks TEXT,
    period_year INTEGER NOT NULL,
    period_quarter INTEGER NOT NULL,
    record_hash TEXT UNIQUE NOT NULL,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# scraping_logsテーブル作成SQL
CREATE_SCRAPING_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS scraping_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_site TEXT NOT NULL,
    prefecture TEXT NOT NULL,
    records_count INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# インデックス作成SQL
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_prefecture ON properties(prefecture);",
    "CREATE INDEX IF NOT EXISTS idx_city ON properties(city);",
    "CREATE INDEX IF NOT EXISTS idx_price ON properties(price);",
    "CREATE INDEX IF NOT EXISTS idx_scraped_at ON properties(scraped_at);",
    "CREATE INDEX IF NOT EXISTS idx_source_site ON properties(source_site);",
    "CREATE INDEX IF NOT EXISTS idx_discount_rate ON predictions(discount_rate);",
    "CREATE INDEX IF NOT EXISTS idx_predicted_at ON predictions(predicted_at);",
    "CREATE INDEX IF NOT EXISTS idx_tx_municipality ON transactions(municipality_code);",
    "CREATE INDEX IF NOT EXISTS idx_tx_period ON transactions(period_year, period_quarter);",
    "CREATE INDEX IF NOT EXISTS idx_tx_prefecture ON transactions(prefecture);",
]

# すべてのテーブル作成SQL
ALL_TABLES = [
    CREATE_PROPERTIES_TABLE,
    CREATE_PREDICTIONS_TABLE,
    CREATE_SCRAPING_LOGS_TABLE,
    CREATE_TRANSACTIONS_TABLE,
]

# 既存DB向けの追加カラム（マイグレーション）
# properties: listing（募集物件）としての必須・任意項目を追加する。
PROPERTY_ADDITIONAL_COLUMNS = {
    "building_year": "INTEGER",
    "rights_type": "TEXT",
    "is_tenanted": "INTEGER DEFAULT 0",
    "monthly_rent": "INTEGER",
    "total_units": "INTEGER",
    "listing_date": "TEXT",
}

# predictions: 「最新予測のみ有効」の運用と、新しい評価（理論価格・区間・ゲート等）。
PREDICTION_ADDITIONAL_COLUMNS = {
    "is_latest": "INTEGER DEFAULT 1",
    "theoretical_price": "INTEGER",
    "conservative_price": "INTEGER",
    "interval_low": "INTEGER",
    "interval_high": "INTEGER",
    "similar_count": "INTEGER",
    "confidence": "TEXT",
    "risk_flags": "TEXT",
    "gate_passed": "INTEGER",
    "score": "REAL",
}

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
]

# すべてのテーブル作成SQL
ALL_TABLES = [
    CREATE_PROPERTIES_TABLE,
    CREATE_PREDICTIONS_TABLE,
    CREATE_SCRAPING_LOGS_TABLE,
]

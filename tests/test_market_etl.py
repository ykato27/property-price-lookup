"""
成約データ基盤（正規化・インポータ・APIクライアント）のテスト。

実データの代わりに XIT001 形式のフィクスチャを使う（APIキーは不要）。
"""
import numpy as np
import pytest

from src.database.db_manager import DatabaseManager
from src.market.fixture_data import generate_fixture_transactions
from src.market.importer import import_records
from src.market.normalize import (
    normalize_record,
    parse_building_year,
    parse_period,
    parse_station_distance,
)
from src.market.reinfolib_client import ReinfolibAuthError, ReinfolibClient


@pytest.fixture()
def db(tmp_path):
    with DatabaseManager(db_path=tmp_path / "test.db") as m:
        m.initialize_database()
        yield m


def test_parse_helpers():
    assert parse_building_year("1972年") == 1972
    assert parse_building_year("2005") == 2005
    assert parse_building_year("") is None
    assert parse_period("2015年第2四半期") == (2015, 2)
    assert parse_period("不明") is None
    assert parse_station_distance("10分") == 10.0
    assert parse_station_distance("30分～60分") == 45.0
    assert parse_station_distance("1時間～1時間30分") == 75.0
    assert parse_station_distance("2時間以上") == 120.0
    assert parse_station_distance("") is None


def test_normalize_xit001_record():
    rec = {
        "Type": "中古マンション等",
        "MunicipalityCode": "13103",
        "Prefecture": "東京都",
        "Municipality": "港区",
        "DistrictName": "芝浦",
        "TradePrice": "85000000",
        "FloorPlan": "3LDK",
        "Area": "70.5",
        "BuildingYear": "1998年",
        "Structure": "ＲＣ",
        "Period": "2024年第1四半期",
        "PriceCategory": "成約価格情報",
    }
    n = normalize_record(rec)
    assert n is not None
    assert n["source"] == "contract"
    assert n["trade_price"] == 85000000
    assert n["building_year"] == 1998
    assert n["period_year"] == 2024 and n["period_quarter"] == 1
    assert n["unit_price"] == pytest.approx(85000000 / 70.5)
    assert n["record_hash"]


def test_normalize_japanese_csv_headers():
    rec = {
        "種類": "中古マンション等",
        "市区町村コード": "13101",
        "都道府県名": "東京都",
        "市区町村名": "千代田区",
        "取引価格（総額）": "60000000",
        "面積（㎡）": "55.0",
        "建築年": "2001年",
        "取引時点": "2023年第3四半期",
        "最寄駅：距離（分）": "30分～60分",
    }
    n = normalize_record(rec)
    assert n is not None
    assert n["station_distance_min"] == 45.0
    assert n["source"] == "trade"  # 区分が無ければ既定は取引価格情報


def test_normalize_drops_incomplete_records():
    assert normalize_record({"Type": "中古マンション等"}) is None
    assert normalize_record({"MunicipalityCode": "13101", "TradePrice": "100"}) is None


def test_import_and_dedup(db):
    raw = generate_fixture_transactions(n=120, seed=7)
    result = import_records(db, raw)
    assert result["inserted"] == result["normalized"] > 100
    assert db.get_transaction_count() == result["inserted"]

    # 同じデータを再取り込みしても重複しない
    again = import_records(db, raw)
    assert again["inserted"] == 0
    assert db.get_transaction_count() == result["inserted"]


def test_import_condo_only_filter(db):
    raw = generate_fixture_transactions(n=30, seed=1)
    raw.append(
        {
            "Type": "宅地(土地と建物)",
            "MunicipalityCode": "13101",
            "Prefecture": "東京都",
            "Municipality": "千代田区",
            "TradePrice": "50000000",
            "Area": "60",
            "BuildingYear": "2010年",
            "Period": "2024年第2四半期",
        }
    )
    result = import_records(db, raw, condo_only=True)
    df = db.get_transactions()
    assert (df["type"] == "中古マンション等").all()
    assert result["normalized"] == 30


def test_api_client_requires_key(monkeypatch):
    monkeypatch.delenv("REINFOLIB_API_KEY", raising=False)
    client = ReinfolibClient(api_key=None)
    # _headers が呼ばれる経路ではキー無しは即エラーになる
    client.api_key = None
    with pytest.raises(ReinfolibAuthError):
        client.fetch_transactions(year=2024, quarter=1, city="13101")


def test_api_client_parses_response(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            return [{"Type": "中古マンション等", "MunicipalityCode": "13101"}]

        def raise_for_status(self):
            pass

    captured = {}

    def fake_get(url, params, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        return FakeResponse()

    monkeypatch.setattr("src.market.reinfolib_client.requests.get", fake_get)
    client = ReinfolibClient(api_key="dummy-key")
    records = client.fetch_transactions(
        year=2024, quarter=2, city="13103", price_classification="02", save_raw=False
    )
    assert len(records) == 1
    assert captured["headers"]["Ocp-Apim-Subscription-Key"] == "dummy-key"
    assert captured["params"]["priceClassification"] == "02"
    assert captured["url"].endswith("/XIT001")


def test_api_client_401_raises_auth_error(monkeypatch):
    class Fake401:
        status_code = 401

        def json(self):
            return {"statusCode": 401}

        def raise_for_status(self):
            raise AssertionError("raise_for_status should not be reached")

    monkeypatch.setattr(
        "src.market.reinfolib_client.requests.get",
        lambda *a, **k: Fake401(),
    )
    client = ReinfolibClient(api_key="bad-key")
    with pytest.raises(ReinfolibAuthError):
        client.fetch_transactions(year=2024, quarter=1, city="13101", save_raw=False)

"""
listing投入・割安スコア/ゲート・リスクフラグ・投資試算・predictions最新化のテスト。
"""
import pandas as pd
import pytest

from src.database.db_manager import DatabaseManager
from src.market.checklist import build_checklist_markdown
from src.market.fixture_data import generate_fixture_transactions
from src.market.importer import import_records
from src.market.investment import analyze_investment
from src.market.listings import import_listings
from src.market.price_model import PriceModel
from src.market.scoring import (
    assess_risk_flags,
    valuate_all_properties,
    valuate_property,
)
from datetime import date


@pytest.fixture(scope="module")
def transactions_df(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tx2")
    with DatabaseManager(db_path=tmp / "t.db") as db:
        db.initialize_database()
        import_records(db, generate_fixture_transactions(n=1600, seed=42))
        df = db.get_transactions()
    return df


@pytest.fixture(scope="module")
def price_model(transactions_df):
    return PriceModel().fit(transactions_df)


@pytest.fixture()
def db(tmp_path):
    with DatabaseManager(db_path=tmp_path / "test.db") as m:
        m.initialize_database()
        yield m


def base_prop(**overrides):
    prop = {
        "property_id": "lst_test",
        "price": 40_000_000,
        "floor_area": 65.0,
        "prefecture": "東京都",
        "city": "港区",
        "layout": "3LDK",
        "structure": "ＲＣ",
        "station_distance": 6,
        "building_year": 2010,
        "building_age": 15,
        "management_fee": 12000,
        "repair_reserve_fund": 12000,
        "rights_type": "所有権",
        "is_tenanted": 0,
        "total_units": 80,
        "listing_date": "2025-11-01",
        "source_site": "手入力",
        "url": "",
    }
    prop.update(overrides)
    return prop


# ---------- リスクフラグ ----------

def test_old_seismic_hard_flag():
    flags = assess_risk_flags(base_prop(building_year=1975), date(2025, 12, 1))
    old = [f for f in flags if f.code == "old_seismic"]
    assert old and old[0].severity == "hard"
    assert "近似" in old[0].detail  # 建築確認ではなく築年からの近似である明示


def test_seismic_boundary_is_warn_not_hard():
    flags = assess_risk_flags(base_prop(building_year=1981), date(2025, 12, 1))
    assert any(f.code == "seismic_boundary" and f.severity == "warn" for f in flags)
    assert not any(f.severity == "hard" for f in flags)


def test_leasehold_hard_and_other_flags():
    flags = assess_risk_flags(
        base_prop(rights_type="定期借地権", is_tenanted=1, total_units=20),
        date(2025, 12, 1),
    )
    codes = {f.code: f.severity for f in flags}
    assert codes.get("leasehold") == "hard"
    assert codes.get("tenanted") == "warn"
    assert codes.get("small_scale") == "warn"


def test_reserve_shortage_flag_uses_per_sqm():
    # 65㎡で積立金 6,500円 → 100円/㎡・月（目安335を大幅に下回る）
    flags = assess_risk_flags(base_prop(repair_reserve_fund=6500), date(2025, 12, 1))
    assert any(f.code == "reserve_shortage" for f in flags)
    # 65㎡で 20,000円 → 約308円/㎡・月（目安圏内）ならフラグは立たない
    flags = assess_risk_flags(base_prop(repair_reserve_fund=20000), date(2025, 12, 1))
    assert not any(f.code == "reserve_shortage" for f in flags)


# ---------- ゲートとスコア ----------

def test_gate_passes_for_deeply_discounted(transactions_df, price_model):
    # 港区・築15年・65㎡の理論価格はフィクスチャ上 8,000万円前後。
    # 4,000万円なら保守値基準でも大きく割安になるはず
    v = valuate_property(
        base_prop(price=40_000_000), transactions_df, model=price_model,
        valuation_period=(2025, 4), valuation_date=date(2025, 12, 1),
    )
    assert v.similar_count >= 10
    assert v.has_recent
    assert v.score is not None and v.score >= 0.15
    assert v.gate_passed, v.gate_reasons
    assert v.confidence in ("高", "中")


def test_gate_fails_when_price_close_to_market(transactions_df, price_model):
    v = valuate_property(
        base_prop(price=85_000_000), transactions_df, model=price_model,
        valuation_period=(2025, 4), valuation_date=date(2025, 12, 1),
    )
    assert not v.gate_passed


def test_gate_fails_for_hard_flag_even_if_cheap(transactions_df, price_model):
    v = valuate_property(
        base_prop(price=30_000_000, building_year=1975, building_age=50),
        transactions_df, model=price_model,
        valuation_period=(2025, 4), valuation_date=date(2025, 12, 1),
    )
    assert not v.gate_passed
    assert any("ハードフィルタ" in r for r in v.gate_reasons)


def test_gate_fails_without_enough_comparables(price_model):
    empty = pd.DataFrame(
        columns=[
            "municipality_code", "municipality", "prefecture", "area",
            "building_year", "unit_price", "period_year", "period_quarter",
        ]
    )
    v = valuate_property(
        base_prop(), empty, model=price_model,
        valuation_period=(2025, 4), valuation_date=date(2025, 12, 1),
    )
    assert v.similar_count == 0
    assert not v.gate_passed


# ---------- 投資試算 ----------

def test_investment_requires_rent():
    assert analyze_investment(price=50_000_000, monthly_rent=None) is None
    assert analyze_investment(price=50_000_000, monthly_rent=0) is None


def test_investment_noi_calculation():
    inv = analyze_investment(
        price=50_000_000, monthly_rent=200_000,
        management_fee=10_000, repair_reserve_fund=10_000,
    )
    assert inv is not None
    assert inv["gross_yield"] == pytest.approx(2_400_000 / 50_000_000)
    # NOI = 240万 - 空室5% - 手数料5% - 管理費等24万 - 固都税概算20万 = 172万
    assert inv["noi"] == pytest.approx(1_720_000)
    assert inv["noi_yield"] < inv["gross_yield"]
    assert "仮定" not in inv["assumptions"]  # 仮定は describe 側の文言ではなくキー名ではない
    assert inv["assumptions"]["空室率"] == "5%"


# ---------- listing 投入 ----------

def test_listing_import_requires_date_and_source(db):
    rows = [
        {
            "募集価格": "50000000", "都道府県": "東京都", "市区町村": "港区",
            "面積": "65", "建築年": "2010",
            # 取得日・出所が無い → 必須エラー
        }
    ]
    success, errors = import_listings(db, rows)
    assert success == 0
    assert any("listing_date" in e or "source_site" in e for e in errors)


def test_listing_import_japanese_csv_columns(db):
    rows = [
        {
            "募集価格": "45,000,000", "都道府県": "東京都", "市区町村": "港区",
            "専有面積": "65.5", "間取り": "3LDK", "建築年": "2010",
            "管理費": "12000", "修繕積立金": "11000", "権利形態": "所有権",
            "取得日": "2025-11-20", "出所": "知人の不動産会社",
        }
    ]
    success, errors = import_listings(db, rows)
    assert success == 1, errors
    df = db.get_properties()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["price"] == 45_000_000
    assert row["building_year"] == 2010
    assert row["building_age"] == 15  # 2025 - 2010 で補完される
    assert row["listing_date"] == "2025-11-20"
    assert row["source_site"] == "知人の不動産会社"
    # 再投入しても同一物件として上書きされ、重複しない
    success2, _ = import_listings(db, rows)
    assert success2 == 1
    assert len(db.get_properties()) == 1


# ---------- predictions 最新化・チェックリスト ----------

def test_valuate_all_keeps_only_latest_prediction(db, transactions_df, price_model):
    import_records(db, generate_fixture_transactions(n=1600, seed=42))
    rows = [
        {
            "募集価格": "40000000", "都道府県": "東京都", "市区町村": "港区",
            "専有面積": "65", "間取り": "3LDK", "建築年": "2010",
            "取得日": "2025-11-20", "出所": "手入力",
        }
    ]
    import_listings(db, rows)

    valuate_all_properties(db, model=price_model)
    valuate_all_properties(db, model=price_model)  # 再評価しても重複しないこと

    latest = db.get_latest_valuations()
    assert len(latest) == 1
    cursor = db.conn.cursor()
    total = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    latest_count = cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE is_latest = 1"
    ).fetchone()[0]
    assert total == 2 and latest_count == 1


def test_checklist_contains_disclaimer_and_flags(transactions_df, price_model):
    prop = base_prop(price=40_000_000, building_year=1975, building_age=50,
                     rights_type="所有権")
    v = valuate_property(
        prop, transactions_df, model=price_model,
        valuation_period=(2025, 4), valuation_date=date(2025, 12, 1),
    )
    md = build_checklist_markdown(prop, v)
    assert "購入推奨ではありません" in md
    assert "長期修繕計画" in md
    assert "総会議事録" in md
    assert "旧耐震" in md
    # 賃料が無い物件には利回りを出さない
    assert "推定利回りは出しません" in md

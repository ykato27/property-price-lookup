"""
汎用ヘルパー関数
"""
from typing import Any


def format_price(price: Any) -> str:
    """
    価格を読みやすい形式にフォーマット

    Args:
        price: 価格

    Returns:
        フォーマット済み価格文字列
    """
    try:
        price_int = int(price)
        if price_int >= 100000000:  # 1億円以上
            return f"{price_int / 100000000:.2f}億円"
        elif price_int >= 10000:  # 1万円以上
            return f"{price_int / 10000:.0f}万円"
        else:
            return f"{price_int:,}円"
    except (ValueError, TypeError):
        return "不明"


def format_area(area: Any) -> str:
    """
    面積をフォーマット

    Args:
        area: 面積

    Returns:
        フォーマット済み面積文字列
    """
    try:
        area_float = float(area)
        return f"{area_float:.2f}㎡"
    except (ValueError, TypeError):
        return "不明"


def format_age(age: Any) -> str:
    """
    築年数をフォーマット

    Args:
        age: 築年数

    Returns:
        フォーマット済み築年数文字列
    """
    try:
        age_int = int(age)
        return f"築{age_int}年"
    except (ValueError, TypeError):
        return "不明"


def format_station_distance(distance: Any) -> str:
    """
    駅徒歩距離をフォーマット

    Args:
        distance: 距離（分）

    Returns:
        フォーマット済み距離文字列
    """
    try:
        distance_int = int(distance)
        return f"徒歩{distance_int}分"
    except (ValueError, TypeError):
        return "不明"


def get_discount_color(discount_rate: float) -> str:
    """
    割引率に応じた色を取得

    Args:
        discount_rate: 割引率（%）

    Returns:
        色コード
    """
    if discount_rate >= 30:
        return "#ff4b4b"  # 赤（大幅割引）
    elif discount_rate >= 20:
        return "#ff8c00"  # オレンジ
    elif discount_rate >= 10:
        return "#ffa500"  # 黄色
    else:
        return "#4caf50"  # 緑

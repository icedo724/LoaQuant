"""화면에 숫자로 나가는 집계. Streamlit 없이 검증할 수 있도록 분리해 둔다."""
from __future__ import annotations

import pandas as pd

from analysis import marketdata as md


def trade_value(
    price_daily: pd.DataFrame,
    volume: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    exclude_from: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """기간 총 거래대금 = Σ(일별 단가 × 일별 거래량).

    예전 구현은 ``(기간 평균 단가) × (기간 총 거래량)`` 이었다. 화면 캡션은
    "일평균 단가 × 일일 거래량을 합산"이라고 적혀 있었으므로 설명과 코드가
    달랐고, 두 값은 가격과 거래량의 공분산만큼 어긋난다. 실측으로 강화 재료
    35품목 중 17품목의 순위가 바뀌었고 최대 오차는 30% 였다.

    ``exclude_from`` 이후 날짜는 제외한다. 오늘 거래량은 아직 집계 중이라
    직전 14일 중앙값의 30% 수준으로 잡히기 때문이다.

    돌려주는 것은 ``(결과표, 실제로 제외된 날짜)``.
    """
    p, v = md.align_price_volume(price_daily, volume)
    if p.empty:
        return pd.DataFrame(), None

    excluded = None
    if exclude_from is not None:
        cut = pd.Timestamp(exclude_from)
        if (p.index >= cut).any():
            excluded = cut
        p, v = p[p.index < cut], v[v.index < cut]

    if start is not None:
        p, v = p[p.index >= pd.Timestamp(start)], v[v.index >= pd.Timestamp(start)]
    if end is not None:
        p, v = p[p.index <= pd.Timestamp(end)], v[v.index <= pd.Timestamp(end)]

    if p.empty:
        return pd.DataFrame(), excluded

    value = (p * v).sum()
    volume_sum = v.sum()
    # 평균 단가는 거래량 가중으로 낸다. 그래야 위 총액과 앞뒤가 맞는다.
    avg_price = value / volume_sum.where(volume_sum != 0)

    out = pd.DataFrame({
        "value": value,
        "volume": volume_sum,
        "avg_price": avg_price,
    })
    return out.sort_values("value", ascending=False), excluded


def recent_window(index: pd.DatetimeIndex, days: int) -> pd.Timestamp | None:
    """인덱스 마지막 날부터 거슬러 ``days`` 일 구간의 시작일."""
    idx = pd.DatetimeIndex(index)
    if idx.empty:
        return None
    return idx.max() - pd.Timedelta(days=days - 1)


# ---------------------------------------------------------------------------
# 시장 시그널
# ---------------------------------------------------------------------------
# 마지막 유효 관측이 이보다 오래됐으면 "현재가"라고 부르지 않는다.
MAX_STALENESS_DAYS = 2

SIGNALS = {
    "strong_buy": "강력 매수 (저점+과매도)",
    "strong_sell": "강력 매도 (고점+과열)",
    "buy": "매수 기회 (밴드 하단)",
    "caution": "매수 주의 (밴드 상단)",
    "hot": "과열 양상 (RSI 높음)",
    "cold": "침체 양상 (RSI 낮음)",
    "neutral": "관망 (적정가)",
}


def market_signal(
    daily: pd.Series,
    reliable: pd.Series,
    max_staleness_days: int = MAX_STALENESS_DAYS,
) -> dict | None:
    """RSI · 볼린저 기반 시그널. 서식 없이 값만 돌려준다.

    두 가지를 앞선 구현과 다르게 한다.

    - **등간격 축.** 지표는 게임일 축에서 계산한다. 시간별 원본 위의
      ``rolling(24)`` 는 실제로 중앙 37시간, 최대 134시간을 덮었다.
    - **오래된 값을 현재가로 쓰지 않는다.** ``dropna().iloc[-1]`` 만 보면,
      골드 환산이 최근 구간에서 실패했을 때 몇 달 전 가격이 현재가 카드에 뜬다.
    """
    from common import timeaxis as ta

    s = pd.Series(daily).dropna()
    if len(s) < ta.BOLLINGER_PERIODS + 1:
        return None

    latest = pd.DatetimeIndex(pd.Series(daily).index).max()
    stale_days = int((latest - s.index[-1]).days)
    if stale_days > max_staleness_days:
        return {"stale": True, "last": s.index[-1], "stale_days": stale_days}

    obs = pd.Series(reliable).reindex(s.index).fillna(False)
    rsi = ta.rsi(s, obs)
    ma, upper, lower = ta.bollinger(s, obs)
    if not (rsi.notna().any() and upper.notna().any() and lower.notna().any()):
        return None

    cur_rsi = float(rsi.dropna().iloc[-1])
    cur_up = float(upper.dropna().iloc[-1])
    cur_lo = float(lower.dropna().iloc[-1])
    price, prev = float(s.iloc[-1]), float(s.iloc[-2])

    if price <= cur_lo and cur_rsi <= 30:
        key = "strong_buy"
    elif price >= cur_up and cur_rsi >= 70:
        key = "strong_sell"
    elif price <= cur_lo:
        key = "buy"
    elif price >= cur_up:
        key = "caution"
    elif cur_rsi >= 70:
        key = "hot"
    elif cur_rsi <= 30:
        key = "cold"
    else:
        key = "neutral"

    return {
        "stale": False, "price": price, "prev": prev, "diff": price - prev,
        "rsi": cur_rsi, "upper": cur_up, "lower": cur_lo,
        "signal": key, "label": SIGNALS[key], "as_of": s.index[-1],
    }

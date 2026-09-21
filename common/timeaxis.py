"""게임일 기준 시간 축.

이 저장소에서 "하루"와 "관측 간격"의 정의는 여기 한 곳에만 둔다.
대시보드와 분석 모듈이 각자 `index - Timedelta(hours=6)` 를 적어두면
한쪽만 고쳐졌을 때 같은 CSV 에서 다른 수가 나온다. 실제로 그랬다.

이 모듈은 I/O 를 하지 않는다. 입력은 항상 **KST naive** DatetimeIndex 를
가진 pandas 객체다. 수집기가 CSV 컬럼명을 KST 문자열로 쓰기 때문이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 로스트아크의 하루는 매일 오전 6시(KST)에 바뀐다.
GAME_DAY_START_HOUR = 6

# 정기 점검: 매주 수요일 06:00~10:00 (KST)
MAINTENANCE_WEEKDAY = 2          # Monday=0
MAINTENANCE_START_HOUR = 6
MAINTENANCE_END_HOUR = 10

_KST = "Asia/Seoul"


# ---------------------------------------------------------------------------
# 현재 시각
# ---------------------------------------------------------------------------
def now_kst() -> pd.Timestamp:
    """KST naive 현재 시각.

    `pd.Timestamp.now()` 를 쓰면 서버 로컬 타임존이 섞인다. Streamlit Cloud 는
    UTC 라서 로컬(KST)에서 개발할 때와 배포했을 때 결과가 달라진다.
    """
    return pd.Timestamp.now(tz=_KST).tz_localize(None)


def game_day_of(ts) -> pd.Timestamp:
    """어떤 KST 시각이 속한 게임일(자정 기준 Timestamp)."""
    return (pd.Timestamp(ts) - pd.Timedelta(hours=GAME_DAY_START_HOUR)).normalize()


def current_game_day() -> pd.Timestamp:
    """지금 진행 중인 게임일."""
    return game_day_of(now_kst())


def to_game_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """KST 타임스탬프 인덱스 → 각 시점이 속한 게임일."""
    return (pd.DatetimeIndex(index) - pd.Timedelta(hours=GAME_DAY_START_HOUR)).normalize()


# ---------------------------------------------------------------------------
# 일 단위 집계
# ---------------------------------------------------------------------------
# 하루 수집 건수가 2~23 건으로 들쭉날쭉하다(실측 중앙값 11). 표본이 너무 적은
# 날의 평균을 23 건짜리 날과 같은 무게로 그리면 안 된다.
DEFAULT_MIN_SAMPLES = 4


def game_daily_mean(df: pd.DataFrame | pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """게임일(06시 기준) 평균과 그 날의 표본 수를 함께 돌려준다.

    표본이 적은 날의 값을 여기서 지우지는 않는다. 20일 창으로 지표를 돌릴 때
    구멍 하나가 창 전체를 죽이기 때문이다. 대신 건수를 항상 함께 넘기고,
    "믿을 만한 날인가"의 판정은 :func:`reliable_mask` 로 호출부에 맡긴다.
    """
    frame = df.to_frame() if isinstance(df, pd.Series) else df
    if frame.empty:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name=None), columns=frame.columns)
        return empty, empty.copy()

    shifted = frame.copy()
    shifted.index = pd.DatetimeIndex(shifted.index) - pd.Timedelta(hours=GAME_DAY_START_HOUR)
    grouped = shifted.groupby(shifted.index.normalize())

    means, counts = grouped.mean(), grouped.count()
    means.index.name = counts.index.name = None
    return means, counts


def reliable_mask(counts: pd.DataFrame, min_samples: int = DEFAULT_MIN_SAMPLES) -> pd.DataFrame:
    """표본이 충분해 평균을 신뢰할 수 있는 날."""
    return counts >= min_samples


# ---------------------------------------------------------------------------
# 진행 중인 하루
# ---------------------------------------------------------------------------
def split_complete(index: pd.DatetimeIndex, today: pd.Timestamp | None = None):
    """게임일 인덱스를 (완료된 날, 진행 중인 날)로 나눈다.

    오늘은 아직 끝나지 않았다. 완료된 날과 같은 막대로 그리면 매일 없던 절벽이
    생기고(거래량 실측: 직전 14일 중앙값의 30%), 오늘을 포함해 학습하면
    부분 집계에 모델을 맞추게 된다.
    """
    idx = pd.DatetimeIndex(index)
    cutoff = today if today is not None else current_game_day()
    return idx[idx < cutoff], idx[idx >= cutoff]


# ---------------------------------------------------------------------------
# 균등 격자
# ---------------------------------------------------------------------------
# 수집은 명목상 1시간 주기지만 실측 성공률은 51% 다(중앙 간격 1.52h, 최대 12.5h).
# RSI·볼린저 같은 지표는 등간격 시계열에서 정의되므로, 원본 인덱스 위에서
# rolling(24) 을 돌리면 "24기간"이 실제로는 중앙값 37시간을 덮는다.
#
# 그렇다고 1시간 격자에 올리는 것도 답이 아니다. 실측:
#
#     격자   실제 관측 슬롯      비고
#     1h        50.4%          절반이 빈칸
#     6h        96.1%
#     1D        98.2%
#
# 1시간 격자에서 창 안 실제 관측 비율은 중앙값 43~46% 라, 신뢰할 만한 창만
# 남기면 지표의 66% 가 사라진다. 빈칸을 ffill 로 메우면 같은 값이 반복되면서
# rolling(24) 표준편차가 11.9 → 10.7 로 줄고, 볼린저 밴드가 좁아져 이탈 신호가
# 과다 발생한다. 없던 신호를 만드는 쪽이 신호가 없는 것보다 나쁘다.
#
# 그래서 **화면 지표는 게임일 단위로 계산한다**(:func:`game_daily_mean`).
# 일 단위는 관측률 98% 이고, RSI 14일·볼린저 20일은 이 지표들의 원래 관례이며,
# 사용자의 판단("오늘 살까")과도 축이 맞는다. 아래 :func:`uniform_grid` 는
# 일중(intraday) 균등 격자가 필요한 경우를 위해 남겨 두지만, 대시보드 지표는
# 이것을 쓰지 않는다.
DEFAULT_GRID = "6h"
DEFAULT_MAX_FILL = pd.Timedelta(hours=3)


def uniform_grid(
    df: pd.DataFrame | pd.Series,
    freq: str = DEFAULT_GRID,
    max_fill: pd.Timedelta = DEFAULT_MAX_FILL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """불규칙 관측을 균등 격자에 올린다.

    돌려주는 것은 ``(값, 관측여부)`` 두 프레임이다.

    - 값: 격자점마다 하나. 짧은 결측(``max_fill`` 이하)은 직전 값으로 메운다.
      그보다 긴 공백은 메우지 않고 NaN 으로 남긴다. 12시간 끊긴 구간을
      이어 붙이면 없던 가격이 생긴다.
    - 관측여부: 그 격자점에 실제 수집값이 있었는지. 지표 계산 시 창(window)
      안에 메운 점이 얼마나 섞였는지 판단하는 데 쓴다.
    """
    frame = df.to_frame() if isinstance(df, pd.Series) else df
    if frame.empty:
        empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=frame.columns)
        return empty, empty.copy()

    grid = frame.resample(freq).mean()
    observed = grid.notna()

    limit = max(1, int(max_fill / pd.Timedelta(freq)))
    filled = grid.ffill(limit=limit)
    return filled, observed


# ---------------------------------------------------------------------------
# 지표 — 등간격 축 위에서만 계산한다
# ---------------------------------------------------------------------------
# 지표의 기본 단위는 **게임일**이다. RSI 14일, 볼린저 20일은 이 지표들의 관례적
# 파라미터이고, 일 단위 관측률이 98% 라 창 길이가 실제로 일정하다.
RSI_PERIODS = 14
BOLLINGER_PERIODS = 20

# 창 안에서 믿을 만한 관측이 이 비율 미만이면 신호를 내지 않는다. 표본 2건짜리
# 날로 채워진 창에서 나온 "과매도"는 가격이 아니라 수집 실패를 반영한다.
DEFAULT_MIN_OBSERVED = 0.6


def _mask_sparse(values: pd.Series, observed: pd.Series, window: int, min_observed: float) -> pd.Series:
    ratio = observed.astype(float).rolling(window, min_periods=window).mean()
    return values.where(ratio >= min_observed)


def rsi(
    series: pd.Series,
    observed: pd.Series,
    window: int = RSI_PERIODS,
    min_observed: float = DEFAULT_MIN_OBSERVED,
) -> pd.Series:
    """Wilder 방식 RSI.

    ``series`` 는 등간격 축(기본: 게임일) 위의 값이어야 하고, ``observed`` 는
    각 시점이 믿을 만한 관측인지를 나타내는 불리언이다(:func:`reliable_mask`).
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    # 하락이 전혀 없는 구간에서 0 으로 나누는 대신 RSI 를 100 으로 둔다.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())
    return _mask_sparse(out, observed, window, min_observed)


def bollinger(
    series: pd.Series,
    observed: pd.Series,
    window: int = BOLLINGER_PERIODS,
    n_std: float = 2.0,
    min_observed: float = DEFAULT_MIN_OBSERVED,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(중심선, 상단, 하단). ``series`` 는 등간격 축 위의 값이어야 한다."""
    ma = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std()
    upper, lower = ma + n_std * sd, ma - n_std * sd
    return (
        _mask_sparse(ma, observed, window, min_observed),
        _mask_sparse(upper, observed, window, min_observed),
        _mask_sparse(lower, observed, window, min_observed),
    )


# ---------------------------------------------------------------------------
# 요일 효과
# ---------------------------------------------------------------------------
# 요일별 '수준'을 그냥 평균내면 기간 추세가 요일 칸으로 새어 들어간다.
# 이 저장소 데이터에서 운명의 파괴석은 7개월간 -54%, 상급 아비도스 융화 재료는
# +88% 움직였고, 추세를 걷어내면 월요일 효과의 부호가 바뀐다.
DETREND_WINDOW = 7


def day_of_week_effect(
    daily: pd.Series,
    baseline_weekday: int = MAINTENANCE_WEEKDAY,
    detrend: bool = True,
    window: int = DETREND_WINDOW,
) -> tuple[pd.Series, pd.Series]:
    """기준 요일 대비 요일별 효과(%)와 요일별 표본 수.

    ``detrend=True`` 면 중심 이동평균으로 추세를 나눈 뒤 요일별로 평균한다
    (곱셈 분해). ``False`` 면 예전처럼 수준을 그대로 평균한다 — 비교용이다.
    """
    s = pd.Series(daily).dropna().sort_index()
    if s.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    if detrend:
        # 창을 다 채운 중심 이동평균만 쓴다. 양 끝의 부분 창은 중심이 어긋나서
        # 추세가 덜 빠지고, 그 잔차가 특정 요일로 몰린다.
        trend = s.rolling(window, center=True, min_periods=window).mean()
        s = (s / trend).replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    wd = pd.Index(s.index).weekday
    grouped = s.groupby(wd)
    means = grouped.mean().reindex(range(7))
    counts = grouped.count().reindex(range(7)).fillna(0).astype(int)

    base = means.get(baseline_weekday)
    if base is None or not np.isfinite(base) or base == 0:
        return pd.Series(np.nan, index=range(7)), counts
    return (means / base - 1.0) * 100.0, counts


# ---------------------------------------------------------------------------
# 점검 구간
# ---------------------------------------------------------------------------
def maintenance_windows(start, end) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end] 구간에 걸치는 정기 점검 시간대 목록."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start > end:
        return []

    out = []
    day = start.normalize()
    while day <= end:
        if day.weekday() == MAINTENANCE_WEEKDAY:
            a = day + pd.Timedelta(hours=MAINTENANCE_START_HOUR)
            b = day + pd.Timedelta(hours=MAINTENANCE_END_HOUR)
            if a <= end and b >= start:
                out.append((a, b))
        day += pd.Timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# 수집 주기 진단
# ---------------------------------------------------------------------------
# 워크플로가 예약한 수집 주기. 지표용 격자(DEFAULT_GRID)와는 다른 값이다.
NOMINAL_COLLECTION_INTERVAL = "1h"


def coverage(index: pd.DatetimeIndex, nominal: str = NOMINAL_COLLECTION_INTERVAL) -> dict:
    """관측 인덱스가 명목 주기 대비 얼마나 촘촘한지.

    화면에 "매시간"이라고 적어두고 실제로는 절반만 수집되는 상태를 막기 위해,
    대시보드가 실제 수집률을 계산해 표시할 수 있게 한다.
    """
    idx = pd.DatetimeIndex(index).dropna().sort_values()
    if len(idx) < 2:
        return {"n": len(idx), "expected": len(idx), "ratio": float("nan"),
                "median_gap_h": float("nan"), "max_gap_h": float("nan"),
                "start": idx[0] if len(idx) else None, "end": idx[-1] if len(idx) else None}

    # 수집이 한 주기 안에 두 번 성공할 수도 있으므로(재시도 크론), 건수를 주기 수로
    # 나누면 1 을 넘는다. "관측이 하나라도 있는 주기의 비율"로 재면 그 영향을 받지 않는다.
    periods = pd.Series(1, index=idx).resample(nominal).sum()
    filled = int((periods > 0).sum())
    expected = len(periods)
    gaps = idx.to_series().diff().dropna()

    return {
        "n": len(idx),
        "expected": expected,
        "ratio": filled / expected if expected else float("nan"),
        "median_gap_h": gaps.median().total_seconds() / 3600.0,
        "max_gap_h": gaps.max().total_seconds() / 3600.0,
        "start": idx[0],
        "end": idx[-1],
    }

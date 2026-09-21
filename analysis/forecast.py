"""일 단위 시세 예측과 예측구간.

이전 구현의 문제는 예측값이 아니라 **구간**이었다.

- Prophet 의 `interval_width` 를 0.80 → 0.95 로 올린 주석이 "실측 커버리지가
  목표치 크게 미달"이라고 적혀 있었다. 커버리지가 미달이면 모델이
  miscalibrated 인 것이고, 명목 폭을 넓힌다고 커버리지가 따라오지 않는다.
- 대체 모델(`get_linreg_forecast`)은 in-sample 잔차의 모집단 표준편차(ddof=0)에
  1.96 을 곱한 **상수 마진**을 95% 구간이라고 표시했다. 자유도 보정도, 외삽
  레버리지도 없어서 구간이 구조적으로 더 좁았다. 갈아탄 이유를 그대로
  재현한 셈이다.

여기서는 OLS 예측구간을 교과서 형태로 계산하고, 명목 대비 실제 커버리지는
:mod:`analysis.backtest` 가 측정한다. 주석에 숫자를 적어두는 대신 재현 가능한
코드로 남기는 것이 요점이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# 추세 1 + 요일 더미 6 + 절편 1 = 8 모수. 그 이상은 있어야 적합이 의미를 갖는다.
MIN_OBSERVATIONS = 14
DEFAULT_LEVEL = 0.95


@dataclass
class Forecast:
    """예측 결과. ``mean``/``lower``/``upper`` 는 모두 날짜 인덱스를 갖는다."""

    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    fitted: pd.Series          # 학습 구간에 대한 적합값
    level: float
    model: str

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.mean.index)


# ---------------------------------------------------------------------------
# 설계 행렬
# ---------------------------------------------------------------------------
def _design(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """절편 + 선형 추세 + 요일 더미(월 기준).

    추세는 관측 **순번**이 아니라 ``origin`` 으로부터의 **경과 일수**다.
    순번을 쓰면 결측일이 있을 때 기울기의 단위가 "관측 1건당"이 되는데,
    예측은 달력 하루씩 전진하므로 단위가 어긋난다.
    """
    idx = pd.DatetimeIndex(dates)
    t = (idx - origin).days.to_numpy(dtype="float64").reshape(-1, 1)
    dow = np.zeros((len(idx), 6), dtype="float64")
    wd = idx.weekday.to_numpy()
    for k in range(1, 7):                      # 월요일(0)은 기준 범주
        dow[:, k - 1] = (wd == k).astype("float64")
    return np.hstack([np.ones((len(idx), 1)), t, dow])


# ---------------------------------------------------------------------------
# OLS: 선형 추세 + 요일 효과
# ---------------------------------------------------------------------------
def ols_trend_dow(
    series: pd.Series,
    horizon: int = 7,
    level: float = DEFAULT_LEVEL,
    start: pd.Timestamp | None = None,
) -> Forecast | None:
    """추세 + 요일 OLS. 예측구간은 t 분포 · 자유도 보정 · 레버리지를 포함한다.

    구간 폭은 ``s · sqrt(1 + x₀ᵀ(XᵀX)⁻¹x₀)`` 에 ``t_{α/2, n-p}`` 를 곱한 값이다.
    마지막 항이 레버리지이고, 이것 때문에 예측 구간은 외삽할수록 넓어진다.
    상수 마진은 이 성질을 잃는다.
    """
    s = pd.Series(series).dropna().sort_index()
    if len(s) < MIN_OBSERVATIONS:
        return None

    idx = pd.DatetimeIndex(s.index)
    origin = idx[0]
    X = _design(idx, origin)
    y = s.to_numpy(dtype="float64")
    n, p = X.shape
    if n <= p:
        return None

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y

    resid = y - X @ beta
    dof = n - p
    sigma = float(np.sqrt(resid @ resid / dof))     # ddof = p. np.std(ddof=0) 이 아니다.
    t_crit = float(stats.t.ppf(0.5 + level / 2.0, dof))

    first = (start or idx[-1] + pd.Timedelta(days=1)).normalize()
    future = pd.date_range(first, periods=horizon, freq="D")
    Xf = _design(future, origin)

    mean = Xf @ beta
    leverage = np.einsum("ij,jk,ik->i", Xf, XtX_inv, Xf)
    margin = t_crit * sigma * np.sqrt(1.0 + leverage)

    return Forecast(
        mean=pd.Series(mean, index=future),
        lower=pd.Series(mean - margin, index=future),
        upper=pd.Series(mean + margin, index=future),
        fitted=pd.Series(X @ beta, index=idx),
        level=level,
        model="ols_trend_dow",
    )


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------
def prophet_forecast(
    series: pd.Series,
    horizon: int = 7,
    level: float = DEFAULT_LEVEL,
    start: pd.Timestamp | None = None,
    changepoint_prior_scale: float = 0.05,
    n_changepoints: int = 10,
    weekly_seasonality: bool | int = True,
) -> Forecast | None:
    """Prophet 래퍼. prophet 은 무거우므로 필요할 때만 import 한다."""
    s = pd.Series(series).dropna().sort_index()
    if len(s) < MIN_OBSERVATIONS:
        return None

    try:
        from prophet import Prophet
    except ImportError:
        return None

    idx = pd.DatetimeIndex(s.index)
    # 주간 계절성은 최소 몇 주기는 봐야 식별된다. 7일로 푸리에 6모수를 적합하면
    # 과적합이 보장된다(이전 MIN_TRAIN_DAYS 가 7이었다).
    if weekly_seasonality and len(s) < 28:
        weekly_seasonality = False

    m = Prophet(
        daily_seasonality=False,
        yearly_seasonality=False,
        weekly_seasonality=weekly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        n_changepoints=min(n_changepoints, max(1, len(s) // 3)),
        interval_width=level,
    )
    m.fit(pd.DataFrame({"ds": idx, "y": s.to_numpy(dtype="float64")}))

    first = (start or idx[-1] + pd.Timedelta(days=1)).normalize()
    future = pd.date_range(first, periods=horizon, freq="D")
    fc = m.predict(pd.DataFrame({"ds": future})).set_index("ds")
    hist = m.predict(pd.DataFrame({"ds": idx})).set_index("ds")

    return Forecast(
        mean=fc["yhat"],
        lower=fc["yhat_lower"],
        upper=fc["yhat_upper"],
        fitted=hist["yhat"],
        level=level,
        model="prophet",
    )


# ---------------------------------------------------------------------------
# 모델 선택
# ---------------------------------------------------------------------------
MODELS = {
    "ols_trend_dow": ols_trend_dow,
    "prophet": prophet_forecast,
}

DEFAULT_MODEL = "prophet"


def predict(
    series: pd.Series,
    horizon: int = 7,
    level: float = DEFAULT_LEVEL,
    model: str | None = None,
    start: pd.Timestamp | None = None,
) -> Forecast | None:
    """모델 이름으로 예측한다. Prophet 이 없으면 OLS 로 내려간다."""
    name = model or DEFAULT_MODEL
    fn = MODELS.get(name, MODELS[DEFAULT_MODEL])
    out = fn(series, horizon=horizon, level=level, start=start)
    if out is None and name != "ols_trend_dow":
        out = ols_trend_dow(series, horizon=horizon, level=level, start=start)
    return out


def measured(item: str, registry: dict | None) -> dict | None:
    """백테스트가 이 품목에 대해 실제로 측정한 값(MAPE·커버리지)."""
    if not registry:
        return None
    return (registry.get("items") or {}).get(item)


def select_model(item: str, registry: dict | None) -> str:
    """백테스트 결과로 품목별 모델을 고른다.

    예전에는 선택 **기준**이 주석상 "CV<5% 안정형"이었는데 **구현**은 품목명
    2개 하드코딩이었다. 다른 품목이 조건을 만족해도 반영되지 않았다.
    이제는 :mod:`analysis.backtest` 가 남긴 결과를 그대로 읽는다.
    """
    if not registry:
        return DEFAULT_MODEL
    entry = (registry.get("items") or {}).get(item)
    if not entry:
        return registry.get("default", DEFAULT_MODEL)
    return entry.get("model", DEFAULT_MODEL)

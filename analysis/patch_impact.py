"""패치·이벤트의 가격 영향 추정 (반사실 비교).

패치 이전 데이터로 모델을 학습해 "패치가 없었다면" 의 가격을 예측하고,
실제 가격과 비교한다.

이전 구현에서 고친 것:

1. **일 경계.** `resample('D')` 로 자정 기준 일평균을 썼다. 대시보드는 게임일
   (06시) 기준이라 같은 CSV 에서 다른 시계열이 나왔다. 이제 양쪽 모두
   :mod:`common.timeaxis` 를 쓴다.
2. **학습 구간 길이.** `MIN_TRAIN_DAYS=7` 로 주간 계절성을 적합했다. 관측
   7개로 푸리에 6모수를 맞추는 것은 과적합이다. 28일로 올렸다.
3. **구간을 계산해 놓고 쓰지 않았다.** 요약표가 차이와 변화율만 보여줘서,
   그 차이가 예측 불확실성 안인지 밖인지 알 수 없었다. 이제 구간 밖 일수를
   함께 보고한다.
4. **같은 날 다른 이벤트.** 2026-03-11 에만 이벤트가 3건 겹쳐 있다. 한
   이벤트에 변화를 통째로 귀속시키기 전에 경고한다.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import backtest, forecast as fc, marketdata as md   # noqa: E402
from common import timeaxis                                        # noqa: E402

# 다른 모듈이 기존 경로로 import 하던 상수들을 그대로 재노출한다.
MARKET_FILES = md.MARKET_FILES
CATEGORY_KR = md.CATEGORY_KR
load_event_log = md.load_event_log
find_item = md.find_item

# 주간 계절성이 식별되려면 최소 몇 주기는 필요하다.
MIN_TRAIN_DAYS = 28
MIN_POST_DAYS = 5

# 패치 이후 몇 일까지를 '영향'으로 볼 것인가.
#
# 남은 데이터를 전부 쓰면 사후 구간이 수백 일이 되는데, 그만큼 외삽한 반사실은
# 예측구간이 넓어져 거의 모든 값을 품는다. 실제로 전체 구간(188일)으로 돌리면
# 상급 아비도스 융화 재료가 +52% 인데도 구간 밖 일수가 2/188 로 나온다. 패치
# 직후의 한정된 창으로 보는 편이 질문("이 패치가 가격을 움직였나")에 맞는다.
POST_WINDOW_DAYS = 28

# 한글 폰트: 'Malgun Gothic' 만 지정하면 Windows 밖에서 글자가 깨진다.
FONT_STACK = "Malgun Gothic, AppleGothic, NanumGothic, Noto Sans KR, sans-serif"


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------
def load_daily_markets() -> dict[str, pd.DataFrame]:
    """카테고리별 **게임일(06시) 기준** 일평균."""
    return {name: timeaxis.game_daily_mean(df)[0]
            for name, df in md.load_all_markets().items()}


def get_items_by_category(category: str, daily: dict) -> list[str]:
    return md.items_of(category, daily)


# ---------------------------------------------------------------------------
# 분석 가능 여부
# ---------------------------------------------------------------------------
def check_feasibility(series: pd.Series, patch_date: pd.Timestamp) -> tuple[bool, str]:
    pre = series[series.index < patch_date]
    post = series[series.index >= patch_date]
    if len(pre) < MIN_TRAIN_DAYS:
        return False, f"사전 데이터 부족 ({len(pre)}일 < 최소 {MIN_TRAIN_DAYS}일)"
    if len(post) < MIN_POST_DAYS:
        return False, f"사후 데이터 부족 ({len(post)}일 < 최소 {MIN_POST_DAYS}일)"
    return True, "OK"


def confounding_events(patch_name: str, patch_date, events: dict | None = None) -> list[str]:
    """같은 날짜의 다른 이벤트. 있으면 인과 귀속이 성립하지 않는다."""
    events = events if events is not None else md.load_event_log()
    return [n for n in md.events_on(patch_date, events) if n != patch_name]


# ---------------------------------------------------------------------------
# 반사실 추정
# ---------------------------------------------------------------------------
def run_counterfactual(
    series: pd.Series,
    patch_date: pd.Timestamp,
    model: str | None = None,
    level: float = fc.DEFAULT_LEVEL,
    post_window: int | None = POST_WINDOW_DAYS,
) -> dict | None:
    """패치 이전 데이터로 학습해 이후 ``post_window`` 일을 예측한다."""
    s = pd.Series(series).dropna().sort_index()
    train = s[s.index < patch_date]
    post = s[s.index >= patch_date]
    if post_window:
        post = post[post.index < patch_date + pd.Timedelta(days=post_window)]
    if train.empty or post.empty:
        return None

    out = fc.predict(train, horizon=len(post), level=level, model=model,
                     start=pd.Timestamp(post.index[0]))
    if out is None:
        return None

    return {
        "actual_pre": train,
        "actual_post": post,
        "cf_mean": out.mean.reindex(post.index),
        "cf_lower": out.lower.reindex(post.index),
        "cf_upper": out.upper.reindex(post.index),
        "level": out.level,
        "model": out.model,
    }


def calc_impact(res: dict, label: str, patch_date: pd.Timestamp) -> dict:
    """영향 요약. 차이뿐 아니라 그 차이가 예측 불확실성 밖인지도 함께 낸다."""
    actual = res["actual_post"]
    cf = res["cf_mean"].reindex(actual.index)
    lo = res["cf_lower"].reindex(actual.index)
    hi = res["cf_upper"].reindex(actual.index)

    ok = actual.notna() & cf.notna()
    a_mean = float(actual[ok].mean()) if ok.any() else float("nan")
    c_mean = float(cf[ok].mean()) if ok.any() else float("nan")
    diff = a_mean - c_mean
    # 일별 변화율의 평균이 아니라 평균끼리의 변화율. 표에 적힌 '차이'와 맞는다.
    pct = (diff / c_mean * 100.0) if c_mean else float("nan")

    outside = int(((actual < lo) | (actual > hi))[ok].sum())
    total = int(ok.sum())
    share = outside / total if total else float("nan")

    if total == 0:
        verdict = "판단 불가"
    elif share >= 0.5:
        verdict = "구간 밖 (영향 있음)"
    elif outside == 0:
        verdict = "구간 안 (영향 확인 안 됨)"
    else:
        verdict = "혼재 (판단 보류)"

    return {
        "아이템": label,
        "패치일": patch_date.date(),
        "실제 평균": round(a_mean, 1),
        "반사실 평균": round(c_mean, 1),
        "차이": round(diff, 1),
        "변화율(%)": round(pct, 1),
        f"구간 밖 일수": f"{outside}/{total}",
        "판정": verdict,
    }


def analyze_patch(
    patch_name: str,
    items: list[str],
    daily: dict,
    patch_date: pd.Timestamp | None = None,
    registry: dict | None = None,
    level: float = fc.DEFAULT_LEVEL,
    post_window: int | None = POST_WINDOW_DAYS,
) -> tuple[pd.DataFrame | None, dict, dict]:
    """``(요약표, 품목별 결과, 진단)`` 을 돌려준다.

    진단에는 건너뛴 품목과 **같은 날 겹친 다른 이벤트**가 들어간다. 호출부가
    이것을 화면에 띄워야 한다.
    """
    events = md.load_event_log()
    if patch_date is None:
        if patch_name not in events:
            return None, {}, {"error": f'event_log 에 "{patch_name}" 항목 없음'}
        patch_date = events[patch_name]
    patch_date = pd.Timestamp(patch_date).normalize()

    if registry is None:
        registry = backtest.load_registry()

    results, rows, skipped = {}, [], []
    for item in items:
        series = md.find_item(item, daily)
        if series is None:
            skipped.append((item, "시세 데이터 없음"))
            continue

        ok, reason = check_feasibility(series, patch_date)
        if not ok:
            skipped.append((item, reason))
            continue

        res = run_counterfactual(series, patch_date, level=level, post_window=post_window,
                                 model=fc.select_model(item, registry))
        if res is None:
            skipped.append((item, "모델 적합 실패"))
            continue

        results[item] = res
        rows.append(calc_impact(res, item, patch_date))

    diagnostics = {
        "patch_date": patch_date,
        "skipped": skipped,
        "confounders": confounding_events(patch_name, patch_date, events),
        "level": level,
        "min_train_days": MIN_TRAIN_DAYS,
        "post_window": post_window,
        "models": {k: v["model"] for k, v in results.items()},
    }

    if not results:
        return None, {}, diagnostics
    return pd.DataFrame(rows), results, diagnostics


# ---------------------------------------------------------------------------
# 차트
# ---------------------------------------------------------------------------
def build_plotly_chart(results: dict, patch_name: str, patch_date: pd.Timestamp) -> go.Figure:
    n_items = len(results)
    n_cols = min(2, n_items)
    n_rows = max(1, (n_items + n_cols - 1) // n_cols)
    v_sp = min(0.08, 0.6 / n_rows) if n_rows > 1 else 0.0
    h_sp = min(0.06, 0.6 / n_cols) if n_cols > 1 else 0.0
    titles = list(results.keys()) + [""] * (n_rows * n_cols - n_items)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        vertical_spacing=v_sp, horizontal_spacing=h_sp)

    C = dict(actual="#4C9EEB", cf="#FF6B6B", band="rgba(255,107,107,0.15)")
    level = next(iter(results.values())).get("level", fc.DEFAULT_LEVEL)

    for idx, (label, res) in enumerate(results.items()):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        first = idx == 0
        actual_all = pd.concat([res["actual_pre"], res["actual_post"]])

        fig.add_trace(go.Scatter(
            x=actual_all.index, y=actual_all.values, mode="lines", name="실제 가격",
            line=dict(color=C["actual"], width=2), legendgroup="actual", showlegend=first,
        ), row=row, col=col)

        cf_x = res["cf_mean"].index
        for y, fill in ((res["cf_upper"].values, None), (res["cf_lower"].values, "tonexty")):
            fig.add_trace(go.Scatter(
                x=cf_x, y=y, mode="lines", line=dict(width=0), fill=fill,
                fillcolor=C["band"], showlegend=False, hoverinfo="skip",
            ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=cf_x, y=res["cf_mean"].values, mode="lines",
            name=f"반사실 예측 ({level:.0%} 구간)",
            line=dict(color=C["cf"], width=2, dash="dot"),
            legendgroup="cf", showlegend=first,
        ), row=row, col=col)

        yvals = actual_all.to_numpy(dtype="float64")
        if np.isfinite(yvals).any():
            fig.add_trace(go.Scatter(
                x=[patch_date, patch_date],
                y=[float(np.nanmin(yvals)), float(np.nanmax(yvals))],
                mode="lines", name=f"{patch_name} 출시",
                line=dict(color="gold", width=1.5, dash="dash"),
                legendgroup="patch", showlegend=first,
            ), row=row, col=col)

    fig.update_layout(
        title=dict(text=f"{patch_name} 임팩트 분석 (출시일: {patch_date.date()}) — 실제 vs 반사실",
                   font=dict(size=15, family=FONT_STACK), x=0.5),
        height=350 * n_rows,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        font=dict(family=FONT_STACK), template="plotly_white",
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


if __name__ == "__main__":
    daily = load_daily_markets()
    ITEMS = ["운명의 파괴석 결정", "운명의 파괴석", "상급 아비도스 융화 재료", "아비도스 융화 재료"]

    summary, results, diag = analyze_patch("지평의 성당", ITEMS, daily)
    if diag.get("confounders"):
        print(f"[주의] 같은 날 다른 이벤트: {', '.join(diag['confounders'])}")
        print("       변화를 한 이벤트에 귀속시킬 수 없습니다.\n")
    for item, reason in diag.get("skipped", []):
        print(f"  [{item}] 분석 제외: {reason}")
    if summary is None:
        print("\n분석 가능한 아이템이 없습니다.")
    else:
        print(summary.to_string(index=False))

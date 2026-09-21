"""LoaQuant 대시보드.

시간 축(게임일 경계, 지표 창)은 :mod:`common.timeaxis`, 데이터 로딩과 이벤트
파싱은 :mod:`analysis.marketdata`, 예측과 예측구간은 :mod:`analysis.forecast`
에 있다. 이 파일은 화면만 그린다.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import backtest, forecast as fc, marketdata as md, metrics  # noqa: E402
from analysis.patch_impact import (                                   # noqa: E402
    MIN_TRAIN_DAYS, POST_WINDOW_DAYS, analyze_patch, build_plotly_chart,
)
from common import timeaxis as ta                                     # noqa: E402

st.set_page_config(page_title="LoaQuant", layout="wide")
st.title("LoaQuant")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #ffffff;
        border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# 교환비는 게임 내 규칙이라 수집 데이터로는 확인할 수 없다. 코드 곳곳에 상수 5 가
# 박혀 있던 것을 페어별 표로 옮겨, 패치로 한 페어만 바뀌어도 여기만 고치면 되게 했다.
# 아래 값은 2026-09-21 저장소 소유자(Cho MinSeo)가 8개 페어 모두 5 가 맞다고 확인했다.
# 새 페어를 추가할 때는 확인 전까지 verified=False 로 두면 화면에 미확인으로 표시된다.
EXCHANGE_PAIRS = [
    # (하위 재료, 상위 재료, 하위 몇 개가 상위 1개인가, 확인됨)
    ("찬란한 명예의 돌파석", "운명의 돌파석", 5, True),
    ("운명의 돌파석", "위대한 운명의 돌파석", 5, True),
    ("정제된 파괴강석", "운명의 파괴석", 5, True),
    ("운명의 파괴석", "운명의 파괴석 결정", 5, True),
    ("정제된 수호강석", "운명의 수호석", 5, True),
    ("운명의 수호석", "운명의 수호석 결정", 5, True),
    ("최상급 오레하 융화 재료", "아비도스 융화 재료", 5, True),
    ("아비도스 융화 재료", "상급 아비도스 융화 재료", 5, True),
]


# ==========================================================================
# 로딩 (캐시)
# ==========================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_market(name: str) -> pd.DataFrame:
    return md.load_market(name)


@st.cache_data(ttl=600, show_spinner=False)
def load_volume() -> pd.DataFrame:
    return md.load_volume()


@st.cache_data(ttl=600, show_spinner=False)
def load_gold() -> pd.Series:
    return md.load_gold()


@st.cache_data(ttl=600, show_spinner=False)
def load_events() -> dict:
    return md.load_event_log()


@st.cache_data(ttl=600, show_spinner=False)
def load_subcategories() -> pd.Series | None:
    return md.sub_categories("lifeskill")


@st.cache_data(ttl=600, show_spinner=False)
def load_registry() -> dict | None:
    return backtest.load_registry()


@st.cache_data(ttl=600, show_spinner=False)
def daily_frame(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """카테고리의 게임일 평균과 표본 수."""
    return ta.game_daily_mean(load_market(name))


@st.cache_data(ttl=3600, show_spinner=False)
def forecast_series(values: pd.Series, horizon: int, model: str, start: pd.Timestamp):
    return fc.predict(values, horizon=horizon, model=model, start=start)


# ==========================================================================
# 공통 컴포넌트
# ==========================================================================
def add_events(fig, events, lo, hi, y_pos=1.05):
    """같은 날 이벤트는 한 줄로 묶어 표시한다."""
    if not events or pd.isna(lo) or pd.isna(hi):
        return
    grouped: dict[pd.Timestamp, list[str]] = {}
    for name, date in events.items():
        d = pd.Timestamp(date).normalize()
        if lo <= d <= hi:
            grouped.setdefault(d, []).append(name)

    for date, names in grouped.items():
        fig.add_vline(x=date, line_width=2, line_dash="dot", line_color="#E74C3C")
        fig.add_annotation(x=date, y=y_pos, yref="paper", text="<br>".join(names),
                           showarrow=False, font=dict(color="#E74C3C", size=11),
                           bgcolor="rgba(255,255,255,0.9)", bordercolor="#E74C3C", borderwidth=1)


def wednesday_ticks(lo, hi):
    if pd.isna(lo) or pd.isna(hi) or lo > hi:
        return [], []
    days = pd.date_range(pd.Timestamp(lo).normalize(), pd.Timestamp(hi).normalize(), freq="D")
    vals = [d for d in days if d.weekday() == ta.MAINTENANCE_WEEKDAY]
    return vals, [d.strftime("%m.%d(수)") for d in vals]


def cash_mode_labels(is_cash: bool):
    return ("원", ",.2f") if is_cash else ("골드", ",.0f")


def interval_note(item: str, level: float, registry: dict | None) -> str:
    """예측구간에 붙일 설명.

    "95% 구간"이라고만 적으면 그 구간이 실제로 95% 를 덮는다는 뜻으로 읽힌다.
    백테스트로 잰 실측 커버리지가 있으면 함께 보여준다. 이 저장소 데이터에서는
    품목에 따라 36~100% 로 갈린다.
    """
    m = fc.measured(item, registry)
    if not m or m.get("coverage") is None:
        return f"명목 {level:.0%} 구간 (실측 커버리지 미측정)"
    cov = m["coverage"]
    warn = " ⚠ 명목보다 크게 낮음" if cov < level - 0.15 else ""
    return (f"명목 {level:.0%} 구간 · 실측 커버리지 {cov:.0%}"
            f" · MAPE {m['mape']:.1f}% (n={m['n']}){warn}")


# ==========================================================================
# 시장 분석 리포트 — 게임일 축에서 계산한다
# ==========================================================================
SIGNAL_STYLE = {
    "strong_buy": ("#d9534f", "#ffe6e6"),
    "strong_sell": ("#0275d8", "#e6f2ff"),
    "buy": ("green", "#eaffea"),
    "caution": ("red", "#ffebe6"),
    "hot": ("orange", "#f9f9f9"),
    "cold": ("blue", "#f9f9f9"),
    "neutral": ("gray", "#f9f9f9"),
}


def render_report_cards(daily: pd.DataFrame, reliable: pd.DataFrame, is_cash: bool):
    st.markdown("##### 시장 분석 리포트")
    st.caption(
        f"게임일(06시 기준) 종가 축에서 RSI {ta.RSI_PERIODS}일 · "
        f"볼린저 {ta.BOLLINGER_PERIODS}일로 계산합니다."
    )
    unit = "원" if is_cash else "G"
    _, fmt = cash_mode_labels(is_cash)

    cols = st.columns(len(daily.columns))
    for idx, column in enumerate(daily.columns):
        rep = metrics.market_signal(daily[column], reliable[column])
        with cols[idx]:
            if rep is None:
                st.caption(f"**{column}**: 지표를 낼 만큼 데이터가 모이지 않았습니다")
                continue
            if rep["stale"]:
                st.caption(f"**{column}**: {rep['last']:%Y-%m-%d} 이후 값이 없어 "
                           f"현재가를 표시하지 않습니다 ({rep['stale_days']}일 경과)")
                continue

            color, bg = SIGNAL_STYLE[rep["signal"]]
            rsi_color = "red" if rep["rsi"] >= 70 else "blue" if rep["rsi"] <= 30 else "gray"
            diff_txt = "0" if rep["diff"] == 0 else f"{rep['diff']:+{fmt}}"
            st.markdown(f"""
            <div style="border:1px solid #ddd;border-radius:10px;padding:15px;
                        background-color:{bg};box-shadow:2px 2px 5px rgba(0,0,0,0.05);">
              <div style="font-size:0.9rem;color:#555;margin-bottom:5px;">{column}</div>
              <div style="display:flex;justify-content:space-between;align-items:end;">
                <span style="font-size:1.4rem;font-weight:bold;color:#333;">{rep['price']:{fmt}} {unit}</span>
                <span style="font-size:0.9rem;font-weight:bold;color:{color};">({diff_txt})</span>
              </div>
              <hr style="margin:10px 0;border:0;border-top:1px solid #ddd;">
              <div style="font-size:0.85rem;color:#666;margin-bottom:5px;">
                RSI 지수: <span style="font-weight:bold;color:{rsi_color}">{rep['rsi']:.1f}</span>
              </div>
              <div style="font-size:1rem;font-weight:bold;color:{color};">{rep['label']}</div>
              <div style="font-size:0.75rem;color:#999;margin-top:6px;">
                {rep['as_of'].strftime('%m/%d')} 게임일 기준
              </div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================================================
# 차트
# ==========================================================================
def draw_price_chart(hourly: pd.DataFrame, title: str, is_cash: bool, events: dict):
    unit, _ = cash_mode_labels(is_cash)
    fig = go.Figure()
    fmt = ("%{x|%m/%d %H:%M} - %{y:,.2f} " if is_cash else "%{x|%m/%d %H:%M} - %{y:,.0f} ") \
        + unit + "<extra></extra>"

    for idx, column in enumerate(hourly.columns):
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly[column], mode="lines", name=column,
                                 line=dict(width=2, color=PALETTE[idx % len(PALETTE)]),
                                 hovertemplate=fmt))

    lo, hi = hourly.index.min(), hourly.index.max()
    for a, b in ta.maintenance_windows(lo, hi):
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(128,128,128,0.2)", layer="below",
                      line_width=0, annotation_text="점검", annotation_position="top left",
                      annotation_font=dict(color="gray", size=10))

    if events:
        add_events(fig, events, lo, hi)

    vals, text = wednesday_ticks(lo, hi)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        hovermode="x unified", template="plotly_white",
        xaxis=dict(showgrid=True, gridcolor="#eee", rangeslider=dict(visible=True),
                   type="date", tickmode="array", tickvals=vals, ticktext=text, tickangle=0),
        yaxis=dict(showgrid=True, gridcolor="#eee",
                   tickformat=",.2f" if is_cash else ",", title=f"가격 ({unit})"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20), height=500,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"price_{title}")


def draw_candles(daily: pd.DataFrame, counts: pd.DataFrame, hourly: pd.DataFrame,
                 title: str, is_cash: bool, events: dict, registry: dict | None):
    st.markdown("#### 시세 캔들스틱 차트 및 예측")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        timeframe = st.radio("기준 시간 단위", ["1일", "1주"], horizontal=True, key=f"tf_{title}")
    with c2:
        show_band = st.checkbox("볼린저 밴드", value=False, key=f"bb_{title}")
    with c3:
        show_fc = st.checkbox("예측 (1일 기준)", value=False, key=f"fc_{title}")

    unit, _ = cash_mode_labels(is_cash)
    complete, _ = ta.split_complete(daily.index)

    for idx, column in enumerate(daily.columns):
        shifted = hourly[[column]].copy()
        shifted.index = shifted.index - pd.Timedelta(hours=ta.GAME_DAY_START_HOUR)
        rule = "1D" if timeframe == "1일" else "W-TUE"
        ohlc = shifted[column].resample(rule).agg(["first", "max", "min", "last"]).dropna()
        if ohlc.empty:
            continue
        ohlc.index = pd.to_datetime(ohlc.index.date)
        if timeframe == "1주":
            ohlc.index = ohlc.index - pd.Timedelta(days=6)
        ohlc.columns = ["Open", "High", "Low", "Close"]

        # 진행 중인 하루/주는 아직 끝나지 않았다. 완료된 봉과 구분한다.
        done = ohlc.index < (complete.max() + pd.Timedelta(days=1) if len(complete) else ohlc.index.max())
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=ohlc.index[done], open=ohlc["Open"][done], high=ohlc["High"][done],
            low=ohlc["Low"][done], close=ohlc["Close"][done], name=column,
            increasing_line_color="#d9534f", decreasing_line_color="#0275d8",
            increasing_fillcolor="#d9534f", decreasing_fillcolor="#0275d8"))
        if (~done).any():
            fig.add_trace(go.Candlestick(
                x=ohlc.index[~done], open=ohlc["Open"][~done], high=ohlc["High"][~done],
                low=ohlc["Low"][~done], close=ohlc["Close"][~done], name="진행 중",
                increasing_line_color="#bbb", decreasing_line_color="#bbb",
                increasing_fillcolor="rgba(187,187,187,0.5)",
                decreasing_fillcolor="rgba(187,187,187,0.5)"))

        if show_band and timeframe == "1일":
            obs = ta.reliable_mask(counts[[column]])[column].reindex(daily.index).fillna(False)
            ma, up, lo_b = ta.bollinger(daily[column], obs)
            fig.add_trace(go.Scatter(x=up.index, y=up, mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=lo_b.index, y=lo_b, mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor="rgba(31,119,180,0.10)",
                                     name=f"볼린저 {ta.BOLLINGER_PERIODS}일", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=ma.index, y=ma, mode="lines",
                                     line=dict(width=1, dash="dot", color="#1f77b4"),
                                     showlegend=False, hoverinfo="skip"))

        if show_fc and timeframe == "1일":
            train = daily[column].reindex(complete).dropna()
            if len(train) >= fc.MIN_OBSERVATIONS:
                model = fc.select_model(column, registry)
                with st.spinner(f"{column} 예측 중..."):
                    out = forecast_series(train, 7, model, pd.Timestamp(train.index[-1]) + pd.Timedelta(days=1))
                if out is not None:
                    color = PALETTE[idx % len(PALETTE)]
                    rgba = "rgba" + str(tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) + (0.15,))
                    fig.add_trace(go.Scatter(x=out.index, y=out.mean, mode="lines",
                                             name=f"예측 ({out.model})",
                                             line=dict(width=2, dash="dot", color=color)))
                    fig.add_trace(go.Scatter(
                        x=list(out.index) + list(out.index[::-1]),
                        y=list(out.upper) + list(out.lower[::-1]),
                        fill="toself", fillcolor=rgba, line=dict(color="rgba(255,255,255,0)"),
                        name=f"{out.level:.0%} 예측구간", hoverinfo="skip"))
                    st.caption(f"[{column}] {interval_note(column, out.level, registry)}")

        if events:
            add_events(fig, events, ohlc.index.min(), ohlc.index.max(), y_pos=1.0)
        vals, text = wednesday_ticks(ohlc.index.min(), ohlc.index.max())
        fig.update_layout(
            title=f"[{column}] {timeframe} 시세 변동", xaxis_rangeslider_visible=False,
            hovermode="x unified", template="plotly_white",
            xaxis=dict(showgrid=True, gridcolor="#eee", type="date", tickmode="array",
                       tickvals=vals, ticktext=text),
            yaxis=dict(showgrid=True, gridcolor="#eee",
                       tickformat=",.2f" if is_cash else ",", title=f"가격 ({unit})"),
            margin=dict(l=20, r=20, t=40, b=20), height=380, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"candle_{title}_{column}")


def draw_day_of_week(daily: pd.DataFrame, title: str):
    """추세를 제거한 요일 효과.

    수준을 그냥 평균하면 기간 추세가 요일 칸으로 새어 들어간다. 이 데이터에서는
    부호가 뒤집히는 (품목, 요일) 조합이 나온다.
    """
    st.markdown("#### 수요일 대비 요일별 시세 변동률")
    show_raw = st.checkbox("추세 제거 전 값도 함께 보기", value=False, key=f"dow_raw_{title}")
    st.caption("중심 7일 이동평균으로 추세를 제거한 뒤 요일별로 평균합니다. "
               "제거하지 않으면 기간 추세가 요일 효과로 잘못 잡힙니다.")

    kor = ["월", "화", "수(기준)", "목", "금", "토", "일"]
    fig = go.Figure()
    for idx, column in enumerate(daily.columns):
        eff, n = ta.day_of_week_effect(daily[column])
        if eff.isna().all():
            continue
        fig.add_trace(go.Bar(
            x=kor, y=eff.to_numpy(), name=column,
            marker_color=PALETTE[idx % len(PALETTE)],
            text=[f"{v:+.2f}%" if pd.notna(v) else "" for v in eff],
            textposition="outside",
            customdata=n.to_numpy(),
            hovertemplate="%{x} — 수요일 대비 %{y:+.2f}% (표본 %{customdata}일)<extra></extra>"))
        if show_raw:
            raw, _ = ta.day_of_week_effect(daily[column], detrend=False)
            fig.add_trace(go.Bar(x=kor, y=raw.to_numpy(), name=f"{column} (추세 제거 전)",
                                 marker_color="#cccccc",
                                 hovertemplate="%{x} — %{y:+.2f}%<extra></extra>"))

    fig.update_layout(
        hovermode="x unified", template="plotly_white", barmode="group",
        xaxis=dict(title="요일", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#eee", tickformat="+.1f",
                   ticksuffix="%", title="변동률 (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20), height=350)
    st.plotly_chart(fig, use_container_width=True, key=f"dow_{title}")


def draw_summary_table(daily: pd.DataFrame, counts: pd.DataFrame, is_cash: bool):
    if daily.empty:
        return
    _, fmt = cash_mode_labels(is_cash)
    asc = daily.sort_index()
    diff = asc.diff()
    desc, diff_desc = asc.iloc[::-1], diff.iloc[::-1]
    cnt = counts.reindex(desc.index)

    out = pd.DataFrame(index=desc.index)
    for col in desc.columns:
        out[col] = [
            (f"{p:{fmt}} ({d:+{fmt}})" if pd.notna(d) else f"{p:{fmt}} (-)") if pd.notna(p) else "-"
            for p, d in zip(desc[col], diff_desc[col])
        ]
    out["표본"] = [f"{int(v)}건" if pd.notna(v) else "-" for v in cnt.max(axis=1)]

    kor = ["월", "화", "수", "목", "금", "토", "일"]
    out.index = [f"{d.strftime('%Y-%m-%d')} ({kor[d.weekday()]})" for d in out.index]

    def style(val):
        try:
            if "(-)" in val or val == "-":
                return "color: gray;"
            change = float(val[val.rfind("(") + 1:val.rfind(")")].replace(",", ""))
            return ("color:#d9534f;font-weight:bold;" if change > 0 else
                    "color:#0275d8;font-weight:bold;" if change < 0 else "color:gray;")
        except (ValueError, AttributeError):
            return ""

    st.caption("게임일(06시 기준) 평균가입니다. '표본'은 그날 수집된 횟수이고, "
               f"{ta.DEFAULT_MIN_SAMPLES}건 미만인 날은 평균을 그대로 믿기 어렵습니다.")
    st.dataframe(out.style.map(style, subset=list(desc.columns)),
                 use_container_width=True)


def draw_volume(volume: pd.DataFrame, items: list[str], title: str, events: dict):
    """일일 거래량.

    마지막 날은 **아직 집계 중인 부분값**이다. 다른 날과 같은 막대로 그리면
    매일 수요가 무너지는 것처럼 보인다(실측: 직전 14일 중앙값의 30%).
    """
    if volume.empty or not items:
        return
    cols = [c for c in items if c in volume.columns]
    if not cols:
        return

    df = volume[cols]
    today = ta.now_kst().normalize()
    partial = df.index >= today

    st.markdown("#### 일일 거래량 추이")
    fig = go.Figure()
    for idx, column in enumerate(df.columns):
        color = PALETTE[idx % len(PALETTE)]
        fig.add_trace(go.Bar(x=df.index[~partial], y=df[column][~partial], name=column,
                             marker_color=color,
                             hovertemplate="%{x|%m/%d} - 거래량: %{y:,.0f} 개<extra></extra>"))
        if partial.any():
            fig.add_trace(go.Bar(x=df.index[partial], y=df[column][partial],
                                 name=f"{column} (집계 중)", marker_color=color,
                                 marker_pattern_shape="/", opacity=0.45,
                                 hovertemplate="%{x|%m/%d} - 집계 중: %{y:,.0f} 개<extra></extra>"))

    if partial.any():
        st.caption("빗금 친 마지막 막대는 오늘 현재까지 집계된 부분값입니다. "
                   "하루가 끝나야 다른 날과 비교할 수 있습니다.")

    if events:
        add_events(fig, events, df.index.min(), df.index.max())
    vals, text = wednesday_ticks(df.index.min(), df.index.max())
    fig.update_layout(
        hovermode="x unified", template="plotly_white", barmode="group",
        xaxis=dict(showgrid=True, gridcolor="#eee", type="date", tickmode="array",
                   tickvals=vals, ticktext=text, tickangle=0),
        yaxis=dict(showgrid=True, gridcolor="#eee", tickformat=",", title="거래량 (개)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20), height=350)
    st.plotly_chart(fig, use_container_width=True, key=f"vol_{title}")


# ==========================================================================
# 오늘의 예측 카드
# ==========================================================================
def draw_forecast_cards(daily: pd.DataFrame, counts: pd.DataFrame, items: list[str],
                        is_cash: bool, registry: dict | None):
    """오늘 게임일의 종가를 예측한다.

    예전 카드는 "오늘의 시세 예측"이라고 적어두고 **내일**을 보여줬다. 학습
    데이터에 진행 중인 오늘이 포함돼 있었고, Prophet 이 그 다음 날을 예측했기
    때문이다. 이제 완료된 날까지만 학습하고 오늘을 예측한다.
    """
    items = [i for i in items if i in daily.columns]
    if not items:
        return

    complete, partial = ta.split_complete(daily.index)
    if len(complete) < fc.MIN_OBSERVATIONS:
        return
    today = ta.current_game_day()

    unit = "원" if is_cash else "G"
    _, fmt = cash_mode_labels(is_cash)

    st.markdown(f"#### 오늘({today.strftime('%m/%d')} 게임일)의 종가 예측")
    cols = st.columns(len(items))
    for i, item in enumerate(items):
        with cols[i]:
            train = daily[item].reindex(complete).dropna()
            if len(train) < fc.MIN_OBSERVATIONS:
                st.caption(f"**{item}**: 학습 데이터 부족")
                continue

            model = fc.select_model(item, registry)
            with st.spinner("예측 중..."):
                out = forecast_series(train, 1, model, today)
            if out is None:
                st.caption(f"**{item}**: 예측 실패")
                continue

            pred = float(out.mean.iloc[0])
            last = float(train.iloc[-1])
            delta = pred - last
            pct = (delta / last * 100) if last else float("nan")

            st.metric(label=f"[{item}] 예측 종가",
                      value=f"{pred:{fmt}} {unit}",
                      delta=f"{delta:+{fmt}} {unit} ({pct:+.2f}%)")
            st.caption(f"{train.index[-1].strftime('%m/%d')} 종가 {last:{fmt}} {unit} 대비 · {out.model}")
            st.caption(f"구간 {out.lower.iloc[0]:{fmt}} ~ {out.upper.iloc[0]:{fmt}} {unit}")
            st.caption(interval_note(item, out.level, registry))

            if len(partial):
                so_far = daily[item].reindex(partial).dropna()
                n = counts[item].reindex(partial).dropna()
                if not so_far.empty:
                    st.caption(f"오늘 현재까지 평균 {so_far.iloc[-1]:{fmt}} {unit} "
                               f"(수집 {int(n.iloc[-1]) if not n.empty else 0}건, 집계 중)")
    st.divider()


# ==========================================================================
# 거래대금
# ==========================================================================
def trade_value_table(daily: pd.DataFrame, volume: pd.DataFrame, target_days,
                      custom_range, is_cash: bool) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """:func:`analysis.metrics.trade_value` 결과를 화면용 표로."""
    start = end = None
    if target_days == "custom":
        if not custom_range or len(custom_range) != 2:
            return pd.DataFrame(), None
        start, end = pd.Timestamp(custom_range[0]), pd.Timestamp(custom_range[1])
    elif target_days is not None:
        p, _ = md.align_price_volume(daily, volume)
        start = metrics.recent_window(p.index, target_days)

    raw, excluded = metrics.trade_value(
        daily, volume, start=start, end=end,
        exclude_from=ta.now_kst().normalize())
    if raw.empty:
        return raw, excluded

    unit = "원" if is_cash else "G"
    out = pd.DataFrame({
        f"거래량 가중 평균 단가 ({unit})": raw["avg_price"],
        "총 거래량 (개)": raw["volume"],
        f"총 거래대금 ({unit})": raw["value"],
    })
    out.index.name = "품목명"
    return out, excluded


# ==========================================================================
# 메인
# ==========================================================================
events = load_events()
problems = md.event_log_problems()
volume_all = load_volume()
gold = load_gold()
registry = load_registry()

materials_h = load_market("materials")

# ── 수집 현황: "매시간"이라고 적어두지 않고 실제 수집률을 계산해 보여준다 ──
if not materials_h.empty:
    cov = ta.coverage(materials_h.index)
    st.markdown(f"""
    <div style="background-color:#f8f9fa;padding:12px;border-radius:8px;
                border-left:5px solid #ff4b4b;margin-bottom:18px;">
      <span style="color:#6c757d;font-size:0.9rem;">수집 현황: </span>
      <b style="font-size:1.05rem;color:#31333F;">{cov['start']:%Y-%m-%d}</b>
      <span style="color:#6c757d;"> ~ </span>
      <b style="font-size:1.05rem;color:#31333F;">{cov['end']:%Y-%m-%d %H:%M}</b>
      <span style="color:#6c757d;"> (KST) · 수집 {cov['n']:,}회</span>
      <div style="color:#6c757d;font-size:0.85rem;margin-top:6px;">
        1시간 주기로 예약돼 있으나 실제 수집률은 <b>{cov['ratio']:.0%}</b>입니다
        (간격 중앙값 {cov['median_gap_h']:.1f}시간, 최대 {cov['max_gap_h']:.1f}시간).
        지표와 일평균은 이 결측을 감안해 게임일 단위로 계산합니다.
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("수집된 시세 데이터가 없습니다.")

if problems:
    st.warning("event_log.txt 에서 형식이 맞지 않아 무시된 줄: " + " / ".join(problems[:5]))

# ── 골드 환산 ──
apply_gold = False
if not gold.empty:
    c1, c2 = st.columns([2, 3])
    with c1:
        apply_gold = st.checkbox("골드 가치 반영하기 (시세를 현금으로 환산)")
    with c2:
        missing, last = md.gold_gap(materials_h.index, gold)
        if missing:
            st.warning(f"골드 환율은 {last:%Y-%m-%d} 까지만 있습니다. "
                       f"그 이후 {missing}일은 환산할 수 없어 **비워 둡니다**.")
    if registry:
        st.caption(f"예측 모델은 `analysis/backtest.py` 결과를 따릅니다 "
                   f"(생성 {registry.get('generated_at', '?')[:10]}, "
                   f"품목 {len(registry.get('items', {}))}개).")
    else:
        st.caption("예측 모델 선택 근거가 아직 없습니다. "
                   "`python analysis/backtest.py` 를 실행하면 품목별로 측정해 반영합니다.")
    st.markdown("---")


def category_view(name: str, label: str, default_items=None, forecast_items=None,
                  item_filter=None, key=None):
    """카테고리 탭 하나를 통째로 그린다."""
    key = key or name
    hourly_all = load_market(name)
    if hourly_all.empty:
        st.info("데이터가 없습니다.")
        return

    options = sorted(hourly_all.columns)
    if item_filter is not None:
        options = [i for i in options if i in item_filter]
    if not options:
        st.info("데이터가 없습니다.")
        return

    defaults = [i for i in (default_items or options[:1]) if i in options]
    selected = st.multiselect("품목 선택", options, default=defaults, key=f"sel_{key}")
    if not selected:
        st.caption("품목을 하나 이상 선택하세요.")
        return

    hourly = hourly_all[selected]
    if apply_gold:
        hourly = md.to_cash(hourly, gold)
        if hourly.dropna(how="all").empty:
            st.warning("선택한 구간에 골드 환율이 없어 현금으로 환산할 수 없습니다.")
            return

    daily, counts = ta.game_daily_mean(hourly)
    reliable = ta.reliable_mask(counts)

    if forecast_items:
        draw_forecast_cards(daily, counts, [i for i in selected if i in forecast_items],
                            apply_gold, registry)

    render_report_cards(daily, reliable, apply_gold)
    draw_price_chart(hourly, label, apply_gold, events)
    draw_candles(daily, counts, hourly, label, apply_gold, events, registry)
    draw_day_of_week(daily, label)
    with st.expander("데이터 요약 표 (게임일 기준)"):
        draw_summary_table(daily, counts, apply_gold)
    st.divider()
    draw_volume(volume_all, selected, label, events)
    return selected, hourly, daily


tab_gold, tab1, tab2, tab3, tab4, tab5, tab6, tab_patch = st.tabs(
    ["골드 시세", "강화 재료", "생활 재료", "배틀 아이템", "각인서", "보석", "거래대금", "패치분석실"])

# ── 골드 ──
with tab_gold:
    st.subheader("일별 골드 시세 (100골드 당 현금 비율)")
    if gold.empty:
        st.warning("골드 시세 데이터(data/gold/daily_gold.csv)를 찾을 수 없습니다.")
    else:
        stale = (ta.now_kst().normalize() - gold.index.max()).days
        if stale > 2:
            st.warning(f"마지막 갱신이 {gold.index.max():%Y-%m-%d} 로 {stale}일 지났습니다. "
                       "골드 환산 값은 그만큼 오래된 환율에 근거합니다.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gold.index, y=gold.to_numpy(), mode="lines+markers",
                                 line=dict(width=3, color="#f1c40f"),
                                 hovertemplate="%{x|%m/%d} - 비율: %{y}<extra></extra>"))
        add_events(fig, events, gold.index.min(), gold.index.max())
        vals, text = wednesday_ticks(gold.index.min(), gold.index.max())
        fig.update_layout(
            title="최근 골드 시세 흐름", hovermode="x unified", template="plotly_white",
            xaxis=dict(title="수집 날짜", showgrid=True, gridcolor="#eee", type="date",
                       tickmode="array", tickvals=vals, ticktext=text, tickangle=0),
            yaxis=dict(title="현금 비율 (100:X)", showgrid=True, gridcolor="#eee"),
            margin=dict(l=20, r=20, t=50, b=20), height=400)
        st.plotly_chart(fig, use_container_width=True, key="gold_chart")

# ── 강화 재료 ──
with tab1:
    st.subheader("강화 재료 시세")
    out = category_view(
        "materials", "강화 재료",
        default_items=["운명의 파괴석", "운명의 파괴석 결정"],
        forecast_items={"상급 아비도스 융화 재료", "아비도스 융화 재료",
                        "운명의 파괴석", "운명의 파괴석 결정"})
    if out:
        selected, hourly, _ = out
        st.markdown("#### 교환 효율 분석")
        st.caption("손익은 시세 차이만 본 것이며 교환에 드는 비용(수수료·부가 재료)은 "
                   "반영하지 않았습니다.")
        if st.checkbox("교환비 비교 보기", value=True, key="ex_show"):
            active = [(l, h, r, v) for l, h, r, v in EXCHANGE_PAIRS
                      if l in selected and h in selected]
            if not active:
                st.caption("하위 재료와 상위 재료를 함께 선택하세요.")
            for low, high, ratio, verified in active:
                mark = "" if verified else " ⚠ 교환비 미확인"
                st.markdown(f"##### [{high}] 교환 효율 — {low} × {ratio}{mark}")
                pair = hourly[[low, high]].dropna()
                if pair.empty:
                    st.warning("두 품목이 동시에 수집된 시점이 없습니다.")
                    continue
                scaled = pair[[low, high]].copy()
                scaled[f"{low} (x{ratio})"] = scaled[low] * ratio
                draw_price_chart(scaled[[f"{low} (x{ratio})", high]],
                                 f"{low} {ratio}묶음 vs {high}", apply_gold, events)
                diff = float(pair[high].iloc[-1] - pair[low].iloc[-1] * ratio)
                unit = "원" if apply_gold else "골드"
                amount = f"{abs(diff):,.2f}" if apply_gold else f"{abs(diff):,.0f}"
                when = pair.index[-1].strftime("%m/%d %H:%M")
                if diff > 0:
                    st.success(f"{when} 기준 · {low} → {high} 교환 : 약 {amount} {unit} 이득 "
                               f"(교환 비용 미반영)")
                elif diff < 0:
                    st.error(f"{when} 기준 · {low} → {high} 교환 : 약 {amount} {unit} 손해")
                else:
                    st.info("차이가 없습니다.")

# ── 생활 재료 ──
with tab2:
    st.subheader("생활 재료 시세")
    subs = load_subcategories()
    if subs is None:
        st.info("데이터가 없습니다.")
    else:
        cat = st.selectbox("카테고리", sorted(subs.dropna().unique()), key="life_cat")
        category_view("lifeskill", f"생활 재료 ({cat})",
                      item_filter=set(subs[subs == cat].index), key=f"life_{cat}")

# ── 배틀 아이템 / 각인서 / 보석 ──
with tab3:
    st.subheader("배틀 아이템 시세")
    category_view("battleitems", "배틀 아이템")

with tab4:
    st.subheader("유물 각인서 시세")
    category_view("engravings", "유물 각인서")

with tab5:
    st.subheader("T4 보석 최저가")
    gems = load_market("gems")
    category_view("gems", "T4 보석",
                  default_items=sorted(gems.columns)[:2] if not gems.empty else None)

# ── 거래대금 ──
with tab6:
    st.subheader("거시 경제 흐름 (거래대금 순위)")
    st.caption("기간 동안의 **일별** `단가 × 거래량` 을 더한 값입니다. "
               "단가는 게임일(06시) 평균, 거래량은 API 가 주는 자정 기준 집계라 "
               "거래량 쪽 날짜에 맞춰 대응시킵니다.")

    options = {"최근 3일": 3, "최근 7일": 7, "최근 30일": 30,
               "전체 (수집 기간)": None, "사용자 지정": "custom"}
    choice = st.radio("조회 기간", list(options.keys()), horizontal=True, index=1)
    target_days = options[choice]

    custom_range = None
    if target_days == "custom":
        custom_range = st.date_input("시작일과 종료일", value=[],
                                     max_value=ta.now_kst().date())

    col_a, col_b = st.columns(2)
    for col, name, heading in ((col_a, "materials", "강화 재료"),
                               (col_b, "lifeskill", "생활 재료")):
        with col:
            st.markdown(f"#### {heading}")
            hourly = load_market(name)
            if hourly.empty or volume_all.empty:
                st.info("데이터가 없습니다.")
                continue
            if apply_gold:
                hourly = md.to_cash(hourly, gold)
            daily, _ = ta.game_daily_mean(hourly)
            res, excluded = trade_value_table(daily, volume_all, target_days, custom_range, apply_gold)
            if res.empty:
                if target_days == "custom":
                    st.info("기간(시작일과 종료일)을 모두 선택해 주세요.")
                else:
                    st.info("해당 기간에 데이터가 없습니다.")
                continue
            unit = "원" if apply_gold else "G"
            st.dataframe(res.style.format({
                f"거래량 가중 평균 단가 ({unit})": "{:,.2f}",
                "총 거래량 (개)": "{:,.0f}",
                f"총 거래대금 ({unit})": "{:,.0f}",
            }), use_container_width=True, height=500)
            if excluded is not None:
                st.caption(f"{excluded:%m/%d} 은 거래량이 집계 중이라 제외했습니다.")

# ── 패치분석실 ──
with tab_patch:
    st.subheader("패치분석실")
    st.caption(f"패치 이전 데이터로 학습해 '출시가 없었다면' 의 가격을 추정합니다. "
               f"학습에 최소 {MIN_TRAIN_DAYS}일이 필요합니다.")

    if not events:
        st.warning("data/event_log.txt 에 이벤트가 없습니다.")
    else:
        ordered = sorted(events.items(), key=lambda kv: kv[1], reverse=True)
        labels = {f"{name}  ({date:%Y-%m-%d})": name for name, date in ordered}

        c1, c2 = st.columns(2)
        with c1:
            sel_label = st.selectbox("패치 선택", list(labels.keys()), key="patch_sel")
            sel_patch = labels[sel_label]
        with c2:
            cat_key = st.selectbox("카테고리", list(md.CATEGORY_KR.keys()),
                                   format_func=lambda k: md.CATEGORY_KR[k], key="patch_cat")

        others = md.events_on(events[sel_patch], events)
        others = [n for n in others if n != sel_patch]
        if others:
            st.warning(f"같은 날 다른 이벤트가 있습니다: {', '.join(others)}. "
                       "가격 변화를 이 패치 하나에 귀속시킬 수 없습니다.")

        cat_daily, _ = daily_frame(cat_key)
        all_items = sorted(cat_daily.columns) if not cat_daily.empty else []
        defaults = [i for i in ["운명의 파괴석 결정", "상급 아비도스 융화 재료"] if i in all_items]
        sel_items = st.multiselect("분석 품목", all_items, default=defaults, key="patch_items")
        post_window = st.slider(
            "패치 이후 몇 일까지 볼 것인가", min_value=7, max_value=90,
            value=POST_WINDOW_DAYS, step=7, key="patch_window",
            help="길게 잡을수록 반사실 예측구간이 넓어져 어지간한 변화는 '구간 안'으로 "
                 "판정됩니다. 패치 직후의 영향을 보려면 짧게 두세요.")

        if st.button("분석 실행", key="patch_run"):
            if not sel_items:
                st.warning("품목을 하나 이상 선택하세요.")
            else:
                daily_all = {k: daily_frame(k)[0] for k in md.MARKET_FILES}
                with st.spinner(f"[{sel_patch}] 반사실 추정 중..."):
                    summary, results, diag = analyze_patch(
                        sel_patch, sel_items, daily_all, registry=registry,
                        post_window=post_window)

                for item, reason in diag.get("skipped", []):
                    st.caption(f"제외 — {item}: {reason}")

                if summary is None or not results:
                    st.error("분석 가능한 품목이 없습니다. 사전·사후 데이터가 부족합니다.")
                else:
                    st.plotly_chart(
                        build_plotly_chart(results, sel_patch, diag["patch_date"]),
                        use_container_width=True, key="patch_plotly")
                    st.markdown("#### 패치 임팩트 요약")
                    st.caption(
                        f"패치일부터 {diag['post_window']}일 구간입니다. "
                        f"'구간 밖 일수'는 실제 가격이 반사실 {diag['level']:.0%} "
                        "예측구간을 벗어난 날이고, 차이가 구간 안이면 패치 영향이라고 "
                        "말할 수 없습니다. 모델: "
                        + ", ".join(f"{k}={v}" for k, v in diag["models"].items()))
                    st.dataframe(summary.style.format({
                        "실제 평균": "{:,.1f}", "반사실 평균": "{:,.1f}",
                        "차이": "{:+,.1f}", "변화율(%)": "{:+.1f}",
                    }), use_container_width=True)

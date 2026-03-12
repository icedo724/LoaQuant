import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

# 기본 설정
st.set_page_config(page_title="LoaQuant", layout="wide")
st.title("LoaQuant")

# CSS 스타일링
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        background-color: #ffffff; 
        border-radius: 4px 4px 0 0; 
        gap: 1px; 
        padding-top: 10px; 
        padding-bottom: 10px; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: #ffffff; 
        border-bottom: 2px solid #ff4b4b; 
    }
    </style>
    """, unsafe_allow_html=True)

st.info("GitHub Actions를 통해 매시간 수집된 데이터를 시각화합니다.")


# ==========================================
# 공통 헬퍼 함수
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_prophet_forecast(series_data, periods=7):
    # 예측 수행
    df_p = series_data.dropna().reset_index()
    df_p.columns = ['ds', 'y']
    m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=periods)
    return m.predict(future)


def add_smart_event_logs(fig, event_logs, min_date, max_date, y_pos=1.05):
    # 이벤트 로그 병합 렌더링
    if not event_logs: return
    grouped_events = {}
    for name, date_str in event_logs.items():
        try:
            event_date = pd.to_datetime(date_str).replace(hour=0, minute=0)
            if pd.notnull(min_date) and pd.notnull(max_date) and min_date <= event_date <= max_date:
                if event_date not in grouped_events: grouped_events[event_date] = []
                grouped_events[event_date].append(name)
        except:
            continue

    for event_date, names in grouped_events.items():
        merged_text = "<br>".join(names)
        fig.add_vline(x=event_date, line_width=2, line_dash="dot", line_color="#E74C3C")
        fig.add_annotation(
            x=event_date, y=y_pos, yref="paper", text=merged_text, showarrow=False,
            font=dict(color="#E74C3C", size=11), bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#E74C3C", borderwidth=1
        )


# ==========================================
# 데이터 로드 및 전처리
# ==========================================
@st.cache_data(ttl=600)
def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data", file_name)
    if not os.path.exists(file_path): return None
    return pd.read_csv(file_path)


@st.cache_data(ttl=600)
def load_gold_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data", "gold", "daily_gold.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        return df
    return None


def load_event_logs():
    events = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data", "event_log.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if ":" in line:
                        name, date_str = line.replace('"', '').split(":")
                        events[name.strip()] = date_str.strip()
                except:
                    continue
    return events


def preprocess_for_chart(df, selected_items):
    if df is None or df.empty or not selected_items: return pd.DataFrame()
    df_filtered = df[df['item_name'].isin(selected_items)].copy()
    if 'sub_category' in df_filtered.columns: df_filtered = df_filtered.drop(columns=['sub_category'])
    df_filtered = df_filtered.set_index('item_name')
    df_transposed = df_filtered.T
    df_transposed.index = pd.to_datetime(df_transposed.index, errors='coerce')
    return df_transposed


def apply_gold_conversion(df, gold_dict, latest_gold):
    df_conv = df.copy()
    dates = df_conv.index.strftime('%Y-%m-%d')
    ratios = dates.map(lambda x: gold_dict.get(x, latest_gold))
    for col in df_conv.columns: df_conv[col] = df_conv[col] * (ratios / 100.0)
    return df_conv


def get_loa_daily_avg_df(df):
    if df.empty: return pd.DataFrame()
    df_adj = df.copy()
    df_adj.index = df_adj.index - pd.Timedelta(hours=6)
    daily_avg = df_adj.groupby(df_adj.index.date).mean()
    daily_avg.index = pd.to_datetime(daily_avg.index)
    return daily_avg


def analyze_market_status(df, column_name, is_cash=False):
    subset = df[column_name].dropna()
    if len(subset) < 24: return None

    delta = subset.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    window = 24
    ma = subset.rolling(window=window).mean().iloc[-1]
    std = subset.rolling(window=window).std().iloc[-1]
    upper = ma + (2 * std)
    lower = ma - (2 * std)

    current_price = subset.iloc[-1]
    prev_price = subset.iloc[-2]
    diff = current_price - prev_price
    diff_msg = "0" if diff == 0 else (f"{diff:+,.2f}" if is_cash else f"{diff:+,.0f}")

    signal_msg = "관망 (적정가)"
    color = "gray"
    bg_color = "#f9f9f9"

    if current_price <= lower and current_rsi <= 30:
        signal_msg = "강력 매수 (저점+과매도)"
        color = "#d9534f"
        bg_color = "#ffe6e6"
    elif current_price >= upper and current_rsi >= 70:
        signal_msg = "강력 매도 (고점+과열)"
        color = "#0275d8"
        bg_color = "#e6f2ff"
    elif current_price <= lower:
        signal_msg = "매수 기회 (밴드 하단)"
        color = "green"
        bg_color = "#eaffea"
    elif current_price >= upper:
        signal_msg = "매수 주의 (밴드 상단)"
        color = "red"
        bg_color = "#ffebe6"
    elif current_rsi >= 70:
        signal_msg = "과열 양상 (RSI 높음)"
        color = "orange"
    elif current_rsi <= 30:
        signal_msg = "침체 양상 (RSI 낮음)"
        color = "blue"

    price_str = f"{current_price:,.2f}" if is_cash else f"{current_price:,.0f}"
    unit_str = "원" if is_cash else "G"
    return {"price": price_str, "unit": unit_str, "diff": diff_msg, "rsi": f"{current_rsi:.1f}", "signal": signal_msg,
            "color": color, "bg_color": bg_color}


# ==========================================
# 차트 및 컴포넌트 렌더링 함수
# ==========================================
def draw_stock_chart(df, title_text="", is_cash=False):
    if df.empty:
        st.warning("표시할 데이터가 없습니다.")
        return

    plot_df = df.copy()

    # 1. 메인 라인 차트 컨트롤
    col1, col2 = st.columns([1, 3])
    with col1:
        show_bollinger = st.checkbox("볼린저 밴드", value=False, key=f"bollinger_{title_text}")
    with col2:
        show_events_main = st.checkbox("이벤트 로그 표시", value=True, key=f"events_main_{title_text}")

    st.markdown("##### 시장 분석 리포트")
    cols = st.columns(len(plot_df.columns))
    for idx, column in enumerate(plot_df.columns):
        analysis = analyze_market_status(plot_df, column, is_cash)
        with cols[idx]:
            if analysis is None:
                st.caption(f"**{column}**: 데이터 부족")
                continue

            rsi_val = float(analysis['rsi'])
            rsi_bar_color = "red" if rsi_val >= 70 else "blue" if rsi_val <= 30 else "gray"

            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; background-color: {analysis['bg_color']}; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                <div style="font-size: 0.9rem; color: #555; margin-bottom: 5px;">{column}</div>
                <div style="display: flex; justify-content: space-between; align-items: end;">
                    <span style="font-size: 1.4rem; font-weight: bold; color: #333;">{analysis['price']} {analysis['unit']}</span>
                    <span style="font-size: 0.9rem; font-weight: bold; color: {analysis['color']};">({analysis['diff']})</span>
                </div>
                <hr style="margin: 10px 0; border: 0; border-top: 1px solid #ddd;">
                <div style="font-size: 0.85rem; color: #666; margin-bottom: 5px;">
                    RSI 지수: <span style="font-weight:bold; color:{rsi_bar_color}">{analysis['rsi']}</span>
                </div>
                <div style="font-size: 1rem; font-weight: bold; color: {analysis['color']};">{analysis['signal']}</div>
            </div>
            """, unsafe_allow_html=True)

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    unit_label = "원" if is_cash else "골드"
    hover_fmt = '%{x|%m/%d %H:%M} - %{y:,.2f} ' + unit_label + '<extra></extra>' if is_cash else '%{x|%m/%d %H:%M} - %{y:,.0f} ' + unit_label + '<extra></extra>'

    for idx, column in enumerate(plot_df.columns):
        line_color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[column], mode='lines', name=column,
                                 line=dict(width=2, color=line_color), hovertemplate=hover_fmt))

        if show_bollinger:
            ma = plot_df[column].rolling(window=24).mean()
            std = plot_df[column].rolling(window=24).std()
            upper = ma + (std * 2)
            lower = ma - (std * 2)
            fill_color_rgba = f"rgba{tuple(list(int(line_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + [0.1])}"

            fig.add_trace(go.Scatter(x=plot_df.index, y=upper, mode='lines', line=dict(width=0), showlegend=False,
                                     hoverinfo='skip'))
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=lower, mode='lines', name=f"{column} 볼린저 영역", line=dict(width=0),
                           fill='tonexty', fillcolor=fill_color_rgba, showlegend=True, hoverinfo='skip'))
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=ma, mode='lines', line=dict(width=1, dash='dot', color=line_color),
                           hoverinfo='skip', showlegend=False))

    min_date, max_date = plot_df.index.min(), plot_df.index.max()

    if not pd.isnull(min_date) and not pd.isnull(max_date):
        current_ptr = min_date.replace(hour=0, minute=0, second=0)
        while current_ptr <= max_date:
            if current_ptr.weekday() == 2:
                patch_start = current_ptr.replace(hour=6, minute=0)
                patch_end = current_ptr.replace(hour=10, minute=0)
                if min_date <= patch_end and patch_start <= max_date:
                    fig.add_vrect(x0=patch_start, x1=patch_end, fillcolor="rgba(128, 128, 128, 0.2)", layer="below",
                                  line_width=0, annotation_text="점검", annotation_position="top left",
                                  annotation_font=dict(color="gray", size=10))
            current_ptr += timedelta(days=1)

    if show_events_main:
        add_smart_event_logs(fig, load_event_logs(), min_date, max_date)

    tick_vals = [d for d in pd.date_range(start=min_date.date(), end=max_date.date(), freq='D') if d.weekday() == 2]
    tick_text = [d.strftime('%m.%d(수)') for d in tick_vals]

    fig.update_layout(
        title=dict(text=f"{title_text} (1시간 단위 갱신)", font=dict(size=18)),
        hovermode="x unified", template="plotly_white",
        xaxis=dict(showgrid=True, gridcolor='#eee', rangeslider=dict(visible=True), type="date", tickmode='array',
                   tickvals=tick_vals, ticktext=tick_text, tickangle=0),
        yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',.2f' if is_cash else ',', title=f"가격 ({unit_label})"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20), height=500
    )
    st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{title_text}")

    # 2. 캔들스틱 차트
    st.markdown("#### 시세 캔들스틱 차트 및 예측")
    col_tf1, col_tf2, col_tf3 = st.columns([1, 1, 2])
    with col_tf1:
        timeframe = st.radio("기준 시간 단위", ["1일", "1주"], horizontal=True, key=f"tf_{title_text}")
    with col_tf2:
        show_prophet = st.checkbox("Prophet 예측 (1일 기준)", value=False, key=f"prophet_{title_text}")
    with col_tf3:
        show_events_candle = st.checkbox("이벤트 로그 표시", value=True, key=f"events_candle_{title_text}")

    for idx, column in enumerate(plot_df.columns):
        df_adj = plot_df[[column]].copy()
        df_adj.index = df_adj.index - pd.Timedelta(hours=6)

        if timeframe == "1일":
            ohlc = df_adj[column].resample('1D').agg(['first', 'max', 'min', 'last']).dropna()
            ohlc.index = pd.to_datetime(ohlc.index.date)
        else:
            ohlc = df_adj[column].resample('W-TUE').agg(['first', 'max', 'min', 'last']).dropna()
            # 주간 기준일(화요일 마감)을 수요일(시작일)로 강제 이동하여 차트와 이벤트 선 정렬
            ohlc.index = pd.to_datetime(ohlc.index.date) - pd.Timedelta(days=6)

        ohlc.columns = ['Open', 'High', 'Low', 'Close']

        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['Open'], high=ohlc['High'], low=ohlc['Low'], close=ohlc['Close'],
            name=column,
            increasing_line_color='#d9534f', decreasing_line_color='#0275d8',
            increasing_fillcolor='#d9534f', decreasing_fillcolor='#0275d8'
        ))

        if show_prophet and timeframe == "1일":
            with st.spinner(f"{column} 예측 모델 연산 중..."):
                forecast = get_prophet_forecast(ohlc['Close'])
                line_color = colors[idx % len(colors)]

                fig_candle.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'],
                    mode='lines', name=f"AI 예측",
                    line=dict(width=2, dash='dot', color=line_color)
                ))
                fill_rgba = f"rgba{tuple(list(int(line_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + [0.15])}"
                fig_candle.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
                    fill='toself', fillcolor=fill_rgba,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False, hoverinfo='skip'
                ))

        c_min_date, c_max_date = ohlc.index.min(), ohlc.index.max()

        if show_events_candle:
            add_smart_event_logs(fig_candle, load_event_logs(), c_min_date, c_max_date, y_pos=1.0)

        c_tick_vals = [d for d in pd.date_range(start=c_min_date, end=c_max_date, freq='D') if d.weekday() == 2]
        c_tick_text = [d.strftime('%m.%d(수)') for d in c_tick_vals]

        fig_candle.update_layout(
            title=f"[{column}] {timeframe} 시세 변동",
            xaxis_rangeslider_visible=False,
            hovermode="x unified", template="plotly_white",
            xaxis=dict(showgrid=True, gridcolor='#eee', type="date", tickmode='array', tickvals=c_tick_vals,
                       ticktext=c_tick_text),
            yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',.2f' if is_cash else ',',
                       title=f"가격 ({unit_label})"),
            margin=dict(l=20, r=20, t=40, b=20), height=350, showlegend=False
        )
        st.plotly_chart(fig_candle, use_container_width=True, key=f"candle_chart_{title_text}_{column}")


def draw_day_of_week_chart(df, title_text="", is_cash=False):
    if df is None or df.empty: return
    daily_df = get_loa_daily_avg_df(df)
    if daily_df.empty: return

    daily_df['weekday'] = daily_df.index.weekday
    weekday_avg = daily_df.groupby('weekday').mean().reindex(range(7))
    kor_days = ['월', '화', '수(기준)', '목', '금', '토', '일']

    fig = go.Figure()

    for idx, column in enumerate(weekday_avg.columns):
        if column == 'weekday': continue

        if 2 in weekday_avg.index and not pd.isna(weekday_avg.loc[2, column]):
            wed_val = weekday_avg.loc[2, column]
            pct_change = ((weekday_avg[column] / wed_val) - 1) * 100
        else:
            pct_change = weekday_avg[column] * 0

        marker_colors = ['#d9534f' if val > 0 else ('#0275d8' if val < 0 else '#808080') for val in pct_change]

        fig.add_trace(go.Bar(
            x=kor_days, y=pct_change, name=column, marker_color=marker_colors,
            text=[f"{val:+.2f}%" if pd.notna(val) else "" for val in pct_change],
            textposition='outside', hovertemplate='%{x}요일 - 수요일 대비 %{y:+.2f}%<extra></extra>'
        ))

    fig.update_layout(
        title="수요일 대비 요일별 시세 변동률",
        hovermode="x unified", template="plotly_white", barmode='group',
        xaxis=dict(title="요일", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#eee', tickformat='+.1f', ticksuffix='%', title="변동률 (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20), height=350
    )
    st.plotly_chart(fig, use_container_width=True, key=f"dow_chart_{title_text}")


def draw_summary_table(df, is_cash=False):
    daily_df = get_loa_daily_avg_df(df)
    if daily_df.empty: return

    daily_sorted = daily_df.sort_index(ascending=True)
    daily_diff = daily_sorted.diff()
    daily_desc = daily_sorted.sort_index(ascending=False)
    diff_desc = daily_diff.sort_index(ascending=False)
    display_df = pd.DataFrame(index=daily_desc.index, columns=daily_desc.columns)

    for col in daily_desc.columns:
        if is_cash:
            display_df[col] = [f"{price:,.2f} ({diff:+,.2f})" if not pd.isna(diff) else f"{price:,.2f} (-)" for
                               price, diff in zip(daily_desc[col], diff_desc[col])]
        else:
            display_df[col] = [f"{price:,.0f} ({diff:+,.0f})" if not pd.isna(diff) else f"{price:,.0f} (-)" for
                               price, diff in zip(daily_desc[col], diff_desc[col])]

    kor_days = ['월', '화', '수', '목', '금', '토', '일']
    display_df.index = [d.strftime(f'%Y-%m-%d ({kor_days[d.weekday()]})') for d in display_df.index]

    def style_variance(val):
        try:
            if "(-)" in val: return "color: gray;"
            start, end = val.rfind('(') + 1, val.rfind(')')
            change = float(val[start:end].replace(',', ''))
            if change > 0:
                return 'color: #d9534f; font-weight: bold;'
            elif change < 0:
                return 'color: #0275d8; font-weight: bold;'
            else:
                return 'color: gray;'
        except:
            return ""

    st.dataframe(display_df.style.map(style_variance), use_container_width=True)


def draw_volume_chart(df_vol, selected_items, title_text=""):
    if df_vol is None or df_vol.empty or not selected_items: return
    subset = df_vol[df_vol['item_name'].isin(selected_items)].copy()
    if subset.empty: return

    subset = subset.set_index('item_name')
    df_t = subset.T
    df_t.index = pd.to_datetime(df_t.index, errors='coerce')

    col1, col2 = st.columns([1, 5])
    with col1:
        show_events_vol = st.checkbox("이벤트 로그 표시", value=True, key=f"events_vol_{title_text}")

    st.markdown("#### 일일 거래량 추이")
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    for idx, column in enumerate(df_t.columns):
        fig.add_trace(go.Bar(
            x=df_t.index, y=df_t[column], name=column,
            marker_color=colors[idx % len(colors)], hovertemplate='%{x|%m/%d} - 거래량: %{y:,.0f} 개<extra></extra>'
        ))

    min_date, max_date = df_t.index.min(), df_t.index.max()

    if show_events_vol:
        add_smart_event_logs(fig, load_event_logs(), min_date, max_date)

    if pd.notnull(min_date) and pd.notnull(max_date):
        tick_vals = [d for d in pd.date_range(start=min_date, end=max_date, freq='D') if d.weekday() == 2]
        tick_text = [d.strftime('%m.%d(수)') for d in tick_vals]
    else:
        tick_vals, tick_text = [], []

    fig.update_layout(
        hovermode="x unified", template="plotly_white", barmode='group',
        xaxis=dict(showgrid=True, gridcolor='#eee', type="date", tickmode='array', tickvals=tick_vals,
                   ticktext=tick_text, tickangle=0),
        yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',', title="거래량 (개)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20), height=350
    )
    st.plotly_chart(fig, use_container_width=True, key=f"vol_chart_{title_text}")


# ==========================================
# 탭 UI 및 메인 로직
# ==========================================
df_materials = load_data("market_materials.csv")
df_lifeskill = load_data("market_lifeskill.csv")
df_battle = load_data("market_battleitems.csv")
df_engravings = load_data("market_engravings.csv")
df_gems = load_data("market_gems.csv")
df_volume = load_data("market_volume.csv")
df_gold = load_gold_data()

apply_gold = False
gold_dict = {}
latest_gold = 100

if df_gold is not None and not df_gold.empty:
    st.markdown("---")
    apply_gold = st.checkbox("골드 가치 반영하기 (시세를 현금 절대 가치로 환산)")
    gold_dict = dict(zip(df_gold['Date'], df_gold['Gold_Price']))
    latest_gold = df_gold['Gold_Price'].iloc[-1]
    st.markdown("---")


def get_chart_df(df, sel):
    c_data = preprocess_for_chart(df, sel)
    if apply_gold and not c_data.empty: c_data = apply_gold_conversion(c_data, gold_dict, latest_gold)
    return c_data


if df_materials is not None and not df_materials.empty:
    time_cols = pd.to_datetime(df_materials.columns, errors='coerce')
    time_cols = time_cols[time_cols.notnull()].sort_values()
    if not time_cols.empty:
        start_date = time_cols.min().strftime('%Y-%m-%d')
        last_update = time_cols.max().strftime('%Y-%m-%d %H:%M')
        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 5px solid #ff4b4b; margin-bottom: 25px;">
                <span style="color: #6c757d; font-size: 0.9rem;">데이터 수집 현황: </span>
                <b style="font-size: 1.1rem; color: #31333F;">{start_date}</b>
                <span style="color: #6c757d;"> 부터 </span>
                <b style="font-size: 1.1rem; color: #31333F;">{last_update}</b>
                <span style="color: #6c757d;"> 까지 수집됨 (KST)</span>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("데이터를 불러오는 중이거나 수집된 데이터가 없습니다.")

tab_gold, tab1, tab2, tab3, tab4, tab5 = st.tabs(["골드 시세", "강화 재료", "생활 재료", "배틀 아이템", "각인서", "보석"])

with tab_gold:
    st.subheader("일별 골드 시세 (100골드 당 현금 비율)")
    if df_gold is not None and not df_gold.empty:
        df_gold_dt = df_gold.copy()
        df_gold_dt['Date'] = pd.to_datetime(df_gold_dt['Date'])

        col1, col2 = st.columns([1, 5])
        with col1:
            show_events_gold = st.checkbox("이벤트 로그 표시", value=True, key="events_gold")

        fig_gold = go.Figure()
        fig_gold.add_trace(go.Scatter(
            x=df_gold_dt['Date'], y=df_gold_dt['Gold_Price'], mode='lines+markers',
            line=dict(width=3, color='#f1c40f'), hovertemplate='%{x|%m/%d} - 비율: %{y}<extra></extra>'
        ))

        min_date, max_date = df_gold_dt['Date'].min(), df_gold_dt['Date'].max()

        if show_events_gold:
            add_smart_event_logs(fig_gold, load_event_logs(), min_date, max_date)

        tick_vals = [d for d in pd.date_range(start=min_date, end=max_date, freq='D') if d.weekday() == 2]
        tick_text = [d.strftime('%m.%d(수)') for d in tick_vals]

        fig_gold.update_layout(
            title="최근 골드 시세 흐름", hovermode="x unified", template="plotly_white",
            xaxis=dict(title="수집 날짜", showgrid=True, gridcolor='#eee', type="date", tickmode='array',
                       tickvals=tick_vals, ticktext=tick_text, tickangle=0),
            yaxis=dict(title="현금 비율 (100:X)", showgrid=True, gridcolor='#eee'),
            margin=dict(l=20, r=20, t=50, b=20), height=400
        )
        st.plotly_chart(fig_gold, use_container_width=True, key="gold_chart")
    else:
        st.warning("골드 시세 데이터(daily_gold.csv)를 찾을 수 없습니다.")

with tab1:
    st.subheader("강화 재료 시세")
    if df_materials is not None:
        all_items = sorted(df_materials['item_name'].unique())
        default_items = ["운명의 파괴석", "운명의 파괴석 결정"]
        valid_defaults = [i for i in default_items if i in all_items]
        selected = st.multiselect("확인할 재료를 선택하세요", all_items, default=valid_defaults, key="mat_select")
        chart_data = get_chart_df(df_materials, selected)

        if not chart_data.empty:
            draw_stock_chart(chart_data, "강화 재료", apply_gold)

            draw_day_of_week_chart(chart_data, "강화 재료", apply_gold)
            with st.expander("데이터 요약 표 (일일 기준)"):
                draw_summary_table(chart_data, apply_gold)

            st.divider()
            draw_volume_chart(df_volume, selected, "강화 재료")
            st.divider()

            st.markdown("#### 교환 효율 분석")
            exchange_pairs = [("찬란한 명예의 돌파석", "운명의 돌파석"), ("운명의 돌파석", "위대한 운명의 돌파석"), ("정제된 파괴강석", "운명의 파괴석"),
                              ("운명의 파괴석", "운명의 파괴석 결정"), ("정제된 수호강석", "운명의 수호석"), ("운명의 수호석", "운명의 수호석 결정"),
                              ("최상급 오레하 융화 재료", "아비도스 융화 재료"), ("아비도스 융화 재료", "상급 아비도스 융화 재료")]
            if st.checkbox("교환비 비교 보기", value=True):
                active = [(l, h) for l, h in exchange_pairs if l in selected and h in selected]
                if not active: st.caption("하위/상위 재료를 함께 선택하세요.")
                for low, high in active:
                    st.markdown(f"##### [{high}] 교환 효율")
                    df_pair = chart_data[[low, high]].copy()
                    df_pair[f"{low} (x5)"] = df_pair[low] * 5
                    draw_stock_chart(df_pair[[f"{low} (x5)", high]], f"{low} 5묶음 vs {high}", apply_gold)
                    diff = df_pair[high].iloc[-1] - (df_pair[low].iloc[-1] * 5)
                    unit_str = "원" if apply_gold else "골드"
                    diff_val_str = f"{abs(diff):,.2f}" if apply_gold else f"{abs(diff):,.0f}"
                    if diff > 0:
                        st.success(f"{low} → {high} 교환 : 약 {diff_val_str} {unit_str} 이득")
                    elif diff < 0:
                        st.error(f"{low} → {high} 교환 : 약 {diff_val_str} {unit_str} 손해")
    else:
        st.warning("데이터 수집 중입니다.")

with tab2:
    st.subheader("생활 재료 시세")
    if df_lifeskill is not None:
        cat = st.selectbox("카테고리", df_lifeskill['sub_category'].unique(), key="life_cat")
        items = sorted(df_lifeskill[df_lifeskill['sub_category'] == cat]['item_name'].unique())
        sel_life = st.multiselect("재료 선택", items, default=items[:1], key="life_sel")
        c_data = get_chart_df(df_lifeskill, sel_life)
        if not c_data.empty:
            draw_stock_chart(c_data, f"생활 재료 ({cat})", apply_gold)
            draw_day_of_week_chart(c_data, f"생활 재료 ({cat})", apply_gold)
            with st.expander("데이터 요약 표 (일일 기준)"):
                draw_summary_table(c_data, apply_gold)
            st.divider()
            draw_volume_chart(df_volume, sel_life, f"생활 재료 ({cat})")

with tab3:
    st.subheader("배틀 아이템 시세")
    if df_battle is not None:
        items = sorted(df_battle['item_name'].unique())
        sel_battle = st.multiselect("아이템 선택", items, default=items[:1], key="battle_sel")
        c_data = get_chart_df(df_battle, sel_battle)
        if not c_data.empty:
            draw_stock_chart(c_data, "배틀 아이템", apply_gold)
            draw_day_of_week_chart(c_data, "배틀 아이템", apply_gold)
            with st.expander("데이터 요약 표 (일일 기준)"):
                draw_summary_table(c_data, apply_gold)
            st.divider()
            draw_volume_chart(df_volume, sel_battle, "배틀 아이템")

with tab4:
    st.subheader("유물 각인서 시세")
    if df_engravings is not None:
        items = sorted(df_engravings['item_name'].unique())
        sel_eng = st.multiselect("각인서 선택", items, default=items[:1], key="eng_sel")
        c_data = get_chart_df(df_engravings, sel_eng)
        if not c_data.empty:
            draw_stock_chart(c_data, "유물 각인서", apply_gold)
            draw_day_of_week_chart(c_data, "유물 각인서", apply_gold)
            with st.expander("데이터 요약 표 (일일 기준)"):
                draw_summary_table(c_data, apply_gold)

with tab5:
    st.subheader("T4 보석 최저가")
    if df_gems is not None:
        items = sorted(df_gems['item_name'].unique())
        sel_gems = st.multiselect("보석 선택", items, default=items[:2], key="gem_sel")
        c_data = get_chart_df(df_gems, sel_gems)
        if not c_data.empty:
            draw_stock_chart(c_data, "T4 보석", apply_gold)
            draw_day_of_week_chart(c_data, "T4 보석", apply_gold)
            with st.expander("데이터 요약 표 (일일 기준)"):
                draw_summary_table(c_data, apply_gold)
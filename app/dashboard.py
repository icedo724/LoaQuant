import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="LoaQuant",
    layout="wide"
)

st.title("LoaQuant")

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


@st.cache_data(ttl=600)
def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data", file_name)

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    return df


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
    if df is None or df.empty or not selected_items:
        return pd.DataFrame()

    df_filtered = df[df['item_name'].isin(selected_items)].copy()
    if 'sub_category' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['sub_category'])

    df_filtered = df_filtered.set_index('item_name')
    df_transposed = df_filtered.T
    df_transposed.index = pd.to_datetime(df_transposed.index, errors='coerce')

    return df_transposed


def apply_gold_conversion(df, gold_dict, latest_gold):
    df_conv = df.copy()
    dates = df_conv.index.strftime('%Y-%m-%d')
    ratios = dates.map(lambda x: gold_dict.get(x, latest_gold))
    for col in df_conv.columns:
        df_conv[col] = df_conv[col] * (ratios / 100.0)
    return df_conv


def get_loa_daily_avg_df(df):
    if df.empty:
        return pd.DataFrame()

    df_adj = df.copy()
    # 로스트아크 기준일(06시) 반영
    df_adj.index = df_adj.index - pd.Timedelta(hours=6)

    daily_avg = df_adj.groupby(df_adj.index.date).mean()
    daily_avg.index = pd.to_datetime(daily_avg.index)
    return daily_avg


def analyze_market_status(df, column_name, is_cash=False):
    subset = df[column_name].dropna()
    if len(subset) < 24:
        return None

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

    if diff == 0:
        diff_msg = "0"
    else:
        diff_msg = f"{diff:+,.2f}" if is_cash else f"{diff:+,.0f}"

    signal_msg = "관망 (적정가)"
    color = "gray"
    bg_color = "#f9f9f9"

    if current_price <= lower and current_rsi <= 30:
        signal_msg = "🔥 강력 매수 (저점+과매도)"
        color = "#d9534f"
        bg_color = "#ffe6e6"
    elif current_price >= upper and current_rsi >= 70:
        signal_msg = "🚨 강력 매도 (고점+과열)"
        color = "#0275d8"
        bg_color = "#e6f2ff"
    elif current_price <= lower:
        signal_msg = "🟢 매수 기회 (밴드 하단)"
        color = "green"
        bg_color = "#eaffea"
    elif current_price >= upper:
        signal_msg = "🔴 매수 주의 (밴드 상단)"
        color = "red"
        bg_color = "#ffebe6"
    elif current_rsi >= 70:
        signal_msg = "📈 과열 양상 (RSI 높음)"
        color = "orange"
    elif current_rsi <= 30:
        signal_msg = "📉 침체 양상 (RSI 낮음)"
        color = "blue"

    price_str = f"{current_price:,.2f}" if is_cash else f"{current_price:,.0f}"
    unit_str = "원" if is_cash else "G"

    return {
        "price": price_str,
        "unit": unit_str,
        "diff": diff_msg,
        "rsi": f"{current_rsi:.1f}",
        "signal": signal_msg,
        "color": color,
        "bg_color": bg_color
    }


def draw_stock_chart(df, title_text="", is_cash=False):
    if df.empty:
        st.warning("표시할 데이터가 없습니다.")
        return

    plot_df = df.copy()

    col1, col2 = st.columns([1, 3])
    with col1:
        show_bollinger = st.checkbox("볼린저 밴드", value=False, key=f"bollinger_{title_text}")

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
            <div style="
                border: 1px solid #ddd; 
                border-radius: 10px; 
                padding: 15px; 
                background-color: {analysis['bg_color']};
                box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                <div style="font-size: 0.9rem; color: #555; margin-bottom: 5px;">{column}</div>
                <div style="display: flex; justify-content: space-between; align-items: end;">
                    <span style="font-size: 1.4rem; font-weight: bold; color: #333;">{analysis['price']} {analysis['unit']}</span>
                    <span style="font-size: 0.9rem; font-weight: bold; color: {analysis['color']};">
                        ({analysis['diff']})
                    </span>
                </div>
                <hr style="margin: 10px 0; border: 0; border-top: 1px solid #ddd;">
                <div style="font-size: 0.85rem; color: #666; margin-bottom: 5px;">
                    RSI 지수: <span style="font-weight:bold; color:{rsi_bar_color}">{analysis['rsi']}</span>
                </div>
                <div style="font-size: 1rem; font-weight: bold; color: {analysis['color']};">
                    {analysis['signal']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    unit_label = "원" if is_cash else "골드"
    hover_fmt = '%{x|%m/%d %H:%M} - %{y:,.2f} ' + unit_label + '<extra></extra>' if is_cash else '%{x|%m/%d %H:%M} - %{y:,.0f} ' + unit_label + '<extra></extra>'

    for idx, column in enumerate(plot_df.columns):
        line_color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[column],
            mode='lines', name=column,
            line=dict(width=2, color=line_color),
            hovertemplate=hover_fmt
        ))

        if show_bollinger:
            ma = plot_df[column].rolling(window=24).mean()
            std = plot_df[column].rolling(window=24).std()
            upper = ma + (std * 2)
            lower = ma - (std * 2)

            fill_color_rgba = f"rgba{tuple(list(int(line_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + [0.1])}"

            fig.add_trace(go.Scatter(
                x=plot_df.index, y=upper, mode='lines',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=plot_df.index, y=lower,
                mode='lines',
                name=f"{column} 볼린저 영역",
                line=dict(width=0),
                fill='tonexty',
                fillcolor=fill_color_rgba,
                showlegend=True,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=plot_df.index, y=ma, mode='lines',
                line=dict(width=1, dash='dot', color=line_color),
                hoverinfo='skip', showlegend=False
            ))

    min_date = df.index.min()
    max_date = df.index.max()

    if not pd.isnull(min_date) and not pd.isnull(max_date):
        current_ptr = min_date.replace(hour=0, minute=0, second=0)
        while current_ptr <= max_date:
            if current_ptr.weekday() == 2:
                patch_start = current_ptr.replace(hour=6, minute=0)
                patch_end = current_ptr.replace(hour=10, minute=0)
                if min_date <= patch_end and patch_start <= max_date:
                    fig.add_vrect(
                        x0=patch_start, x1=patch_end,
                        fillcolor="rgba(128, 128, 128, 0.2)",
                        layer="below", line_width=0,
                        annotation_text="점검", annotation_position="top left",
                        annotation_font=dict(color="gray", size=10)
                    )
            current_ptr += timedelta(days=1)

        event_logs = load_event_logs()
        for name, date_str in event_logs.items():
            try:
                event_date = pd.to_datetime(date_str).replace(hour=0, minute=0)
                if min_date <= event_date <= max_date:
                    fig.add_vline(x=event_date, line_width=2, line_dash="dot", line_color="#E74C3C")
                    fig.add_annotation(
                        x=event_date, y=1.05, yref="paper",
                        text=name, showarrow=False,
                        font=dict(color="#E74C3C", size=11),
                        bgcolor="rgba(255, 255, 255, 0.9)"
                    )
            except:
                continue

    kor_days = ['월', '화', '수', '목', '금', '토', '일']
    tick_vals = pd.date_range(start=min_date.date(), end=max_date.date(), freq='D')
    tick_text = [d.strftime(f'%m/%d ({kor_days[d.weekday()]})') for d in tick_vals]

    fig.update_layout(
        title=dict(text=f"{title_text} (1시간 단위 갱신)", font=dict(size=18)),
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(
            showgrid=True, gridcolor='#eee',
            rangeslider=dict(visible=True),
            type="date",
            tickmode='array', tickvals=tick_vals, ticktext=tick_text,
            tickangle=0,
        ),
        yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',.2f' if is_cash else ',', title=f"가격 ({unit_label})"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"#### 일평균 가격 (06시 기준)")
    daily_df = get_loa_daily_avg_df(df)

    if not daily_df.empty:
        fig_daily = go.Figure()

        daily_hover_fmt = '%{x|%m/%d} 평균: %{y:,.2f} ' + unit_label + '<extra></extra>' if is_cash else '%{x|%m/%d} 평균: %{y:,.0f} ' + unit_label + '<extra></extra>'

        for column in daily_df.columns:
            fig_daily.add_trace(go.Scatter(
                x=daily_df.index, y=daily_df[column],
                mode='lines+markers', name=f"{column} (평균)",
                line=dict(width=3),
                hovertemplate=daily_hover_fmt
            ))

        event_logs = load_event_logs()
        d_min_date = daily_df.index.min()
        d_max_date = daily_df.index.max()

        for name, date_str in event_logs.items():
            try:
                event_date = pd.to_datetime(date_str).replace(hour=0, minute=0)
                if d_min_date <= event_date <= d_max_date:
                    fig_daily.add_vline(x=event_date, line_width=2, line_dash="dot", line_color="#E74C3C")
                    fig_daily.add_annotation(
                        x=event_date, y=1.05, yref="paper",
                        text=name, showarrow=False,
                        font=dict(color="#E74C3C", size=11),
                        bgcolor="rgba(255, 255, 255, 0.9)"
                    )
            except:
                continue

        d_tick_vals = daily_df.index
        d_tick_text = [d.strftime(f'%m/%d ({kor_days[d.weekday()]})') for d in d_tick_vals]

        fig_daily.update_layout(
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(
                showgrid=True, gridcolor='#eee', type="date",
                tickmode='array', tickvals=d_tick_vals, ticktext=d_tick_text,
                dtick=86400000
            ),
            yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',.2f' if is_cash else ',',
                       title=f"평균가 ({unit_label})"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("##### 데이터 요약 표")

        daily_sorted = daily_df.sort_index(ascending=True)
        daily_diff = daily_sorted.diff()

        daily_desc = daily_sorted.sort_index(ascending=False)
        diff_desc = daily_diff.sort_index(ascending=False)

        display_df = pd.DataFrame(index=daily_desc.index, columns=daily_desc.columns)

        for col in daily_desc.columns:
            if is_cash:
                display_df[col] = [
                    f"{price:,.2f} ({diff:+,.2f})" if not pd.isna(diff) else f"{price:,.2f} (-)"
                    for price, diff in zip(daily_desc[col], diff_desc[col])
                ]
            else:
                display_df[col] = [
                    f"{price:,.0f} ({diff:+,.0f})" if not pd.isna(diff) else f"{price:,.0f} (-)"
                    for price, diff in zip(daily_desc[col], diff_desc[col])
                ]

        display_df.index = [d.strftime(f'%Y-%m-%d ({kor_days[d.weekday()]})') for d in display_df.index]

        def style_variance(val):
            try:
                if "(-)" in val: return "color: gray;"
                start = val.rfind('(') + 1
                end = val.rfind(')')
                change_str = val[start:end].replace(',', '')
                change = float(change_str)

                if change > 0:
                    return 'color: #d9534f; font-weight: bold;'
                elif change < 0:
                    return 'color: #0275d8; font-weight: bold;'
                else:
                    return 'color: gray;'
            except:
                return ""

        st.dataframe(display_df.style.map(style_variance))

def draw_day_of_week_chart(df, is_cash=False):
    if df is None or df.empty:
        return

    # 일평균 데이터 베이스 사용 (intra-day 노이즈 제거)
    daily_df = get_loa_daily_avg_df(df)
    if daily_df.empty:
        return

    # 날짜 인덱스에서 요일 추출 (0=월, 6=일)
    daily_df['weekday'] = daily_df.index.weekday

    # 요일별 평균 계산
    weekday_avg = daily_df.groupby('weekday').mean()

    # 0~6 모두 표시되도록 인덱스 재정비
    weekday_avg = weekday_avg.reindex(range(7))
    kor_days = ['월', '화', '수', '목', '금', '토', '일']

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    unit_label = "원" if is_cash else "골드"
    hover_fmt = '%{x}요일 - 평균: %{y:,.2f} ' + unit_label + '<extra></extra>' if is_cash else '%{x}요일 - 평균: %{y:,.0f} ' + unit_label + '<extra></extra>'

    # 각 아이템별로 Bar 추가
    for idx, column in enumerate(weekday_avg.columns):
        if column == 'weekday': continue
        fig.add_trace(go.Bar(
            x=kor_days,
            y=weekday_avg[column],
            name=column,
            marker_color=colors[idx % len(colors)],
            hovertemplate=hover_fmt
        ))

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        barmode='group',
        xaxis=dict(title="요일", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',.2f' if is_cash else ',', title=f"평균가 ({unit_label})"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 거래량 차트 그리기 함수
# -----------------------------------------------------------------------------
def draw_volume_chart(df_vol, selected_items):
    if df_vol is None or df_vol.empty or not selected_items:
        return

    subset = df_vol[df_vol['item_name'].isin(selected_items)].copy()
    if subset.empty:
        return

    subset = subset.set_index('item_name')
    df_t = subset.T
    df_t.index = pd.to_datetime(df_t.index, errors='coerce')

    st.markdown("#### 📊 일일 거래량 추이")

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    for idx, column in enumerate(df_t.columns):
        fig.add_trace(go.Bar(
            x=df_t.index,
            y=df_t[column],
            name=column,
            marker_color=colors[idx % len(colors)],
            hovertemplate='%{x|%m/%d} - 거래량: %{y:,.0f} 개<extra></extra>'
        ))

    kor_days = ['월', '화', '수', '목', '금', '토', '일']
    min_date = df_t.index.min()
    max_date = df_t.index.max()

    event_logs = load_event_logs()
    for name, date_str in event_logs.items():
        try:
            event_date = pd.to_datetime(date_str).replace(hour=0, minute=0)
            if pd.notnull(min_date) and pd.notnull(max_date) and min_date <= event_date <= max_date:
                fig.add_vline(x=event_date, line_width=2, line_dash="dot", line_color="#E74C3C")
                fig.add_annotation(
                    x=event_date, y=1.05, yref="paper",
                    text=name, showarrow=False,
                    font=dict(color="#E74C3C", size=11),
                    bgcolor="rgba(255, 255, 255, 0.9)"
                )
        except:
            continue

    if pd.notnull(min_date) and pd.notnull(max_date):
        tick_vals = pd.date_range(start=min_date, end=max_date, freq='D')
        tick_text = [d.strftime(f'%m/%d ({kor_days[d.weekday()]})') for d in tick_vals]
    else:
        tick_vals, tick_text = [], []

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        barmode='group',
        xaxis=dict(
            showgrid=True, gridcolor='#eee',
            type="date",
            tickmode='array', tickvals=tick_vals, ticktext=tick_text,
            tickangle=0
        ),
        yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',', title="거래량 (개)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 데이터 로드
# -----------------------------------------------------------------------------
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
    apply_gold = st.checkbox("골드 가치 반영하기 (모든 아이템 시세를 현금 절대 가치로 환산)")
    gold_dict = dict(zip(df_gold['Date'], df_gold['Gold_Price']))
    latest_gold = df_gold['Gold_Price'].iloc[-1]
    st.markdown("---")


def get_chart_df(df, sel):
    c_data = preprocess_for_chart(df, sel)
    if apply_gold and not c_data.empty:
        c_data = apply_gold_conversion(c_data, gold_dict, latest_gold)
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

        fig_gold = go.Figure()
        fig_gold.add_trace(go.Scatter(
            x=df_gold_dt['Date'],
            y=df_gold_dt['Gold_Price'],
            mode='lines+markers',
            line=dict(width=3, color='#f1c40f'),
            hovertemplate='%{x|%m/%d} - 비율: %{y}<extra></extra>'
        ))

        min_date = df_gold_dt['Date'].min()
        max_date = df_gold_dt['Date'].max()
        kor_days = ['월', '화', '수', '목', '금', '토', '일']

        event_logs = load_event_logs()
        for name, date_str in event_logs.items():
            try:
                event_date = pd.to_datetime(date_str).replace(hour=0, minute=0)
                if min_date <= event_date <= max_date:
                    fig_gold.add_vline(x=event_date, line_width=2, line_dash="dot", line_color="#E74C3C")
                    fig_gold.add_annotation(
                        x=event_date, y=1.05, yref="paper",
                        text=name, showarrow=False,
                        font=dict(color="#E74C3C", size=11),
                        bgcolor="rgba(255, 255, 255, 0.9)"
                    )
            except:
                continue
        tick_vals = pd.date_range(start=min_date, end=max_date, freq='D')
        tick_text = [d.strftime(f'%m/%d ({kor_days[d.weekday()]})') for d in tick_vals]

        fig_gold.update_layout(
            title="최근 골드 시세 흐름",
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(
                title="수집 날짜",
                showgrid=True,
                gridcolor='#eee',
                type="date",
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                tickangle=0
            ),
            yaxis=dict(title="현금 비율 (100:X)", showgrid=True, gridcolor='#eee'),
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )
        st.plotly_chart(fig_gold, use_container_width=True)
    else:
        st.warning("골드 시세 데이터(daily_gold.csv)를 찾을 수 없습니다.")

with tab1:
    st.subheader("강화 재료 시세")
    if df_materials is not None:
        all_items = sorted(df_materials['item_name'].unique())
        default_items = ["운명의 파괴석", "운명의 파괴석 결정"]
        valid_defaults = [i for i in default_items if i in all_items]
        selected = st.multiselect("확인할 재료를 선택하세요", all_items, default=valid_defaults)

        chart_data = get_chart_df(df_materials, selected)

        if not chart_data.empty:
            draw_stock_chart(chart_data, "강화 재료", apply_gold)

            with st.expander("요일별 평균 가격 추세 보기"):
                draw_day_of_week_chart(chart_data, apply_gold)

            # 거래량 차트 배치
            st.divider()
            draw_volume_chart(df_volume, selected)

            st.divider()
            st.markdown("#### 교환 효율 분석")
            exchange_pairs = [
                ("찬란한 명예의 돌파석", "운명의 돌파석"),
                ("운명의 돌파석", "위대한 운명의 돌파석"),
                ("정제된 파괴강석", "운명의 파괴석"),
                ("운명의 파괴석", "운명의 파괴석 결정"),
                ("정제된 수호강석", "운명의 수호석"),
                ("운명의 수호석", "운명의 수호석 결정"),
                ("최상급 오레하 융화 재료", "아비도스 융화 재료"),
                ("아비도스 융화 재료", "상급 아비도스 융화 재료")
            ]
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
                        st.success(f"**{low}** → **{high}** 교환 : 약 **{diff_val_str} {unit_str}** 이득")
                    elif diff < 0:
                        st.error(f"**{low}** → **{high}** 교환 : 약 **{diff_val_str} {unit_str}** 손해")
    else:
        st.warning("데이터 수집 중입니다.")

with tab2:
    st.subheader("생활 재료 시세")
    if df_lifeskill is not None:
        cat = st.selectbox("카테고리", df_lifeskill['sub_category'].unique())
        items = sorted(df_lifeskill[df_lifeskill['sub_category'] == cat]['item_name'].unique())
        sel_life = st.multiselect("재료 선택", items, default=items[:1])
        c_data = get_chart_df(df_lifeskill, sel_life)

        if not c_data.empty:
            draw_stock_chart(c_data, f"생활 재료 ({cat})", apply_gold)

            with st.expander("요일별 평균 가격 추세 보기"):
                draw_day_of_week_chart(c_data, apply_gold)

            st.divider()
            draw_volume_chart(df_volume, sel_life)

with tab3:
    st.subheader("배틀 아이템 시세")
    if df_battle is not None:
        items = sorted(df_battle['item_name'].unique())
        sel_battle = st.multiselect("아이템 선택", items, default=items[:1])
        c_data = get_chart_df(df_battle, sel_battle)

        if not c_data.empty:
            draw_stock_chart(c_data, "배틀 아이템", apply_gold)

            with st.expander("요일별 평균 가격 추세 보기"):
                draw_day_of_week_chart(c_data, apply_gold)

            st.divider()
            draw_volume_chart(df_volume, sel_battle)

with tab4:
    st.subheader("유물 각인서 시세")
    if df_engravings is not None:
        items = sorted(df_engravings['item_name'].unique())
        sel_eng = st.multiselect("각인서 선택", items, default=items[:1])
        c_data = get_chart_df(df_engravings, sel_eng)
        if not c_data.empty:
            draw_stock_chart(c_data, "유물 각인서", apply_gold)

            with st.expander("요일별 평균 가격 추세 보기"):
                draw_day_of_week_chart(c_data, apply_gold)

with tab5:
    st.subheader("T4 보석 최저가")
    if df_gems is not None:
        items = sorted(df_gems['item_name'].unique())
        sel_gems = st.multiselect("보석 선택", items, default=items[:2])
        c_data = get_chart_df(df_gems, sel_gems)
        if not c_data.empty:
            draw_stock_chart(c_data, "T4 보석", apply_gold)

            with st.expander("요일별 평균 가격 추세 보기"):
                draw_day_of_week_chart(c_data, apply_gold)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loconomy",
    layout="wide"
)

st.title("Lost Ark Market Trends")

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


# -----------------------------------------------------------------------------
# 2. 데이터 및 이벤트 로드 함수
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data", file_name)

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    return df


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


# -----------------------------------------------------------------------------
# 3. 데이터 가공 함수
# -----------------------------------------------------------------------------
def get_loa_daily_avg_df(df):
    if df.empty:
        return pd.DataFrame()

    df_adj = df.copy()
    df_adj.index = df_adj.index - pd.Timedelta(hours=6)

    daily_avg = df_adj.groupby(df_adj.index.date).mean()
    daily_avg.index = pd.to_datetime(daily_avg.index)
    return daily_avg


# -----------------------------------------------------------------------------
# 4. 차트 그리기
# -----------------------------------------------------------------------------
def draw_stock_chart(df, title_text=""):
    if df.empty:
        st.warning("표시할 데이터가 없습니다.")
        return

    # (A) 실시간 시세 차트 구성
    fig = go.Figure()

    for column in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[column],
            mode='lines', name=column,
            line=dict(width=2),
            hovertemplate='%{x|%m/%d %H:%M} - %{y:,.0f} 골드<extra></extra>'
        ))

    min_date = df.index.min()
    max_date = df.index.max()

    if not pd.isnull(min_date) and not pd.isnull(max_date):
        # 점검 시간 영역 표시 (수요일 06:00 ~ 10:00)
        current_ptr = min_date.replace(hour=0, minute=0, second=0)
        while current_ptr <= max_date:
            if current_ptr.weekday() == 2:  # 수요일
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

        # 이벤트 종료일 표시
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
        yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',', title="가격 (골드)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # (B) 로아 기준 일평균 시계열 차트
    st.markdown(f"#### 일평균 가격")
    daily_df = get_loa_daily_avg_df(df)

    if not daily_df.empty:
        fig_daily = go.Figure()
        for column in daily_df.columns:
            fig_daily.add_trace(go.Scatter(
                x=daily_df.index, y=daily_df[column],
                mode='lines+markers', name=f"{column} (평균)",
                line=dict(width=3),
                hovertemplate='%{x|%m/%d} 평균: %{y:,.0f} 골드<extra></extra>'
            ))

        kor_days = ['월', '화', '수', '목', '금', '토', '일']
        d_tick_vals = daily_df.index
        d_tick_text = [d.strftime(f'%m/%d ({kor_days[d.weekday()]})') for d in d_tick_vals]

        fig_daily.update_layout(
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(
                showgrid=True,
                gridcolor='#eee',
                type="date",
                tickmode='array',
                tickvals=d_tick_vals,
                ticktext=d_tick_text,
                dtick=86400000
            ),
            yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=',', title="평균가 (골드)"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        with st.expander("데이터 요약 표 보기"):
            display_df = daily_df.copy()
            display_df.index = [d.strftime(f'%Y-%m-%d ({kor_days[d.weekday()]})') for d in display_df.index]

            st.dataframe(display_df.sort_index(ascending=False).style.format("{:,.1f}"))


# -----------------------------------------------------------------------------
# 5. 데이터 로드 및 탭 구성
# -----------------------------------------------------------------------------
df_materials = load_data("market_materials.csv")
df_lifeskill = load_data("market_lifeskill.csv")
df_battle = load_data("market_battleitems.csv")
df_engravings = load_data("market_engravings.csv")
df_gems = load_data("market_gems.csv")

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["강화 재료", "생활 재료", "배틀 아이템", "각인서", "보석"])

with tab1:
    st.subheader("강화 재료 시세")
    if df_materials is not None:
        all_items = df_materials['item_name'].unique()
        default_items = ["운명의 파괴석", "운명의 파괴석 결정"]
        valid_defaults = [i for i in default_items if i in all_items]
        selected = st.multiselect("확인할 재료를 선택하세요", all_items, default=valid_defaults)
        chart_data = preprocess_for_chart(df_materials, selected)
        if not chart_data.empty:
            draw_stock_chart(chart_data, "강화 재료")

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
                    draw_stock_chart(df_pair[[f"{low} (x5)", high]], f"{low} 5묶음 vs {high}")

                    diff = df_pair[high].iloc[-1] - (df_pair[low].iloc[-1] * 5)
                    if diff > 0:
                        st.success(f"**{low}** → **{high}** 교환 : 약 **{diff:,.0f} 골드** 이득")
                    elif diff < 0:
                        st.error(f"**{low}** → **{high}** 교환 : 약 **{abs(diff):,.0f} 골드** 손해")
    else:
        st.warning("데이터 수집 중입니다.")

with tab2:
    st.subheader("생활 재료 시세")
    if df_lifeskill is not None:
        cat = st.selectbox("카테고리", df_lifeskill['sub_category'].unique())
        items = df_lifeskill[df_lifeskill['sub_category'] == cat]['item_name'].unique()
        sel_life = st.multiselect("재료 선택", items, default=items[:5])
        c_data = preprocess_for_chart(df_lifeskill, sel_life)
        if not c_data.empty: draw_stock_chart(c_data, f"생활 재료 ({cat})")

with tab3:
    st.subheader("배틀 아이템 시세")
    if df_battle is not None:
        items = df_battle['item_name'].unique()
        sel_battle = st.multiselect("아이템 선택", items, default=items[:5])
        c_data = preprocess_for_chart(df_battle, sel_battle)
        if not c_data.empty: draw_stock_chart(c_data, "배틀 아이템")

with tab4:
    st.subheader("유물 각인서 시세")
    if df_engravings is not None:
        items = df_engravings['item_name'].unique()
        sel_eng = st.multiselect("각인서 선택", items, default=items[:1])
        c_data = preprocess_for_chart(df_engravings, sel_eng)
        if not c_data.empty: draw_stock_chart(c_data, "유물 각인서")

with tab5:
    st.subheader("T4 보석 최저가")
    if df_gems is not None:
        items = df_gems['item_name'].unique()
        sel_gems = st.multiselect("보석 선택", items, default=items)
        c_data = preprocess_for_chart(df_gems, sel_gems)
        if not c_data.empty: draw_stock_chart(c_data, "T4 보석")
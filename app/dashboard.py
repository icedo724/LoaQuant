import streamlit as st
import pandas as pd
import plotly.express as px
import os

# -----------------------------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="로스트아크 시세 대시보드",
    layout="wide"
)

st.title("Lost Ark Market Trends")
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)
st.info("GitHub Actions를 통해 매시간 수집된 데이터를 시각화합니다.")


# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 함수
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
# 3. 데이터 로드
# -----------------------------------------------------------------------------
df_materials = load_data("market_materials.csv")
df_lifeskill = load_data("market_lifeskill.csv")
df_battle = load_data("market_battleitems.csv")
df_engravings = load_data("market_engravings.csv")
df_gems = load_data("market_gems.csv")

# -----------------------------------------------------------------------------
# 4. 탭 구성
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "강화 재료", "생활 재료", "배틀 아이템", "각인서", "보석"
])

# --- Tab 1: 강화 재료 ---
with tab1:
    st.subheader("강화 재료 시세")
    if df_materials is not None:
        all_items = df_materials['item_name'].unique()

        default_items = ["운명의 파괴석", "운명의 파괴석 결정"]
        valid_defaults = [item for item in default_items if item in all_items]
        if not valid_defaults:
            valid_defaults = all_items[:3] if len(all_items) > 0 else []

        selected = st.multiselect("확인할 재료를 선택하세요", all_items, default=valid_defaults, key="mat_select")

        chart_data = preprocess_for_chart(df_materials, selected)
        if not chart_data.empty:
            st.line_chart(chart_data)
        st.divider()
        st.markdown("#### 교환 효율 분석 (5:1 비율)")

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

        show_exchange = st.checkbox("선택한 재료의 교환비 비교 보기", value=True)

        if show_exchange:
            active_pairs = []
            for low, high in exchange_pairs:
                if low in selected and high in selected:
                    active_pairs.append((low, high))

            if not active_pairs:
                st.caption("비교하고 싶은 **하위 재료**와 **상위 재료**를 위 목록에서 **함께 선택**해주세요.")
            else:
                for low, high in active_pairs:
                    st.markdown(f"##### [{high}] 제작 효율 분석")

                    df_pair = chart_data[[low, high]].copy()
                    df_pair[f"{low} (x5)"] = df_pair[low] * 5

                    plot_cols = [f"{low} (x5)", high]
                    st.line_chart(df_pair[plot_cols])

                    last_low_unit_cost = df_pair[low].iloc[-1] * 5
                    last_high_unit_cost = df_pair[high].iloc[-1]

                    diff_unit = last_high_unit_cost - last_low_unit_cost
                    diff_100 = diff_unit * 100

                    if diff_unit > 0:
                        st.success(f"결론: **{high} 100개(묶음)**를 교환으로 얻으면, 그냥 사는 것보다 **{diff_100:,.0f}골드** 이득입니다!")
                    elif diff_unit < 0:
                        st.error(f"결론: **{high} 100개(묶음)**를 그냥 경매장에서 사는 게 교환보다 **{abs(diff_100):,.0f}골드** 이득입니다!")
                    else:
                        st.info("가격이 동일합니다.")
    else:
        st.warning("아직 데이터가 수집되지 않았습니다.")

# --- Tab 2: 생활 재료 ---
with tab2:
    st.subheader("생활 재료 시세")
    if df_lifeskill is not None:
        if 'sub_category' in df_lifeskill.columns:
            categories = df_lifeskill['sub_category'].unique()
            selected_cat = st.selectbox("생활 카테고리 선택", categories, index=0)
            filtered_items = df_lifeskill[df_lifeskill['sub_category'] == selected_cat]['item_name'].unique()
        else:
            filtered_items = df_lifeskill['item_name'].unique()

        selected_life = st.multiselect("재료 선택", filtered_items, default=filtered_items[:5], key="life_select")
        chart_data = preprocess_for_chart(df_lifeskill, selected_life)
        if not chart_data.empty: st.line_chart(chart_data)
    else:
        st.warning("데이터 준비 중입니다.")

# --- Tab 3: 배틀 아이템 ---
with tab3:
    st.subheader("배틀 아이템 시세")
    if df_battle is not None:
        all_battle = df_battle['item_name'].unique()
        defaults = [i for i in ["정령의 회복약", "암흑 수류탄", "파괴 폭탄"] if i in all_battle]
        selected_battle = st.multiselect("아이템 선택", all_battle, default=defaults, key="battle_select")
        chart_data = preprocess_for_chart(df_battle, selected_battle)
        if not chart_data.empty: st.line_chart(chart_data)
    else:
        st.warning("데이터 준비 중입니다.")

# --- Tab 4: 각인서 ---
with tab4:
    st.subheader("유물 각인서 시세")
    if df_engravings is not None:
        all_eng = df_engravings['item_name'].unique()
        selected_eng = st.multiselect("각인서 선택", all_eng, default=all_eng[:1], key="eng_select")
        chart_data = preprocess_for_chart(df_engravings, selected_eng)
        if not chart_data.empty: st.line_chart(chart_data)
    else:
        st.warning("데이터 준비 중입니다.")

# --- Tab 5: 보석 ---
with tab5:
    st.subheader("T4 보석 (8~10레벨 겁화/작열) 최저가")
    if df_gems is not None:
        all_gems = df_gems['item_name'].unique()
        selected_gems = st.multiselect("보석 선택", all_gems, default=all_gems, key="gem_select")
        chart_data = preprocess_for_chart(df_gems, selected_gems)
        if not chart_data.empty:
            st.line_chart(chart_data)
            st.caption("※ 경매장 즉시 구매가 최저가 기준입니다.")
    else:
        st.warning("데이터 준비 중입니다.")
import re
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

MIN_TRAIN_DAYS = 7
MIN_POST_DAYS  = 3

MARKET_FILES = {
    'materials':   'data/market_materials.csv',
    'engravings':  'data/market_engravings.csv',
    'gems':        'data/market_gems.csv',
    'lifeskill':   'data/market_lifeskill.csv',
    'battleitems': 'data/market_battleitems.csv',
}

CATEGORY_KR = {
    'materials':   '강화 재료',
    'engravings':  '각인서',
    'gems':        '보석',
    'lifeskill':   '생활 재료',
    'battleitems': '배틀 아이템',
}


def load_event_log(path='data/event_log.txt') -> dict[str, pd.Timestamp]:
    events = {}
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'"(.+?)":\s*(\d{4}-\d{2}-\d{2})', line)
            if m:
                events[m.group(1)] = pd.Timestamp(m.group(2))
    return events


def load_all_markets(files: dict) -> dict[str, pd.DataFrame]:
    # wide → 일별 평균 DataFrame, 카테고리별 반환
    daily = {}
    for name, path in files.items():
        try:
            df = pd.read_csv(path, index_col=0, encoding='utf-8-sig')
            df = df.drop(columns=[c for c in df.columns if not c[:4].isdigit()], errors='ignore')
            df_T = df.T.copy()
            df_T.index = pd.to_datetime(df_T.index)
            daily[name] = df_T.resample('D').mean()
        except FileNotFoundError:
            pass
    return daily


def load_gold(path='data/gold/daily_gold.csv') -> pd.Series:
    df = pd.read_csv(path, encoding='utf-8-sig', parse_dates=['Date'])
    return df.set_index('Date')['Gold_Price']


def find_item(name: str, daily: dict) -> pd.Series | None:
    for df in daily.values():
        if name in df.columns:
            return df[name].dropna()
    return None


def get_items_by_category(category: str, daily: dict) -> list[str]:
    # 카테고리에 속한 전체 아이템 이름 반환
    df = daily.get(category)
    if df is None:
        return []
    return sorted(df.columns.tolist())


def check_feasibility(series: pd.Series, patch_date: pd.Timestamp) -> tuple[bool, str]:
    pre  = series[series.index < patch_date]
    post = series[series.index >= patch_date]
    if len(pre) < MIN_TRAIN_DAYS:
        return False, f"사전 데이터 부족 ({len(pre)}일 < 최소 {MIN_TRAIN_DAYS}일)"
    if len(post) < MIN_POST_DAYS:
        return False, f"사후 데이터 부족 ({len(post)}일 < 최소 {MIN_POST_DAYS}일)"
    return True, "OK"


def run_counterfactual(series: pd.Series, patch_date: pd.Timestamp) -> dict:
    # 패치 이전 데이터로 Prophet 학습 → 이후 반사실 예측
    train = series[series.index < patch_date]
    post  = series[series.index >= patch_date]

    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.90,
    )
    m.fit(pd.DataFrame({'ds': train.index, 'y': train.values}))
    fc = m.predict(pd.DataFrame({'ds': post.index})).set_index('ds')

    return {
        'actual_pre':  train,
        'actual_post': post,
        'cf_mean':     fc['yhat'],
        'cf_lower':    fc['yhat_lower'],
        'cf_upper':    fc['yhat_upper'],
    }


def calc_impact(res: dict, label: str, patch_date: pd.Timestamp) -> dict:
    actual = res['actual_post']
    cf     = res['cf_mean'].reindex(actual.index)
    diff   = actual - cf
    pct    = diff / cf * 100
    return {
        '아이템':    label,
        '패치일':    patch_date.date(),
        '실제 평균': round(actual.mean(), 1),
        '반사실 평균': round(cf.mean(), 1),
        '차이':      round(diff.mean(), 1),
        '변화율(%)': round(pct.mean(), 1),
    }


def build_plotly_chart(results: dict, patch_name: str, patch_date: pd.Timestamp) -> go.Figure:
    # 실제 vs 반사실 인터랙티브 시계열 차트 반환 (저장 없음)
    n_items = len(results)
    n_cols  = min(2, n_items)
    n_rows  = max(1, (n_items + n_cols - 1) // n_cols)
    v_sp    = min(0.08, 0.6 / n_rows)   if n_rows > 1 else 0.0
    h_sp    = min(0.06, 0.6 / n_cols)   if n_cols > 1 else 0.0
    titles  = list(results.keys()) + [''] * (n_rows * n_cols - n_items)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        vertical_spacing=v_sp,
        horizontal_spacing=h_sp,
    )

    C = dict(actual='#4C9EEB', cf='#FF6B6B', band='rgba(255,107,107,0.15)')

    for idx, (label, res) in enumerate(results.items()):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        first    = idx == 0
        actual_all = pd.concat([res['actual_pre'], res['actual_post']])

        fig.add_trace(go.Scatter(
            x=actual_all.index, y=actual_all.values,
            mode='lines', name='실제 가격',
            line=dict(color=C['actual'], width=2),
            legendgroup='actual', showlegend=first,
        ), row=row, col=col)

        cf_x = res['cf_mean'].index
        for y, fill, show in [
            (res['cf_upper'].values, None,      False),
            (res['cf_lower'].values, 'tonexty', False),
        ]:
            fig.add_trace(go.Scatter(
                x=cf_x, y=y, mode='lines',
                line=dict(width=0), fill=fill,
                fillcolor=C['band'], showlegend=show,
            ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=cf_x, y=res['cf_mean'].values,
            mode='lines', name='반사실 예측',
            line=dict(color=C['cf'], width=2, dash='dot'),
            legendgroup='cf', showlegend=first,
        ), row=row, col=col)

        yvals = actual_all.values
        ymin, ymax = float(np.nanmin(yvals)), float(np.nanmax(yvals))
        fig.add_trace(go.Scatter(
            x=[patch_date, patch_date], y=[ymin, ymax],
            mode='lines', name=f'{patch_name} 출시',
            line=dict(color='gold', width=1.5, dash='dash'),
            legendgroup='patch', showlegend=first,
        ), row=row, col=col)

    fig.update_layout(
        title=dict(
            text=f'{patch_name} 임팩트 분석 (출시일: {patch_date.date()}) — 실제 vs 반사실',
            font=dict(size=15, family='Malgun Gothic'), x=0.5,
        ),
        height=350 * n_rows,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        font=dict(family='Malgun Gothic'),
        template='plotly_white',
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def analyze_patch(patch_name: str, items: list[str],
                  daily: dict, patch_date: pd.Timestamp | None = None,
                  out_dir: str = 'analysis/output') -> tuple[pd.DataFrame | None, dict]:
    # event_log에서 날짜 조회 후 아이템별 반사실 분석 실행, (summary_df, results) 반환
    if patch_date is None:
        events = load_event_log()
        if patch_name not in events:
            print(f'[오류] event_log에 "{patch_name}" 항목 없음')
            return None, {}
        patch_date = events[patch_name]

    results, impact_rows, skipped = {}, [], []

    for item in items:
        series = find_item(item, daily)
        if series is None:
            skipped.append(item)
            continue

        ok, reason = check_feasibility(series, patch_date)
        if not ok:
            print(f'  [{item}] 분석 불가능: {reason}')
            skipped.append(item)
            continue

        res = run_counterfactual(series, patch_date)
        results[item] = res
        impact_rows.append(calc_impact(res, item, patch_date))

    if not results:
        print(f'\n[{patch_name}] 분석 가능한 아이템이 없습니다.')
        return None, {}

    summary_df = pd.DataFrame(impact_rows)
    return summary_df, results


if __name__ == '__main__':
    daily = load_all_markets(MARKET_FILES)

    ITEMS = [
        '운명의 파괴석 결정',
        '운명의 파괴석',
        '상급 아비도스 융화 재료',
        '아비도스 융화 재료',
    ]

    summary, results = analyze_patch('지평의 성당', ITEMS, daily)
    if summary is not None:
        print(summary.to_string(index=False))

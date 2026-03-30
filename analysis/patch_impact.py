import ast
import re
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

MIN_TRAIN_DAYS = 7   # 학습에 필요한 최소 사전 데이터(일)
MIN_POST_DAYS  = 3   # 검정에 필요한 최소 사후 데이터(일)

MARKET_FILES = {
    'materials':  'data/market_materials.csv',
    'engravings': 'data/market_engravings.csv',
    'gems':       'data/market_gems.csv',
    'lifeskill':  'data/market_lifeskill.csv',
    'battleitems':'data/market_battleitems.csv',
}


# ── 데이터 로딩 ──────────────────────────────────────────────────────────────

def load_event_log(path='data/event_log.txt') -> dict[str, pd.Timestamp]:
    """패치 이름 → 날짜 매핑 반환"""
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
    """카테고리별 일별 평균 가격 DataFrame 반환"""
    daily = {}
    for name, path in files.items():
        try:
            df = pd.read_csv(path, index_col=0, encoding='utf-8-sig')
            # sub_category 열 제거 (lifeskill 등 일부 파일)
            df = df.drop(columns=[c for c in df.columns
                                   if not c[:4].isdigit()], errors='ignore')
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
    """모든 카테고리에서 아이템 이름으로 시리즈 검색"""
    for df in daily.values():
        if name in df.columns:
            return df[name].dropna()
    return None


# ── 검정 가능 여부 확인 ───────────────────────────────────────────────────────

def check_feasibility(series: pd.Series, patch_date: pd.Timestamp) -> tuple[bool, str]:
    """사전/사후 데이터 충분성 검사"""
    pre  = series[series.index < patch_date]
    post = series[series.index >= patch_date]
    if len(pre) < MIN_TRAIN_DAYS:
        return False, f"사전 데이터 부족 ({len(pre)}일 < 최소 {MIN_TRAIN_DAYS}일)"
    if len(post) < MIN_POST_DAYS:
        return False, f"사후 데이터 부족 ({len(post)}일 < 최소 {MIN_POST_DAYS}일)"
    return True, "OK"


# ── Prophet 반사실적 모델 ─────────────────────────────────────────────────────

def run_counterfactual(series: pd.Series, patch_date: pd.Timestamp) -> dict:
    """사전 데이터로 Prophet 학습 후 사후 구간 반사실 예측"""
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


# ── 임팩트 수치 계산 ─────────────────────────────────────────────────────────

def calc_impact(res: dict, label: str, patch_date: pd.Timestamp) -> dict:
    """실제 vs 반사실 평균 비교 통계 반환"""
    actual = res['actual_post']
    cf     = res['cf_mean'].reindex(actual.index)
    diff   = actual - cf
    pct    = (diff / cf * 100)
    return {
        '아이템':          label,
        '패치일':          patch_date.date(),
        '실제 평균':        round(actual.mean(), 1),
        '반사실 평균':      round(cf.mean(), 1),
        '차이':            round(diff.mean(), 1),
        '변화율(%)':       round(pct.mean(), 1),
    }


# ── 시각화 ───────────────────────────────────────────────────────────────────

def plot_timeseries(results: dict, patch_name: str, patch_date: pd.Timestamp,
                    out_path: str):
    """아이템별 실제 vs 반사실 시계열 차트 저장"""
    n_cols = 2
    n_rows = max(1, (len(results) + 1) // n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=list(results.keys()),
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
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
            (res['cf_upper'].values, None,       False),
            (res['cf_lower'].values, 'tonexty',  False),
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
            font=dict(size=16, family='Malgun Gothic'), x=0.5,
        ),
        height=300 * n_rows, width=1400,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        font=dict(family='Malgun Gothic'),
        template='plotly_dark',
        plot_bgcolor='#1a1a2e', paper_bgcolor='#16213e',
    )
    fig.write_html(out_path)


def plot_summary(summary_df: pd.DataFrame, patch_name: str,
                 patch_date: pd.Timestamp, out_path: str):
    """아이템별 변화율 바차트 저장"""
    colors = ['#4C9EEB' if v >= 0 else '#FF6B6B'
              for v in summary_df['변화율(%)']]

    fig = go.Figure(go.Bar(
        x=summary_df['아이템'],
        y=summary_df['변화율(%)'],
        text=summary_df['변화율(%)'].apply(lambda v: f'{v:+.1f}%'),
        textposition='outside',
        marker_color=colors,
    ))
    fig.add_hline(y=0, line_color='white', line_width=1)
    fig.update_layout(
        title=dict(
            text=f'{patch_name} 임팩트 요약 — 반사실 대비 실제 가격 변화율(%)',
            font=dict(size=15, family='Malgun Gothic'), x=0.5,
        ),
        height=500, width=1400,
        yaxis_title='변화율 (%)',
        font=dict(family='Malgun Gothic'),
        template='plotly_dark',
        plot_bgcolor='#1a1a2e', paper_bgcolor='#16213e',
    )
    fig.write_html(out_path)


# ── 메인 분석 함수 ────────────────────────────────────────────────────────────

def analyze_patch(patch_name: str, items: list[str],
                  daily: dict, patch_date: pd.Timestamp | None = None,
                  out_dir: str = 'analysis'):
    """
    단일 패치에 대해 지정 아이템의 반사실적 임팩트 분석 실행.

    patch_name : event_log 키 (patch_date 미지정 시 event_log에서 날짜 조회)
    items      : 분석할 아이템 이름 목록
    daily      : load_all_markets() 반환값
    patch_date : 직접 지정할 경우 사용 (없으면 event_log 참조)
    out_dir    : HTML 결과물 저장 디렉터리
    """
    if patch_date is None:
        events = load_event_log()
        if patch_name not in events:
            print(f'[오류] event_log에 "{patch_name}" 항목 없음')
            return None
        patch_date = events[patch_name]

    results, impact_rows, skipped = {}, [], []

    for item in items:
        series = find_item(item, daily)
        if series is None:
            print(f'  [{item}] 데이터 없음 — 건너뜀')
            skipped.append(item)
            continue

        ok, reason = check_feasibility(series, patch_date)
        if not ok:
            print(f'  [{item}] 분석 불가능: {reason}')
            skipped.append(item)
            continue

        print(f'  [{item}] 분석 중...')
        res = run_counterfactual(series, patch_date)
        results[item] = res
        impact_rows.append(calc_impact(res, item, patch_date))

    if not results:
        print(f'\n[{patch_name}] 분석 가능한 아이템이 없습니다.')
        return None

    summary_df = pd.DataFrame(impact_rows)
    slug = patch_name.replace(' ', '_').replace('/', '-')

    plot_timeseries(results, patch_name, patch_date,
                    f'{out_dir}/{slug}_timeseries.html')
    plot_summary(summary_df, patch_name, patch_date,
                 f'{out_dir}/{slug}_summary.html')

    print(f'\n{"="*65}')
    print(f'  {patch_name}  |  출시일: {patch_date.date()}')
    print(f'{"="*65}')
    print(summary_df.to_string(index=False))
    print(f'{"="*65}')
    print('* 양수(+): 패치로 가격 상승 (수요 증가)')
    print('* 음수(-): 패치로 가격 하락 (공급 증가 또는 수요 감소)')
    if skipped:
        print(f'* 건너뜀: {", ".join(skipped)}')

    return summary_df


# ── 실행 예시 ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    daily = load_all_markets(MARKET_FILES)

    ITEMS = [
        '운명의 파괴석 결정',
        '운명의 파괴석',
        '상급 아비도스 융화 재료',
        '아비도스 융화 재료',
    ]

    analyze_patch('지평의 성당', ITEMS, daily)

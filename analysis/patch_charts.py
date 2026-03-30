import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.insert(0, '.')

OUTPUT_DIR = 'analysis/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from analysis.patch_impact import load_all_markets, run_counterfactual, MARKET_FILES

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

TARGET_ITEMS = [
    '운명의 파괴석 결정',
    '운명의 파괴석',
    '아비도스 융화 재료',
    '상급 아비도스 융화 재료',
]

C_ACTUAL = '#1D4ED8'
C_CF     = '#B91C1C'
C_BAND   = '#FECACA'
C_PRE    = '#F0F9FF'
C_POST   = '#FFF7ED'
C_VLINE  = '#92400E'


def draw_patch_chart(patch_name: str, patch_date: pd.Timestamp,
                     items: list, daily: dict, out_path: str):
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    fig.text(0.5, 0.98, f'{patch_name} 출시 전후 가격 변화',
             ha='center', va='top', fontsize=17, fontweight='bold', color='#111')
    fig.text(0.5, 0.955,
             f'출시가 없었을 경우의 추정 가격  |  출시일: {patch_date.date()}',
             ha='center', va='top', fontsize=10, color='#666')

    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3,
                  top=0.91, bottom=0.08, left=0.07, right=0.97)

    for idx, item in enumerate(items):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        for df in daily.values():
            if item in df.columns:
                series = df[item].dropna()
                break
        else:
            ax.set_visible(False)
            continue

        res = run_counterfactual(series, patch_date)
        actual_all = pd.concat([res['actual_pre'], res['actual_post']])
        cf_idx = res['cf_mean'].index

        ymin = min(actual_all.min(), res['cf_lower'].min()) * 0.995
        ymax = max(actual_all.max(), res['cf_upper'].max()) * 1.025

        ax.axvspan(actual_all.index.min(), patch_date, color=C_PRE, alpha=0.5, zorder=0)
        ax.axvspan(patch_date, actual_all.index.max(), color=C_POST, alpha=0.5, zorder=0)

        ax.fill_between(cf_idx, res['cf_lower'], res['cf_upper'],
                        color=C_BAND, alpha=0.5, zorder=1)
        ax.plot(cf_idx, res['cf_mean'], color=C_CF, linewidth=1.6,
                linestyle='--', zorder=3, alpha=0.85)
        ax.plot(actual_all.index, actual_all.values, color=C_ACTUAL,
                linewidth=2.2, zorder=4)

        ax.axvline(patch_date, color=C_VLINE, linewidth=1.4, linestyle='-', zorder=5)
        ax.text(patch_date, ymax, f' {patch_name}\n 출시',
                color=C_VLINE, fontsize=7.5, va='top', fontweight='bold')

        post_actual_mean = res['actual_post'].mean()
        post_cf_mean     = res['cf_mean'].mean()
        pct = (post_actual_mean - post_cf_mean) / post_cf_mean * 100
        mid_post = patch_date + (actual_all.index.max() - patch_date) / 2
        ax.annotate('', xy=(mid_post, post_actual_mean),
                    xytext=(mid_post, post_cf_mean),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=1.2))
        ax.text(mid_post, (post_actual_mean + post_cf_mean) / 2,
                f'  {pct:+.1f}%', fontsize=8.5, color='#444',
                va='center', fontweight='bold')

        ax.set_title(item, fontsize=11, fontweight='bold', color='#111', pad=10)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=8.5)
        ax.tick_params(axis='y', labelsize=8.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#ddd')
        ax.grid(axis='y', color='#eee', linewidth=0.8, zorder=0)
        ax.set_facecolor('white')

    handles = [
        mpatches.Patch(color=C_PRE,  alpha=0.8, label='패치 전'),
        mpatches.Patch(color=C_POST, alpha=0.8, label='패치 후'),
        plt.Line2D([0], [0], color=C_ACTUAL, linewidth=2,            label='실제 가격'),
        plt.Line2D([0], [0], color=C_CF,     linewidth=1.6, linestyle='--', label='반사실 예측'),
        mpatches.Patch(color=C_BAND, alpha=0.6, label='예측 90% 신뢰구간'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=9, framealpha=0.9, edgecolor='#ddd',
               bbox_to_anchor=(0.5, 0.01))

    save_path = os.path.join(OUTPUT_DIR, out_path)
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] {save_path} 저장 완료')


if __name__ == '__main__':
    daily = load_all_markets(MARKET_FILES)

    draw_patch_chart('지평의 성당', pd.Timestamp('2026-03-18'),
                     TARGET_ITEMS, daily, 'cathedral_counterfactual.png')

    draw_patch_chart('보너스룸', pd.Timestamp('2026-03-11'),
                     TARGET_ITEMS, daily, 'bonusroom_counterfactual.png')

"""패치 전후 가격 변화 정적 차트(PNG) 생성.

    python analysis/patch_charts.py "지평의 성당"
    python analysis/patch_charts.py --list

고친 것:

- 범례가 "예측 90% 신뢰구간"이라고 적혀 있었는데 실제 밴드는 95% 였다. 이제
  결과에 실린 수준을 그대로 표시한다.
- `'보너스룸'` 이라는, `data/event_log.txt` 에 없는 이벤트를 날짜까지
  하드코딩해 분석했다. 이제 이벤트 로그에 있는 것만 분석한다.
- 폰트가 `'Malgun Gothic'` 단독이라 Windows 밖에서 한글이 깨졌다.
- 상대 경로와 `sys.path.insert(0, '.')` 때문에 저장소 루트에서만 동작했다.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.patches as mpatches      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.gridspec import GridSpec   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis import marketdata as md                              # noqa: E402
from analysis.patch_impact import (                                # noqa: E402
    check_feasibility, confounding_events, load_daily_markets, run_counterfactual,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "output")

# 설치된 것 중 먼저 잡히는 한글 폰트를 쓴다. 하나만 지정하면 다른 OS 에서 깨진다.
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                "Noto Sans CJK KR", "NanumBarunGothic", "Malgun Gothic Semilight"]
plt.rcParams["font.family"] = KOREAN_FONTS + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def warn_if_no_korean_font() -> bool:
    """한글 폰트가 없으면 알린다.

    matplotlib 은 폰트가 없어도 그림을 만들어낸다. 한글이 전부 네모로 나온
    PNG 가 조용히 저장되는 것보다, 저장 전에 말해 주는 편이 낫다.
    """
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    if any(f in installed for f in KOREAN_FONTS):
        return True
    print("[경고] 한글 폰트를 찾지 못했습니다. 차트의 한글이 네모로 표시됩니다.")
    print("       Linux: apt-get install fonts-nanum · macOS/Windows 는 기본 폰트 사용")
    return False

TARGET_ITEMS = [
    "운명의 파괴석 결정",
    "운명의 파괴석",
    "아비도스 융화 재료",
    "상급 아비도스 융화 재료",
]

C_ACTUAL = "#1D4ED8"
C_CF = "#B91C1C"
C_BAND = "#FECACA"
C_PRE = "#F0F9FF"
C_POST = "#FFF7ED"
C_VLINE = "#92400E"


def draw_patch_chart(patch_name, patch_date, items, daily, out_path) -> str | None:
    patch_date = pd.Timestamp(patch_date).normalize()

    panels = []
    for item in items:
        series = md.find_item(item, daily)
        if series is None:
            print(f"  [{item}] 건너뜀: 시세 데이터 없음")
            continue
        ok, reason = check_feasibility(series, patch_date)
        if not ok:
            print(f"  [{item}] 건너뜀: {reason}")
            continue
        res = run_counterfactual(series, patch_date)
        if res is None:
            print(f"  [{item}] 건너뜀: 모델 적합 실패")
            continue
        panels.append((item, res))

    if not panels:
        print(f"[{patch_name}] 그릴 수 있는 품목이 없습니다.")
        return None

    level = panels[0][1].get("level", 0.95)

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.text(0.5, 0.98, f"{patch_name} 출시 전후 가격 변화",
             ha="center", va="top", fontsize=17, fontweight="bold", color="#111")
    fig.text(0.5, 0.955, f"출시가 없었을 경우의 추정 가격  |  출시일: {patch_date.date()}",
             ha="center", va="top", fontsize=10, color="#666")

    rows = (len(panels) + 1) // 2
    gs = GridSpec(rows, 2, figure=fig, hspace=0.45, wspace=0.3,
                  top=0.91, bottom=0.08, left=0.07, right=0.97)

    for idx, (item, res) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        actual_all = pd.concat([res["actual_pre"], res["actual_post"]])
        cf_idx = res["cf_mean"].index

        ymin = min(actual_all.min(), res["cf_lower"].min()) * 0.995
        ymax = max(actual_all.max(), res["cf_upper"].max()) * 1.025

        ax.axvspan(actual_all.index.min(), patch_date, color=C_PRE, alpha=0.5, zorder=0)
        ax.axvspan(patch_date, actual_all.index.max(), color=C_POST, alpha=0.5, zorder=0)
        ax.fill_between(cf_idx, res["cf_lower"], res["cf_upper"], color=C_BAND, alpha=0.5, zorder=1)
        ax.plot(cf_idx, res["cf_mean"], color=C_CF, linewidth=1.6, linestyle="--", zorder=3, alpha=0.85)
        ax.plot(actual_all.index, actual_all.values, color=C_ACTUAL, linewidth=2.2, zorder=4)
        ax.axvline(patch_date, color=C_VLINE, linewidth=1.4, zorder=5)
        ax.text(patch_date, ymax, f" {patch_name}\n 출시", color=C_VLINE,
                fontsize=7.5, va="top", fontweight="bold")

        post_actual = res["actual_post"].mean()
        post_cf = res["cf_mean"].mean()
        if post_cf:
            pct = (post_actual - post_cf) / post_cf * 100
            mid = patch_date + (actual_all.index.max() - patch_date) / 2
            ax.annotate("", xy=(mid, post_actual), xytext=(mid, post_cf),
                        arrowprops=dict(arrowstyle="<->", color="#666", lw=1.2))
            ax.text(mid, (post_actual + post_cf) / 2, f"  {pct:+.1f}%",
                    fontsize=8.5, color="#444", va="center", fontweight="bold")

        ax.set_title(item, fontsize=11, fontweight="bold", color="#111", pad=10)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#ddd")
        ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
        ax.set_facecolor("white")

    handles = [
        mpatches.Patch(color=C_PRE, alpha=0.8, label="패치 전"),
        mpatches.Patch(color=C_POST, alpha=0.8, label="패치 후"),
        plt.Line2D([0], [0], color=C_ACTUAL, linewidth=2, label="실제 가격"),
        plt.Line2D([0], [0], color=C_CF, linewidth=1.6, linestyle="--", label="반사실 예측"),
        mpatches.Patch(color=C_BAND, alpha=0.6, label=f"예측 {level:.0%} 구간"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               framealpha=0.9, edgecolor="#ddd", bbox_to_anchor=(0.5, 0.01))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, out_path)
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {save_path}")
    return save_path


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def main(argv=None) -> int:
    events = md.load_event_log()

    ap = argparse.ArgumentParser(description="패치 전후 반사실 차트 생성")
    ap.add_argument("patch", nargs="?", help="event_log.txt 에 있는 이벤트 이름")
    ap.add_argument("--list", action="store_true", help="분석 가능한 이벤트 목록")
    ap.add_argument("--items", type=str, default=None, help="쉼표로 구분한 품목명")
    args = ap.parse_args(argv)

    if args.list or not args.patch:
        print("event_log.txt 의 이벤트:")
        for name, date in sorted(events.items(), key=lambda kv: kv[1], reverse=True):
            others = [n for n in md.events_on(date, events) if n != name]
            tail = f"   ⚠ 같은 날: {', '.join(others)}" if others else ""
            print(f"  {date.date()}  {name}{tail}")
        return 0 if args.list else 1

    if args.patch not in events:
        print(f'[오류] event_log 에 "{args.patch}" 없음. --list 로 확인하세요.')
        return 1

    patch_date = events[args.patch]
    others = confounding_events(args.patch, patch_date, events)
    if others:
        print(f"[주의] 같은 날 다른 이벤트: {', '.join(others)}")
        print("       가격 변화를 한 이벤트에 귀속시킬 수 없습니다.")

    warn_if_no_korean_font()
    items = [x.strip() for x in args.items.split(",")] if args.items else TARGET_ITEMS
    daily = load_daily_markets()
    out = draw_patch_chart(args.patch, patch_date, items, daily, f"{_slug(args.patch)}.png")
    return 0 if out else 1


if __name__ == "__main__":
    raise SystemExit(main())

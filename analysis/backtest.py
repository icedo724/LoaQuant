"""예측 모델의 정확도와 **실측 예측구간 커버리지**를 재는 롤링 오리진 백테스트.

이 파일이 존재하는 이유는 하나다. 이전 코드에는 다음과 같은 주석이 네 군데
있었는데, 그 숫자를 만들어낸 코드가 저장소에 없었다.

    # interval_width=0.95: 실측 커버리지가 목표(80%)의 절반 미달이므로 구간 확대
    # Prophet 대비 MAPE 10배 이상 우월하고 CI 커버리지도 높음

모델 선택의 근거가 재현 불가능한 주석으로만 남아 있으면 검증도 반박도 할 수
없다. 이제 근거는 이 스크립트의 출력이고, 그 출력을 대시보드가 그대로 읽는다.

    python analysis/backtest.py --horizon 1 --items "운명의 파괴석,상급 아비도스 융화 재료"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import forecast as fc                 # noqa: E402
from analysis import marketdata as md               # noqa: E402
from common import timeaxis                         # noqa: E402

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_registry.json")

# 학습 구간이 이보다 짧으면 주간 계절성이 식별되지 않는다.
MIN_TRAIN_DAYS = 28

# 명목 수준보다 이만큼 이상 낮은 커버리지는 "구간이 못 믿을 만큼 좁다"로 본다.
COVERAGE_TOLERANCE = 0.15


def rolling_origin(
    series: pd.Series,
    model: str,
    horizon: int = 1,
    level: float = fc.DEFAULT_LEVEL,
    min_train: int = MIN_TRAIN_DAYS,
    step: int = 1,
    max_origins: int | None = None,
) -> dict:
    """원점을 하루씩 옮기며 h일 앞을 예측하고 실측과 대조한다.

    돌려주는 것:
      - ``mape``      : 평균 절대 백분율 오차 (%)
      - ``coverage``  : 실측값이 예측구간 안에 들어온 비율
      - ``width``     : 구간 폭의 중앙값을 실측 수준으로 나눈 값 (%)
      - ``n``         : 대조한 예측 건수
    """
    s = pd.Series(series).dropna().sort_index()
    origins = list(range(min_train, len(s) - horizon + 1, step))
    if max_origins:
        origins = origins[-max_origins:]

    errors, hits, widths = [], [], []
    for cut in origins:
        train, actual = s.iloc[:cut], s.iloc[cut:cut + horizon]
        if actual.empty:
            continue
        out = fc.predict(train, horizon=horizon, level=level, model=model,
                         start=pd.Timestamp(actual.index[0]))
        if out is None:
            continue

        mean = out.mean.reindex(actual.index)
        lo = out.lower.reindex(actual.index)
        hi = out.upper.reindex(actual.index)
        ok = mean.notna() & actual.notna() & (actual != 0)
        if not ok.any():
            continue

        errors.extend((np.abs(actual[ok] - mean[ok]) / np.abs(actual[ok]) * 100).tolist())
        hits.extend(((actual[ok] >= lo[ok]) & (actual[ok] <= hi[ok])).tolist())
        widths.extend(((hi[ok] - lo[ok]) / np.abs(actual[ok]) * 100).tolist())

    if not errors:
        return {"model": model, "mape": float("nan"), "coverage": float("nan"),
                "width": float("nan"), "n": 0}

    return {
        "model": model,
        "mape": float(np.mean(errors)),
        "coverage": float(np.mean(hits)),
        "width": float(np.median(widths)),
        "n": len(errors),
    }


def choose(results: list[dict], level: float) -> dict | None:
    """MAPE 가 가장 낮은 모델. 단, 구간이 명목보다 크게 좁은 모델은 제외한다.

    정확도만 보면 구간을 0으로 만든 모델이 이긴다. 대시보드가 구간을 그려서
    보여주는 이상, 그 구간이 표시한 확률을 대충이라도 지켜야 한다.
    """
    usable = [r for r in results if r["n"] > 0 and np.isfinite(r["mape"])]
    if not usable:
        return None

    calibrated = [r for r in usable if r["coverage"] >= level - COVERAGE_TOLERANCE]
    pool = calibrated or usable
    if not calibrated:
        # 전부 미달이면 정확도 대신 커버리지가 가장 덜 나쁜 쪽을 고른다.
        return max(usable, key=lambda r: r["coverage"])
    return min(pool, key=lambda r: r["mape"])


def evaluate(
    items: list[str] | None = None,
    models: list[str] | None = None,
    horizon: int = 1,
    level: float = fc.DEFAULT_LEVEL,
    max_origins: int | None = 90,
) -> dict:
    models = models or list(fc.MODELS.keys())
    markets = md.load_all_markets()
    if not markets:
        raise SystemExit("data/ 에 시세 CSV 가 없습니다.")

    daily = {name: timeaxis.game_daily_mean(df)[0] for name, df in markets.items()}

    if items is None:
        items = sorted({c for df in daily.values() for c in df.columns})

    registry = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon": horizon,
        "level": level,
        "min_train_days": MIN_TRAIN_DAYS,
        "default": fc.DEFAULT_MODEL,
        "items": {},
    }

    for item in items:
        series = md.find_item(item, daily)
        if series is None or len(series) < MIN_TRAIN_DAYS + horizon:
            continue

        results = [rolling_origin(series, m, horizon=horizon, level=level,
                                  max_origins=max_origins) for m in models]
        best = choose(results, level)
        if best is None:
            continue

        registry["items"][item] = {
            "model": best["model"],
            "mape": round(best["mape"], 3),
            "coverage": round(best["coverage"], 3),
            "width": round(best["width"], 3),
            "n": best["n"],
            "candidates": {r["model"]: {k: (round(v, 3) if isinstance(v, float) and np.isfinite(v) else v)
                                        for k, v in r.items() if k != "model"}
                           for r in results},
        }
        print(f"  {item:<24} → {best['model']:<14} "
              f"MAPE {best['mape']:5.2f}%  커버리지 {best['coverage']:.0%} (명목 {level:.0%})  n={best['n']}")

    return registry


def load_registry(path: str = REGISTRY_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="예측 모델 롤링 오리진 백테스트")
    ap.add_argument("--horizon", type=int, default=1, help="예측 기간(일)")
    ap.add_argument("--level", type=float, default=fc.DEFAULT_LEVEL, help="명목 예측구간 수준")
    ap.add_argument("--items", type=str, default=None, help="쉼표로 구분한 품목명. 생략 시 전체")
    ap.add_argument("--models", type=str, default=None, help="쉼표로 구분한 모델명")
    ap.add_argument("--max-origins", type=int, default=90, help="품목당 최대 원점 수")
    ap.add_argument("--out", type=str, default=REGISTRY_PATH)
    args = ap.parse_args(argv)

    items = [x.strip() for x in args.items.split(",")] if args.items else None
    models = [x.strip() for x in args.models.split(",")] if args.models else None

    print(f"롤링 오리진 백테스트 — horizon={args.horizon}일, 명목 {args.level:.0%}")
    registry = evaluate(items=items, models=models, horizon=args.horizon,
                        level=args.level, max_origins=args.max_origins)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"\n{len(registry['items'])}개 품목 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

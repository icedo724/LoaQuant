"""예측구간·데이터 로딩·거래대금 검증.

    python tests/test_analysis.py

여기 있는 검증은 전부 "예전 코드가 조용히 틀렸던 지점"에 대응한다. 주석에
숫자를 적어두는 대신 재현 가능한 검사로 남긴다.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis import forecast as fc, marketdata as md, metrics  # noqa: E402

FAILURES = []


def check(name, passed, detail=""):
    print(("  PASS  " if passed else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not passed:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 예측구간
# ---------------------------------------------------------------------------
def test_interval_widens_with_extrapolation():
    print("\n[1] 예측구간이 멀리 갈수록 넓어지는가")
    # 예전 get_linreg_forecast 는 상수 마진(1.96 * np.std(잔차, ddof=0))을 썼다.
    # 레버리지 항이 없으면 7일 뒤 구간이 1일 뒤와 같은 폭으로 그려진다.
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=120, freq="D")
    s = pd.Series(np.linspace(1000, 1200, 120) + rng.normal(0, 20, 120), index=idx)

    out = fc.ols_trend_dow(s, horizon=14)
    widths = (out.upper - out.lower).to_numpy()
    # 요일 더미마다 레버리지가 달라 주중에는 폭이 오르내린다. 추세 항의 효과는
    # 같은 요일끼리(7일 간격) 비교해야 드러난다.
    same_weekday = [widths[i + 7] > widths[i] for i in range(7)]
    check("같은 요일 기준 7일 뒤가 더 넓음", all(same_weekday),
          f"{widths[0]:.2f} → {widths[7]:.2f}")
    check("14일 뒤가 1일 뒤보다 넓음", widths[-1] > widths[0],
          f"{widths[0]:.2f} → {widths[-1]:.2f}")

    # 학습이 짧고 멀리 외삽할수록 레버리지 효과가 커진다. 상수 마진은 이 성질이 없다.
    short = s.iloc[:30]
    far = fc.ols_trend_dow(short, horizon=30)
    fw = (far.upper - far.lower).to_numpy()
    check("짧은 학습 + 먼 외삽에서 폭이 뚜렷이 벌어짐", fw[-1] / fw[0] > 1.2,
          f"{fw[0]:.1f} → {fw[-1]:.1f} ({fw[-1] / fw[0]:.2f}배)")


def test_interval_coverage_matches_nominal():
    print("\n[2] 명목 95% 구간이 실제로 95% 를 덮는가")
    # 진짜 검증. 알려진 오차 구조에서 실측 커버리지를 재고 명목과 비교한다.
    rng = np.random.default_rng(7)
    hits, n = 0, 0
    for trial in range(60):
        idx = pd.date_range("2026-01-01", periods=80, freq="D")
        truth = np.linspace(500, 560, 81)
        noise = rng.normal(0, 8, 81)
        s = pd.Series(truth[:80] + noise[:80], index=idx)

        out = fc.ols_trend_dow(s, horizon=1, level=0.95)
        if out is None:
            continue
        actual = truth[80] + noise[80]
        hits += int(out.lower.iloc[0] <= actual <= out.upper.iloc[0])
        n += 1

    cov = hits / n if n else float("nan")
    check("실측 커버리지가 명목 근처", 0.88 <= cov <= 1.0, f"{cov:.0%} (n={n})")


def test_constant_margin_would_be_too_narrow():
    print("\n[3] 예전 방식(상수 마진)이 실제로 더 좁은가")
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-01-01", periods=60, freq="D")
    s = pd.Series(np.linspace(300, 380, 60) + rng.normal(0, 15, 60), index=idx)

    out = fc.ols_trend_dow(s, horizon=7, level=0.95)
    proper = float((out.upper - out.lower).iloc[-1])

    # 예전 구현 재현: in-sample 잔차의 ddof=0 표준편차 × 1.96 × 2
    resid = (s - out.fitted.reindex(s.index)).to_numpy()
    legacy = 2 * 1.96 * float(np.std(resid))

    check("교정된 구간이 예전 상수 마진보다 넓음", proper > legacy,
          f"{proper:.1f} vs {legacy:.1f}")


def test_trend_is_per_day_not_per_observation():
    print("\n[4] 추세 기울기가 '관측 1건당'이 아니라 '하루당'인가")
    # 결측일이 있어도 하루당 기울기는 같아야 한다. 예전 코드는 np.arange(n) 을
    # 시간축으로 써서 결측이 있으면 단위가 어긋났다.
    full = pd.date_range("2026-01-01", periods=60, freq="D")
    s_full = pd.Series(np.arange(60, dtype="float64") * 10 + 100, index=full)

    keep = full[[i for i in range(60) if i % 3 != 1]]          # 3일 중 1일 결측
    s_gap = s_full.reindex(keep)

    a = fc.ols_trend_dow(s_full, horizon=1)
    b = fc.ols_trend_dow(s_gap, horizon=1, start=pd.Timestamp("2026-03-01"))
    c = fc.ols_trend_dow(s_full, horizon=1, start=pd.Timestamp("2026-03-01"))

    check("결측이 있어도 같은 날짜 예측이 일치",
          abs(float(b.mean.iloc[0]) - float(c.mean.iloc[0])) < 1.0,
          f"{float(b.mean.iloc[0]):.1f} vs {float(c.mean.iloc[0]):.1f}")
    check("완전한 시계열에서 정확히 외삽",
          abs(float(a.mean.iloc[0]) - (100 + 60 * 10)) < 1.0, f"{float(a.mean.iloc[0]):.1f}")


def test_short_series_is_refused():
    print("\n[5] 데이터가 모자라면 예측하지 않는가")
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    s = pd.Series([1.0, 2, 3, 4, 5], index=idx)
    check("5일로는 None", fc.ols_trend_dow(s) is None)
    check("predict 도 None", fc.predict(s, model="ols_trend_dow") is None)


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------
def test_cash_conversion_leaves_gaps_empty():
    print("\n[6] 환율이 없는 날짜를 조용히 메우지 않는가")
    # 예전 apply_gold_conversion 은 gold_dict.get(x, latest_gold) 라서, 골드
    # 데이터가 끊긴 뒤 82일을 전부 마지막 환율로 환산해 사실처럼 보여줬다.
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"item": [1000.0] * 5}, index=idx)
    gold = pd.Series([20.0, 20.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))

    cash = md.to_cash(prices, gold)
    check("환율 있는 날은 환산", np.isclose(cash["item"].iloc[0], 200.0),
          f"{cash['item'].iloc[0]}")
    check("환율 없는 날은 NaN", bool(cash["item"].iloc[2:].isna().all()))

    missing, last = md.gold_gap(idx, gold)
    check("빠진 날짜 수를 보고", missing == 3, f"{missing}일")
    check("마지막 환율 날짜를 보고", last == pd.Timestamp("2026-01-02"))


def test_event_log_parser():
    print("\n[7] 이벤트 로그 파서")
    body = (
        '"지평의 성당": 2026-03-18\n'
        '"카다룸 제도": 2026-03-11\n'
        '"운수대통 복주머니": 2026-03-11\n'
        '"차원술사 출시":2026-07-08\n'
        '\n'
        'broken line without quotes\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        events = md.load_event_log(path)
        check("정상 4건 파싱", len(events) == 4, f"{len(events)}건")
        check("공백 없는 콜론도 처리", events.get("차원술사 출시") == pd.Timestamp("2026-07-08"))
        problems = md.event_log_problems(path)
        check("형식 오류를 조용히 삼키지 않음", problems == ["broken line without quotes"],
              f"{problems}")
        same_day = md.events_on("2026-03-11", events)
        check("같은 날 이벤트 충돌 감지", same_day == ["운수대통 복주머니", "카다룸 제도"],
              f"{same_day}")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 거래대금
# ---------------------------------------------------------------------------
def test_trade_value_matches_its_caption():
    print("\n[8] 거래대금이 Σ(단가×거래량) 인가")
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    # 가격과 거래량이 상관된 경우. 두 계산식의 차이가 드러난다.
    price = pd.DataFrame({"A": [100.0, 200.0, 300.0, 400.0]}, index=idx)
    volume = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0]}, index=idx)

    res, _ = metrics.trade_value(price, volume)
    expected = float((price["A"] * volume["A"]).sum())          # 30,000
    legacy = float(price["A"].mean() * volume["A"].sum())       # 25,000

    check("Σ(pᵢvᵢ) 와 일치", np.isclose(res.loc["A", "value"], expected),
          f"{res.loc['A', 'value']:,.0f}")
    check("예전 식(p̄·Σv)과 다름", not np.isclose(expected, legacy),
          f"{expected:,.0f} vs {legacy:,.0f}")
    check("평균 단가가 거래량 가중",
          np.isclose(res.loc["A", "avg_price"], expected / volume["A"].sum()),
          f"{res.loc['A', 'avg_price']:.1f}")


def test_trade_value_ranking_is_not_legacy_ranking():
    print("\n[9] 계산식이 순위를 바꾸는가 (회귀 방지)")
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    # B 는 비쌀 때 많이 팔리고, A 는 쌀 때 많이 팔린다.
    price = pd.DataFrame({"A": [100.0, 100.0, 10.0], "B": [10.0, 100.0, 100.0]}, index=idx)
    volume = pd.DataFrame({"A": [1.0, 1.0, 100.0], "B": [1.0, 1.0, 100.0]}, index=idx)

    res, _ = metrics.trade_value(price, volume)
    proper_top = res.index[0]
    legacy = (price.mean() * volume.sum()).sort_values(ascending=False)
    check("올바른 식에서는 B 가 1위", proper_top == "B", f"{proper_top}")
    check("예전 식에서는 순위가 갈리지 않음", legacy.index[0] == "A",
          f"{legacy.index[0]}")


def test_partial_day_is_excluded():
    print("\n[10] 집계 중인 오늘을 거래대금에서 빼는가")
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    price = pd.DataFrame({"A": [100.0] * 4}, index=idx)
    volume = pd.DataFrame({"A": [100.0, 100.0, 100.0, 5.0]}, index=idx)   # 마지막은 부분값

    res, excluded = metrics.trade_value(price, volume,
                                        exclude_from=pd.Timestamp("2026-01-04"))
    check("제외된 날짜를 알려줌", excluded == pd.Timestamp("2026-01-04"), f"{excluded}")
    check("부분값이 총액에 안 들어감", np.isclose(res.loc["A", "value"], 30000.0),
          f"{res.loc['A', 'value']:,.0f}")

    res_all, _ = metrics.trade_value(price, volume)
    check("제외하지 않으면 총액이 달라짐", np.isclose(res_all.loc["A", "value"], 30500.0),
          f"{res_all.loc['A', 'value']:,.0f}")


def test_stale_price_is_not_reported_as_current():
    print("\n[11] 오래된 값을 '현재가'로 쓰지 않는가")
    # 골드 환산이 최근 구간에서 실패하면 그 날들이 NaN 이 된다. dropna().iloc[-1]
    # 만 보면 몇 달 전 가격이 현재가 카드에 그대로 뜬다.
    idx = pd.date_range("2026-01-01", periods=120, freq="D")
    s = pd.Series(np.linspace(1000, 1100, 120), index=idx)
    reliable = pd.Series(True, index=idx)

    fresh = metrics.market_signal(s, reliable)
    check("최신 데이터면 시그널이 나옴", fresh is not None and not fresh["stale"])
    check("현재가가 마지막 값", np.isclose(fresh["price"], 1100.0), f"{fresh['price']:.1f}")

    stale = s.copy()
    stale.iloc[-40:] = np.nan                      # 최근 40일 환산 실패
    rep = metrics.market_signal(stale, reliable)
    check("오래되면 stale 로 표시", rep is not None and rep["stale"])
    check("경과 일수를 보고", rep["stale_days"] == 40, f"{rep['stale_days']}일")
    check("가격을 내놓지 않음", "price" not in rep)


def main():
    test_interval_widens_with_extrapolation()
    test_interval_coverage_matches_nominal()
    test_constant_margin_would_be_too_narrow()
    test_trend_is_per_day_not_per_observation()
    test_short_series_is_refused()
    test_cash_conversion_leaves_gaps_empty()
    test_event_log_parser()
    test_trade_value_matches_its_caption()
    test_trade_value_ranking_is_not_legacy_ranking()
    test_partial_day_is_excluded()
    test_stale_price_is_not_reported_as_current()

    print("\n" + "=" * 52)
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        return 1
    print("전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())

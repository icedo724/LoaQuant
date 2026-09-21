"""게임일 시간 축 검증.

    python tests/test_timeaxis.py

여기서 깨지는 동작은 곧바로 화면의 숫자가 틀리는 것으로 이어진다. 이 저장소의
"하루"는 자정이 아니라 오전 6시에 바뀌고, 그 정의가 대시보드와 분석 모듈에
각각 적혀 있어서 서로 다른 값이 나왔던 적이 있다.
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from common import timeaxis as ta  # noqa: E402

FAILURES = []


def check(name, passed, detail=""):
    print(("  PASS  " if passed else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not passed:
        FAILURES.append(name)


def _hourly(start, periods, freq="1h", value=100.0):
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame({"item": np.full(len(idx), value, dtype="float64")}, index=idx)


def test_game_day_boundary():
    print("\n[1] 하루가 오전 6시에 바뀌는가")
    check("05:59 는 전날", ta.game_day_of("2026-03-18 05:59") == pd.Timestamp("2026-03-17"))
    check("06:00 은 당일", ta.game_day_of("2026-03-18 06:00") == pd.Timestamp("2026-03-18"))
    check("익일 05:59 는 여전히 당일",
          ta.game_day_of("2026-03-19 05:59") == pd.Timestamp("2026-03-18"))


def test_now_is_timezone_independent():
    print("\n[2] '지금'이 서버 타임존에 흔들리지 않는가")
    # pd.Timestamp.now() 를 쓰면 UTC 컨테이너와 KST 로컬에서 결과가 달라졌다.
    kst = ta.now_kst()
    expected = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    check("now_kst 가 KST 벽시계와 일치", abs((kst - expected).total_seconds()) < 5,
          f"{kst}")
    check("current_game_day 는 now_kst 에서 파생",
          ta.current_game_day() == ta.game_day_of(kst))


def test_daily_mean_and_counts():
    print("\n[3] 게임일 평균과 표본 수")
    # 3/18 06:00 부터 24시간 = 3/18 게임일 1일치
    df = _hourly("2026-03-18 06:00", 24)
    df.iloc[:, 0] = np.arange(24, dtype="float64")
    means, counts = ta.game_daily_mean(df)
    check("하루로 묶임", len(means) == 1, f"{len(means)}일")
    check("표본 24건", int(counts.iloc[0, 0]) == 24)
    check("평균이 산술평균", np.isclose(means.iloc[0, 0], np.arange(24).mean()))

    # 06:00 직전 한 건은 전날로 가야 한다
    df2 = _hourly("2026-03-18 05:00", 3)
    means2, counts2 = ta.game_daily_mean(df2)
    check("05:00 과 06:00 이 다른 날로 갈림", len(means2) == 2, f"{len(means2)}일")
    check("전날에 1건", int(counts2.iloc[0, 0]) == 1)


def test_sparse_days_are_flagged_not_dropped():
    print("\n[4] 표본이 적은 날을 지우지 않고 표시만 하는가")
    df = _hourly("2026-03-18 06:00", 24)
    df = pd.concat([df, _hourly("2026-03-19 07:00", 2)])
    means, counts = ta.game_daily_mean(df)
    check("이틀 모두 평균이 살아 있음", means.notna().all().all())
    mask = ta.reliable_mask(counts, min_samples=4)
    check("첫날은 신뢰 가능", bool(mask.iloc[0, 0]))
    check("둘째날은 신뢰 불가로 표시", not bool(mask.iloc[1, 0]))


def test_split_complete():
    print("\n[5] 진행 중인 하루를 분리하는가")
    today = pd.Timestamp("2026-03-20")
    idx = pd.date_range("2026-03-17", "2026-03-20", freq="D")
    done, partial = ta.split_complete(idx, today=today)
    check("완료된 날 3일", len(done) == 3, f"{len(done)}일")
    check("진행 중인 날 1일", len(partial) == 1 and partial[0] == today)


def test_indicators_need_even_axis():
    print("\n[6] 지표가 불균등 관측을 신호로 바꾸지 않는가")
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    s = pd.Series(np.linspace(100, 140, 40), index=idx)

    all_obs = pd.Series(True, index=idx)
    ma, up, lo = ta.bollinger(s, all_obs, window=20)
    check("관측이 충분하면 밴드가 나옴", ma.notna().any())
    check("상단이 중심보다 큼", bool((up.dropna() > ma.dropna()).all()))

    sparse = pd.Series([i % 4 == 0 for i in range(40)], index=idx)   # 25% 만 신뢰
    ma2, _, _ = ta.bollinger(s, sparse, window=20)
    check("관측이 부족하면 신호를 내지 않음", ma2.isna().all())

    r = ta.rsi(s, all_obs, window=14)
    check("단조 상승이면 RSI 100 에 수렴", float(r.dropna().iloc[-1]) > 99,
          f"{float(r.dropna().iloc[-1]):.1f}")


def test_day_of_week_detrending():
    print("\n[7] 요일 효과에서 추세가 빠지는가")
    idx = pd.date_range("2026-01-05", periods=140, freq="D")      # 월요일 시작
    # 추세만 있고 요일 효과는 전혀 없는 시계열
    s = pd.Series(np.linspace(100, 50, 140), index=idx)

    raw, _ = ta.day_of_week_effect(s, detrend=False)
    det, n = ta.day_of_week_effect(s, detrend=True)

    check("추세 제거 전에는 요일 차이가 생김", float(raw.abs().max()) > 0.5,
          f"최대 {float(raw.abs().max()):.2f}%")
    check("추세 제거 후에는 사라짐", float(det.abs().max()) < 0.05,
          f"최대 {float(det.abs().max()):.4f}%")
    check("요일별 표본 수를 함께 돌려줌", int(n.sum()) > 0 and len(n) == 7)


def test_maintenance_windows():
    print("\n[8] 정기 점검 구간")
    wins = ta.maintenance_windows("2026-03-16", "2026-03-29")
    check("2주간 수요일 2회", len(wins) == 2, f"{len(wins)}회")
    check("06시 시작 10시 종료",
          all(a.hour == 6 and b.hour == 10 for a, b in wins))
    check("전부 수요일", all(a.weekday() == 2 for a, _ in wins))


def test_coverage():
    print("\n[9] 수집률 계산")
    idx = pd.DatetimeIndex(pd.date_range("2026-01-01", periods=100, freq="1h")[::2])
    cov = ta.coverage(idx)
    check("절반 수집이면 비율 ~0.5", 0.45 < cov["ratio"] < 0.55, f"{cov['ratio']:.2f}")
    check("간격 중앙값 2시간", abs(cov["median_gap_h"] - 2.0) < 1e-6)

    # 재시도 크론이라 한 시간에 두 번 성공할 수도 있다. 그래도 1 을 넘으면 안 된다.
    dense = pd.DatetimeIndex(sorted(set(
        list(pd.date_range("2026-01-01", periods=48, freq="1h")) +
        list(pd.date_range("2026-01-01 00:20", periods=48, freq="1h")))))
    cov2 = ta.coverage(dense)
    check("한 주기에 두 건이어도 비율이 1 이하", cov2["ratio"] <= 1.0, f"{cov2['ratio']:.2f}")


def main():
    test_game_day_boundary()
    test_now_is_timezone_independent()
    test_daily_mean_and_counts()
    test_sparse_days_are_flagged_not_dropped()
    test_split_complete()
    test_indicators_need_even_axis()
    test_day_of_week_detrending()
    test_maintenance_windows()
    test_coverage()

    print("\n" + "=" * 52)
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        return 1
    print("전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())

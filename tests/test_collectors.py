"""수집 멱등성 가드·수집 로그·환율 추출 검증.

의존성 없이 단독 실행한다.
    python tests/test_collectors.py

크론 시도를 시간당 3회로 늘린 대신 과수집을 가드가 막는 구조라, 가드가 깨지면
곧바로 API 호출량과 커밋이 불어난다. 수집기를 수정할 때는 함께 실행한다.
"""
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from economy import collect_guard as cg  # noqa: E402

FAILURES = []


def check(name, passed, detail=''):
    print(('  PASS  ' if passed else '  FAIL  ') + name + (f'  [{detail}]' if detail else ''))
    if not passed:
        FAILURES.append(name)


def write_csv(path, header_cols):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(','.join(header_cols) + '\n')
        f.write('테스트아이템' + ',1' * (len(header_cols) - 1) + '\n')


def test_guard_blocks_over_collection():
    print("\n[1] 짧은 간격의 중복 실행을 막는가")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'market_materials.csv')
        write_csv(path, ['item_name', '2026-09-21 12:00', '2026-09-21 12:35'])

        check('마지막 수집 시각을 헤더에서 읽음',
              cg.last_collected_at(path) == datetime(2026, 9, 21, 12, 35),
              str(cg.last_collected_at(path)))

        ok, _ = cg.should_collect(path, now=datetime(2026, 9, 21, 12, 40))
        check('5분 뒤 재실행은 차단', ok is False)

        ok, _ = cg.should_collect(path, now=datetime(2026, 9, 21, 13, 20))
        check('45분 뒤도 차단 (기준 50분)', ok is False)

        ok, _ = cg.should_collect(path, now=datetime(2026, 9, 21, 13, 30))
        check('55분 뒤에는 통과', ok is True)
    finally:
        shutil.rmtree(tmp)


def test_guard_allows_hourly_cadence():
    print("\n[2] 크론이 :13/:33/:53 에 떠도 매시간 수집이 막히지 않는가")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'market_materials.csv')
        # 앞 시간은 :53 에, 다음 시간은 :13 에 성공한 경우 = 간격 20분이므로 차단되어야 한다.
        write_csv(path, ['item_name', '2026-09-21 12:53'])
        ok, _ = cg.should_collect(path, now=datetime(2026, 9, 21, 13, 13))
        check('같은 시간대 중복 시도는 차단', ok is False)

        # 앞 시간 :13 성공 후 다음 시간 :13 시도 = 60분이므로 통과해야 한다.
        write_csv(path, ['item_name', '2026-09-21 12:13'])
        ok, _ = cg.should_collect(path, now=datetime(2026, 9, 21, 13, 13))
        check('다음 시간 시도는 통과', ok is True)

        # 앞 시간 :53 성공 후 다음 시간 :53 시도 = 60분. 기준이 60분이면 경계에서 막힌다.
        write_csv(path, ['item_name', '2026-09-21 12:53'])
        ok, _ = cg.should_collect(path, now=datetime(2026, 9, 21, 13, 53))
        check('60분 경계에서도 통과 (기준이 60분이 아닌 이유)', ok is True)
    finally:
        shutil.rmtree(tmp)


def test_guard_proceeds_without_evidence():
    print("\n[3] 판단 근거가 없으면 수집을 진행하는가")
    tmp = tempfile.mkdtemp()
    try:
        ok, reason = cg.should_collect(os.path.join(tmp, 'none.csv'))
        check('파일이 없으면 진행', ok is True, reason)

        path = os.path.join(tmp, 'broken.csv')
        write_csv(path, ['item_name', 'sub_category', 'item_grade'])
        ok, reason = cg.should_collect(path)
        check('시각 컬럼이 하나도 없으면 진행', ok is True, reason)
    finally:
        shutil.rmtree(tmp)


def test_run_log_roundtrip():
    print("\n[4] 수집 로그가 작업별로 기록·조회되는가")
    tmp = tempfile.mkdtemp()
    try:
        cg.append_run_log(tmp, 'prices', datetime(2026, 9, 21, 12, 0), 'success', 97, 300, 'ok')
        cg.append_run_log(tmp, 'volume', datetime(2026, 9, 21, 5, 17), 'failed', 12, 0, 'api down')
        cg.append_run_log(tmp, 'prices', datetime(2026, 9, 21, 13, 0), 'skipped', 0, 0, 'too soon')

        path = os.path.join(tmp, 'collection_log.csv')
        check('로그 파일 생성', os.path.exists(path))
        with open(path, encoding='utf-8-sig') as f:
            lines = f.read().strip().split('\n')
        check('헤더 + 3행', len(lines) == 4, f'{len(lines)}행')
        check('job 컬럼이 첫 컬럼', lines[0].startswith('job,'), lines[0])

        # last_success_at 은 started_at 이 아니라 finished_at(실제 종료 시각)을 본다.
        # 가드가 재야 하는 것은 "언제 끝났는가"이기 때문이다.
        check('prices 마지막 성공 조회',
              cg.last_success_at(tmp, 'prices') is not None)
        check('failed 는 성공으로 세지 않음',
              cg.last_success_at(tmp, 'volume') is None)

        cg.append_run_log(tmp, 'skiponly', datetime(2026, 9, 21, 1, 0), 'skipped', 0, 0, '')
        check('skipped 만 있으면 성공 기록 없음',
              cg.last_success_at(tmp, 'skiponly') is None)
        check('모르는 작업은 None', cg.last_success_at(tmp, 'nosuch') is None)
    finally:
        shutil.rmtree(tmp)


def test_daily_job_guard():
    print("\n[5] 일간 작업이 하루 1회로 제한되는가")
    tmp = tempfile.mkdtemp()
    try:
        ok, reason = cg.should_run_job(tmp, 'volume', 20)
        check('기록이 없으면 진행', ok is True, reason)

        cg.append_run_log(tmp, 'volume', datetime(2026, 9, 21, 5, 0), 'success', 120, 120, '')
        finished = cg.last_success_at(tmp, 'volume')

        ok, _ = cg.should_run_job(tmp, 'volume', 20, now=finished + timedelta(hours=3))
        check('3시간 뒤 재실행은 차단', ok is False)

        later = finished + timedelta(hours=21)
        ok, _ = cg.should_run_job(tmp, 'volume', 20, now=later)
        check('21시간 뒤 실행은 통과', ok is True)
    finally:
        shutil.rmtree(tmp)


# gold_processing 은 pandas 에 의존하므로, 의존성이 없으면 정규식만 같은 정의로 검증한다.
GOLD_PATTERN = re.compile(r'(?<!\d)100\s*(?:[:대/\-;|lI]\s*)?(\d{1,3}(?:\.\d+)?)(?!\d)')
MIN_VALID_RATE, MAX_VALID_RATE = 1.0, 200.0


def _fallback_extract(text):
    m = GOLD_PATTERN.search(text)
    if not m:
        return None
    value = float(m.group(1))
    return value if MIN_VALID_RATE <= value <= MAX_VALID_RATE else None


def test_gold_rate_extraction():
    print("\n[6] 환율 추출이 자릿수 오류 없이 동작하는가")
    try:
        from common.gold_processing import extract_rate, GOLD_PATTERN as P, MIN_VALID_RATE as LO
        check('gold_processing 의 정의를 직접 검증', True)
        check('테스트와 모듈의 패턴이 동일', P.pattern == GOLD_PATTERN.pattern)
        check('테스트와 모듈의 하한이 동일', LO == MIN_VALID_RATE, f'{LO}')
    except ImportError:
        extract_rate = _fallback_extract
        check('pandas 부재 — 동일 정의로 대체 검증', True)

    cases = [
        ("100:12 삽니다", 12.0, '기본 형식'),
        ("100대12", 12.0, '구분자 변형'),
        ("100 12", 12.0, '구분자 없음'),
        ("100:9 팔아요", 9.0, '한 자리 (예전에는 누락)'),
        ("100:105", 105.0, '세 자리 (예전에는 10 으로 오추출)'),
        ("골드 100:11.5", 11.5, '소수점 (예전에는 11 로 절삭)'),
        ("가격 100000 원", None, '더 긴 숫자의 일부 (예전에는 00 추출)'),
        ("아이템 100개 12만원", None, '환율과 무관한 문장'),
        ("1000:12", None, '앞자리가 더 긴 숫자'),
        ("100:0", None, '범위 밖'),
        ("레벨 1600 100:12", 12.0, '앞에 다른 숫자가 있어도 추출'),
    ]
    for text, expected, label in cases:
        got = extract_rate(text)
        check(f'{label}: {text!r}', got == expected, f'{got} (기대 {expected})')


def main():
    test_guard_blocks_over_collection()
    test_guard_allows_hourly_cadence()
    test_guard_proceeds_without_evidence()
    test_run_log_roundtrip()
    test_daily_job_guard()
    test_gold_rate_extraction()

    print("\n" + "=" * 52)
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        return 1
    print("전체 통과")
    return 0


if __name__ == '__main__':
    sys.exit(main())

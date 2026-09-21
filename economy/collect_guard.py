"""수집 실행의 멱등성 가드와 실행 로그.

크론 시도 횟수를 시간당 3회로 늘리는 것은 GitHub 이 스케줄 이벤트를 버리는 문제에
대한 유일한 직접 대응이다. 다만 시도가 늘면 운 좋게 여러 번 성공하는 시간대가 생기고,
그만큼 API 호출과 커밋이 불어난다. 그래서 "마지막 수집으로부터 충분히 지났을 때만
진행한다"는 가드가 크론 다중화와 반드시 짝을 이뤄야 한다.

부수 효과로 수동 workflow_dispatch 남발과 재실행에 의한 중복 컬럼도 함께 막힌다.
"""
import csv
import os
from datetime import datetime, timedelta, timezone

# 매시간 수집이 목표이므로, 마지막 수집 후 이만큼 지나야 다음 수집을 허용한다.
# 60분이 아니라 50분인 이유는, 크론이 :13/:33/:53 에 시도하므로 성공 시각이
# 정확히 1시간 간격이 아니기 때문이다. 60분으로 두면 매시간 수집이 오히려 막힌다.
MIN_GAP_MINUTES = 50

# 수집 시각 컬럼의 형식. data_collector.get_korea_time_str() 과 같아야 한다.
TIME_COL_FORMAT = '%Y-%m-%d %H:%M'

# 메타 컬럼(아이템 식별용). 나머지는 모두 수집 시각 컬럼으로 본다.
META_COLUMNS = {'item_name', 'sub_category', 'item_grade', 'item_tier'}

LOG_FIELDS = ['job', 'started_at', 'finished_at', 'status', 'requests', 'rows', 'note']


def kst_now():
    # 러너는 UTC 로 돌기 때문에 KST 를 명시적으로 만든다.
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))


def last_collected_at(csv_path):
    """CSV 헤더에서 가장 최근 수집 시각을 읽는다. 없으면 None."""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, encoding='utf-8-sig', newline='') as f:
            header = next(csv.reader(f))
    except (StopIteration, OSError):
        return None

    stamps = []
    for col in header:
        if col in META_COLUMNS:
            continue
        try:
            stamps.append(datetime.strptime(col, TIME_COL_FORMAT))
        except ValueError:
            continue
    return max(stamps) if stamps else None


def should_collect(csv_path, now=None, min_gap_minutes=MIN_GAP_MINUTES):
    """(진행 여부, 사유) 를 돌려준다.

    파일이 없거나 시각 컬럼을 하나도 읽지 못하면 진행한다. 판단 근거가 없을 때
    수집을 건너뛰면 영원히 시작하지 못하기 때문이다.
    """
    now = now or kst_now().replace(tzinfo=None)
    last = last_collected_at(csv_path)
    if last is None:
        return True, '이전 수집 기록 없음'

    gap_min = (now - last).total_seconds() / 60.0
    if gap_min < min_gap_minutes:
        return False, f'마지막 수집 {last:%Y-%m-%d %H:%M} 이후 {gap_min:.0f}분 (기준 {min_gap_minutes}분)'
    return True, f'마지막 수집 이후 {gap_min:.0f}분'


def append_run_log(data_dir, job, started_at, status, requests=0, rows=0, note=''):
    """실행 결과를 data/collection_log.csv 에 한 줄 남긴다.

    수집이 안 된 시각은 CSV 에서 '존재하지 않는 컬럼'이라 결측인지 아닌지 구분할 수
    없다. 실행 자체를 기록해 두면 수집 주기를 손으로 분석하지 않고도 모니터링할 수 있다.
    """
    path = os.path.join(data_dir, 'collection_log.csv')
    is_new = not os.path.exists(path)
    row = {
        'job': job,
        'started_at': started_at.strftime(TIME_COL_FORMAT),
        'finished_at': kst_now().strftime(TIME_COL_FORMAT),
        'status': status,
        'requests': requests,
        'rows': rows,
        'note': note,
    }
    try:
        with open(path, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except OSError as e:
        # 로그 실패가 수집 실패가 되어서는 안 된다.
        print(f"   -> [Warn] 수집 로그 기록 실패: {e}")


def last_success_at(data_dir, job):
    """수집 로그에서 해당 작업이 마지막으로 성공한 시각을 읽는다. 없으면 None."""
    path = os.path.join(data_dir, 'collection_log.csv')
    if not os.path.exists(path):
        return None
    latest = None
    try:
        with open(path, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                if row.get('job') != job or row.get('status') != 'success':
                    continue
                try:
                    at = datetime.strptime(row['finished_at'], TIME_COL_FORMAT)
                except (KeyError, TypeError, ValueError):
                    continue
                if latest is None or at > latest:
                    latest = at
    except OSError:
        return None
    return latest


def should_run_job(data_dir, job, min_gap_hours, now=None):
    """(진행 여부, 사유). 로그가 없으면 진행한다 — 근거가 없을 때 건너뛰면
    영원히 시작하지 못하기 때문이다."""
    now = now or kst_now().replace(tzinfo=None)
    last = last_success_at(data_dir, job)
    if last is None:
        return True, '이전 성공 기록 없음'
    gap_h = (now - last).total_seconds() / 3600.0
    if gap_h < min_gap_hours:
        return False, f'마지막 성공 {last:%Y-%m-%d %H:%M} 이후 {gap_h:.1f}시간 (기준 {min_gap_hours}시간)'
    return True, f'마지막 성공 이후 {gap_h:.1f}시간'

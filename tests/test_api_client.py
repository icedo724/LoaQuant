"""API 호출 한도·재시도 정책 검증.

의존성 없이 단독 실행한다.
    python tests/test_api_client.py

이 파일이 검증하는 동작이 깨지면 곧바로 오픈 API 이용 한도 위반으로 이어지므로,
api_client.py 를 수정할 때는 반드시 함께 실행한다.
"""
import os
import sys
import time
import types

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# API 키 파일 없이 실행할 수 있도록 설정 로더를 대체한다.
_fake_loader = types.ModuleType('common.config_loader')
_fake_loader.load_api_key = lambda: 'TEST_KEY'
sys.modules['common.config_loader'] = _fake_loader

import common.api_client as ac  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = f'body-{status_code}'

    def json(self):
        return {'ok': True}


class FakeSession:
    """미리 정해둔 상태 코드를 순서대로 돌려주는 가짜 세션."""

    def __init__(self, codes):
        self.codes = list(codes)
        self.calls = []

    def _next(self):
        self.calls.append(time.monotonic())
        code = self.codes.pop(0) if self.codes else 200
        if isinstance(code, tuple):
            return FakeResponse(code[0], code[1])
        return FakeResponse(code)

    def post(self, *args, **kwargs):
        return self._next()

    def get(self, *args, **kwargs):
        return self._next()


def build_api(codes):
    api = ac.LostArkAPI()
    api.session = FakeSession(codes)
    return api


FAILURES = []


def check(name, passed, detail=''):
    print(('  PASS  ' if passed else '  FAIL  ') + name + (f'  [{detail}]' if detail else ''))
    if not passed:
        FAILURES.append(name)


def test_rate_limit_stays_under_quota():
    print("\n[1] 호출 간격이 분당 100회 한도 아래를 유지하는가")
    rpm = 60 / ac.MIN_REQUEST_INTERVAL
    check('환산 분당 호출량이 100 미만', rpm < 100, f'{rpm:.1f} req/min')

    api = build_api([200] * 5)
    started = time.monotonic()
    for _ in range(5):
        api.get_market_items(50000, item_name='x')
    elapsed = time.monotonic() - started
    expected = ac.MIN_REQUEST_INTERVAL * 4  # 첫 요청은 대기하지 않는다
    check('연속 요청이 최소 간격만큼 벌어짐', elapsed >= expected * 0.95,
          f'{elapsed:.2f}s >= {expected:.2f}s')


def test_429_does_not_recurse_forever():
    print("\n[2] 429 가 무한 재귀 없이 유한 재시도로 끝나는가")
    api = build_api([429] * 20)
    result = api.get_market_items(50000, item_name='x')
    check('None 반환 (재귀 아님)', result is None)
    check(f'시도 횟수가 MAX_RETRIES({ac.MAX_RETRIES})로 제한',
          len(api.session.calls) == ac.MAX_RETRIES, f'{len(api.session.calls)}회')


def test_circuit_breaker_stops_collection():
    print("\n[3] 연속 실패가 누적되면 수집을 중단하는가")
    api = build_api([429] * 500)
    raised = None
    try:
        for _ in range(ac.MAX_CONSECUTIVE_FAILURES + 2):
            api.get_market_items(50000, item_name='x')
    except ac.APIUnavailableError as e:
        raised = e
    check('APIUnavailableError 발생', raised is not None)


def test_success_resets_failure_counter():
    print("\n[4] 성공하면 연속 실패 카운터가 초기화되는가")
    api = build_api([429] * ac.MAX_RETRIES + [200] + [429] * 500)
    api.get_market_items(50000, item_name='x')
    check('실패 1건 누적', api._consecutive_failures == 1, f'={api._consecutive_failures}')
    api.get_market_items(50000, item_name='x')
    check('성공 후 0으로 초기화', api._consecutive_failures == 0, f'={api._consecutive_failures}')


def test_retry_after_header_is_honored():
    print("\n[5] Retry-After 헤더를 그대로 따르는가")
    delay = ac.LostArkAPI._retry_delay(FakeResponse(429, {'Retry-After': '7'}), attempt=0)
    check('상한으로 자르지 않고 7초 그대로', delay == 7.0, f'{delay}')

    fallback = ac.LostArkAPI._retry_delay(FakeResponse(429, {}), attempt=2)
    check('헤더가 없으면 지수 백오프',
          fallback == min(ac.INITIAL_BACKOFF * 4, ac.MAX_BACKOFF), f'{fallback}')


def test_gives_up_when_server_asks_longer_than_budget():
    print("\n[6] 서버 요구 대기가 상한을 넘으면 일찍 재시도하지 않고 포기하는가")
    saved = ac.MAX_BACKOFF
    ac.MAX_BACKOFF = 10.0
    try:
        api = build_api([(429, {'Retry-After': '600'}), 200])
        started = time.monotonic()
        result = api.get_market_items(50000, item_name='x')
        took = time.monotonic() - started
        check('None 반환', result is None)
        check('재시도하지 않음', len(api.session.calls) == 1, f'{len(api.session.calls)}회')
        check('상한보다 일찍 재요청하지 않음', took < ac.MAX_BACKOFF, f'{took:.2f}s')
    finally:
        ac.MAX_BACKOFF = saved


def test_non_retryable_error_fails_fast():
    print("\n[7] 재시도해도 소용없는 오류는 즉시 중단하는가")
    api = build_api([401, 200])
    result = api.get_market_items(50000, item_name='x')
    check('401 즉시 None', result is None)
    check('재시도하지 않음', len(api.session.calls) == 1, f'{len(api.session.calls)}회')


def main():
    # 백오프 대기로 테스트가 길어지지 않도록 간격을 줄인다.
    ac.INITIAL_BACKOFF, ac.MAX_BACKOFF = 0.01, 0.02

    test_rate_limit_stays_under_quota()
    test_429_does_not_recurse_forever()
    test_circuit_breaker_stops_collection()
    test_success_resets_failure_counter()
    test_retry_after_header_is_honored()
    test_gives_up_when_server_asks_longer_than_budget()
    test_non_retryable_error_fails_fast()

    print("\n" + "=" * 52)
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        return 1
    print("전체 통과")
    return 0


if __name__ == '__main__':
    sys.exit(main())

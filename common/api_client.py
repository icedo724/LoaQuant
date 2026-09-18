import time

import requests

from common.config_loader import load_api_key

# 로스트아크 오픈 API 호출 한도는 분당 100회다.
# 실제 한도보다 낮게 잡아 응답 지연·재시도로 인한 순간 초과를 방지한다.
RATE_LIMIT_PER_MIN = 95
MIN_REQUEST_INTERVAL = 60.0 / RATE_LIMIT_PER_MIN

# 429(한도 초과) 응답에 대한 재시도 정책.
# 고정 간격으로 무한히 재시도하면 한도를 계속 두드리게 되므로,
# 지수 백오프를 적용하고 정해진 횟수 이후에는 포기한다.
MAX_RETRIES = 4
INITIAL_BACKOFF = 5.0
MAX_BACKOFF = 120.0

# 일시적 서버 오류. 429와 동일하게 재시도 대상으로 본다.
RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# 재시도를 유한하게 만들어도 요청 수가 많으면 전체 실행이 길어진다.
# API 가 응답하지 않는 상황에서 수백 건을 계속 두드리는 대신,
# 연속 실패가 누적되면 수집을 중단해 실패를 빨리 드러낸다.
MAX_CONSECUTIVE_FAILURES = 5


class APIUnavailableError(RuntimeError):
    pass


class LostArkAPI:
    def __init__(self):
        self.api_key = load_api_key()
        self.base_url = "https://developer-lostark.game.onstove.com"
        self.headers = {
            'accept': 'application/json',
            'authorization': f'bearer {self.api_key}',
            'content-type': 'application/json'
        }
        self.session = requests.Session()
        self._last_request_at = 0.0
        self._consecutive_failures = 0

    # -----------------------------------------------------------------
    # 호출 한도 제어
    # -----------------------------------------------------------------
    def _throttle(self):
        # 모든 요청이 이 지점을 통과하므로, 호출 간격을 여기 한 곳에서만 관리한다.
        # 수집 스크립트마다 sleep 을 흩어 두면 실제 분당 호출량을 계산할 수 없다.
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _retry_delay(response, attempt):
        # 서버가 Retry-After 로 대기 시간을 지정하면 그 값을 그대로 따른다.
        # 상한으로 잘라내면 서버가 지시한 시점보다 일찍 다시 요청하게 되므로 자르지 않는다.
        retry_after = response.headers.get('Retry-After') if response is not None else None
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)

    # -----------------------------------------------------------------
    # 공통 요청 처리
    # -----------------------------------------------------------------
    def _request(self, method, url, payload=None):
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                if method == 'POST':
                    response = self.session.post(url, headers=self.headers, json=payload, timeout=30)
                else:
                    response = self.session.get(url, headers=self.headers, timeout=30)
            except requests.RequestException as e:
                delay = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                print(f"연결 실패 ({attempt + 1}/{MAX_RETRIES}): {e} → {delay:.0f}초 후 재시도")
                time.sleep(delay)
                continue

            if response.status_code == 200:
                self._consecutive_failures = 0
                return response.json()

            if response.status_code in RETRYABLE_STATUS:
                delay = self._retry_delay(response, attempt)
                label = "Rate Limit 도달" if response.status_code == 429 else f"서버 오류 {response.status_code}"
                if delay > MAX_BACKOFF:
                    # 서버가 요구한 대기 시간이 이 작업의 예산을 넘는 경우,
                    # 더 일찍 재요청하는 대신 이번 요청을 포기한다.
                    print(f"{label}. 서버 요청 대기 시간 {delay:.0f}초가 상한({MAX_BACKOFF:.0f}초)을 초과 → 포기")
                    break
                print(f"{label} ({attempt + 1}/{MAX_RETRIES}) → {delay:.0f}초 후 재시도")
                time.sleep(delay)
                continue

            # 재시도해도 결과가 달라지지 않는 오류(400/401/403 등)는 즉시 중단한다.
            print(f"API 오류 ({response.status_code}): {response.text}")
            return None

        self._consecutive_failures += 1
        print(f"요청 포기: {url}")
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise APIUnavailableError(
                f"연속 {self._consecutive_failures}건 요청 실패. API 응답 불가로 판단하고 수집을 중단한다."
            )
        return None

    # -----------------------------------------------------------------
    # 엔드포인트
    # -----------------------------------------------------------------
    def get_market_items(self, category_code, item_name=None, item_tier=None, item_grade=None, page_no=1,
                         sort_condition="ASC"):
        url = f"{self.base_url}/markets/items"
        payload = {
            "Sort": "CURRENT_MIN_PRICE",
            "CategoryCode": category_code,
            "PageNo": page_no,
            "SortCondition": sort_condition
        }
        if item_tier: payload["ItemTier"] = item_tier
        if item_name: payload["ItemName"] = item_name
        if item_grade: payload["ItemGrade"] = item_grade

        return self._request('POST', url, payload)

    def get_auction_items(self, category_code, item_name, item_tier=None, page_no=1):
        url = f"{self.base_url}/auctions/items"
        payload = {
            "ItemLevelMin": 0, "ItemLevelMax": 0,
            "ItemTier": item_tier if item_tier else 0,
            "CategoryCode": category_code,
            "ItemName": item_name,
            "PageNo": page_no,
            "Sort": "BUY_PRICE",
            "SortCondition": "ASC"
        }

        return self._request('POST', url, payload)

    def get_market_item_stats(self, item_id):
        url = f"{self.base_url}/markets/items/{item_id}"
        return self._request('GET', url)

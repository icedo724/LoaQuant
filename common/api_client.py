import requests
import time
from common.config_loader import load_api_key


class LostArkAPI:
    def __init__(self):
        self.api_key = load_api_key()
        self.base_url = "https://developer-lostark.game.onstove.com"
        self.headers = {
            'accept': 'application/json',
            'authorization': f'bearer {self.api_key}',
            'content-type': 'application/json'
        }

    def get_market_items(self, category_code, item_name=None, item_tier=None, item_grade=None, page_no=1,
                         sort_condition="ASC"):
        url = f"{self.base_url}/markets/items"

        payload = {
            "Sort": "CURRENT_MIN_PRICE",
            "CategoryCode": category_code,
            "PageNo": page_no,
            "SortCondition": sort_condition
        }

        # 선택적 파라미터 추가
        if item_tier:
            payload["ItemTier"] = item_tier
        if item_name:
            payload["ItemName"] = item_name
        if item_grade:
            payload["ItemGrade"] = item_grade

        try:
            response = requests.post(url, headers=self.headers, json=payload)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("⏳ Rate Limit 도달. 60초 대기 중...")
                time.sleep(60)
                return self.get_market_items(category_code, item_name, item_tier, item_grade, page_no, sort_condition)
            else:
                print(f"API 오류 ({response.status_code}): {response.text}")
                return None
        except Exception as e:
            print(f"연결 실패: {e}")
            return None
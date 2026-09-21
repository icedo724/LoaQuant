"""일일 거래량 수집.

이 수집기는 하루 1회만 돌면 된다. markets/items/{id} 가 돌려주는 Stats 는 일 단위
(Date + TradeCount)이므로 매시간 받아도 같은 값이 다시 올 뿐이다. 게다가 응답에는
최근 여러 날치가 한 번에 들어 있고 저장은 combine_first 병합이라, 하루 이틀 놓쳐도
다음 실행에서 자동으로 메워진다. 즉 이 작업은 주기를 늦춰도 데이터 손실이 없다.

아이템 ID 는 data_collector 가 시세를 받으면서 함께 남겨 둔 item_id_map.json 에서
읽는다. 예전에는 ID 를 얻으려고 같은 검색 요청 81건을 처음부터 다시 보냈는데,
그 응답은 data_collector 가 이미 받았던 것과 동일했다.
"""
import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.api_client import LostArkAPI, APIUnavailableError
from economy.collect_guard import append_run_log, kst_now, should_run_job

# 거래량은 일 단위 데이터다. 하루 1회로 충분하고, 크론이 중복으로 떠도 여기서 끊는다.
MIN_GAP_HOURS = 20

# ID 맵이 이보다 오래되면 신뢰하지 않고 다시 매핑한다.
# 신규 아이템 추가나 이름 변경을 반영하기 위한 값이다.
ID_MAP_MAX_AGE_DAYS = 7

LIFE_SKILL_MAP = {
    "식물채집": ["들꽃", "수줍은 들꽃", "화사한 들꽃", "아비도스 들꽃"],
    "벌목": ["목재", "부드러운 목재", "튼튼한 목재", "아비도스 목재"],
    "채광": ["철광석", "묵직한 철광석", "단단한 철광석", "아비도스 철광석"],
    "수렵": ["진귀한 가죽", "두툼한 생고기", "수렵의 결정", "다듬은 생고기", "오레하 두툼한 생고기", "아비도스 두툼한 생고기"],
    "낚시": ["낚시의 결정", "생선", "붉은 살 생선", "오레하 태양 잉어", "아비도스 태양 잉어"],
    "고고학": ["진귀한 유물", "고고학의 결정", "고대 유물", "희귀한 유물", "오레하 유물", "아비도스 유물"],
    "기타": ["견습생용 제작 키트", "숙련가용 제작 키트", "도구 제작 부품", "전문가용 제작 키트", "초보자용 제작 키트", "달인용 제작 키트"],
}

ALL_MATERIALS = [
    "운명의 파편 주머니(대)", "빙하의 숨결", "용암의 숨결", "운명의 돌파석", "위대한 운명의 돌파석",
    "운명의 파괴석", "운명의 파괴석 결정", "운명의 수호석", "운명의 수호석 결정",
    "아비도스 융화 재료", "상급 아비도스 융화 재료", "명예의 파편 주머니(대)",
    "태양의 은총", "태양의 축복", "태양의 가호", "찬란한 명예의 돌파석",
    "정제된 수호강석", "정제된 파괴강석", "최상급 오레하 융화 재료",
    "장인의 재봉술", "장인의 야금술",
    "재봉술 : 업화 [11-14]", "재봉술 : 업화 [15-18]", "재봉술 : 업화 [19-20]",
    "야금술 : 업화 [11-14]", "야금술 : 업화 [15-18]", "야금술 : 업화 [19-20]",
]


def id_map_path():
    return os.path.join(project_root, 'data', 'item_id_map.json')


def load_cached_id_map(max_age_days=ID_MAP_MAX_AGE_DAYS, now=None):
    """저장된 ID 맵을 읽는다. 없거나 오래됐으면 None 을 돌려 재매핑을 유도한다."""
    path = id_map_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            payload = json.load(f)
        items = payload.get('items') or {}
        updated_at = datetime.strptime(payload['updated_at'], '%Y-%m-%d %H:%M')
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"ID 맵을 읽지 못했습니다 ({e}). 다시 매핑합니다.")
        return None

    if not items:
        return None
    now = now or kst_now().replace(tzinfo=None)
    age_days = (now - updated_at).total_seconds() / 86400.0
    if age_days > max_age_days:
        print(f"ID 맵이 {age_days:.1f}일 지났습니다(기준 {max_age_days}일). 다시 매핑합니다.")
        return None

    # 저장은 {이름: ID} 이고 조회는 ID 로 하므로 뒤집는다.
    print(f"ID 맵 재사용: {len(items)}건 ({age_days:.1f}일 전 수집) — 매핑 요청 생략")
    return {item_id: name for name, item_id in items.items()}


def build_id_map(api):
    """ID 맵이 없을 때만 거래소를 다시 훑는다."""
    print("거래소 아이템 ID 매핑 중...")
    target_items = {}

    for items in LIFE_SKILL_MAP.values():
        for name in items:
            data = api.get_market_items(category_code=90000, item_name=name)
            if data and 'Items' in data:
                for item in data['Items']:
                    if name == item['Name']:
                        target_items[item['Id']] = item['Name']

    for name in ALL_MATERIALS:
        data = api.get_market_items(category_code=50000, item_name=name)
        if data and 'Items' in data:
            for item in data['Items']:
                if name in item['Name']:
                    target_items[item['Id']] = item['Name']

    for page in range(1, 20):
        b_data = api.get_market_items(category_code=60000, page_no=page)
        if b_data and 'Items' in b_data and len(b_data['Items']) > 0:
            for item in b_data['Items']:
                target_items[item['Id']] = item['Name']
        else:
            break

    if target_items:
        payload = {
            'updated_at': kst_now().strftime('%Y-%m-%d %H:%M'),
            'items': {name: item_id for item_id, name in target_items.items()},
        }
        try:
            with open(id_map_path(), 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
        except OSError as e:
            print(f"ID 맵 저장 실패: {e}")
    return target_items


def save_volume(all_volume_data):
    df_new = pd.DataFrame(all_volume_data)
    df_wide_new = df_new.pivot_table(index='item_name', columns='Date',
                                     values='TradeCount', aggfunc='last')

    save_path = os.path.join(project_root, 'data', 'market_volume.csv')

    if os.path.exists(save_path):
        df_old = pd.read_csv(save_path).set_index('item_name')
        # 새 값이 우선이다. 오늘치는 하루가 진행 중이라 값이 늘어날 수 있다.
        df_combined = df_wide_new.combine_first(df_old).reset_index()
    else:
        df_combined = df_wide_new.reset_index()

    date_cols = sorted([c for c in df_combined.columns if c != 'item_name'])
    df_combined = df_combined[['item_name'] + date_cols]
    df_combined.to_csv(save_path, index=False, encoding='utf-8-sig')
    return len(df_combined), len(date_cols)


def fetch_daily_volume_wide():
    started_at = kst_now().replace(tzinfo=None)
    data_path = os.path.join(project_root, 'data')

    ok, reason = should_run_job(data_path, 'volume', MIN_GAP_HOURS, now=started_at)
    if not ok:
        print(f"--- 거래량 수집 건너뜀: {reason} ---")
        append_run_log(data_path, 'volume', started_at, 'skipped', note=reason)
        return

    print(f"--- [{started_at:%Y-%m-%d %H:%M} (KST)] 일일 거래량 수집 시작 ({reason}) ---")
    api = LostArkAPI()

    collect_error = None
    all_volume_data = []
    try:
        target_items = load_cached_id_map()
        if target_items is None:
            target_items = build_id_map(api)

        print(f"총 {len(target_items)}개 품목 거래량 조회 시작...")
        for item_id, item_name in target_items.items():
            v_data = api.get_market_item_stats(item_id)
            if v_data and len(v_data) > 0 and 'Stats' in v_data[0]:
                for stat in v_data[0]['Stats']:
                    all_volume_data.append({
                        'Date': stat['Date'],
                        'item_name': item_name,
                        'TradeCount': stat['TradeCount'],
                    })
    except APIUnavailableError as e:
        # 여기까지 받은 거래량은 버리지 않는다.
        collect_error = e
        print(f"\n수집 중단: {e}")

    if not all_volume_data:
        note = str(collect_error) if collect_error else '수집된 데이터 없음'
        append_run_log(data_path, 'volume', started_at, 'failed', api.request_count, 0, note)
        print("수집된 데이터가 없습니다.")
        raise SystemExit(1)

    rows, date_cols = save_volume(all_volume_data)
    print(f"저장 완료: {rows}개 품목 × {date_cols}개 날짜")

    if collect_error is not None:
        append_run_log(data_path, 'volume', started_at, 'failed',
                       api.request_count, rows, str(collect_error))
        raise SystemExit(1)

    append_run_log(data_path, 'volume', started_at, 'success', api.request_count, rows, reason)
    print(f"모든 작업 완료. (요청 {api.request_count}건)")


if __name__ == "__main__":
    fetch_daily_volume_wide()

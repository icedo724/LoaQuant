import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.api_client import LostArkAPI, APIUnavailableError
from common.db_connector import get_db_engine
from economy.collect_guard import append_run_log, kst_now, should_collect


def ensure_data_dir():
    data_path = os.path.join(project_root, 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path


def get_korea_time_str():
    utc_now = datetime.now(timezone.utc)
    kst_now = utc_now + timedelta(hours=9)
    return kst_now.strftime('%Y-%m-%d %H:%M')


def save_item_id_map(id_map):
    """수집 중에 만난 (아이템명 -> 거래소 ID) 를 저장한다.

    volume_collector 는 거래량을 조회하려면 아이템 ID 가 필요한데, 지금은 그 ID 를
    얻으려고 같은 검색 요청 81건을 처음부터 다시 보낸다. 여기서 만난 응답에 이미
    Id 가 들어 있으므로, 버리지 말고 남겨 두면 그 81건이 통째로 사라진다.
    """
    if not id_map:
        return
    path = os.path.join(ensure_data_dir(), 'item_id_map.json')
    payload = {
        'updated_at': kst_now().strftime('%Y-%m-%d %H:%M'),
        'items': id_map,
    }
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"   -> [ID 맵 저장] {len(id_map)}건")
    except OSError as e:
        print(f"   -> [Warn] ID 맵 저장 실패: {e}")


def update_wide_csv(new_data_list, file_name, current_time_col, category_col=None):
    data_path = ensure_data_dir()
    full_path = os.path.join(data_path, file_name)

    current_df = pd.DataFrame(new_data_list)
    if current_df.empty:
        return
    merge_keys = ['item_name']
    cols_to_keep = ['item_name', 'current_min_price']

    if category_col and category_col in current_df.columns:
        merge_keys.append(category_col)
        cols_to_keep.insert(1, category_col)

    current_df = current_df.drop_duplicates(subset=merge_keys)
    mini_df = current_df[cols_to_keep].copy()

    mini_df.rename(columns={'current_min_price': current_time_col}, inplace=True)

    if os.path.exists(full_path):
        try:
            old_df = pd.read_csv(full_path)
            actual_merge_keys = [k for k in merge_keys if k in old_df.columns]

            merged_df = pd.merge(old_df, mini_df, on=actual_merge_keys, how='outer')
            merged_df.to_csv(full_path, index=False, encoding='utf-8-sig')
            print(f"   -> [파일 저장] {file_name}")
        except Exception as e:
            print(f"   -> [Error] 병합 실패 ({file_name}): {e}")
    else:
        mini_df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"   -> [신규 생성] {file_name}")


def collect_market_data():
    started_at = kst_now().replace(tzinfo=None)
    data_path = ensure_data_dir()

    # 크론 시도를 시간당 3회로 늘린 만큼, 짧은 간격의 중복 실행은 여기서 끊는다.
    ok, reason = should_collect(os.path.join(data_path, 'market_materials.csv'), now=started_at)
    if not ok:
        print(f"--- 수집 건너뜀: {reason} ---")
        append_run_log(data_path, 'prices', started_at, 'skipped', note=reason)
        return

    api = LostArkAPI()
    engine = get_db_engine()

    now_str = get_korea_time_str()
    print(f"--- [{now_str} (KST)] 데이터 수집 시작 ({reason}) ---")

    id_map = {}
    data_materials = []
    data_lifeskill = []
    data_battle = []
    data_engravings = []
    data_gems = []

    # API 가 응답하지 않아 수집이 중단되더라도, 여기까지 모은 데이터는 버리지 않는다.
    # 예전에는 예외가 그대로 올라가 저장 단계에 도달하지 못했고, 성공적으로 받아둔
    # 시세까지 통째로 사라졌다.
    collect_error = None
    try:

        # ---------------------------------------------------------
        # 1. 생활 재료
        # ---------------------------------------------------------
        life_skill_map = {
            "식물채집": ["들꽃", "수줍은 들꽃", "화사한 들꽃", "아비도스 들꽃"],
            "벌목": ["목재", "부드러운 목재", "튼튼한 목재", "아비도스 목재"],
            "채광": ["철광석", "묵직한 철광석", "단단한 철광석", "아비도스 철광석"],
            "수렵": ["진귀한 가죽", "두툼한 생고기", "수렵의 결정", "다듬은 생고기", "오레하 두툼한 생고기", "아비도스 두툼한 생고기"],
            "낚시": ["낚시의 결정", "생선", "붉은 살 생선", "오레하 태양 잉어", "아비도스 태양 잉어"],
            "고고학": ["진귀한 유물", "고고학의 결정", "고대 유물", "희귀한 유물", "오레하 유물", "아비도스 유물"],
            "기타": ["견습생용 제작 키트", "숙련가용 제작 키트", "도구 제작 부품", "전문가용 제작 키트", "초보자용 제작 키트", "달인용 제작 키트"]
        }

        print(f"\n[생활 재료] 수집 중")
        for category, items in life_skill_map.items():
            for name in items:
                data = api.get_market_items(category_code=90000, item_name=name)
                if data and 'Items' in data:
                    for item in data['Items']:
                        if name == item['Name']:
                            id_map[item['Name']] = item['Id']
                            data_lifeskill.append({
                                'item_name': item['Name'],
                                'sub_category': category,
                                'item_grade': item['Grade'],
                                'item_tier': 3,
                                'current_min_price': item['CurrentMinPrice'],
                                'collected_at': datetime.now()
                            })

        # ---------------------------------------------------------
        # 2. 강화 재료 (T4/T3)
        # ---------------------------------------------------------
        items_t4 = [
            # 기본 재료
            "운명의 파편 주머니(대)", "빙하의 숨결", "용암의 숨결",
            # [그룹 1] 돌파석
            "운명의 돌파석", "위대한 운명의 돌파석",
            # [그룹 2] 파괴석
            "운명의 파괴석", "운명의 파괴석 결정",
            # [그룹 3] 수호석
            "운명의 수호석", "운명의 수호석 결정",
            # [그룹 4] 융화 재료
            "아비도스 융화 재료", "상급 아비도스 융화 재료"
        ]

        items_t3 = [
            "명예의 파편 주머니(대)", "태양의 은총", "태양의 축복", "태양의 가호",
            # [교환 대상]
            "찬란한 명예의 돌파석",
            "정제된 수호강석",
            "정제된 파괴강석",
            "최상급 오레하 융화 재료"
        ]

        items_special = [
            "장인의 재봉술",
            "장인의 야금술",
            "재봉술 : 업화 [11-14]",
            "재봉술 : 업화 [15-18]",
            "재봉술 : 업화 [19-20]",
            "야금술 : 업화 [11-14]",
            "야금술 : 업화 [15-18]",
            "야금술 : 업화 [19-20]"
        ]

        def fetch_market_items(target_list, result_list, category_code=50000, tier_val=None):
            print(f"\n[강화 재료] 수집 중 ({target_list[0]} 등)")
            for name in target_list:
                data = api.get_market_items(category_code, item_name=name, item_tier=tier_val)
                if data and 'Items' in data:
                    for item in data['Items']:
                        if name in item['Name']:
                            id_map[item['Name']] = item['Id']
                            result_list.append({
                                'item_name': item['Name'],
                                'item_grade': item['Grade'],
                                'item_tier': tier_val if tier_val else 3,
                                'current_min_price': item['CurrentMinPrice'],
                                'collected_at': datetime.now()
                            })

        fetch_market_items(items_t4, data_materials, 50000, 4)
        fetch_market_items(items_t3, data_materials, 50000, 3)
        fetch_market_items(items_special, data_materials, 50000, None)

        # ---------------------------------------------------------
        # 3. 배틀 아이템
        # ---------------------------------------------------------
        print(f"\n[배틀 아이템] 수집 중")
        # 배틀 아이템(Category: 60000) 전체 페이지 순회
        for page in range(1, 20):
            b_data = api.get_market_items(category_code=60000, page_no=page)

            if b_data and 'Items' in b_data and len(b_data['Items']) > 0:
                for item in b_data['Items']:
                    id_map[item['Name']] = item['Id']
                    data_battle.append({
                        'item_name': item['Name'],
                        'current_min_price': item['CurrentMinPrice'],
                        'collected_at': datetime.now()
                    })
            else:
                break

        # ---------------------------------------------------------
        # 4. 각인서
        # ---------------------------------------------------------
        print(f"\n[각인서] 수집 중")
        for page in range(1, 11):
            eng_data = api.get_market_items(40000, item_grade="유물", page_no=page, sort_condition="DESC")
            if eng_data and 'Items' in eng_data:
                for item in eng_data['Items']:
                    data_engravings.append({
                        'item_name': item['Name'],
                        'item_grade': item['Grade'],
                        'item_tier': 3,
                        'current_min_price': item['CurrentMinPrice'],
                        'collected_at': datetime.now()
                    })
            else:
                break

        # ---------------------------------------------------------
        # 5. 보석 (T4 8~10레벨)
        # ---------------------------------------------------------
        target_gems = [
            "8레벨 겁화의 보석", "9레벨 겁화의 보석", "10레벨 겁화의 보석",
            "8레벨 작열의 보석", "9레벨 작열의 보석", "10레벨 작열의 보석"
        ]
        print(f"\n[보석] 경매장 시세 수집 중")
        for gem_name in target_gems:
            data = api.get_auction_items(category_code=210000, item_name=gem_name, item_tier=4)
            if data and 'Items' in data:
                min_price = None
                for auction_item in data['Items']:
                    buy_price = auction_item.get('AuctionInfo', {}).get('BuyPrice')
                    if buy_price:
                        if min_price is None or buy_price < min_price:
                            min_price = buy_price

                if min_price:
                    data_gems.append({
                        'item_name': gem_name,
                        'item_grade': '고대',
                        'item_tier': 4,
                        'current_min_price': min_price,
                        'collected_at': datetime.now()
                    })

    except APIUnavailableError as e:
        collect_error = e
        print(f"\n수집 중단: {e}")

    # ---------------------------------------------------------
    # 6. 저장 (DB & CSV)
    # ---------------------------------------------------------

    # CSV 저장
    print("\nCSV 파일 업데이트")
    if data_materials: update_wide_csv(data_materials, "market_materials.csv", now_str)
    if data_lifeskill: update_wide_csv(data_lifeskill, "market_lifeskill.csv", now_str, category_col="sub_category")
    if data_battle: update_wide_csv(data_battle, "market_battleitems.csv", now_str)
    if data_engravings: update_wide_csv(data_engravings, "market_engravings.csv", now_str)
    if data_gems: update_wide_csv(data_gems, "market_gems.csv", now_str)

    # DB 저장
    all_rows = data_materials + data_lifeskill + data_battle + data_engravings + data_gems
    if all_rows and engine:
        try:
            df_db = pd.DataFrame(all_rows)
            df_db.to_sql(name='market_prices', con=engine, if_exists='append', index=False)
            print(f"\nDB 저장 완료: 총 {len(df_db)}건")
        except Exception as e:
            print(f"DB 저장 실패: {e}")

    save_item_id_map(id_map)

    collected = len(all_rows)
    if collect_error is not None:
        append_run_log(data_path, 'prices', started_at, 'failed', api.request_count, collected, str(collect_error))
        print(f"\n부분 저장 완료 ({collected}건). 수집은 실패로 종료한다.")
        # 부분 데이터도 커밋되도록 워크플로의 커밋 스텝은 if: always() 로 둔다.
        # 다만 실행 자체는 실패로 표시해 원인을 드러낸다.
        raise SystemExit(1)

    append_run_log(data_path, 'prices', started_at, 'success', api.request_count, collected, reason)
    print(f"\n모든 작업 완료. (요청 {api.request_count}건, 수집 {collected}건)")


if __name__ == "__main__":
    collect_market_data()
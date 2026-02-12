import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta, timezone

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.api_client import LostArkAPI
from common.db_connector import get_db_engine


def ensure_data_dir():
    data_path = os.path.join(project_root, 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path


def get_korea_time_str():
    utc_now = datetime.now(timezone.utc)
    kst_now = utc_now + timedelta(hours=9)
    return kst_now.strftime('%Y-%m-%d %H:%M')


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
    api = LostArkAPI()
    engine = get_db_engine()

    now_str = get_korea_time_str()
    print(f"--- [{now_str} (KST)] 데이터 수집 시작 ---")

    data_materials = []
    data_lifeskill = []
    data_battle = []
    data_engravings = []
    data_gems = []

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
                        data_lifeskill.append({
                            'item_name': item['Name'],
                            'sub_category': category,
                            'item_grade': item['Grade'],
                            'item_tier': 3,
                            'current_min_price': item['CurrentMinPrice'],
                            'collected_at': datetime.now()
                        })
            time.sleep(0.12)

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
    items_special = ["장인의 재봉술", "장인의 야금술"]

    def fetch_market_items(target_list, result_list, category_code=50000, tier_val=None):
        print(f"\n[강화 재료] 수집 중 ({target_list[0]} 등)")
        for name in target_list:
            data = api.get_market_items(category_code, item_name=name, item_tier=tier_val)
            if data and 'Items' in data:
                for item in data['Items']:
                    if name in item['Name']:
                        result_list.append({
                            'item_name': item['Name'],
                            'item_grade': item['Grade'],
                            'item_tier': tier_val if tier_val else 3,
                            'current_min_price': item['CurrentMinPrice'],
                            'collected_at': datetime.now()
                        })
            time.sleep(0.12)

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
                data_battle.append({
                    'item_name': item['Name'],
                    'current_min_price': item['CurrentMinPrice'],
                    'collected_at': datetime.now()
                })
            time.sleep(0.12)
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
            time.sleep(0.2)
        else:
            break

    # ---------------------------------------------------------
    # 5. 보석 (T4 8~10레벨)
    # ---------------------------------------------------------
    target_gems = [
        "8레벨 겁화의 보석", "9레벨 겁화의 보석", "10레벨 겁화의 보석",
        "8레벨 작열의 보석", "9레벨 작열의 보석", "10레벨 작열의 보석"
    ]
    print(f"\[보석] 경매장 시세 수집 중")
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
        time.sleep(0.3)

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

    print("\n모든 작업 완료.")


if __name__ == "__main__":
    collect_market_data()
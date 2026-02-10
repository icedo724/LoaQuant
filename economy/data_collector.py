import sys
import os
import time
import pandas as pd
from datetime import datetime

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


def update_wide_csv(new_data_list, file_name, current_time_col):
    data_path = ensure_data_dir()
    full_path = os.path.join(data_path, file_name)

    current_df = pd.DataFrame(new_data_list)
    if current_df.empty:
        return

    current_df = current_df.drop_duplicates(subset=['item_name'])

    mini_df = current_df[['item_name', 'current_min_price']].copy()
    mini_df.rename(columns={'current_min_price': current_time_col}, inplace=True)

    if os.path.exists(full_path):
        try:
            old_df = pd.read_csv(full_path)
            merged_df = pd.merge(old_df, mini_df, on='item_name', how='outer')
            merged_df.to_csv(full_path, index=False, encoding='utf-8-sig')
            print(f"   -> 덧붙이기 성공: {file_name} (컬럼 추가: {current_time_col})")
        except Exception as e:
            print(f"   -> CSV 병합 실패: {e}")
    else:
        mini_df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"   -> 신규 생성: {file_name}")


def collect_market_data():
    api = LostArkAPI()
    engine = get_db_engine()

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"--- [{now_str}] 데이터 수집 시작 ---")

    materials_data = []  # 강화재료
    engravings_data = []  # 각인서

    items_t4 = ["운명의 파편 주머니(대)", "아비도스 융화 재료", "운명의 돌파석", "운명의 수호석", "운명의 파괴석", "빙하의 숨결", "용암의 숨결"]
    items_t3 = ["명예의 파편 주머니(대)", "최상급 오레하 융화 재료", "찬란한 명예의 돌파석", "정제된 수호강석", "정제된 파괴강석", "태양의 은총", "태양의 축복", "태양의 가호"]
    items_special = ["장인의 재봉술", "장인의 야금술"]

    def fetch_materials(item_list, tier_val=None):
        print(f"\n재료 수집 중 (Target: {item_list[0]} 등)")
        for name in item_list:
            data = api.get_market_items(category_code=50000, item_name=name, item_tier=tier_val)
            if data and 'Items' in data and len(data['Items']) > 0:
                for item in data['Items']:
                    if name in item['Name']:
                        materials_data.append({
                            'item_name': item['Name'],
                            'item_grade': item['Grade'],
                            'item_tier': tier_val if tier_val else 3,
                            'current_min_price': item['CurrentMinPrice'],
                            'recent_price': item['RecentPrice'],
                            'yday_avg_price': item['YDayAvgPrice'],
                            'bundle_count': item['BundleCount'],
                            'collected_at': datetime.now()
                        })
                        print(f"   -> {item['Name']}: {item['CurrentMinPrice']} G")
            time.sleep(0.15)

    fetch_materials(items_t4, tier_val=4)
    fetch_materials(items_t3, tier_val=3)
    fetch_materials(items_special, tier_val=None)

    print(f"\n🔍 [유물 각인서] 수집 중")
    for page in range(1, 11):
        engraving_data = api.get_market_items(
            category_code=40000, item_grade="유물", page_no=page, sort_condition="DESC"
        )
        if engraving_data and 'Items' in engraving_data and len(engraving_data['Items']) > 0:
            for item in engraving_data['Items']:
                engravings_data.append({
                    'item_name': item['Name'],
                    'item_grade': item['Grade'],
                    'item_tier': 3,
                    'current_min_price': item['CurrentMinPrice'],
                    'recent_price': item['RecentPrice'],
                    'yday_avg_price': item['YDayAvgPrice'],
                    'bundle_count': item['BundleCount'],
                    'collected_at': datetime.now()
                })
            print(f"   -> Page {page} 완료")
            time.sleep(0.2)
        else:
            break

    all_rows = materials_data + engravings_data
    if all_rows and engine:
        try:
            df_db = pd.DataFrame(all_rows)
            df_db.to_sql(name='market_prices', con=engine, if_exists='append', index=False)
            print(f"\nDB 저장 완료: 총 {len(df_db)}건")
        except Exception as e:
            print(f"DB 저장 실패: {e}")

    print("\nCSV 파일 업데이트 중")

    if materials_data:
        update_wide_csv(materials_data, "market_materials.csv", now_str)

    if engravings_data:
        update_wide_csv(engravings_data, "market_engravings.csv", now_str)

    print("\n모든 작업 완료.")


if __name__ == "__main__":
    collect_market_data()
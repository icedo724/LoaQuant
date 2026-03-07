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


def fetch_daily_volume_wide():
    print(f"--- 일일 거래량 데이터 자동 수집 (모듈 통합 버전) ---")

    api = LostArkAPI()
    engine = get_db_engine()
    target_items = {}

    print("🔍 거래소 실제 아이템 ID 매핑 중...")

    life_skill_map = {"식물채집": ["들꽃", "수줍은 들꽃", "화사한 들꽃", "아비도스 들꽃"], "벌목": ["목재", "부드러운 목재", "튼튼한 목재", "아비도스 목재"],
                      "채광": ["철광석", "묵직한 철광석", "단단한 철광석", "아비도스 철광석"],
                      "수렵": ["진귀한 가죽", "두툼한 생고기", "수렵의 결정", "다듬은 생고기", "오레하 두툼한 생고기", "아비도스 두툼한 생고기"],
                      "낚시": ["낚시의 결정", "생선", "붉은 살 생선", "오레하 태양 잉어", "아비도스 태양 잉어"],
                      "고고학": ["진귀한 유물", "고고학의 결정", "고대 유물", "희귀한 유물", "오레하 유물", "아비도스 유물"],
                      "기타": ["견습생용 제작 키트", "숙련가용 제작 키트", "도구 제작 부품", "전문가용 제작 키트", "초보자용 제작 키트", "달인용 제작 키트"]}
    for cat, items in life_skill_map.items():
        for name in items:
            data = api.get_market_items(category_code=90000, item_name=name)
            if data and 'Items' in data:
                for item in data['Items']:
                    if name == item['Name']: target_items[item['Id']] = item['Name']
            time.sleep(0.12)

    all_materials = ["운명의 파편 주머니(대)", "빙하의 숨결", "용암의 숨결", "운명의 돌파석", "위대한 운명의 돌파석", "운명의 파괴석", "운명의 파괴석 결정", "운명의 수호석",
                     "운명의 수호석 결정", "아비도스 융화 재료", "상급 아비도스 융화 재료", "명예의 파편 주머니(대)", "태양의 은총", "태양의 축복", "태양의 가호",
                     "찬란한 명예의 돌파석", "정제된 수호강석", "정제된 파괴강석", "최상급 오레하 융화 재료", "장인의 재봉술", "장인의 야금술", "재봉술 : 업화 [11-14]",
                     "재봉술 : 업화 [15-18]", "재봉술 : 업화 [19-20]", "야금술 : 업화 [11-14]", "야금술 : 업화 [15-18]", "야금술 : 업화 [19-20]"]
    for name in all_materials:
        data = api.get_market_items(category_code=50000, item_name=name)
        if data and 'Items' in data:
            for item in data['Items']:
                if name in item['Name']: target_items[item['Id']] = item['Name']
        time.sleep(0.12)

    for page in range(1, 20):
        b_data = api.get_market_items(category_code=60000, page_no=page)
        if b_data and 'Items' in b_data and len(b_data['Items']) > 0:
            for item in b_data['Items']: target_items[item['Id']] = item['Name']
            time.sleep(0.12)
        else:
            break

    print(f"✅ 매핑 완료! 총 {len(target_items)}개 품목 거래량 조회 시작...")

    all_volume_data = []
    for item_id, item_name in target_items.items():

        v_data = api.get_market_item_stats(item_id)

        if v_data and len(v_data) > 0 and 'Stats' in v_data[0]:
            for stat in v_data[0]['Stats']:
                all_volume_data.append({
                    'Date': stat['Date'],
                    'item_name': item_name,
                    'TradeCount': stat['TradeCount']
                })
        time.sleep(0.12)

    if not all_volume_data:
        print("⚠️ 수집된 데이터가 없습니다.")
        return

    df_new = pd.DataFrame(all_volume_data)
    df_wide_new = df_new.pivot(index='item_name', columns='Date', values='TradeCount')

    save_path = os.path.join(project_root, 'data', 'market_volume.csv')

    if os.path.exists(save_path):
        df_old = pd.read_csv(save_path)
        df_old = df_old.set_index('item_name')

        df_combined = df_wide_new.combine_first(df_old).reset_index()

        date_cols = sorted([c for c in df_combined.columns if c != 'item_name'])
        df_combined = df_combined[['item_name'] + date_cols]

        df_combined.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"📁 기존 파일에 새로운 날짜 병합 완료!")
    else:
        df_wide_new = df_wide_new.reset_index()
        date_cols = sorted([c for c in df_wide_new.columns if c != 'item_name'])
        df_wide_new = df_wide_new[['item_name'] + date_cols]
        df_wide_new.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"📁 신규 파일 생성 완료!")


if __name__ == "__main__":
    fetch_daily_volume_wide()
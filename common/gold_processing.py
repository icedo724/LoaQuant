import pandas as pd
import re
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)

input_path = os.path.join(project_root, 'data', 'gold', 'chatlog.csv')
save_path = os.path.join(project_root, 'data', 'gold', 'daily_gold.csv')


def process_gold_prices(input_file, output_file):
    print("데이터 전처리를 시작합니다...")

    if not os.path.exists(input_file):
        print(f"❌ 에러: '{input_file}' 파일을 찾을 수 없습니다.")
        print(f"👉 '{input_file}' 경로에 파일이 있는지 확인해 주세요!")
        return

    df = pd.read_csv(input_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')

    df['Date'] = df['Timestamp'].dt.tz_convert('Asia/Seoul').dt.date

    pattern = r'100\s*[:대/\-;|l]?\s*(\d{2})'
    df['Gold_Price'] = df['Content'].str.extract(pattern)[0].astype(float)

    df_clean = df.dropna(subset=['Gold_Price'])
    print(f"총 {len(df)}개 메시지 중 {len(df_clean)}개 가격 추출 성공!")

    daily_mode = df_clean.groupby('Date')['Gold_Price'].apply(lambda x: x.mode().mean()).reset_index()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    daily_mode.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ [{output_file}] 파일로 완벽하게 정리되었습니다!")

if __name__ == "__main__":
    process_gold_prices(input_path, save_path)
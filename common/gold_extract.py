import json
import pandas as pd
import os


def extract_discord_messages_from_har(har_file_path):
    print("데이터 추출을 시작합니다. 잠시만 기다려주세요...")

    with open(har_file_path, 'r', encoding='utf-8') as f:
        har_data = json.load(f)

    extracted_data = []

    for entry in har_data['log']['entries']:
        request_url = entry['request']['url']

        if '/api/v9/channels/' in request_url and '/messages' in request_url:
            if 'response' in entry and 'content' in entry['response'] and 'text' in entry['response']['content']:
                response_text = entry['response']['content']['text']

                try:
                    messages = json.loads(response_text)

                    for msg in messages:
                        if 'content' in msg and msg['content'].strip() != "":
                            extracted_data.append({
                                'Timestamp': msg['timestamp'],
                                'Author': msg['author']['username'],
                                'Content': msg['content']
                            })
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(extracted_data)

    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
        df = df.sort_values(by='Timestamp')
        df = df.drop_duplicates(subset=['Timestamp', 'Author', 'Content'])
        df = df.reset_index(drop=True)

    return df

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)

har_path = os.path.join(project_root, 'data', 'gold', 'discord_data.har')
save_name = os.path.join(project_root, 'data', 'gold', 'chatlog.csv')


def main(har_file_path=har_path, output_path=save_name):
    if not os.path.exists(har_file_path):
        print(f"에러: '{har_file_path}' 파일을 찾을 수 없습니다.")
        return 1

    df_chat = extract_discord_messages_from_har(har_file_path)

    print(f"\n총 {len(df_chat)}개의 메시지를 추출했습니다")
    print("-" * 50)
    print(df_chat.head())
    print("-" * 50)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_chat.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[{output_path}] 파일로 저장이 완료되었습니다")
    return 0


# 모듈 최상위에서 실행하면 import 만 해도 HAR 파싱과 CSV 쓰기가 일어난다.
# gold_processing.py 와 규약을 맞춰 진입점을 가드 안에 둔다.
if __name__ == "__main__":
    raise SystemExit(main())
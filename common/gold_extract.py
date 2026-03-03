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

if not os.path.exists(har_path):
    print(f"❌ 에러: '{har_path}' 파일을 찾을 수 없습니다.")
else:
    df_chat = extract_discord_messages_from_har(har_path)

    print(f"\n✅ 총 {len(df_chat)}개의 메시지를 성공적으로 추출했습니다")
    print("-" * 50)
    print(df_chat.head())
    print("-" * 50)

    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    df_chat.to_csv(save_name, index=False, encoding='utf-8-sig')
    print(f"✅ [{save_name}] 파일로 저장이 완료되었습니다")
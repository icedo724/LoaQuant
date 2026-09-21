"""Discord 채팅 로그에서 골드-현금 환율(100골드당 원)을 뽑아 일별로 정리한다.

이 경로는 사람이 브라우저에서 HAR 을 직접 뽑아야 시작되므로 자동화되어 있지 않다.
(2026-07-01 이후 82일간 멈춰 있었다.) 자동 갱신 출처가 정해지기 전까지의 임시 경로이며,
아래 세 가지를 고쳐 두었다.

1. '하루'의 정의를 로아 서버 기준(오전 6시)으로 맞췄다. 예전에는 자정 KST 로 끊어서,
   시세 쪽 일평균(오전 6시 기준)과 조인할 때 00:00~06:00 구간에 다른 날짜의 환율이
   적용됐다.
2. 정규식이 두 자리 수만 잡아 100:9 를 놓치고 100:105 를 10 으로 잘못 읽었다.
3. 일별 대표값을 최빈값에서 중앙값으로 바꿨다. 최빈값은 1원 격자로 양자화되어
   (현재 수준 12에서 한 칸이 8.3%) 미세한 추세가 통째로 사라진다.

출력에는 source / collected_at / sample_n 컬럼을 덧붙인다. 출처를 바꿀 때 과거
데이터를 버리지 않기 위해서다. Date / Gold_Price 컬럼명은 읽는 쪽 호환을 위해 유지한다.
"""
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

input_path = os.path.join(project_root, 'data', 'gold', 'chatlog.csv')
save_path = os.path.join(project_root, 'data', 'gold', 'daily_gold.csv')

# 로아 서버는 오전 6시에 날짜가 바뀐다.
LOA_DAY_OFFSET_HOURS = 6

# "100:12", "100대 12", "100 12", "100:11.5" 를 잡는다.
# (?<!\d) 와 (?!\d) 로 더 긴 숫자의 일부를 잘라 읽는 것을 막는다.
# 예전 패턴 r'100\s*[:대/\-;|l]?\s*(\d{2})' 은 "가격 100000 원"에서 00 을 뽑았다.
GOLD_PATTERN = re.compile(r'(?<!\d)100\s*(?:[:대/\-;|lI]\s*)?(\d{1,3}(?:\.\d+)?)(?!\d)')

# 명백한 오추출을 거른다. 범위를 좁게 잡으면 시세가 실제로 움직였을 때 조용히
# 전부 버리게 되므로 넉넉히 두고, 걸러진 건수를 출력해 드러낸다.
MIN_VALID_RATE = 1.0
MAX_VALID_RATE = 200.0

SOURCE_LABEL = 'discord_har_median'


def extract_rate(text):
    """문자열에서 환율을 뽑는다. 못 찾거나 범위를 벗어나면 None."""
    if not isinstance(text, str):
        return None
    m = GOLD_PATTERN.search(text)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    if not (MIN_VALID_RATE <= value <= MAX_VALID_RATE):
        return None
    return value


def process_gold_prices(input_file, output_file):
    print("데이터 전처리를 시작합니다...")

    if not os.path.exists(input_file):
        print(f"에러: '{input_file}' 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(input_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')

    # 로아 서버 기준으로 날을 끊는다. 시세 쪽 일평균과 같은 규약이어야 조인이 맞는다.
    kst = df['Timestamp'].dt.tz_convert('Asia/Seoul')
    df['Date'] = (kst - pd.Timedelta(hours=LOA_DAY_OFFSET_HOURS)).dt.date

    df['Gold_Price'] = df['Content'].map(extract_rate)

    df_clean = df.dropna(subset=['Gold_Price'])
    matched_any = df['Content'].astype(str).str.contains(r'(?<!\d)100(?!\d)', regex=True, na=False).sum()
    print(f"총 {len(df)}개 메시지 중 {len(df_clean)}개 가격 추출 성공 "
          f"('100' 포함 메시지 {matched_any}개 중 {matched_any - len(df_clean)}개는 범위 밖이거나 형식 불일치)")

    if df_clean.empty:
        print("추출된 가격이 없어 저장하지 않습니다.")
        return

    grouped = df_clean.groupby('Date')['Gold_Price']
    daily = grouped.median().reset_index()
    daily['sample_n'] = grouped.size().values
    daily['source'] = SOURCE_LABEL
    daily['collected_at'] = datetime.now(timezone.utc).astimezone(
        timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M')

    daily = daily[['Date', 'Gold_Price', 'source', 'collected_at', 'sample_n']]
    daily = daily.sort_values('Date').reset_index(drop=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    daily.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[{output_file}] 저장 완료: {len(daily)}일치 "
          f"({daily['Date'].iloc[0]} ~ {daily['Date'].iloc[-1]})")


if __name__ == "__main__":
    process_gold_prices(input_path, save_path)

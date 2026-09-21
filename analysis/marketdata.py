"""시세·거래량·이벤트·골드 데이터 로딩.

대시보드와 분석 모듈이 같은 파일을 각자의 방식으로 읽던 것을 한 곳으로 모은다.
이전에는 이벤트 로그 파서가 둘(`dashboard.load_event_logs` 는 검증 없는
`split(":")`, `patch_impact.load_event_log` 는 정규식)이었고 반환 타입도
달랐다.

여기서 나오는 시계열은 모두 **KST naive** 인덱스를 갖는 시간 단위 원본이다.
일 단위 집계는 :mod:`common.timeaxis` 가 담당한다. 이 모듈은 "하루"를
정의하지 않는다.
"""
from __future__ import annotations

import os
import re

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 상대 경로를 쓰면 저장소 루트에서 실행할 때만 동작한다. 항상 절대 경로로 푼다.
MARKET_FILES = {
    "materials": "market_materials.csv",
    "engravings": "market_engravings.csv",
    "gems": "market_gems.csv",
    "lifeskill": "market_lifeskill.csv",
    "battleitems": "market_battleitems.csv",
}

CATEGORY_KR = {
    "materials": "강화 재료",
    "engravings": "각인서",
    "gems": "보석",
    "lifeskill": "생활 재료",
    "battleitems": "배틀 아이템",
}

VOLUME_FILE = "market_volume.csv"
GOLD_FILE = os.path.join("gold", "daily_gold.csv")
EVENT_FILE = "event_log.txt"

# wide CSV 에서 품목을 식별하는 열. 나머지 열은 전부 수집 시각이다.
_META_COLUMNS = ("item_name", "sub_category")


def _path(*parts: str) -> str:
    return os.path.join(DATA_DIR, *parts)


# ---------------------------------------------------------------------------
# 시세 (wide CSV)
# ---------------------------------------------------------------------------
def load_wide(name: str) -> pd.DataFrame | None:
    """wide CSV 를 원본 그대로 읽는다. 없으면 None."""
    path = _path(MARKET_FILES.get(name, name))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def to_timeseries(df: pd.DataFrame | None, items=None) -> pd.DataFrame:
    """wide CSV → 시각 인덱스 · 품목 컬럼 DataFrame.

    컬럼명이 수집 시각(KST 문자열)이므로 전치해서 인덱스로 올린다. 시각으로
    해석되지 않는 컬럼은 메타 정보이므로 버린다.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    frame = df
    if items is not None:
        frame = frame[frame["item_name"].isin(list(items))]
    if frame.empty:
        return pd.DataFrame()

    drop = [c for c in _META_COLUMNS if c in frame.columns and c != "item_name"]
    frame = frame.drop(columns=drop).set_index("item_name")

    out = frame.T
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notna()].sort_index()
    return out.astype("float64")


def load_market(name: str, items=None) -> pd.DataFrame:
    """카테고리 하나를 시각 인덱스 시계열로."""
    return to_timeseries(load_wide(name), items)


def load_all_markets() -> dict[str, pd.DataFrame]:
    """전 카테고리를 시각 인덱스 시계열로. 비어 있는 카테고리는 제외."""
    out = {}
    for name in MARKET_FILES:
        ts = load_market(name)
        if not ts.empty:
            out[name] = ts
    return out


def sub_categories(name: str = "lifeskill") -> pd.Series | None:
    """품목명 → 하위 카테고리 매핑 (생활 재료 탭용)."""
    df = load_wide(name)
    if df is None or "sub_category" not in df.columns:
        return None
    return df.set_index("item_name")["sub_category"]


def find_item(item: str, markets: dict[str, pd.DataFrame]) -> pd.Series | None:
    """여러 카테고리에서 품목 하나의 시계열을 찾는다."""
    for df in markets.values():
        if item in df.columns:
            s = df[item].dropna()
            if not s.empty:
                return s
    return None


def items_of(category: str, markets: dict[str, pd.DataFrame]) -> list[str]:
    df = markets.get(category)
    return sorted(df.columns.tolist()) if df is not None else []


# ---------------------------------------------------------------------------
# 거래량
# ---------------------------------------------------------------------------
def load_volume(items=None) -> pd.DataFrame:
    """일별 거래량. 인덱스는 **API 가 주는 자정 기준 날짜**다.

    시세의 게임일(06시 기준)과 경계가 다르다. 둘을 곱하는 쪽에서 어느 기준에
    맞출지 명시적으로 결정해야 한다(:func:`align_price_volume` 참조).
    """
    path = _path(VOLUME_FILE)
    if not os.path.exists(path):
        return pd.DataFrame()
    return to_timeseries(pd.read_csv(path, encoding="utf-8-sig"), items)


def align_price_volume(price_daily: pd.DataFrame, volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """게임일 단가와 자정 기준 거래량을 같은 축에 맞춘다.

    거래량은 API 가 자정 기준으로 집계해 주므로 우리가 06시 기준으로 다시
    나눌 수 없다. 그래서 **거래량 쪽 날짜를 기준으로 삼고** 단가를 그 날짜에
    맞춘다. 게임일 D 의 단가(D 06:00 ~ D+1 06:00)를 달력일 D 에 대응시키는
    근사이며, 실측 괴리는 중앙값 1% 미만이다.

    두 축이 다르다는 사실 자체를 호출부가 잊지 않도록 함수로 고정해 둔다.
    """
    if price_daily.empty or volume.empty:
        return pd.DataFrame(), pd.DataFrame()

    common_items = [c for c in price_daily.columns if c in volume.columns]
    if not common_items:
        return pd.DataFrame(), pd.DataFrame()

    p = price_daily[common_items]
    v = volume[common_items]
    idx = p.index.intersection(v.index)
    return p.loc[idx], v.loc[idx]


# ---------------------------------------------------------------------------
# 이벤트 로그 — 파서는 이것 하나뿐이다
# ---------------------------------------------------------------------------
_EVENT_RE = re.compile(r'^"(?P<name>.+)"\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$')


def load_event_log(path: str | None = None) -> dict[str, pd.Timestamp]:
    """``"이벤트명": YYYY-MM-DD`` 형식만 받는다.

    형식을 벗어난 줄은 조용히 버리지 않고 :func:`event_log_problems` 로
    확인할 수 있게 한다. 예전 대시보드 파서는 `split(":")` 에 bare except 라
    이름에 콜론이 들어간 줄을 소리 없이 삼켰다.
    """
    events: dict[str, pd.Timestamp] = {}
    for name, date, _ in _read_event_lines(path):
        if date is not None:
            events[name] = date
    return events


def event_log_problems(path: str | None = None) -> list[str]:
    """형식이 맞지 않아 무시된 줄."""
    return [raw for _, date, raw in _read_event_lines(path) if date is None]


def _read_event_lines(path: str | None):
    path = path or _path(EVENT_FILE)
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = _EVENT_RE.match(line)
            if m:
                yield m.group("name"), pd.Timestamp(m.group("date")), line
            else:
                yield line, None, line


def events_on(date, events: dict[str, pd.Timestamp]) -> list[str]:
    """같은 날짜의 이벤트 이름 전부.

    반사실 분석은 한 이벤트에 가격 변화를 통째로 귀속시킨다. 같은 날 다른
    이벤트가 있으면 그 귀속이 성립하지 않으므로 호출부가 경고할 수 있어야 한다.
    """
    target = pd.Timestamp(date).normalize()
    return sorted(n for n, d in events.items() if pd.Timestamp(d).normalize() == target)


# ---------------------------------------------------------------------------
# 골드 환율
# ---------------------------------------------------------------------------
def load_gold() -> pd.Series:
    """일별 골드 환율(100골드 당 현금). 인덱스는 날짜."""
    path = _path(GOLD_FILE)
    if not os.path.exists(path):
        return pd.Series(dtype="float64")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty or "Date" not in df.columns:
        return pd.Series(dtype="float64")
    s = df.set_index(pd.to_datetime(df["Date"]))["Gold_Price"].astype("float64")
    s.index.name = None
    return s.sort_index()


def to_cash(prices: pd.DataFrame, gold: pd.Series) -> pd.DataFrame:
    """골드 가격을 현금으로 환산한다.

    **환율이 없는 날짜는 NaN 으로 남긴다.** 예전 구현은 없는 날짜를 조용히
    최신 환율로 대체해서, 골드 데이터가 끊긴 뒤 82일 동안 고정 환율로 환산된
    값을 사용자에게 사실처럼 보여줬다. 모르는 건 모른다고 비워 둔다.
    """
    if prices.empty or gold.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype="float64")

    days = pd.DatetimeIndex(prices.index).normalize()
    ratio = pd.Series(gold.reindex(days).to_numpy(), index=prices.index, dtype="float64")
    return prices.mul(ratio / 100.0, axis=0)


def gold_gap(prices_index: pd.DatetimeIndex, gold: pd.Series) -> tuple[int, pd.Timestamp | None]:
    """환율이 없어 환산 불가한 날짜 수와 골드 데이터의 마지막 날짜."""
    if gold.empty:
        return len(pd.DatetimeIndex(prices_index).normalize().unique()), None
    days = pd.DatetimeIndex(prices_index).normalize().unique()
    missing = int((~days.isin(gold.index)).sum())
    return missing, gold.index.max()

# -*- coding: utf-8 -*-
"""Dukascopy 일별 1분캔들(bi5) 백필 — 틱 대비 요청수 1/24라 스로틀 회피.
usage: fetch_duka_daily.py SYM YYYY-MM-DD YYYY-MM-DD
결과: data/{SYM}_1m_dailyfill.pkl.bak (플랫봉 제거, 캐시 포맷) → merge_1m.py로 병합"""
import sys, os, time, pickle, struct, lzma, urllib.request
from datetime import datetime, timedelta, timezone

INSTR = {
    "HK50": "HKGIDXHKD", "JP225": "JPNIDXJPY", "GER40": "DEUIDXEUR",
    "UK100": "GBRIDXGBP", "US100": "USATECHIDXUSD", "XAGUSD": "XAGUSD",
    "WTI": "LIGHTCMDUSD", "XAUUSD": "XAUUSD",
}
DEC = {"XAGUSD": 1000, "XAUUSD": 1000}  # 기본 1000, 심볼별 동일

SYM = sys.argv[1]; d0 = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
d1 = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
ins = INSTR[SYM]; dec = DEC.get(SYM, 1000)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f'{SYM}_1m_dailyfill.pkl.bak')

bars = []; day = d0; nreq = 0; nerr = 0
while day <= d1:
    url = f'https://datafeed.dukascopy.com/datafeed/{ins}/{day.year}/{day.month-1:02d}/{day.day:02d}/BID_candles_min_1.bi5'
    raw = None
    for att in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=60).read()
            raw = lzma.decompress(data) if data else b''
            break
        except Exception as e:
            if att == 3: nerr += 1; raw = b''
            else: time.sleep(2 * (att + 1))
    nreq += 1
    day_ms = int(day.timestamp() * 1000)
    for i in range(len(raw) // 24):
        t, o, c, l, h, v = struct.unpack('>IIIIIf', raw[i*24:(i+1)*24])
        if o == c == l == h: continue  # 휴장 플랫봉
        bars.append({'start': day_ms + t*1000, 'open': o/dec, 'high': h/dec, 'low': l/dec, 'close': c/dec})
    if nreq % 50 == 0:
        print(f'{day.date()} 까지 {nreq}일 요청, {len(bars):,}봉, 실패 {nerr}', flush=True)
        pickle.dump(bars, open(out_path, 'wb'))
    time.sleep(0.15)
    day += timedelta(days=1)

pickle.dump(bars, open(out_path, 'wb'))
from collections import Counter
cnt = Counter(datetime.fromtimestamp(b['start']/1000, tz=timezone.utc).year for b in bars)
print(f'완료: {SYM} {len(bars):,}봉 → {out_path}')
print('연도별: ' + ' '.join(f'{y}:{cnt[y]//1000}k' for y in sorted(cnt)) + f' | 요청 {nreq} 실패 {nerr}')

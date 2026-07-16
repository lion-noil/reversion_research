# -*- coding: utf-8 -*-
"""Dukascopy 월단위 시간봉(bi5) 페치 → 1h 캐시(_60m.pkl) + 4h 리샘플(_240m.pkl) 저장.
usage: fetch_duka_hourly.py SYM [START_YM] [END_YM]  (기본 2015-01 ~ 현재)
요청수 = 월 1개 (연 12) — 스로틀 부담 극소."""
import sys, os, time, pickle, struct, lzma, urllib.request
from datetime import datetime, timezone

INSTR = {
    "HK50": "HKGIDXHKD", "JP225": "JPNIDXJPY", "GER40": "DEUIDXEUR",
    "UK100": "GBRIDXGBP", "US100": "USATECHIDXUSD", "XAGUSD": "XAGUSD",
    "WTI": "LIGHTCMDUSD", "XAUUSD": "XAUUSD",
    "USDJPY": "USDJPY", "EURUSD": "EURUSD", "GBPUSD": "GBPUSD",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF", "NZDUSD": "NZDUSD",
}
SYM = sys.argv[1]
ym0 = sys.argv[2] if len(sys.argv) > 2 else '2015-01'
ym1 = sys.argv[3] if len(sys.argv) > 3 else datetime.now(timezone.utc).strftime('%Y-%m')
ins = INSTR[SYM]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

y, mo = map(int, ym0.split('-')); y1, mo1 = map(int, ym1.split('-'))
bars = []; nreq = 0; nerr = 0
while (y, mo) <= (y1, mo1):
    url = f'https://datafeed.dukascopy.com/datafeed/{ins}/{y}/{mo-1:02d}/BID_candles_hour_1.bi5'
    raw = b''
    for att in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=90).read()
            raw = lzma.decompress(data) if data else b''
            break
        except urllib.error.HTTPError:
            break  # 404 = 해당 월 데이터 없음
        except Exception:
            if att == 3: nerr += 1
            else: time.sleep(3 * (att + 1))
    nreq += 1
    m0 = int(datetime(y, mo, 1, tzinfo=timezone.utc).timestamp() * 1000)
    for i in range(len(raw) // 24):
        t, o, c, l, h, v = struct.unpack('>IIIIIf', raw[i*24:(i+1)*24])
        if o == c == l == h: continue
        bars.append({'start': m0 + t*1000, 'open': o/1000, 'high': h/1000, 'low': l/1000, 'close': c/1000})
    time.sleep(0.2)
    mo += 1
    if mo > 12: mo = 1; y += 1

pickle.dump(bars, open(os.path.join(DATA, f'{SYM}_60m.pkl'), 'wb'))
# 4h 리샘플
BAR = 4*3600*1000
agg = {}
for b in bars:
    k = b['start'] // BAR
    if k not in agg:
        agg[k] = dict(start=int(k*BAR), open=b['open'], high=b['high'], low=b['low'], close=b['close'])
    else:
        a = agg[k]
        a['high'] = max(a['high'], b['high']); a['low'] = min(a['low'], b['low']); a['close'] = b['close']
h4 = [agg[k] for k in sorted(agg)]
pickle.dump(h4, open(os.path.join(DATA, f'{SYM}_240m.pkl'), 'wb'))
from collections import Counter
cnt = Counter(datetime.fromtimestamp(b['start']/1000, tz=timezone.utc).year for b in bars)
yl = ' '.join(f'{yy%100:02d}:{cnt[yy]//100/10:.1f}k' for yy in sorted(cnt))
print(f'{SYM}: 1h {len(bars):,}봉 / 4h {len(h4):,}봉 | 요청 {nreq} 실패 {nerr} | {yl}', flush=True)

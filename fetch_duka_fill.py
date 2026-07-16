# -*- coding: utf-8 -*-
"""Dukascopy 시간봉 재개형 필러 — 캐시에 '빠진 월'만 페치해 유니온 병합 (여러 번 돌려도 안전).
- _60m.pkl 갱신 후 _240m.pkl 재생성 (2023-07+ 구간은 1m 리샘플이 우선 — 완전밀도 보존)
- 단일 프로세스(심볼 루프 내장, 좀비 배시루프 없음), data/duka_fill.pid 기록
- 연속 실패 12회면 서버 적대 판정 → 즉시 종료(다음 실행에서 이어받음)
usage: fetch_duka_fill.py [SYM1,SYM2,...]  (기본: FX7+MT5 8종)"""
import sys, os, time, pickle, struct, lzma, urllib.request
from datetime import datetime, timezone
from collections import Counter

INSTR = {
    "HK50": "HKGIDXHKD", "JP225": "JPNIDXJPY", "GER40": "DEUIDXEUR",
    "UK100": "GBRIDXGBP", "US100": "USATECHIDXUSD", "XAGUSD": "XAGUSD",
    "WTI": "LIGHTCMDUSD", "XAUUSD": "XAUUSD",
    "USDJPY": "USDJPY", "EURUSD": "EURUSD", "GBPUSD": "GBPUSD",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF", "NZDUSD": "NZDUSD",
}
DEFAULT = "USDCAD,USDCHF,NZDUSD,JP225,US100,GER40,UK100,HK50,XAGUSD,WTI,XAUUSD,USDJPY,EURUSD,GBPUSD,AUDUSD"
# 5자리 FX 페어는 decimalFactor 100000 (JPY계·금속·지수·원유는 1000)
DEC = {s: 100000 for s in ["EURUSD","GBPUSD","AUDUSD","USDCAD","USDCHF","NZDUSD"]}
SYMS = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).split(',')
ROOT = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(ROOT, 'data')
open(os.path.join(DATA, 'duka_fill.pid'), 'w').write(str(os.getpid()))
MIN_BARS = 250  # 월 봉수가 이보다 적으면 '빠진 월'로 간주 (HK50도 정상월 ~330)
H = 3600*1000

def month_range():
    now = datetime.now(timezone.utc)
    y, mo = 2013, 1
    while (y, mo) <= (now.year, now.month):
        yield y, mo
        mo += 1
        if mo > 12: mo = 1; y += 1

def fetch_month(ins, y, mo):
    url = f'https://datafeed.dukascopy.com/datafeed/{ins}/{y}/{mo-1:02d}/BID_candles_hour_1.bi5'
    for att in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=45).read()
            return lzma.decompress(data) if data else b''
        except urllib.error.HTTPError:
            return b''   # 404 = 해당 월 없음(정상)
        except Exception:
            if att < 2: time.sleep(4*(att+1))
    return None  # 실패

consec_fail = 0
for SYM in SYMS:
    ins = INSTR[SYM]
    p60 = os.path.join(DATA, f'{SYM}_60m.pkl')
    bars = pickle.load(open(p60, 'rb')) if os.path.exists(p60) else []
    have = Counter((datetime.fromtimestamp(b['start']/1000, tz=timezone.utc).year,
                    datetime.fromtimestamp(b['start']/1000, tz=timezone.utc).month) for b in bars)
    missing = [(y, mo) for y, mo in month_range() if have.get((y, mo), 0) < MIN_BARS]
    if not missing:
        print(f'{SYM}: 빠진 월 없음', flush=True); continue
    print(f'{SYM}: 빠진 월 {len(missing)}개 페치 시작', flush=True)
    d = {b['start']: b for b in bars}
    got = 0
    for y, mo in missing:
        raw = fetch_month(ins, y, mo)
        if raw is None:
            consec_fail += 1
            if consec_fail >= 12:
                print(f'연속실패 {consec_fail} — 서버 쿨다운, 종료(재실행 시 이어받음)', flush=True)
                pickle.dump([d[k] for k in sorted(d)], open(p60, 'wb')); sys.exit(2)
            continue
        consec_fail = 0
        m0 = int(datetime(y, mo, 1, tzinfo=timezone.utc).timestamp()*1000)
        dec = DEC.get(SYM, 1000)
        for i in range(len(raw)//24):
            t, o, c, l, h, v = struct.unpack('>IIIIIf', raw[i*24:(i+1)*24])
            if o == c == l == h: continue
            d[m0+t*1000] = {'start': m0+t*1000, 'open': o/dec, 'high': h/dec, 'low': l/dec, 'close': c/dec}
        got += 1
        time.sleep(2.0)
    bars = [d[k] for k in sorted(d)]
    pickle.dump(bars, open(p60, 'wb'))
    # 4h 재생성: duka 1h 리샘플 + (2023-07+ 1m 리샘플 우선)
    # ⚠️ 키는 반드시 타임스탬프(kk)로 통일 — 봉번호(k)와 섞으면 중복봉 생성(2026-07-15 버그 수정)
    BAR = 4*H
    agg = {}
    for b in bars:
        kk = int((b['start']//BAR)*BAR)
        if kk not in agg: agg[kk] = dict(start=kk, open=b['open'], high=b['high'], low=b['low'], close=b['close'])
        else:
            a = agg[kk]; a['high'] = max(a['high'], b['high']); a['low'] = min(a['low'], b['low']); a['close'] = b['close']
    p1 = os.path.join(DATA, f'{SYM}_1m.pkl')
    if os.path.exists(p1):
        CUT = int(datetime(2023, 7, 1, tzinfo=timezone.utc).timestamp()*1000)
        m1 = [x for x in pickle.load(open(p1, 'rb')) if x['start'] >= CUT]
        for b in m1:
            k = b['start']//BAR
            kk = int(k*BAR)
            if agg.get(kk, {}).get('_src') != 'm1':
                agg[kk] = dict(start=kk, open=b['open'], high=b['high'], low=b['low'], close=b['close'], _src='m1')
            else:
                a = agg[kk]; a['high'] = max(a['high'], b['high']); a['low'] = min(a['low'], b['low']); a['close'] = b['close']
    h4 = [{kk: v for kk, v in x.items() if kk != '_src'} for x in (agg[k] for k in sorted(agg))]
    pickle.dump(h4, open(os.path.join(DATA, f'{SYM}_240m.pkl'), 'wb'))
    yc = Counter(datetime.fromtimestamp(b['start']/1000, tz=timezone.utc).year for b in bars)
    print(f'{SYM}: +{got}개월 → 1h {len(bars)/1000:.0f}k / 4h {len(h4)/1000:.1f}k | ' +
          ' '.join(f'{yy%100:02d}:{yc[yy]//100/10:.1f}k' for yy in sorted(yc)), flush=True)
print('FILL_DONE', flush=True)

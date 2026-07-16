# -*- coding: utf-8 -*-
"""10년+ 일봉(d1) 직접 받기 → data/{SYM}_D10m.pkl (interval 'D10'). -v로 flats제거.
장기 일봉 가능 심볼만(SOL/XRP/XAUT 제외). 2015-01-01~now (크립토는 상장부터)."""
import subprocess, os, tempfile, datetime as dt
from reversion_calibrator import _save_cache
INST={'EURUSD':'eurusd','XAUUSD':'xauusd','XAGUSD':'xagusd','WTI':'lightcmdusd','US100':'usatechidxusd',
'JP225':'jpnidxjpy','GER40':'deuidxeur','UK100':'gbridxgbp','HK50':'hkgidxhkd','BTCUSD':'btcusd','ETHUSD':'ethusd',
'GBPUSD':'gbpusd','USDJPY':'usdjpy','AUDUSD':'audusd','USDCAD':'usdcad','USDCHF':'usdchf','NZDUSD':'nzdusd'}
def D(ms): return dt.datetime.utcfromtimestamp(ms/1000).strftime('%Y-%m-%d')
for sym,inst in INST.items():
    with tempfile.TemporaryDirectory(prefix='d10_') as tmp:
        cmd=(f'npx -y -p dukascopy-node@latest dukascopy-node -i {inst} -from 2015-01-01 -to now '
             f'-t d1 -p bid -v -f csv -dir "{tmp}" -fn out -r 4 -rp 800 -re')
        subprocess.run(cmd,shell=True,capture_output=True,encoding='utf-8',errors='replace')
        p=os.path.join(tmp,'out.csv')
        if not os.path.exists(p): print(f'{sym:9} 실패'); continue
        cs=[]
        with open(p,encoding='utf-8') as f:
            next(f,None)
            for line in f:
                q=line.rstrip('\n').split(',')
                if len(q)<5: continue
                try: cs.append({'start':int(q[0]),'open':float(q[1]),'high':float(q[2]),'low':float(q[3]),'close':float(q[4])})
                except ValueError: pass
        if not cs: print(f'{sym:9} 0행'); continue
        _save_cache(sym,'D10',cs)
        print(f'{sym:9} {len(cs):>5}봉  {D(cs[0]["start"])}~{D(cs[-1]["start"])} → {sym}_D10m.pkl')
print('완료')

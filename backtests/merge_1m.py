# -*- coding: utf-8 -*-
"""1분 캐시 백필 병합: 현재 캐시 + 백업파일 유니온(타임스탬프 기준) → 저장 + 연도 밀도 출력.
usage: merge_1m.py SYM BACKUP_PATH"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reversion_calibrator import _load_cache, CACHE_DIR
from collections import Counter
from datetime import datetime, timezone
S=sys.argv[1]; bak=sys.argv[2]
new=_load_cache(S,'1')
old=pickle.load(open(bak,'rb'))
d={x['start']:x for x in old}
d.update({x['start']:x for x in new})
merged=[d[k] for k in sorted(d)]
pickle.dump(merged,open(os.path.join(CACHE_DIR,f'{S}_1m.pkl'),'wb'))
c=Counter(datetime.fromtimestamp(x['start']/1000,tz=timezone.utc).year for x in merged)
yl=' '.join(f'{y%100:02d}:{c[y]//1000}k' for y in sorted(c))
ok = c.get(2021,0)>150000 and c.get(2022,0)>150000
print(f'{S}: {len(merged):,}봉 | {yl} | 21-22: {"OK" if ok else "부족"}')

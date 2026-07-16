"""스윕 json -> K-능선 리포트 데이터(_..._report.json). 심볼별 실제 연수로 회/년 계산.
사용: python build_ridge_report.py <sweep_json_path> <out_report_json>
"""
import sys, os, json
import numpy as np
os.environ['S1_DIR'] = 'short'; os.environ['S1_MAXHOLD_D'] = '14'
os.environ.setdefault('S1_SYMS', 'x')  # prep import 시 SYMS 오버라이드 무해
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persym_sweep as P
from bt_s1_quarters_mt5 import prep, WIN

H = 3600 * 1000
KMAX = float(os.getenv('RIDGE_KMAX', '4.6'))  # 능선 스캔 상한(크립토 고K는 5.8)
KS = [round(3.0 + 0.1 * i, 2) for i in range(int(round((KMAX - 3.0) / 0.1)) + 1)]

def cd_to_h(cd):
    return float(cd[:-1]) / 60 if cd.endswith('m') else float(cd[:-1])

def main():
    sweep_json, out_json = sys.argv[1], sys.argv[2]
    D = json.load(open(sweep_json, encoding='utf-8'))
    out = {}
    for s, rows in D.items():
        if not rows:
            continue
        d = prep(s)
        yrs = (int(d['ts'][-1]) - int(d['ts'][WIN])) / (365.25 * 86400e3)
        best = rows[0]
        hik = [r for r in rows if r['k1'] >= 3.5 and r['exp'] > 0]
        pick = hik[0] if hik else best
        b = pick['b']; cdh = cd_to_h(pick['cd'])
        ridge = []
        for k in KS:
            tp_, sl_, top_, tom_, mx = P.run(d, k, b, int(cdh * H))
            r = np.array(tp_ + sl_ + top_ + tom_); n = len(r)
            if n == 0:
                ridge.append(dict(k=k, exp=None, pf=None, n=0, wr=None)); continue
            e = round(float(r.mean()) * 100, 3)
            pf = round(float(r[r > 0].sum() / (-r[r < 0].sum())) if (r < 0).any() else 9.9, 2)
            wr = round(100 * float((r > 0).mean()))
            ridge.append(dict(k=k, exp=e, pf=pf, n=n, wr=wr))
        # 채택K 중심 ±0.6 국소 평탄도(고K 전용 엣지도 공정 평가)
        lo, hi = pick['k1'] - 0.6, pick['k1'] + 0.6
        band = [x for x in ridge if lo - 1e-9 <= x['k'] <= hi + 1e-9 and x['exp'] is not None]
        posratio = round(sum(1 for x in band if x['exp'] > 0) / len(band), 2) if band else 0
        out[s] = dict(top=rows[:8], pick=pick, best=best, ridge=ridge,
                      posratio=posratio, yrs=round(yrs, 2), pick_yr=round(pick['n'] / yrs, 1))
    json.dump(dict(syms=out, KS=KS), open(out_json, 'w'), ensure_ascii=False)
    print(f'wrote {out_json} ({len(out)} syms)')
    for s, v in out.items():
        p = v['pick']
        print(f"{s:>9} pick K{p['k1']} B{p['b']} {p['cd']} n{p['n']}({v['pick_yr']}/yr {v['yrs']}y) "
              f"exp{p['exp']:+.3f} PF{p['pf']} posratio{v['posratio']}")

if __name__ == '__main__':
    main()

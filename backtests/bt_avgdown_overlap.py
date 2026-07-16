"""
중첩(동시보유 허용) 엔진 위의 추매 비교 — 채택 파라미터가 선정된 그 엔진(persym_sweep) 기준.

두 운용을 같은 '수락된 신호열'(쿨다운=진입기준, persym과 동일) 위에서 비교:
  (A) 병렬독립  = 현 시스템. 재발생 신호는 매번 새 병렬 포지션. (do_add=False → persym.run과 동일 n)
  (B) 추매흡수  = 재발생 신호가 '아직 열려있고 쿨다운 지난' 기존 포지션에 추매(평단↓·TPSL 재앵커).
                 흡수되면 새 포지션을 열지 않음(최대 1회/포지션). 흡수 대상은 가장 오래된 적격 포지션.

수락 신호 타이밍은 A·B 동일(라우팅만 다름) → 깨끗한 비교. 14일 최대보유, fee0.11%, 손절우선.
exit는 tp/sl 벡터 argmax로 산출(추매 시 새 타깃으로 ci+1부터 재산출, 최대보유는 원진입 기준).

사용: python backtests/bt_avgdown_overlap.py            # rev_long+short 비교 + 포트폴리오
"""
from __future__ import annotations
import sys, os, heapq
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt_s1_quarters_mt5 import prep, WIN
from bt_avgdown import _targets, SIDE, ZSIGN, FEE, MAXHOLD_MS, H_MS, S2_REV_LONG, S2_REV_SHORT

REV = [("rev_long", S2_REV_LONG), ("rev_short", S2_REV_SHORT)]
_DAY = 86400 * 1000


def _run_portfolio(lot_events, init=10_000.0, lot_frac=0.05, max_lots=200):
    """lot_events: (entry_ts, exit_ts, ret). 진입순 처리·힙 청산. → 지표 dict."""
    ev = sorted(lot_events, key=lambda x: x[0])
    equity = init; heap = []; open_cnt = 0; peak_lots = 0; skipped = taken = 0
    curve = []; span1 = max(e[1] for e in ev)

    def realize(t):
        nonlocal equity, open_cnt
        while heap and heap[0][0] <= t:
            ex_ts, ret, notional = heapq.heappop(heap)
            equity += notional * ret; open_cnt -= 1
            curve.append((ex_ts, equity))

    for en, ex, ret in ev:
        realize(en)
        if open_cnt < max_lots and equity > 0:
            heapq.heappush(heap, (ex, ret, lot_frac * equity))
            open_cnt += 1; taken += 1; peak_lots = max(peak_lots, open_cnt)
        else:
            skipped += 1
    realize(span1 + 1)
    eq = np.array([e for _, e in curve], float)
    peak = np.maximum.accumulate(eq); dd = (eq - peak) / peak
    yrs = (span1 - min(e[0] for e in ev)) / (365.25 * _DAY)
    cagr = (equity / init) ** (1 / yrs) - 1 if equity > 0 else -1.0
    return dict(final=equity, mult=equity / init, cagr=cagr, mdd=float(dd.min()),
                taken=taken, skipped=skipped, peak_lots=peak_lots, yrs=yrs)


def sim_overlap(d, kind, k1, b, cd, mode="parallel", max_adds=1):
    """mode: 'parallel'=현 시스템(추매 없음) | 'ontop'=중첩 그대로 + 각 게임이 쿨다운후 신호에 추매
             | 'absorb'=재발생을 새게임 대신 기존게임에 흡수(게임수 줄어듦)."""
    C, H, L, ts, ma, sd, z, n = d["C"], d["H"], d["L"], d["ts"], d["ma"], d["sd"], d["z"], d["n"]
    side = SIDE[kind]; zs = ZSIGN[kind]
    cand = np.flatnonzero(z >= k1) if zs > 0 else np.flatnonzero(z <= -k1)
    cand = cand[cand >= WIN]
    MH = MAXHOLD_MS

    def resolve(entry_idx, start_k, tp, sl):
        """[start_k, hi)에서 첫 tp/sl(손절우선). 미도달=최대보유 종가청산."""
        hi = int(np.searchsorted(ts, ts[entry_idx] + MH, "right"))
        if hi <= start_k:
            kk = min(start_k, n - 1)
            return kk, C[kk], "HOLD"
        if side > 0:
            SLh = L[start_k:hi] <= sl; TPh = H[start_k:hi] >= tp
        else:
            SLh = H[start_k:hi] >= sl; TPh = L[start_k:hi] <= tp
        hit = SLh | TPh
        if not hit.any():
            return hi - 1, C[hi - 1], "HOLD"
        j = int(np.argmax(hit)); idx = start_k + j
        return (idx, sl, "SL") if SLh[j] else (idx, tp, "TP")

    def add_to(p, ci, price):
        new_avg = (sum(p["legs"]) + price) / (len(p["legs"]) + 1)
        ntp, nsl, ok = _targets(kind, ma[ci], sd[ci], new_avg, price, b)
        if not ok:
            return False
        p["legs"].append(price); p["leg_ts"].append(int(ts[ci]))
        p["avg"] = new_avg; p["tp"], p["sl"] = ntp, nsl; p["adds"] += 1
        idx, expx, rs = resolve(p["entry_idx"], ci + 1, ntp, nsl)
        p["exit_idx"], p["exit_px"], p["reason"] = idx, expx, rs
        p["exit_ts"] = int(ts[idx])
        return True

    positions = []
    last = None
    for ci in cand:
        if ci + 1 >= n:
            break
        if last is not None and (ts[ci] - last) < cd:
            continue
        price = C[ci]
        # ── 추매: 열려있고·쿨다운지나고·추매한도 남은 모든 게임에 얹기(게임당 first eligible) ──
        if mode in ("ontop", "absorb"):
            elig = [p for p in positions if p["adds"] < max_adds and p["side"] == side
                    and p["entry_ts"] <= ts[ci] - cd and p["exit_ts"] > ts[ci]]
            if mode == "absorb":               # 흡수: 가장 오래된 1개에만, 새 게임은 안 엶
                elig = sorted(elig, key=lambda x: x["entry_ts"])[:1]
            absorbed = False
            for p in elig:
                if add_to(p, ci, price):
                    absorbed = True
            if mode == "absorb" and absorbed:
                last = ts[ci]
                continue
        # ── 새 게임 (parallel·ontop 공통: 중첩 유지) ──
        tp, sl, ok = _targets(kind, ma[ci], sd[ci], price, price, b)
        if ok:
            idx, expx, rs = resolve(ci, ci + 1, tp, sl)
            positions.append(dict(entry_idx=int(ci), entry_ts=int(ts[ci]), legs=[price],
                leg_ts=[int(ts[ci])], avg=price, tp=tp, sl=sl, side=side,
                exit_idx=idx, exit_px=expx, reason=rs, exit_ts=int(ts[idx]), adds=0))
            last = ts[ci]

    out = []
    for p in positions:
        if side > 0:
            lg = [p["exit_px"] / x - 1 for x in p["legs"]]
        else:
            lg = [(x - p["exit_px"]) / x for x in p["legs"]]
        ret = float(np.mean(lg) - FEE)
        out.append(dict(entry_ts=p["entry_ts"], exit_ts=p["exit_ts"], ret=ret, added=p["adds"] > 0,
                        nlegs=len(p["legs"]),
                        legs=[(p["leg_ts"][i], float(lg[i] - FEE)) for i in range(len(p["legs"]))]))
    return out


def _maxc(positions):
    ev = []
    for p in positions:
        ev.append((p["entry_ts"], 1)); ev.append((p["exit_ts"], -1))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, dd in ev:
        cur += dd; mx = max(mx, cur)
    return mx


def agg(positions):
    r = np.array([p["ret"] for p in positions], float)         # 게임(포지션)당 수익
    legrets = np.array([lr for p in positions for (_, lr) in p["legs"]], float)  # 레그(랏)당
    legs = sum(p["nlegs"] for p in positions)
    added = sum(p["added"] for p in positions)
    wins = r > 0
    pf = r[wins].sum() / (-r[~wins].sum()) if (~wins).any() else 99.0
    return dict(npos=len(r), legs=legs, added=added, maxc=_maxc(positions),
                wr=round(100 * wins.mean(), 0), exp=round(float(r.mean()) * 100, 3),
                exp_leg=round(float(legrets.mean()) * 100, 3),
                pf=round(float(pf), 2), worst=round(float(r.min()) * 100, 1))


def main():
    cache = {}
    for _, p in REV:
        for s in p:
            cache.setdefault(s, prep(s))

    port_events = {"A": [], "C": []}
    for kind, params in REV:
        print("\n" + "=" * 100)
        print(f"  {kind}  중첩 유지 · (A)현시스템(추매무) vs (C)각 게임에 추매 얹기 | 채택 파라미터, 14일보유")
        print("=" * 100)
        print(f"{'심볼':>9}{'K1':>5}{'B':>5}{'CD':>5} | "
              f"{'게임':>4}{'wrA':>4}{'expA':>7}{'pfA':>5}{'wrstA':>6}  ||  "
              f"{'게임':>4}{'레그':>4}{'add':>4}{'wrC':>4}{'expC':>7}{'leg':>7}{'pfC':>5}{'wrstC':>6} |{'Δwr':>4}{'Δexp':>7}")
        for s, (k1, b, cdh) in params.items():
            d = cache[s]; cd = int(cdh * H_MS)
            A = sim_overlap(d, kind, k1, b, cd, mode="parallel")
            C = sim_overlap(d, kind, k1, b, cd, mode="ontop", max_adds=1)
            a, c = agg(A), agg(C)
            for p in A:
                for lt, lr in p["legs"]: port_events["A"].append((lt, p["exit_ts"], lr))
            for p in C:
                for lt, lr in p["legs"]: port_events["C"].append((lt, p["exit_ts"], lr))
            print(f"{s:>9}{k1:>5}{b:>5}{cdh:>5} | "
                  f"{a['npos']:>4}{a['wr']:>4.0f}{a['exp']:>+7.3f}{a['pf']:>5.1f}{a['worst']:>6.1f}  ||  "
                  f"{c['npos']:>4}{c['legs']:>4}{c['added']:>4}{c['wr']:>4.0f}{c['exp']:>+7.3f}"
                  f"{c['exp_leg']:>+7.3f}{c['pf']:>5.1f}{c['worst']:>6.1f} |"
                  f"{c['wr']-a['wr']:>+4.0f}{c['exp']-a['exp']:>+7.3f}")

    print("\n" + "=" * 100)
    print("  역추세 슬리브 포트폴리오 (랏=자산5%, 최대200랏) — 레그=랏. C는 추매분만큼 자본 더 투입")
    print("=" * 100)
    for key, lbl in [("A", "(A) 현 시스템 = 추매 무"), ("C", "(C) 각 게임 추매 얹기")]:
        m = _run_portfolio(port_events[key])
        print(f"  {lbl:>22}: 랏 {len(port_events[key]):>5}  최종 ${m['final']:>10,.0f} "
              f"({m['mult']:.2f}배, CAGR {m['cagr']*100:>+5.1f}%)  MDD {m['mdd']*100:>5.1f}%  "
              f"최대동시 {m['peak_lots']}랏")


if __name__ == "__main__":
    main()

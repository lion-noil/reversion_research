"""
S1 (σ-복귀 역추세 롱) — MT5 10심볼 분기별 파라미터 튜닝.
규칙은 bt_s1_sweep.py / tradingBot strategies/s1_reversion.py와 동일:
  win=10080(7일, 고정) 종가 MA/σ(모집단), 진입 z<=-K1, TP=MA-B*σ, SL대칭(pct),
  쿨다운 12h(고정), fee 0.11% 왕복, 봉(고저) 청산·손절우선.
튜닝 대상 = K1(진입임계) × B(TP밴드)만. 기간(win)·쿨다운은 원래 설정 그대로 고정.

분기 = UTC 달력분기, 진입시점 기준. z는 전체 시계열로 계산(분기 초반 진입도 직전 7일
MA/σ 사용) 후, 체결을 진입 ts의 분기로 버킷팅 → 분기별 (K1,B) 격자에서 최적 선택.

⚠️ 분기별 최적은 in-sample 최적화(과적합)다. 헤드라인은 '3년 전체 최적'으로 보고,
   분기 표는 파라미터가 시장 국면 따라 어디로 쏠리는지 보는 용도.

사용: python backtests/bt_s1_quarters_mt5.py
      python backtests/bt_s1_quarters_mt5.py --symbols US100,XAUUSD
      python backtests/bt_s1_quarters_mt5.py --min-n 6
"""
from __future__ import annotations
import sys, os, argparse
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reversion_calibrator import _load_cache

WIN = int(os.getenv("S1_WIN", "10080"))   # MA/σ 창(분봉 수). 기본 7일=10080
COOLDOWN_H = float(os.getenv("S1_COOLDOWN_H", "12"))  # 쿨다운(시간). env로 덮어쓰기.
# 주의(2026-06 측정): 3h로 줄이면 빈도는 ↑이나 풀링 기대값이 음수로 떨어짐(저품질 재진입).
# US100/JP225만 3h가 유리. 그 외(특히 XAUUSD/BTCUSD/WTI)는 12h 유지가 우위. cd_compare.py 참고.
COOLDOWN_MS = int(COOLDOWN_H * 3600 * 1000)
FEE = 0.0011
SYMS = ["US100", "JP225", "HK50", "GER40", "UK100",
        "XAUUSD", "XAGUSD", "WTI", "BTCUSD", "ETHUSD"]

# 튜닝 격자 (b < k1만 유효). 지수~크립토 변동성 폭을 포괄.
K1S = (2.0, 2.5, 3.0, 3.48)
BS = (1.5, 1.8, 2.0, 2.2, 2.5, 3.0)
BASELINE = (2.5, 2.0)   # 원래 설정(비교 기준)


def q_of(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"


def prep(sym):
    m = _load_cache(sym, "1")
    if not m:
        return None
    C = np.array([x["close"] for x in m], float)
    H = np.array([x["high"] for x in m], float)
    L = np.array([x["low"] for x in m], float)
    ts = np.array([x["start"] for x in m], np.int64)
    n = len(C)
    if n <= WIN:
        return None
    cs = np.concatenate([[0.0], np.cumsum(C)])
    cs2 = np.concatenate([[0.0], np.cumsum(C * C)])
    w = WIN
    ma = np.full(n, np.nan); sd = np.full(n, np.nan)
    s = cs[w:] - cs[:-w]; s2 = cs2[w:] - cs2[:-w]
    mm = s / w; var = np.maximum(s2 / w - mm * mm, 0.0)
    ma[w - 1:] = mm; sd[w - 1:] = np.sqrt(var)
    z = np.where(sd > 0, (C - ma) / sd, np.nan)
    return dict(C=C, H=H, L=L, ts=ts, ma=ma, sd=sd, z=z, n=n)


def sim(d, k1, b, cooldown_ms=None):
    """bt_s1_sweep.sim과 동일. (entry_ts_ms, net_ret) 리스트 반환. cooldown_ms 미지정시 전역값."""
    cd = COOLDOWN_MS if cooldown_ms is None else cooldown_ms
    C, H, L, ts, ma, sd, z, n = d["C"], d["H"], d["L"], d["ts"], d["ma"], d["sd"], d["z"], d["n"]
    cand = np.flatnonzero(z <= -k1)
    cand = cand[cand >= WIN]
    out = []
    last_exit_ms = None
    for ci in cand:
        if ci + 1 >= n:
            break
        if last_exit_ms is not None and (ts[ci] - last_exit_ms) < cd:
            continue
        price = C[ci]; tp = ma[ci] - b * sd[ci]
        if tp <= price:
            continue
        pct = tp / price - 1.0
        sl = price * (1.0 - pct)
        Lsl = L[ci + 1:] <= sl
        Htp = H[ci + 1:] >= tp
        hit = Lsl | Htp
        j = int(np.argmax(hit))
        if not hit[j]:
            break
        if Lsl[j]:
            out.append((int(ts[ci]), sl / price - 1 - FEE))
        else:
            out.append((int(ts[ci]), tp / price - 1 - FEE))
        last_exit_ms = ts[ci + 1 + j]
    return out


def stat(rets):
    r = np.asarray(rets, float)
    if len(r) == 0:
        return (0, float("nan"), float("nan"), float("nan"), 0.0)
    wr = (r > 0).mean() * 100
    pf = r[r > 0].sum() / (-r[r < 0].sum()) if (r < 0).any() else float("inf")
    return (len(r), wr, r.mean() * 100, pf, (np.prod(1 + r) - 1) * 100)  # n,wr,exp%,pf,cum%


def fmt(st):
    n, wr, exp, pf, cum = st
    if n == 0:
        return f"{0:>4}{'—':>7}{'—':>9}{'—':>7}{'—':>9}"
    pfs = "inf" if pf == float("inf") else f"{pf:.2f}"
    return f"{n:>4}{wr:>6.0f}%{exp:>+8.3f}%{pfs:>7}{cum:>+8.1f}%"


def grid():
    return [(k1, b) for k1 in K1S for b in BS if b < k1]


def pick_best(buckets_by_combo, q, min_n):
    """해당 분기에서 n>=min_n인 (k1,b) 중 기대값(exp) 최대 선택. 없으면 None."""
    best = None
    for (k1, b), by_q in buckets_by_combo.items():
        st = stat(by_q.get(q, []))
        if st[0] < min_n or not np.isfinite(st[2]):
            continue
        key = (st[2], st[3] if np.isfinite(st[3]) else 0.0)  # exp, then pf
        if best is None or key > best[0]:
            best = (key, (k1, b), st)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols")
    ap.add_argument("--min-n", type=int, default=5, help="분기 최적 선정 최소 거래수(기본5)")
    ap.add_argument("--k1s", default=None, help="시그마(K1) 후보 콤마구분(미지정시 기본격자)")
    ap.add_argument("--bs", default=None, help="B 후보 콤마구분(음수 가능, --bs=-2,..)")
    args = ap.parse_args()
    syms = ([s.strip().upper() for s in args.symbols.split(",")] if args.symbols else SYMS)

    k1s = tuple(float(x) for x in args.k1s.split(",")) if args.k1s else K1S
    bs = tuple(float(x) for x in args.bs.split(",")) if args.bs else BS
    G = [(k1, b) for k1 in k1s for b in bs if b < k1]
    if BASELINE not in G:           # 원래설정 비교줄용으로 항상 포함
        G.append(BASELINE)
    print(f"S1 역추세 롱 · MT5 분기별 튜닝 | win={WIN}(고정), CD{COOLDOWN_H:g}h(고정), fee0.11% | "
          f"격자 K1{k1s}×B{bs} ({len(G)}조합) | 분기최소n={args.min_n}")

    stat_hdr = f"{'n':>4}{'승률':>7}{'기대/건':>9}{'PF':>7}{'누적':>9}"
    sym_hdr = f"{'심볼':>7}{'K1':>6}{'B':>5}{stat_hdr}"
    print("\n" + "=" * 78)
    print("【A】 3년 전체 최적 (헤드라인 추천값) — 기대값 최대, fee반영")
    print("=" * 78)
    print(sym_hdr)

    combo_cache = {}  # sym -> {(k1,b): [(ts,ret)...]}
    for s in syms:
        d = prep(s)
        if d is None:
            print(f"{s:>7}  (1m 캐시 없음/부족 — 스킵)")
            continue
        per_combo = {(k1, b): sim(d, k1, b) for (k1, b) in G}
        combo_cache[s] = per_combo
        # 전체 최적
        best = None
        for (k1, b), tr in per_combo.items():
            st = stat([r for _, r in tr])
            if st[0] < 10 or not np.isfinite(st[2]):
                continue
            key = (st[2], st[3] if np.isfinite(st[3]) else 0.0)
            if best is None or key > best[0]:
                best = (key, (k1, b), st)
        if best:
            (k1, b), st = best[1], best[2]
            print(f"{s:>7}{k1:>6}{b:>5}{fmt(st)}")
        else:
            print(f"{s:>7}{'—':>6}{'—':>5}  (거래<10, 최적 산출 불가)")

    # ── 베이스라인(원래 K1=2.5,B=2.0) 전체 성적 비교 ──
    print("\n" + "-" * 78)
    print(f"  [참고] 원래 설정 K1={BASELINE[0]} B={BASELINE[1]} 3년 전체 성적")
    print(sym_hdr)
    for s in syms:
        if s not in combo_cache:
            continue
        tr = combo_cache[s].get(BASELINE)
        st = stat([r for _, r in tr]) if tr is not None else (0, 0, 0, 0, 0)
        print(f"{s:>7}{BASELINE[0]:>6}{BASELINE[1]:>5}{fmt(st)}")

    # ── 【B】 심볼별 분기 최적 ──
    print("\n" + "=" * 78)
    print(f"【B】 분기별 최적 파라미터 (in-sample, 과적합 주의) | 분기최소n={args.min_n}")
    print("=" * 78)
    all_q = set()
    for s in combo_cache:
        for (k1, b), tr in combo_cache[s].items():
            for t, _ in tr:
                all_q.add(q_of(t))
    qs = sorted(all_q)

    for s in syms:
        if s not in combo_cache:
            continue
        by_combo = {kb: _bucket(tr) for kb, tr in combo_cache[s].items()}
        print(f"\n--- {s} ---")
        print(f"{'분기':>8}{'K1':>6}{'B':>5}{'거래':>5}{'승률':>7}{'기대/건':>9}{'PF':>7}{'누적':>9}"
              f"   | 원래{BASELINE} 기대/PF")
        for q in qs:
            bl = stat(by_combo[BASELINE].get(q, []))
            bl_s = "—" if bl[0] == 0 else (f"{bl[2]:+.3f}%/"
                    + ("inf" if bl[3] == float('inf') else f"{bl[3]:.2f}") + f"(n{bl[0]})")
            best = pick_best(by_combo, q, args.min_n)
            if best is None:
                print(f"{q:>8}{'—':>6}{'—':>5}{fmt(stat([]))}   | {bl_s}")
            else:
                (k1, b), st = best[1], best[2]
                print(f"{q:>8}{k1:>6}{b:>5}{fmt(st)}   | {bl_s}")

    print("\n[해석] 기대/건(=평균 net 수익률/거래)이 핵심. 분기 거래수가 적으면(n작음) 최적값은 "
          "노이즈다. 분기마다 최적 K1/B가 크게 출렁이면 그 심볼은 파라미터 안정성이 낮은 것 "
          "→ 3년 전체 최적(A)이나 풀링값을 쓰는 게 실전 안전. 우상향 누적+분기 일관성이 좋은 "
          "심볼만 라이브 후보.")


def _bucket(trades):
    by = defaultdict(list)
    for t, r in trades:
        by[q_of(t)].append(r)
    return by


if __name__ == "__main__":
    main()

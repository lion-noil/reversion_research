"""
추매(평단 낮추기) 변형 백테스트 — S1(추세)·S2(역추세) 단일포지션 엔진 확장.

아이디어(사용자 검증 요청):
  진입 후 '설정 쿨다운'이 지난 뒤 같은 신호가 한 번 더 발생하면, 새 게임을 여는 대신
  기존 (지고 있는) 포지션에 추매 → 평단 이동 → TP/SL을 새 평단 기준으로 재설정.
  지는 게임에 '한 번 더 기회'를 주는 게 통하는지 측정.

확정 모델(사용자 선택):
  · 사이징 = 1:1 균등 (평단 = 진입가 단순평균, 2진입이면 중간값)
  · 최대 추매 = 1회
  · 재설정 = TP는 추매시점 MA∓Bσ 재앵커, SL은 새 평단 대칭(±pct)
  · 공통 규칙은 핸드오프 v2 유지: win=10080 z, 손절 우선, fee 0.11% 왕복, 최대보유 14일

⚠️ 본질적 주의:
  '추매로 평단 낮추기'는 **역추세(S2)** 에 자연스럽다(더 지면 z가 더 극단→신호 재발생→평단↓).
  **추세(S1)** 는 신호 재발생 = 추세 지속(이기는 방향)이라, 같은 메커니즘이 '추매(평단낮춤)'가
  아니라 '피라미딩(이기는 포지션 불리기)'으로 작동한다. 표에서 added 건수/방향으로 드러남.

baseline(max_adds=0)은 기존 sim/run과 동일 결과를 재현해야 함(검증용).

사용:
  python backtests/bt_avgdown.py                 # 전 전략·핵심심볼 비교표
  python backtests/bt_avgdown.py --json out.json
"""
from __future__ import annotations
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt_s1_quarters_mt5 import prep, WIN

FEE = 0.0011
DAY_MS = 86400 * 1000
H_MS = 3600 * 1000
MAXHOLD_MS = 14 * DAY_MS          # 핸드오프 v2 = 14일 종가청산

# 진입 신호: z 임계 부호로 결정.
#   rev_long  : z<=-K1, side=LONG   (S2 역추세 롱)
#   rev_short : z>=+K1, side=SHORT  (S2 역추세 숏)
#   trend_long: z>=+K1, side=LONG   (S1 추세 롱)
#   trend_short: z<=-K1, side=SHORT (S1 추세 숏)
SIDE = {"rev_long": +1, "rev_short": -1, "trend_long": +1, "trend_short": -1}
# 신호조건 z 부호: +1 → z>=+K1, -1 → z<=-K1
ZSIGN = {"rev_long": -1, "rev_short": +1, "trend_long": +1, "trend_short": -1}


def _targets(kind, ma_k, sd_k, avg, ref_price, b):
    """현재 bar(ma_k,sd_k)·평단(avg)·앵커가격(ref_price)으로 (tp, sl, ok) 산출.
    ok=False면 무효(롱인데 tp<=avg 등) → 추매/진입 스킵.
    역추세: TP=MA∓Bσ 절대밴드 재앵커, SL=평단 대칭.
    추세  : pct를 앵커가격의 밴드거리로 산출 후 평단 대칭(±pct)."""
    if kind == "rev_long":
        tp = ma_k - b * sd_k
        if tp <= avg:
            return 0.0, 0.0, False
        pct = tp / avg - 1.0
        return tp, avg * (1 - pct), True
    if kind == "rev_short":
        tp = ma_k + b * sd_k
        if tp >= avg:
            return 0.0, 0.0, False
        pct = 1 - tp / avg
        return tp, avg * (1 + pct), True
    if kind == "trend_long":
        band = ma_k + b * sd_k
        pct = 1 - band / ref_price          # 핸드오프 추세롱 pct식
        if pct <= 0:
            return 0.0, 0.0, False
        return avg * (1 + pct), avg * (1 - pct), True
    # trend_short
    band = ma_k - b * sd_k
    pct = band / ref_price - 1.0            # 핸드오프 추세숏 pct식
    if pct <= 0:
        return 0.0, 0.0, False
    return avg * (1 - pct), avg * (1 + pct), True


def sim_avg(d, kind, k1, b, cd_ms, max_adds=1, maxhold_ms=MAXHOLD_MS):
    """단일포지션 추매 시뮬. 비중첩(한 게임 청산 후 다음 진입, 진입 쿨다운=청산기준).
    반환: 트레이드 dict 리스트 {ret, added(bool), reason, entry_ts}."""
    C, H, L, ts, ma, sd, z, n = d["C"], d["H"], d["L"], d["ts"], d["ma"], d["sd"], d["z"], d["n"]
    side = SIDE[kind]; zs = ZSIGN[kind]
    cand = np.flatnonzero(z >= k1) if zs > 0 else np.flatnonzero(z <= -k1)
    cand = cand[cand >= WIN]
    sig = (z >= k1) if zs > 0 else (z <= -k1)   # 재발생 판정용 bar별 불리언

    out = []
    last_exit_ms = None
    ci_idx = 0
    cand = cand.tolist()
    for ci in cand:
        if ci + 1 >= n:
            break
        if last_exit_ms is not None and (ts[ci] - last_exit_ms) < cd_ms:
            continue
        price0 = C[ci]
        tp, sl, ok = _targets(kind, ma[ci], sd[ci], price0, price0, b)
        if not ok:
            continue
        entry_ts = ts[ci]
        legs = [price0]                       # 진입가들
        leg_ts = [int(entry_ts)]              # 레그별 진입 시각(포트폴리오 랏 점유용)
        avg = price0
        adds = 0
        # 전진 워크 (14일 캡)
        end_ms = entry_ts + maxhold_ms
        exit_i = None; exit_px = None; reason = None
        k = ci + 1
        while k < n and ts[k] <= end_ms:
            lo, hi = L[k], H[k]
            if side > 0:                       # LONG: 손절 우선
                if lo <= sl:
                    exit_i, exit_px, reason = k, sl, "SL"; break
                if hi >= tp:
                    exit_i, exit_px, reason = k, tp, "TP"; break
            else:                              # SHORT: 손절(고가) 우선
                if hi >= sl:
                    exit_i, exit_px, reason = k, sl, "SL"; break
                if lo <= tp:
                    exit_i, exit_px, reason = k, tp, "TP"; break
            # 추매 판정: 미추매 & 한도 & 쿨다운경과 & 신호 재발생
            if adds < max_adds and (ts[k] - entry_ts) >= cd_ms and sig[k]:
                p1 = C[k]
                new_avg = (sum(legs) + p1) / (len(legs) + 1)
                ntp, nsl, nok = _targets(kind, ma[k], sd[k], new_avg, p1, b)
                if nok:
                    legs.append(p1); leg_ts.append(int(ts[k])); avg = new_avg
                    tp, sl = ntp, nsl
                    adds += 1
            k += 1
        if exit_i is None:                     # 14일 미도달 → 종가청산
            kk = min(k, n - 1)
            exit_i, exit_px, reason = kk, C[kk], "HOLD"
        # P&L: 레그별 방향수익 평균 − fee(레그당 왕복, 평균이라 한 번 차감)
        if side > 0:
            leg_gross = [exit_px / p - 1 for p in legs]
        else:
            leg_gross = [(p - exit_px) / p for p in legs]
        gross = float(np.mean(leg_gross))
        out.append({"ret": float(gross - FEE), "added": adds > 0,
                    "reason": reason, "entry_ts": int(entry_ts),
                    "exit_ts": int(ts[exit_i]),
                    # 포트폴리오용: 레그별 (진입ts, net수익=gross−fee). 추매면 2랏.
                    "legs": [(leg_ts[i], float(leg_gross[i] - FEE)) for i in range(len(legs))]})
        last_exit_ms = int(ts[exit_i])
    return out


def stats(trs):
    r = np.array([t["ret"] for t in trs], float)
    if len(r) == 0:
        return None
    wins = r > 0
    pf = r[wins].sum() / (-r[~wins].sum()) if (~wins).any() else float("inf")
    n_add = sum(t["added"] for t in trs)
    return {
        "n": len(r), "added": int(n_add),
        "wr": round(100 * wins.mean(), 1),
        "exp": round(float(r.mean()) * 100, 3),
        "pf": round(float(pf), 2) if np.isfinite(pf) else 99.0,
        "cum": round((np.prod(1 + r) - 1) * 100, 1),     # 비중첩 복리
        "worst": round(float(r.min()) * 100, 2),
        "avgloss": round(float(r[~wins].mean()) * 100, 2) if (~wins).any() else 0.0,
    }


# 채택 파라미터 (K1, B, CD시간) — 핸드오프 표에서.
S2_REV_LONG = {   # S2 역추세 롱 (라이브 주력 계열)
    "XRPUSDT": (3.5, -0.4, 2.25), "WTI": (2.9, -2.0, 3.0), "US100": (3.25, -0.8, 1.5),
    "HK50": (2.6, -2.0, 2.0), "BTCUSDT": (3.3, -2.0, 3.0), "BTCUSD": (3.5, -2.0, 2.25),
    "GER40": (3.5, -2.0, 1.25), "JP225": (2.7, -2.0, 3.0), "UK100": (3.35, -2.0, 1.5),
    "XAGUSD": (2.85, -1.8, 3.0), "XAUTUSDT": (2.75, -1.8, 3.0), "XAUUSD": (2.35, -1.8, 3.0),
}
S2_REV_SHORT = {  # S2 역추세 숏 (양방향 강한 심볼만)
    "XRPUSDT": (5.0, -2.0, 0.5), "WTI": (3.4, 0.6, 1.0),
    "BTCUSDT": (4.6, -0.4, 0.5), "UK100": (3.8, 1.8, 0.5),
}
S1_TREND_LONG = { # S1 추세 롱 (모멘텀 심볼)
    "SOLUSDT": (3.4, -2.0, 3.0), "XAUUSD": (3.45, -1.8, 1.25), "XAGUSD": (2.75, -1.2, 3.0),
    "ETHUSD": (3.5, 1.6, 1.75), "ETHUSDT": (2.35, 1.2, 2.5), "XAUTUSDT": (3.25, -2.0, 2.0),
    "BTCUSD": (3.25, -2.0, 2.75), "BTCUSDT": (3.2, -2.0, 3.0), "XRPUSDT": (2.55, -0.4, 3.0),
    "JP225": (3.35, -2.0, 2.0), "GER40": (2.75, -1.8, 3.0), "US100": (2.8, -1.8, 2.75),
}
GROUPS = [
    ("S2 역추세 롱", "rev_long", S2_REV_LONG),
    ("S2 역추세 숏", "rev_short", S2_REV_SHORT),
    ("S1 추세 롱", "trend_long", S1_TREND_LONG),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--symbols", default=None, help="콤마구분 필터")
    args = ap.parse_args()
    filt = set(s.strip().upper() for s in args.symbols.split(",")) if args.symbols else None

    cache = {}
    def get(s):
        if s not in cache:
            cache[s] = prep(s)
        return cache[s]

    result = {}
    for title, kind, params in GROUPS:
        print("\n" + "=" * 100)
        print(f"  {title}  ({kind}) — baseline(추매0) vs 추매1회 | 1:1균등, TP=MA∓Bσ재앵커·SL대칭, 14일보유, fee0.11%")
        print("=" * 100)
        hdr = (f"{'심볼':>8} {'K1':>5}{'B':>5}{'CD':>5} | "
               f"{'n':>4}{'wr':>5}{'exp%':>7}{'PF':>5}{'cum%':>8}{'avgL':>6}{'wrst':>6}  ||  "
               f"{'n':>4}{'add':>4}{'wr':>5}{'exp%':>7}{'PF':>5}{'cum%':>8}{'avgL':>6}{'wrst':>6}  | Δexp Δcum")
        print(hdr)
        grp_rows = []
        for s, (k1, b, cdh) in params.items():
            if filt and s not in filt:
                continue
            d = get(s)
            if d is None:
                print(f"{s:>8}  (캐시없음)")
                continue
            cd = int(cdh * H_MS)
            base = stats(sim_avg(d, kind, k1, b, cd, max_adds=0))
            avg = stats(sim_avg(d, kind, k1, b, cd, max_adds=1))
            if base is None or avg is None:
                print(f"{s:>8}  (거래없음)")
                continue
            dexp = round(avg["exp"] - base["exp"], 3)
            dcum = round(avg["cum"] - base["cum"], 1)
            print(f"{s:>8} {k1:>5}{b:>5}{cdh:>5} | "
                  f"{base['n']:>4}{base['wr']:>5.0f}{base['exp']:>7.3f}{base['pf']:>5.1f}"
                  f"{base['cum']:>8.1f}{base['avgloss']:>6.1f}{base['worst']:>6.1f}  ||  "
                  f"{avg['n']:>4}{avg['added']:>4}{avg['wr']:>5.0f}{avg['exp']:>7.3f}{avg['pf']:>5.1f}"
                  f"{avg['cum']:>8.1f}{avg['avgloss']:>6.1f}{avg['worst']:>6.1f}  | "
                  f"{dexp:>+5.3f} {dcum:>+6.1f}")
            grp_rows.append({"sym": s, "k1": k1, "b": b, "cd_h": cdh,
                             "base": base, "avg": avg, "d_exp": dexp, "d_cum": dcum})
        # 그룹 풀링 요약(Δexp 평균, 개선 심볼 수)
        if grp_rows:
            imp = sum(1 for r in grp_rows if r["d_exp"] > 0)
            mean_dexp = round(float(np.mean([r["d_exp"] for r in grp_rows])), 3)
            mean_dcum = round(float(np.mean([r["d_cum"] for r in grp_rows])), 1)
            print("-" * 100)
            print(f"  그룹요약: {len(grp_rows)}심볼 중 추매로 exp개선 {imp} | "
                  f"평균Δexp {mean_dexp:+.3f}%p · 평균Δcum {mean_dcum:+.1f}%p")
        result[kind] = {"title": title, "rows": grp_rows}

    if args.json:
        json.dump(result, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n저장: {args.json}")


if __name__ == "__main__":
    main()

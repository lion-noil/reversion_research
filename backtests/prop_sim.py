# -*- coding: utf-8 -*-
"""프랍(FXIFY/FTMO류) 통과확률 시뮬 (2026-09-04) — CFD 책(U2 비환율 + U3 환율)을
자산비 사이징 스케일별로 굴려 프랍 규칙 위반/통과를 롤링 윈도로 카운트.

프랍 규칙(FXIFY 2-step 표준, 파라미터화):
  - 이익목표 PROFIT (예 1단계 +8%, 2단계 +5%)
  - 최대 총낙폭 MAXDD (예 −10%, 시작잔고 고정 기준 = static)
  - 일일 최대낙폭 DAILYDD (예 −5%, 매일 자정 리셋)
  - 최소 거래일 MIN_DAYS
평가 = 5년 데이터에서 매일을 시작점으로 슬라이딩(중복 챌린지 N개) → 각 챌린지가
  이익목표 먼저 = ✅통과 / 일일·총 낙폭 먼저 = ❌실격 / 기간내 미달 = ⏳
핵심 산출: 스케일별 [통과율, 실격율, 평균 통과일수]. MTM(장중 저가/고가)로 낙폭 판정.

실행: python prop_sim.py [U2|U3|BOTH] [scale=8]   (scale=자산비 배수, 라이브 1%→scale%)
"""
import os
import sys
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# universe_report의 셀 생성부를 재사용 — trades 리스트를 그대로 가져온다.
import importlib.util
_ur = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_report.py")

H = 3600 * 1000
DAY = 86400 * 1000
KST_OFF = 0   # 프랍 일일 리셋은 보통 브로커=미국동부 5pm이나, 근사로 UTC 자정 사용

# 프랍 규칙 프리셋 (%는 시작잔고 대비)
RULES = dict(
    profit_s1=0.08, profit_s2=0.05,   # FXIFY 2-step
    max_dd=0.10,       # 총낙폭(static, 시작잔고 고정 기준)
    daily_dd=0.05,     # 일일낙폭
    min_days=3,        # 최소거래일
    challenge_days=60, # 평가창(무제한이지만 현실적 상한)
)


def load_trades(uni, scale_mult):
    """universe_report를 그 유니버스로 실행해 trades를 회수. sizing은 scale_mult배.
    반환: [(en_ms, ex_ms, ret, sym, side, ep, book, frac_scaled)]"""
    os.environ["U"] = uni
    spec = importlib.util.spec_from_file_location(f"ur_{uni}", _ur)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["universe_report.py", uni]
    # universe_report는 import 시 전부 실행되고 print까지 하므로, trades만 뽑아 쓴다.
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    trades = [(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7] * scale_mult) for t in mod.trades]
    # MTM 마크 함수도 재사용
    return trades, mod.mark, mod.T5Y


def simulate_equity(trades, mark, t5y, max_concurrent=None, max_daily_new=None):
    """시간단위 MTM 시퀀스 생성 → [(hour, realized_eq, mtm_eq)]. eq는 시작 1.0 정규화.
    필터(프랍 일일DD 방어): max_concurrent=동시보유 셀 상한(초과 시 신규 스킵),
      max_daily_new=하루 신규진입 상한(같은 UTC일 내). 스킵된 진입은 그 거래 자체를 버림."""
    ev = []
    for i, t in enumerate(trades):
        ev.append((t[0], 0, i)); ev.append((t[1], 1, i))
    ev.sort(key=lambda x: (x[0], x[1]))
    eq = 1.0
    open_ = {}
    ptr = 0
    t0 = t5y // H
    t1 = max(e[0] for e in ev) // H
    seq = []
    day_new = defaultdict(int)   # UTC일 → 그날 채택된 신규 진입 수
    for hour in range(t0, t1 + 1):
        tms = (hour + 1) * H
        while ptr < len(ev) and ev[ptr][0] < tms:
            en_ms, typ, i = ev[ptr]; ptr += 1
            if typ == 0:
                # 필터: 동시보유·일일신규 상한
                if max_concurrent is not None and len(open_) >= max_concurrent:
                    continue
                d = en_ms // DAY
                if max_daily_new is not None and day_new[d] >= max_daily_new:
                    continue
                open_[i] = trades[i][7] * eq   # notional
                day_new[d] += 1
            else:
                no = open_.pop(i, None)
                if no is not None:
                    eq += no * trades[i][2]
        un = 0.0
        for i, no in open_.items():
            en, ex, r, sym, side, ep, bk, fr = trades[i]
            m = mark(sym, hour, side, ep)
            un += no * side * (m / ep - 1.0)
        seq.append((hour, eq, eq + un))
    return seq


def run_challenges(seq, rules, profit_target):
    """매일을 시작점으로 챌린지 롤링. 반환 dict(pass, fail_daily, fail_total, timeout, n, pass_days[])."""
    # 일 단위 인덱싱
    by_hour = {h: (r, m) for (h, r, m) in seq}
    hours = [h for (h, r, m) in seq]
    h0, h1 = hours[0], hours[-1]
    starts = list(range(h0 - h0 % 24, h1 - rules["challenge_days"] * 24, 24))  # 매일 00:00 시작
    res = dict(passed=0, fail_daily=0, fail_total=0, timeout=0, n=0, pass_days=[])
    maxwin = rules["challenge_days"] * 24
    for s in starts:
        if s not in by_hour:
            continue
        start_eq = by_hour[s][0]   # 그날 시작 실현자산
        if start_eq <= 0:
            continue
        day_start_mtm = by_hour[s][1]
        peak_mtm = by_hour[s][1]   # 트레일링DD용 최고 MTM
        cur_day = s // 24
        traded_days = set()
        outcome = None
        for h in range(s, min(s + maxwin, h1) + 1):
            if h not in by_hour:
                continue
            realized, mtm = by_hour[h]
            # 정규화(시작 대비)
            r_ret = realized / start_eq - 1
            peak_mtm = max(peak_mtm, mtm)
            traded_days.add(h // 24)
            # 일일 리셋
            if h // 24 != cur_day:
                cur_day = h // 24
                day_start_mtm = mtm
            # 실격 판정 (MTM 기준). 트레일링=peak 대비, static=시작 대비.
            dd_base = peak_mtm if rules.get("trailing") else start_eq
            if (mtm / dd_base - 1) <= -rules["max_dd"]:
                outcome = "fail_total"; break
            if (mtm / day_start_mtm - 1) <= -rules["daily_dd"]:
                outcome = "fail_daily"; break
            # 통과 판정 (실현 이익 + 최소거래일)
            if r_ret >= profit_target and len(traded_days) >= rules["min_days"]:
                outcome = "passed"; res["pass_days"].append((h - s) / 24); break
        res["n"] += 1
        if outcome:
            res[outcome] += 1
        else:
            res["timeout"] += 1
    return res


def run_funded_rounds(seq, rules, payout_target):
    """funded 계좌 라운드 — 각 시작점에서 '출금(payout_target 도달)' vs '파산(DD)' vs '지속(timeout)'.
    payout·death 확률을 반환(EV 모델 입력). 규칙은 챌린지와 동일 DD, 목표만 낮음."""
    r = run_challenges(seq, dict(rules, min_days=1), payout_target)
    n = r["n"] or 1
    death = r["fail_daily"] + r["fail_total"]
    return dict(payout_prob=r["passed"] / n, death_prob=death / n, timeout=r["timeout"] / n)


def run_filter_sweep(uni):
    """일일DD 방어 필터 스윕 — FXIFY룰(목표8%·static10%·일일5%) 고정, 스케일×필터 격자.
    필터: max_concurrent(동시보유 셀 상한) × max_daily_new(하루 신규 상한)."""
    rules = dict(RULES, max_dd=0.10, daily_dd=0.05, trailing=False)
    print(f"\n{'='*70}\n[{uni}] 일일DD 방어 필터 스윕 (FXIFY룰 목표8%·static10%·일일5%)")
    print(f"{'스케일':>5} {'동시상한':>6} {'일일신규':>6} | {'통과율':>6} {'일일실격':>7} {'미달':>5} | {'평균일':>6} {'5년배수':>7}")
    filters = [(None, None), (6, None), (4, None), (3, None), (None, 3), (None, 2), (4, 2), (3, 2)]
    for scale in (5.0, 8.0):
        trades, mark, t5y = load_trades(uni, scale)
        for mc, mdn in filters:
            seq = simulate_equity(trades, mark, t5y, max_concurrent=mc, max_daily_new=mdn)
            r1 = run_challenges(seq, rules, 0.08)
            n = r1["n"] or 1
            pd = r1["pass_days"]
            avgd = f"{np.mean(pd):.0f}일" if pd else "-"
            print(f"{scale:>4.0f}% {str(mc):>6} {str(mdn):>6} | {100*r1['passed']/n:>5.1f}% "
                  f"{100*r1['fail_daily']/n:>6.1f}% {100*r1['timeout']/n:>4.1f}% | {avgd:>6} | {seq[-1][1]:>6.1f}x")


def main():
    unis = ["U2", "U3"] if (len(sys.argv) < 2 or sys.argv[1].upper() == "BOTH") else [sys.argv[1].upper()]
    if len(sys.argv) > 2 and sys.argv[2] == "filter":
        for uni in unis:
            run_filter_sweep(uni)
        return
    if len(sys.argv) > 2 and sys.argv[2] == "funded":
        # funded 라운드 확률 실측 (EV 입력): 스케일별 phase1통과·payout·death
        rules = dict(RULES, max_dd=0.10, daily_dd=0.05, trailing=False)
        print(f"[{unis[0]}] funded 라운드 실측 (static10/일일5) — phase1(+8%)·payout(+5%)·death")
        print(f"{'스케일':>5} {'필터':>8} | {'p1(+8%)':>7} {'payout(+5%)':>11} {'death':>6}")
        for scale in (5.0, 8.0):
            trades, mark, t5y = load_trades(unis[0], scale)
            for mdn in (None, 2):
                seq = simulate_equity(trades, mark, t5y, max_daily_new=mdn)
                p1 = run_challenges(seq, rules, 0.08)
                fr = run_funded_rounds(seq, rules, 0.05)
                p1r = p1["passed"] / (p1["n"] or 1)
                print(f"{scale:>4.0f}% {('일일'+str(mdn)) if mdn else '무필터':>8} | {p1r*100:>6.1f}% "
                      f"{fr['payout_prob']*100:>10.1f}% {fr['death_prob']*100:>5.1f}%")
        return
    scales = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["4", "8", "12", "16", "20"])]
    # 프랍사 변형: FTMO류 트레일링DD(peak 기준) vs static, 목표 8/10%
    variants = [
        ("FXIFY 2step (목표8%·static10%)", 0.08, 0.10, 0.05, False),
        ("FTMO (목표10%·트레일링10%)", 0.10, 0.10, 0.05, True),
        ("완화 (목표8%·static12%·일일6%)", 0.08, 0.12, 0.06, False),
    ]

    for uni in unis:
        print(f"\n{'='*70}\n[{uni}] 프랍 통과확률 — 사이징 스케일별 (라이브 1% × 배수)")
        print(f"규칙: 1단계 +{RULES['profit_s1']*100:.0f}% / 총DD −{RULES['max_dd']*100:.0f}%(static) / "
              f"일일DD −{RULES['daily_dd']*100:.0f}% / 최소{RULES['min_days']}일 / 평가창 {RULES['challenge_days']}일")
        # 스케일별 equity 시퀀스는 1회만 계산해 재사용(변형은 규칙만 바뀜)
        seqs = {}
        for scale in scales:
            trades, mark, t5y = load_trades(uni, scale)
            seqs[scale] = (simulate_equity(trades, mark, t5y))
        for vname, profit, maxdd, daily, trailing in variants:
            print(f"\n  ▶ {vname}")
            print(f"  {'스케일':>5} | {'통과율':>6} {'일일실격':>7} {'총DD실격':>7} {'미달':>5} | {'평균통과일':>8} | 5년배수")
            rules = dict(RULES, max_dd=maxdd, daily_dd=daily, trailing=trailing)
            for scale in scales:
                seq = seqs[scale]
                r1 = run_challenges(seq, rules, profit)
                n = r1["n"] or 1
                pd = r1["pass_days"]
                avgd = f"{np.mean(pd):.0f}일" if pd else "-"
                print(f"  {scale:>4.0f}% | {100*r1['passed']/n:>5.1f}% {100*r1['fail_daily']/n:>6.1f}% "
                      f"{100*r1['fail_total']/n:>6.1f}% {100*r1['timeout']/n:>4.1f}% | {avgd:>8} | {seq[-1][1]:.2f}x")


if __name__ == "__main__":
    main()

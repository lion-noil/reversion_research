# -*- coding: utf-8 -*-
"""프랍 다회도전 기대값(EV) 계산 — 2026-09-04.
prop_sim 통과율을 실제 프랍사 요금/배분/환급 구조에 대입해 원화 기대이익 산출.

모델(2-step 챌린지):
  - phase1 통과확률 p1, phase2 통과확률 p2(≈p1, 목표 낮아 소폭↑) → funded 도달 = p1·p2
  - funded 후 첫 출금까지 목표수익 도달확률 pw, 출금액 = 계좌×수익목표×배분율
  - 챌린지피 fee, 통과 시 환급 refund(보통 첫 출금 때)
  - in-sample 보정계수 IS(실전 통과율 = 시뮬×IS)
반복 도전 EV = (도전당 비용) 대비 (funded 도달 시 순이익) 기대.
usage: py -3 prop_ev.py
"""

# ── 시뮬 통과율(무필터 8% vs 일일신규2 필터) — prop_sim.py U2 결과 ──
# (phase1 기준. 실전은 IS 보정)
# prop_sim.py U2 funded 실측 (static10/일일5): p1(+8% phase1), payout_prob(+5%), death_prob
SIM = {
    "무필터 5%":   dict(p1=0.270, payout_prob=0.350, death_prob=0.293),
    "무필터 8%":   dict(p1=0.277, payout_prob=0.352, death_prob=0.386),
    "일일2필터 5%": dict(p1=0.246, payout_prob=0.286, death_prob=0.166),
    "일일2필터 8%": dict(p1=0.263, payout_prob=0.359, death_prob=0.248),
}
IS = 0.65   # in-sample 보정(시뮬은 튜닝데이터라 실전은 0.6~0.7배 보수)

# ── 프랍사 프리셋 (2026 기준 대표값, 10만달러 계좌 · 원화환산 1450원/$) ──
FX = 1450
PROPS = {
    "FXIFY 2-step $100k": dict(fee_usd=439, refund=True, split=0.90, steps=2, payout_target=0.05),
    "FTMO 2-step $100k":  dict(fee_usd=540, refund=True, split=0.90, steps=2, payout_target=0.05, trailing=True),
    "FXIFY 1-step $100k(가정)": dict(fee_usd=549, refund=True, split=0.90, steps=1, payout_target=0.05),
    "인스턴트 $100k(가정,피↑무평가)": dict(fee_usd=999, refund=False, split=0.90, steps=0, payout_target=0.05),
}


def ev_for(prop, sim, is_factor):
    p1 = min(1.0, sim["p1"] * is_factor)
    p2 = min(1.0, p1 * 1.15)               # phase2 목표 낮아(5%) 소폭↑
    steps = prop.get("steps", 2)
    # funded 계좌 라운드: 출금(5%도달) vs 파산(일일/총DD) — sim 라운드 결과에서 유도.
    #   payout_prob = 라운드가 출금으로 끝날 확률, death_prob = 파산으로 끝날 확률.
    #   (timeout은 계좌 유지·재시도 → 흡수). is 보정 적용.
    payout_prob = min(1.0, sim.get("payout_prob", 0.32) * is_factor)
    death_prob = sim.get("death_prob", 0.35)   # 파산은 보정 안 함(리스크는 보수적으로)
    if prop.get("trailing"):
        p1 *= 0.85; p2 *= 0.85; payout_prob *= 0.85; death_prob *= 1.15
    p_funded = {2: p1 * p2, 1: p1, 0: 1.0}[steps]   # 2step/1step/인스턴트
    # funded 계좌 기대 출금 횟수: 매 라운드 출금/파산 경쟁 → E[출금] = payout/(payout+death) × 1/death
    #   = 파산 전까지 기대 출금 수. (기하: 각 라운드 death로 끝날 때까지의 payout 수)
    pr, dr = payout_prob, death_prob
    exp_payouts = pr / dr if dr > 0 else 20.0

    fee = prop["fee_usd"] * FX
    payout = 100000 * prop["payout_target"] * prop["split"] * FX
    refund = fee if prop["refund"] else 0.0

    exp_attempts = 1.0 / p_funded if p_funded > 0 else 999
    cost_to_funded = exp_attempts * fee
    # funded 도달 후 총 수령 = 환급 + 기대출금횟수 × 출금액
    gain = refund + exp_payouts * payout
    ev = gain - cost_to_funded
    return dict(p1=p1, p_funded=p_funded, exp_attempts=exp_attempts, exp_payouts=exp_payouts,
                cost=cost_to_funded, payout=payout, gain=gain,
                ev=ev, roi=ev / cost_to_funded if cost_to_funded else 0)


print(f"환율 {FX}원/$ · in-sample 보정 ×{IS} · 100k 계좌 기준\n")
for pname, prop in PROPS.items():
    print(f"### {pname}  (fee ${prop['fee_usd']}={prop['fee_usd']*FX:,.0f}원, 배분 {prop['split']*100:.0f}%, "
          f"환급 {'O' if prop['refund'] else 'X'}{', 트레일링DD' if prop.get('trailing') else ''})")
    print(f"{'전략/사이징':>14} | {'funded확률':>9} {'평균도전':>7} {'기대출금':>7} | {'funded비용':>11} | "
          f"{'출금1회':>9} | {'EV':>11} {'ROI':>6}")
    for sname, sim in SIM.items():
        e = ev_for(prop, sim, IS)
        print(f"{sname:>14} | {e['p_funded']*100:>7.1f}% {e['exp_attempts']:>5.1f}회 {e['exp_payouts']:>5.1f}회 | "
              f"{e['cost']:>10,.0f}원 | {e['payout']:>8,.0f}원 | {e['ev']:>+10,.0f}원 {e['roi']*100:>+5.0f}%")
    print()

print("모델: funded=phase1×phase2 통과. funded 후 매 라운드 출금(+5%)vs파산(DD) 경쟁 → 기대출금=payout/death회.")
print("  비용=funded까지 평균 도전 fee 누적. 기대출금×출금액(배분반영) − 비용 = EV. in-sample ×0.65 보정.")
print("  ⚠️파산확률은 보정 안 함(리스크 보수적). 계좌 재구매·funded 재도전 미반영. 실전은 슬리피지로 소폭 불리.")

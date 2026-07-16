# S3 · σ-복귀 역추세 **숏온리** — tradingBot 핸드아웃

> reversion_research 백테스트(`backtests/persym_sweep.py` DIR=short)에서 검증한 역추세 숏 전략을
> 라이브 봇 **S3**으로 이관하기 위한 명세. 작성 2026-06-26.
> 기존 `strategies/s1_reversion.py`(S1 역추세롱 / S2 추세숏)와 같은 자족 모듈 스타일로 추가한다.

---

## 1. 한 줄 정의

**과열(z ≥ +K1)에서 숏 진입 → MA 쪽으로 되돌림(평균회귀)에 베팅.** S1 역추세롱의 정확한 거울.

| 전략 | 진입 신호 | 방향 | 베팅 | TP |
|---|---|---|---|---|
| S1 | z ≤ −K1 (과매도) | 롱 | 평균회귀(상승) | MA − B·σ (진입가 **위**) |
| S2 | z ≤ −K1 (과매도) | 숏 | 추세지속(하락) | 미러(진입가 아래) |
| **S3** | **z ≥ +K1 (과열)** | **숏** | **평균회귀(하락)** | **MA + B·σ (진입가 아래)** |

⚠️ S3는 진입 신호가 S2와 **반대**다(S2=z≤−K1, S3=z≥+K1). 혼동 금지.

---

## 2. 신호 규칙 (백테스트와 1:1)

- 지표: `MA` = win분 종가 이동평균, `SD` = 같은 창 표준편차(모집단), `z = (price − MA)/SD`. **win=10080 고정**(1분봉 7일).
- 진입(flat & 쿨다운 통과 시): **z ≥ +K1**
  - `TP_price = MA + B·σ`  — **가드: `TP_price < entry`** (아니면 진입 스킵; B<0도 허용)
  - `pct = 1 − TP_price/entry`  (> 0)
  - `SL_price = entry × (1 + pct)`  (대칭, 진입가 위)
- 청산(봉 고저 기준, **손절 우선**):
  - `high ≥ SL_price` → **SL**
  - `low ≤ TP_price` → **TP**
  - 보유 ≥ **14일** 경과 → **TIMEOUT**(종가 시장가 청산)  ← 백테스트의 최대보유 룰
- 쿨다운: 아래 §5 참고(백테스트는 "직전 **진입**" 기준, 피라미딩 허용).
- 수수료: 왕복 0.11%(리포팅용, 체결가엔 미반영).

---

## 3. 검증된 파라미터 (in-sample, 데이터 ~2.97년 · fee0.11% · 최대보유14일)

견고 등급(채택K ±0.6 능선양수율 = 1.0, 즉 K 주변이 평탄대 흑자)만 채택 후보로 추림.

### 크립토 (USDT — 봇이 직접 거래)
| 심볼 | K1 | B | 쿨다운 | n (회/년) | 승률 | 기대/건 | PF | maxc |
|---|---|---|---|---|---|---|---|---|
| **XRPUSDT** ⭐ | 5.0 | −2.0 | 30m | 93 (31) | 57% | **+3.09%** | 1.77 | 13 |
| **BTCUSDT** | 4.6 | −0.4 | 30m | 89 (30) | 60% | +1.10% | 1.53 | 14 |

XRP는 K4.6~5.4 넓은 평탄대(정점 K5.3 +3.39%), BTC는 K4.5~4.9 단단. 둘 다 단일 스파이크 아님.
⚠️ **MT5 BTCUSD/ETHUSD(브로커 피드)는 숏 노이즈** — USDT 피드와 다름. 반드시 USDT 사용.

### MT5 지수/원자재 (봇이 MT5 심볼 거래 가능할 때만)
| 심볼 | K1 | B | 쿨다운 | n (회/년) | 승률 | 기대/건 | PF | maxc |
|---|---|---|---|---|---|---|---|---|
| **WTI** ⭐ | 3.5 | 0.6 | 1h | 66 (22) | 59% | **+3.04%** | 2.21 | 14 |
| **UK100** | 3.8 | 1.8 | 30m | 84 (28) | 73% | +0.36% | 2.88 | 14 |
| **JP225** | 3.8 | 1.0 | 45m | 81 (27) | 48% | +0.29% | 1.29 | 14 |

- WTI는 K올릴수록 단조상승(+1%→+8%). 빈도 30회/년 원하면 K3.4(n81, +2.38%, PF1.95).
- 제외: SOLUSDT·XAUTUSDT·ETHUSDT(저K)·US100·GER40·XAUUSD(최적에서도 음수).

---

## 4. 드롭인 코드 (strategies/s1_reversion.py 에 추가)

```python
# ─────────────────────────────────────────────────────────────────────────────
# S3 (역추세 숏) — 진입 z ≥ +K1(과열)에서 숏, MA로 되돌림 베팅. S1 롱의 거울.
#   TP = MA + B·σ (진입가 아래, 가드 pct>0). SL = 진입가×(1+pct) [위, 대칭].
# ─────────────────────────────────────────────────────────────────────────────
def s3_entry_levels(z, ma, sd, price, p: "S1Params"):
    """S3 역추세숏 진입 레벨. 충족 시 (tp_price, sl_price) 반환(tp<진입가<sl), 아니면 None.
    호출측에서 flat & 쿨다운 선확인. p.k1/p.b 재사용(b<k1 가드는 levels에서 pct>0로 대체)."""
    if z is None or ma is None or sd is None or sd <= 0:
        return None
    if z < p.k1:                      # 과열 임계 미달
        return None
    tp_price = ma + p.b * sd          # B<0 허용(되돌림 목표가 MA 아래까지)
    if tp_price >= price:             # 가드: TP는 진입가 아래여야(pct>0)
        return None
    pct = 1.0 - tp_price / price
    sl_price = price * (1.0 + pct)    # 위(숏 손절), 대칭
    return tp_price, sl_price


def s3_exit_on_candle(pos: "S1Position", high: float, low: float):
    """S3 숏 청산(봉 고저, 손절 우선). 위(sl) 닿으면 SL, 아래(tp) 닿으면 TP."""
    if high >= pos.sl_price:
        return "SL"
    if low <= pos.tp_price:
        return "TP"
    return None


def s3_exit_on_tick(pos: "S1Position", price: float):
    """S3 숏 청산(라이브 틱). 손절 우선."""
    if price >= pos.sl_price:
        return "SL"
    if price <= pos.tp_price:
        return "TP"
    return None
```

- `S1Params`/`S1Position` 재사용(필드 동일). `k1`은 심볼별 §3 값으로 주입.
- 14일 최대보유 TIMEOUT은 상위 루프에서 `now - pos.entry_ts_ms >= 14*86400_000` 체크로 추가.

---

## 5. ⚠️ 라이브 구현 핵심 결정 — 피라미딩(중복보유)

백테스트는 **"직전 진입 이후 쿨다운"** 기준이고 **플랫 게이트가 없어** 포지션이 쌓인다 → 그래서
`maxc`(최대 동시보유)가 13~14까지 나온다. 즉 위 성적은 **한 과열장에 숏을 여러 번 겹쳐 진입**한
결과다. 반면 S1 라이브 모듈은 `s1_cooldown_ok(last_exit...)` + 플랫게이트 = **단일 포지션**.

→ **선택 필요:**
- **(A) 충실 재현**: S3는 z≥K1 + (직전 **진입** 후 쿨다운) 충족 시 보유 여부 무관하게 신규 진입,
  심볼당 최대 `maxc`개까지 스택. 각 포지션은 1/maxc로 사이징. → 백테스트와 일치하나 다포지션 관리 필요.
- **(B) 단일 포지션(간단)**: 기존 S1처럼 플랫게이트+청산후쿨다운. 구현 쉬우나 **거래수↓·성적 달라짐**
  → 이 경우 단일포지션 가정으로 **재백테스트 후** 파라미터 재확정할 것.

**권장: (A)로 가되 심볼당 동시포지션 상한(예 ≤8)·총 자본배분 상한을 반드시 건다.** maxc14는 과열장 집중위험.

---

## 6. 그 외 주의

- **전부 in-sample 최적.** 라이브 전 분기별 OOS(워크포워드) 검증 필수. 소표본(n80대) 고PF는 ±변동 큼.
- 쿨다운 30m~1h로 짧다 → 진입 잦음. (A) 채택 시 슬리피지·펀딩·수수료 누적 재점검.
- 진입 z는 직전 7일(10080분) MA/σ 기준 — 봉 확정마다 갱신. 부팅 시 7일치 종가 워밍업 필요.
- 심볼별 K가 제각각(3.5~5.0)이니 S3 파라미터는 **심볼별 테이블**로 관리(전역 단일값 금지).

---

## 7. 셀프테스트(이관 검증)

`strategies/s1_reversion.py`의 `_selftest` 패턴을 S3용으로 복제해, reversion_research 캐시로
백테스트 수치 재현 확인:

```
기대(피라미딩 OFF·단일포지션 기준은 위 표와 다를 수 있음. 표는 피라미딩 ON 기준)
XRPUSDT K5.0 B-2.0 CD30m → n93 승률57% 기대+3.09%/건 PF1.77 (maxc13)
BTCUSDT K4.6 B-0.4 CD30m → n89 승률60% 기대+1.10%/건 PF1.53 (maxc14)
```

재현 스크립트: `backtests/persym_sweep.py`(env S1_DIR=short, S1_K1_MIN/MAX/STEP, S1_NMIN, S1_SYMS,
S1_TAG, S1_MAXHOLD_D=14) → 리포트 `build_ridge_report.py`(RIDGE_KMAX) → HTML `build_short_hik_html.py`.
실행 파이썬은 anaconda3(numpy). 산출물: `s1_best_short_hik.html`(MT5), `s1_best_crypto_short_hik.html`(크립토).

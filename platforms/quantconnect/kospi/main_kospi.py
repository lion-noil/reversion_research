# -*- coding: utf-8 -*-
"""QuantConnect(LEAN) 이식 — 코스피200 ETF(KODEX200, 069500) ALIGN3 3층정렬 추세롱 (kis_trader/align3_executor.py 라이브 로직, 2026-10-25).

규칙(라이브와 동일):
- 5분봉 완결 시점에 z1d(완결 일봉 종가 20개) ≥ +1 & z4h(세션봉 종가 20개: 09–13시 / 13–15:30) ≥ +1 & z5m(5분 종가 390개) ≥ +1 → 종가 시장가 매수
- 청산: TP/SL 없음. 7거래일 뒤 진입 시각 봉(15:10 상한)에서 시장가 매도 (거래일 = 데이터에 봉이 있는 날)
- 멀티로트: 진입 쿨다운 12봉, 동시로트 상한 = min(20, 평가액//현재가), 슬롯 = 평가액/상한, 주수 = min(슬롯, 현금)//현재가
- 진입은 15:15 봉까지. z = (마지막 − 평균)/모집단 표준편차.
데이터: 프로젝트 파일 kdata_NN.py (KODEX200 5분 종가, 하루 한 줄 평문 — 크레온 5분봉 2021-07~2026-08 + ALIGN3 실행기 수집 5분봉 ~2026-10-23).
initialize 에서 로컬 CSV 로 쓰고 LOCAL_FILE 커스텀 데이터로 읽는다(무료 조직은 Object Store 를 백테스트에서 못 읽음).
봉 시각은 KST 라벨을 그대로 쓰고 알고리즘 시간대는 바꾸지 않는다(날짜·시각 숫자만 의미 있음 — 월·일 집계 경계는 KST 날짜와 일치).
비용: 매수·매도 각 0.025% (왕복 0.05%, 연구 엔진과 동일). 체결 = 신호 봉 종가(LEAN 즉시체결, 커스텀 데이터 마지막 가격). 통화 단위는 원(표시는 $).
"""
from AlgorithmImports import *
import math, os, importlib

W_1D, W_4H, W_5M = 20, 20, 390
CD_BARS, HOLD_TDAYS, CAP_MAX = 12, 7, 20
ENTRY_LAST, EXIT_CLAMP = "1515", "1510"
FEE_SIDE = 0.00025
YEARS = (2021, 2026)
PATHS = {}


def unpack_data():
    """kdata_NN.py 평문("yyyyMMdd @HHMM c c ... [@HHMM c ...]") → 연도별 로컬 csv(time,close). 없는 연도는 get_source 가 가장 가까운 연도를 쓴다."""
    rows = {}
    for i in range(1, 100):
        try:
            mod = importlib.import_module(f"kdata_{i:02d}")
        except Exception:
            break
        for ln in mod.TXT.splitlines():
            toks = ln.split()
            if len(toks) < 3 or not toks[0].isdigit():
                continue
            d = toks[0]
            t = None
            for tok in toks[1:]:
                if tok.startswith("@"):
                    t = datetime.strptime(d + tok[1:], "%Y%m%d%H%M")
                    continue
                rows.setdefault(d[:4], []).append(f"{t.strftime('%Y%m%d %H:%M')},{tok}")
                t = t + timedelta(minutes=5)
    for y, lst in rows.items():
        p = os.path.join(os.getcwd(), f"kospi_069500_5m_{y}.csv")
        with open(p, "w") as f:
            f.write("time,close\n" + "\n".join(lst) + "\n")
        PATHS[int(y)] = p


class Kodex5m(PythonData):
    def get_source(self, config, date, is_live):
        y = min(max(date.year, YEARS[0]), YEARS[1])
        if y not in PATHS:
            y = min(PATHS, key=lambda k: abs(k - y))
        return SubscriptionDataSource(PATHS[y], SubscriptionTransportMedium.LOCAL_FILE, FileFormat.CSV)

    def reader(self, config, line, date, is_live):
        if not line or not line[0].isdigit():
            return None
        p = line.split(",")
        if len(p) < 2:
            return None
        d = Kodex5m()
        d.symbol = config.symbol
        t = datetime.strptime(p[0], "%Y%m%d %H:%M")
        d.time = t
        d.end_time = t + timedelta(minutes=5)
        d.value = float(p[-1])
        return d


class SideFee(FeeModel):
    def get_order_fee(self, parameters):
        v = abs(float(parameters.order.get_value(parameters.security)))
        return OrderFee(CashAmount(v * FEE_SIDE, "USD"))


def zscore(vals, w):
    if len(vals) < w:
        return None
    seg = vals[-w:]
    m = sum(seg) / w
    var = sum((x - m) * (x - m) for x in seg) / w
    if var <= 0:
        return None
    return (seg[-1] - m) / math.sqrt(var)


class KospiAlign3(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2021, 7, 13)
        self.set_end_date(2026, 10, 24)
        self.set_cash(10_000_000)
        unpack_data()
        if not PATHS:
            raise Exception("kdata_<연도>.py 를 풀지 못함 — 프로젝트 파일 확인")
        self.debug(f"data files: {sorted(PATHS)}")
        sec = self.add_data(Kodex5m, "KODEX200", Resolution.MINUTE)
        sec.set_fee_model(SideFee())
        self.sym = sec.symbol
        self.set_benchmark(lambda dt: 1.0)
        self.closes = []
        self.daily_dates, self.daily_closes = [], []
        self.sess_closes = []
        self.cur_date = None
        self.cur_last = None
        self.am_last = None
        self.pm_last = None
        self.am_done = False
        self.lots = []
        self.last_entry_idx = -10 ** 9
        self.trades = []
        self.n_entries = 0

    def on_data(self, data):
        bar = data.get(self.sym)
        if bar is None:
            return
        px = float(bar.value)
        d = bar.time.strftime("%Y%m%d")
        hhmm = bar.time.strftime("%H%M")
        # 날짜 바뀜 → 전일 일봉·오후 세션 확정
        if self.cur_date is not None and d != self.cur_date:
            self.daily_dates.append(self.cur_date)
            self.daily_closes.append(self.cur_last)
            if self.pm_last is not None:
                self.sess_closes.append(self.pm_last)
            self.am_last = None
            self.pm_last = None
            self.am_done = False
        self.cur_date = d
        self.cur_last = px
        if hhmm < "1300":
            self.am_last = px
        else:
            if not self.am_done and self.am_last is not None:
                self.sess_closes.append(self.am_last)   # 오전 세션 확정(13:00 봉부터 사용)
                self.am_done = True
            self.pm_last = px
        self.closes.append(px)
        idx = len(self.closes) - 1

        # 1) 청산
        for lot in list(self.lots):
            held = sum(1 for dd in self.daily_dates if dd > lot["d"]) + (1 if d > lot["d"] else 0)
            if held >= HOLD_TDAYS and hhmm >= min(lot["t"], EXIT_CLAMP):
                self.market_order(self.sym, -lot["shares"], tag="exit")
                r = px / lot["px"] - 1 - 2 * FEE_SIDE
                self.trades.append((lot["d"], lot["t"], d, hhmm, lot["px"], px, lot["shares"], r))
                self.lots.remove(lot)

        # 2) 진입
        if hhmm > ENTRY_LAST:
            return
        z1 = zscore(self.daily_closes, W_1D)
        z4 = zscore(self.sess_closes, W_4H)
        z5 = zscore(self.closes, W_5M)
        if z1 is None or z4 is None or z5 is None or not (z1 >= 1 and z4 >= 1 and z5 >= 1):
            return
        if idx - self.last_entry_idx < CD_BARS:
            return
        equity = float(self.portfolio.total_portfolio_value)
        cash = float(self.portfolio.cash)
        cap = min(CAP_MAX, max(1, int(equity // px)))
        if len(self.lots) >= cap:
            return
        slot = equity / cap
        shares = int(min(slot, cash) // px)
        if shares <= 0:
            return
        self.market_order(self.sym, shares, tag="entry")
        self.lots.append({"d": d, "t": hhmm, "px": px, "shares": shares})
        self.last_entry_idx = idx
        self.n_entries += 1

    def on_end_of_algorithm(self):
        n = len(self.trades)
        rs = [t[7] * 100 for t in self.trades]
        mean = sum(rs) / n if n else 0.0
        tot = sum(rs)
        wins = sum(1 for r in rs if r > 0)
        self.log("cell,sym,tf,strat,side,n,mean_r_pct,sum_r_pct,tp,sl,time")
        self.log(f"0,KODEX200,5m,align3,long,{n},{mean:.3f},{tot:.2f},0,0,{n}")
        self.log(f"open lots at end: {len(self.lots)} · entries {self.n_entries} · wins {wins}/{n}")
        years = {}
        for t in self.trades:
            years.setdefault(t[2][:4], []).append(t[7] * 100)
        for y in sorted(years):
            v = years[y]
            self.log(f"year,{y},{len(v)},{sum(v):.2f},{sum(v)/len(v):.3f}")
        for t in self.trades:
            self.log(f"trade,{t[0]} {t[1]},{t[2]} {t[3]},{t[4]:.0f},{t[5]:.0f},{t[6]},{t[7]*100:+.3f}")

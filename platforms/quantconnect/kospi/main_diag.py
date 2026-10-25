# -*- coding: utf-8 -*-
"""QuantConnect(LEAN) 진단 — Object Store 커스텀 데이터가 읽히는지 확인(결과는 예외 메시지로 전달)."""
from AlgorithmImports import *


class Kodex5m(PythonData):
    def get_source(self, config, date, is_live):
        y = min(max(date.year, 2021), 2026)
        return SubscriptionDataSource(f"kospi/069500_5m_{y}.csv", SubscriptionTransportMedium.OBJECT_STORE, FileFormat.CSV)

    def reader(self, config, line, date, is_live):
        if not line or not line[0].isdigit():
            return None
        p = line.split(",")
        d = Kodex5m()
        d.symbol = config.symbol
        t = datetime.strptime(p[0], "%Y%m%d %H:%M")
        d.time = t
        d.end_time = t + timedelta(minutes=5)
        d.value = float(p[4])
        return d


class Diag(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2021, 7, 13)
        self.set_end_date(2021, 9, 30)
        self.set_cash(10_000_000)
        self.info = {}
        try:
            keys = list(self.object_store.keys)
            self.info["keys"] = [str(k) for k in keys][:12]
        except Exception as e:
            self.info["keys_err"] = str(e)[:200]
        try:
            self.info["contains"] = self.object_store.contains_key("kospi/069500_5m_2021.csv")
            txt = self.object_store.read("kospi/069500_5m_2021.csv")
            self.info["len"] = len(txt)
            self.info["head"] = txt[:80]
        except Exception as e:
            self.info["read_err"] = str(e)[:300]
        sec = self.add_data(Kodex5m, "KODEX200", Resolution.MINUTE)
        self.sym = sec.symbol
        self.n = 0
        self.n_slice = 0
        self.first = None
        self.last = None

    def on_data(self, data):
        self.n_slice += 1
        bar = data.get(self.sym)
        if bar is None:
            try:
                bar = data[self.sym]
            except Exception:
                bar = None
        if bar is None:
            return
        self.n += 1
        if self.first is None:
            self.first = str(bar.time)
        self.last = str(bar.time)

    def on_end_of_algorithm(self):
        raise Exception(f"DIAG bars={self.n} slices={self.n_slice} first={self.first} last={self.last} info={self.info}")

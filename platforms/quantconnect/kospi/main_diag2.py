# -*- coding: utf-8 -*-
"""QuantConnect(LEAN) 진단 2 — 샌드박스 로컬 파일 쓰기 + LOCAL_FILE 커스텀 데이터 읽기 가능 여부(결과는 예외 메시지로 전달)."""
from AlgorithmImports import *
import os, tempfile

CSV = "time,close\n20210713 09:00,100\n20210713 09:05,101\n20210714 09:00,102\n20210714 09:05,103\n"
PATHS = []


class Tiny(PythonData):
    def get_source(self, config, date, is_live):
        return SubscriptionDataSource(PATHS[0], SubscriptionTransportMedium.LOCAL_FILE, FileFormat.CSV)

    def reader(self, config, line, date, is_live):
        if not line or not line[0].isdigit():
            return None
        p = line.split(",")
        d = Tiny()
        d.symbol = config.symbol
        t = datetime.strptime(p[0], "%Y%m%d %H:%M")
        d.time = t
        d.end_time = t + timedelta(minutes=5)
        d.value = float(p[1])
        return d


class Diag2(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2021, 7, 12)
        self.set_end_date(2021, 7, 16)
        self.set_cash(1_000_000)
        self.info = {"cwd": os.getcwd(), "tmp": tempfile.gettempdir()}
        for base in (os.getcwd(), tempfile.gettempdir(), "/tmp", os.path.expanduser("~")):
            try:
                p = os.path.join(base, "kospi_diag.csv")
                with open(p, "w") as f:
                    f.write(CSV)
                self.info["ok_" + base] = os.path.getsize(p)
                if not PATHS:
                    PATHS.append(p)
            except Exception as e:
                self.info["err_" + base] = str(e)[:120]
        self.info["ls"] = sorted(os.listdir(os.getcwd()))[:25]
        if PATHS:
            sec = self.add_data(Tiny, "TINY", Resolution.MINUTE)
            self.sym = sec.symbol
        else:
            self.sym = None
        self.n = 0
        self.times = []

    def on_data(self, data):
        if self.sym is None:
            return
        bar = data.get(self.sym)
        if bar is None:
            return
        self.n += 1
        if len(self.times) < 6:
            self.times.append(str(bar.end_time))

    def on_end_of_algorithm(self):
        raise Exception(f"DIAG2 bars={self.n} times={self.times} info={self.info}")

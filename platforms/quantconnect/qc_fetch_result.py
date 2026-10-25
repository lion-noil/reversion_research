# -*- coding: utf-8 -*-
"""QuantConnect 백테스트 결과를 통째로 받아 로컬에 보관 — 공개 임베드(quantconnect.cloud)가 죽어도 사이트가 자체 검증 페이지를 그릴 수 있게.
인증: .qc_state.json(저장 세션, Playwright storage_state). 만료면 qc_session_run.py --headed 로 재로그인.
출력: results/read_<backtestId>.json  = {backtest(통계·차트 포함), charts{Strategy Equity, Drawdown,...}, orders[]}
usage: .venv-quant\\Scripts\\python platforms/quantconnect/qc_fetch_result.py [--project 38045136] [--share c3f1bee0...] [--name "..."] [--no-orders]
"""
import os, sys, json, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".qc_state.json")
BASE = "https://www.quantconnect.com"
OUT_DIR = os.path.join(HERE, "results")


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def main():
    pid = int(arg("--project", "38045136"))
    share = arg("--share", "c3f1bee0b4b7507977f78ada885221c5")
    want_name = arg("--name")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(storage_state=STATE)
        page = ctx.new_page()
        # page.request 로 api/v2 를 치면 "API token hash is not valid"(세션 쿠키 미적용) → 페이지 안에서 fetch (2026-10-25 확인)
        page.goto(f"{BASE}/project/{pid}", wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(3000)
        if "/login" in page.url:
            sys.exit("세션 만료 — qc_session_run.py --headed 로 재로그인 필요")
        JS = """async ([ep, body]) => {
            const r = await fetch('/api/v2/' + ep, {method:'POST', credentials:'include',
                headers:{'Content-Type':'application/json','Accept':'application/json','X-Requested-With':'XMLHttpRequest'}, body});
            return [r.status, await r.text()];
        }"""

        def call(endpoint, payload=None):
            status, text = page.evaluate(JS, [endpoint, json.dumps(payload or {})])
            try:
                out = json.loads(text)
            except Exception:
                sys.exit(f"{endpoint}: HTTP {status} 비JSON 응답 {text[:200]}")
            if not out.get("success"):
                sys.exit(f"{endpoint}: 실패 {json.dumps(out)[:400]}")
            return out

        lst = call("backtests/list", {"projectId": pid})
        bts = lst.get("backtests") or []
        print(f"backtests in project {pid}: {len(bts)}")
        for b in bts:
            print(" ", b.get("backtestId"), "|", b.get("name"), "|", b.get("created"), "| share:", (b.get("sharedUrl") or b.get("shareUrl") or "")[:70])
        pick = None
        for b in bts:
            su = (b.get("sharedUrl") or b.get("shareUrl") or "")
            if share and share in su:
                pick = b
            if want_name and b.get("name") == want_name:
                pick = b
        if pick is None:
            # 공유 URL 필드가 없으면 최신 완료 백테스트
            done = [b for b in bts if b.get("completed")]
            pick = sorted(done, key=lambda b: b.get("created") or "")[-1] if done else bts[-1]
            print("공유 해시로 못 찾아 최신 완료 백테스트 선택:", pick.get("name"))
        bid = pick["backtestId"]
        print("선택:", bid, pick.get("name"))

        read = call("backtests/read", {"projectId": pid, "backtestId": bid})
        bt = read.get("backtest") or {}
        out = {"projectId": pid, "backtestId": bid, "name": bt.get("name"), "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "backtest": bt, "charts": {}, "orders": []}
        print("backtest keys:", sorted(bt.keys())[:40])
        charts = bt.get("charts") or {}
        print("charts:", list(charts.keys()))
        # 차트별 전체 포인트 (read 의 charts 는 샘플링될 수 있어 chart/read 로 다시 요청)
        for cname in charts:
            try:
                # start/end(epoch 초)·count 를 다 줘야 전체 구간이 온다 (count 만 주면 45점으로 샘플링됨 — 2026-10-25 확인)
                cr = call("backtests/chart/read", {"projectId": pid, "backtestId": bid, "name": cname, "count": 5000, "start": 1500000000, "end": 1900000000})
                out["charts"][cname] = cr.get("chart") or cr
                ser = (cr.get("chart") or {}).get("series") or {}
                print(f"  chart {cname}: series={list(ser.keys())} points={[len(s.get('values') or []) for s in ser.values()]}")
            except SystemExit as e:
                print("  chart/read 실패:", cname, str(e)[:120])
                out["charts"][cname] = charts[cname]
        if "--no-orders" not in sys.argv:
            start, orders = 0, []
            while True:
                try:
                    orr = call("backtests/orders/read", {"projectId": pid, "backtestId": bid, "start": start, "end": start + 50})
                except SystemExit as e:
                    print("orders/read 실패:", str(e)[:160]); break
                chunk = orr.get("orders") or []
                orders.extend(chunk)
                total = int(orr.get("length") or 0)
                if not chunk or (total and start + 50 >= total):
                    break
                start += 50
            out["orders"] = orders
            print("orders:", len(orders))
        os.makedirs(OUT_DIR, exist_ok=True)
        fn = os.path.join(OUT_DIR, f"read_{bid}.json")
        json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
        print("saved", fn, os.path.getsize(fn), "bytes")
        browser.close()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""QuantConnect api/v2 를 페이지 안 fetch(세션 쿠키)로 호출 — 무료 계층에서 토큰 없이 프로젝트 생성·코드 업로드·Object Store 업로드·백테스트 실행.
(page.request 는 세션 쿠키가 안 붙어 거부됨 → 로그인된 페이지 컨텍스트의 fetch 만 통과, qc_fetch_result.py 와 같은 방식. 2026-10-25)
usage: .venv-quant/Scripts/python platforms/quantconnect/kospi/qc_api.py <cmd> [...]
  org                         조직 id 출력
  create --name KospiAlign3   프로젝트 생성 → kospi/.project_id
  upload [data/*.csv]         Object Store 에 kospi/<파일명> 키로 업로드 (기본: kospi/data/*.csv 전부)
  objects                     Object Store 키 목록(kospi/ 접두)
  push --code main_kospi.py   프로젝트 main.py 교체
  run --name <백테스트명> [--timeout-min 60]   컴파일→백테스트→완료 폴링 → results/kospi_bt_<id>.json(요약)
  read --backtest <id>        backtests/read + 차트·주문·로그 전체 → results/read_<id>.json (qc_fetch_result 와 같은 형식)
  public --backtest <id>      공개 링크 생성 시도
"""
import os, sys, json, time, glob
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
STATE = os.path.join(PARENT, ".qc_state.json")
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
PID_FILE = os.path.join(HERE, ".project_id")
BASE = "https://www.quantconnect.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"

JS_JSON = """async ([ep, body]) => {
    const r = await fetch('/api/v2/' + ep, {method:'POST', credentials:'include',
        headers:{'Content-Type':'application/json','Accept':'application/json','X-Requested-With':'XMLHttpRequest'}, body});
    return [r.status, await r.text()];
}"""
JS_FORM = """async ([ep, fields, fileField, fileName, content]) => {
    const fd = new FormData();
    for (const k of Object.keys(fields)) fd.append(k, fields[k]);
    if (fileField) fd.append(fileField, new Blob([content], {type: 'text/plain'}), fileName);
    const r = await fetch('/api/v2/' + ep, {method:'POST', credentials:'include',
        headers:{'Accept':'application/json','X-Requested-With':'XMLHttpRequest'}, body: fd});
    return [r.status, await r.text()];
}"""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


class QC:
    def __init__(self, page):
        self.page = page

    def call(self, ep, payload=None, must=True):
        status, text = self.page.evaluate(JS_JSON, [ep, json.dumps(payload or {})])
        try:
            out = json.loads(text)
        except Exception:
            raise SystemExit(f"{ep}: HTTP {status} 비JSON {text[:300]}")
        if must and not out.get("success"):
            raise SystemExit(f"{ep}: 실패 {json.dumps(out, ensure_ascii=False)[:600]}")
        return out

    def form(self, ep, fields, file_field=None, file_name=None, content=None):
        status, text = self.page.evaluate(JS_FORM, [ep, fields, file_field, file_name, content])
        try:
            return json.loads(text)
        except Exception:
            raise SystemExit(f"{ep}: HTTP {status} 비JSON {text[:300]}")

    def org_id(self):
        r = self.call("projects/read")
        orgs = {p.get("organizationId") for p in r.get("projects", []) if p.get("organizationId")}
        if not orgs:
            r2 = self.call("organizations/list", must=False)
            orgs = {o.get("id") for o in r2.get("organizations", []) if o.get("id")}
        if len(orgs) != 1:
            log("조직 후보:", orgs)
        return sorted(orgs)[0]


def pid():
    v = arg("--project")
    if v:
        return int(v)
    if os.path.exists(PID_FILE):
        return int(open(PID_FILE).read().strip())
    raise SystemExit("프로젝트 id 없음 — create 먼저")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "org"
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(storage_state=STATE, user_agent=UA)
        page = ctx.new_page()
        page.goto(f"{BASE}/project/", wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(3000)
        if "/login" in page.url:
            raise SystemExit("세션 만료 — qc_session_run.py --headed 로 재로그인")
        qc = QC(page)
        if cmd == "org":
            print("ORG", qc.org_id())
        elif cmd == "create":
            name = arg("--name", "KospiAlign3")
            r = qc.call("projects/create", {"name": name, "language": "Py"})
            prj = (r.get("projects") or [{}])[0]
            new_id = prj.get("projectId") or r.get("projectId")
            open(PID_FILE, "w").write(str(new_id))
            log("프로젝트 생성", new_id, name)
            print("PROJECT", new_id)
        elif cmd == "upload":
            org = qc.org_id()
            files = [a for a in sys.argv[2:] if a.endswith(".csv")] or sorted(glob.glob(os.path.join(HERE, "data", "*.csv")))
            for fn in files:
                key = "kospi/" + os.path.basename(fn)
                content = open(fn, encoding="utf-8").read()
                r = qc.form("object/set", {"organizationId": org, "key": key}, "objectData", os.path.basename(fn), content)
                log("object/set", key, f"{len(content)//1024} KB", "OK" if r.get("success") else json.dumps(r, ensure_ascii=False)[:300])
        elif cmd == "objects":
            org = qc.org_id()
            r = qc.call("object/list", {"organizationId": org, "path": "kospi/"}, must=False)
            print(json.dumps(r, ensure_ascii=False)[:2000])
        elif cmd == "push":
            code_path = arg("--code", os.path.join(HERE, "main_kospi.py"))
            code = open(code_path, encoding="utf-8").read()
            r = qc.call("files/update", {"projectId": pid(), "name": "main.py", "content": code}, must=False)
            if not r.get("success"):
                log("files/update 실패:", json.dumps(r, ensure_ascii=False)[:400])
                r = qc.call("files/create", {"projectId": pid(), "name": "main.py", "content": code})
            log("main.py 업로드", len(code), "자 OK")
        elif cmd == "files":
            # 프로젝트에 보조 파일 업로드/교체: qc_api.py files embed/kdata_2021.py ...
            for path in [a for a in sys.argv[2:] if not a.startswith("--")]:
                name = os.path.basename(path)
                content = open(path, encoding="utf-8").read()
                r = qc.call("files/update", {"projectId": pid(), "name": name, "content": content}, must=False)
                if not r.get("success"):
                    r = qc.call("files/create", {"projectId": pid(), "name": name, "content": content}, must=False)
                log("파일", name, len(content) // 1024, "KB", "OK" if r.get("success") else json.dumps(r, ensure_ascii=False)[:300])
        elif cmd == "run":
            name = arg("--name", time.strftime("kospi-align3-%m%d-%H%M"))
            tmo = int(arg("--timeout-min", "60"))
            c = qc.call("compile/create", {"projectId": pid()})
            cid = c["compileId"]
            for _ in range(90):
                cr = qc.call("compile/read", {"projectId": pid(), "compileId": cid})
                if cr.get("state") == "BuildSuccess":
                    break
                if cr.get("state") == "BuildError":
                    raise SystemExit("컴파일 실패: " + json.dumps(cr.get("logs"), ensure_ascii=False)[:1500])
                page.wait_for_timeout(2000)
            log("컴파일 OK", cid)
            bt = qc.call("backtests/create", {"projectId": pid(), "compileId": cid, "backtestName": name})
            bid = (bt.get("backtest") or {}).get("backtestId") or bt.get("backtestId")
            log("백테스트 시작", bid, name)
            t0 = time.time()
            last = None
            bk = {}
            while time.time() - t0 < tmo * 60:
                page.wait_for_timeout(15000)
                r = qc.call("backtests/read", {"projectId": pid(), "backtestId": bid})
                bk = r.get("backtest") or {}
                st = (bk.get("progress"), bk.get("completed"), bk.get("status"), (bk.get("error") or "")[:200])
                if st != last:
                    log("진행", st); last = st
                if bk.get("completed"):
                    break
            stats = bk.get("statistics") or {}
            out = {"project": pid(), "backtest_id": bid, "name": name, "completed": bk.get("completed"), "error": bk.get("error"),
                   "stacktrace": (bk.get("stacktrace") or "")[:2000], "statistics": stats,
                   "runtime": bk.get("runtimeStatistics"), "elapsed_min": round((time.time() - t0) / 60, 1)}
            fn = os.path.join(RES, f"kospi_bt_{bid}.json")
            json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            log("저장", fn)
            for k in ("Net Profit", "Compounding Annual Return", "Sharpe Ratio", "Drawdown", "Total Orders", "Win Rate"):
                log(" ", k, stats.get(k))
            if bk.get("error"):
                log("오류:", bk.get("error"), (bk.get("stacktrace") or "")[:800])
            print("BACKTEST", bid)
        elif cmd == "read":
            bid = arg("--backtest")
            read = qc.call("backtests/read", {"projectId": pid(), "backtestId": bid})
            bt = read.get("backtest") or {}
            out = {"projectId": pid(), "backtestId": bid, "name": bt.get("name"), "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "backtest": bt, "charts": {}, "orders": [], "logs": []}
            for cname in (bt.get("charts") or {}):
                try:
                    cr = qc.call("backtests/chart/read", {"projectId": pid(), "backtestId": bid, "name": cname,
                                                          "count": 5000, "start": 1500000000, "end": 1900000000})
                    out["charts"][cname] = cr.get("chart") or cr
                except SystemExit:
                    out["charts"][cname] = (bt.get("charts") or {})[cname]
            start = 0
            while True:
                r = qc.call("backtests/orders/read", {"projectId": pid(), "backtestId": bid, "start": start, "end": start + 100}, must=False)
                orders = r.get("orders") or []
                out["orders"].extend(orders)
                if len(orders) < 100:
                    break
                start += 100
            lg = qc.call("backtests/read/log", {"projectId": pid(), "backtestId": bid}, must=False)
            out["logs"] = lg.get("logs") or lg.get("log") or []
            fn = os.path.join(RES, f"read_{bid}.json")
            json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
            log("저장", fn, os.path.getsize(fn) // 1024, "KB · orders", len(out["orders"]), "· logs", len(out["logs"]) if isinstance(out["logs"], list) else "?")
        elif cmd == "public":
            bid = arg("--backtest")
            for ep, body in (("backtests/share", {"projectId": pid(), "backtestId": bid}),
                             ("backtests/update", {"projectId": pid(), "backtestId": bid, "public": True})):
                r = qc.call(ep, body, must=False)
                log(ep, json.dumps(r, ensure_ascii=False)[:400])
        ctx.storage_state(path=STATE)
        b.close()


if __name__ == "__main__":
    main()

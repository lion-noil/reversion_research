# -*- coding: utf-8 -*-
"""코스피 검증 보고서(verify:kospi)를 사이트 보고서 저장소(Upstash trading:reports)에 발행 — publish_verify_report.py(환율)와 같은 경로.
usage: py -3 platforms/quantconnect/kospi/publish_kospi_report.py [--label 2026-10-25] [--dry]"""
import os, sys, json, subprocess, tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
SITE_V = os.path.join(HERE, "..", "..", "..", "..", "hyeongeonnoil", "src", "data", "verified.json")
WSL = ["wsl", "-d", "Ubuntu-24.04", "--"]


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def to_wsl(path):
    p = os.path.abspath(path).replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:]


def main():
    label = arg("--label")
    if not label:
        import glob
        label = sorted(glob.glob(os.path.join(HERE, "results", "verify_kospi-*.md")))[-1].split("verify_kospi-")[1][:-3]
    md = open(os.path.join(HERE, "results", f"verify_kospi-{label}.md"), encoding="utf-8").read()
    data = json.load(open(os.path.join(HERE, "results", f"verify_kospi-{label}.data.json"), encoding="utf-8"))
    k = json.load(open(SITE_V, encoding="utf-8")).get("kospi") or {}
    data["caveats"] = k.get("caveats") or []
    data["expectation"] = k.get("expectation") or ""
    data["verified_at"] = label
    data["universe"] = "kospi"
    st = data["stats"]
    title = "코스피200 ETF(KODEX200) ALIGN3 외부 엔진 재현 검증 — QuantConnect 독립 백테스트"
    subtitle = (f"검증일 {label} · 백테스트 {data['meta']['start']}~{data['meta']['end']} · 순이익 {st['net_profit_pct']:+.0f}% · CAGR {st['cagr_pct']:.1f}% · "
                f"낙폭 {st['max_dd_pct']:.1f}% · 데이터는 자체 수집(엔진만 외부)")
    payload = {"kind": "verify", "label": "kospi", "md": md, "title": title, "subtitle": subtitle, "data": data}
    tmp = os.path.join(tempfile.gettempdir(), f"verify_kospi_{label}.payload.json")
    json.dump(payload, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"payload {os.path.getsize(tmp)//1024} KB → {title}")
    if "--dry" in sys.argv:
        return
    subprocess.run(WSL + ["docker", "cp", to_wsl(tmp), "news-scrap:/tmp/verify_kospi_payload.json"], check=True)
    local_rs = os.path.join(HERE, "..", "..", "..", "..", "News_scrap", "app", "report_store.py")
    subprocess.run(WSL + ["docker", "cp", to_wsl(local_rs), "news-scrap:/app/report_store.py"], check=True)
    code = ("import json,report_store as rs; p=json.load(open('/tmp/verify_kospi_payload.json',encoding='utf-8')); "
            "rid=rs.publish_report(p['kind'],p['label'],p['md'],title=p['title'],subtitle=p['subtitle'],data=p['data']); "
            "print('published',rid); print([ (d.get('id'),d.get('title')) for d in rs.list_reports()[:5] ])")
    r = subprocess.run(WSL + ["docker", "exec", "news-scrap", "python", "-c", code])
    if r.returncode != 0:
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()

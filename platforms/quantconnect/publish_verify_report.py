# -*- coding: utf-8 -*-
"""검증 보고서(verify:<label>)를 사이트 보고서 저장소(Upstash trading:reports)에 발행.
입력: results/verify_<label>.md + results/verify_<label>.data.json (build_verified_data.py) + 사이트 verified.json 의 주의문.
발행은 News_scrap 컨테이너(news-scrap)의 report_store.publish_report 로 한다(Upstash 자격증명이 거기에만 있음):
  docker cp 로 두 파일을 컨테이너 /tmp 에 넣고 python -c 로 호출. 같은 kind:label 은 덮어씀.
usage: py -3 platforms/quantconnect/publish_verify_report.py [--label 2026-10-18(검증일)] [--universe fx-majors] [--dry]
"""
import os, sys, json, subprocess, tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
SITE_V = os.path.join(HERE, "..", "..", "..", "hyeongeonnoil", "src", "data", "verified.json")
WSL = ["wsl", "-d", "Ubuntu-24.04", "--"]


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def to_wsl(path):
    p = os.path.abspath(path).replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:]


def main():
    label = arg("--label", "2026-10-18")
    md_path = os.path.join(HERE, "results", f"verify_{label}.md")
    data_path = os.path.join(HERE, "results", f"verify_{label}.data.json")
    md = open(md_path, encoding="utf-8").read()
    data = json.load(open(data_path, encoding="utf-8"))
    q = json.load(open(SITE_V, encoding="utf-8"))["quantconnect"]
    data["caveats"] = q.get("caveats") or []
    data["expectation"] = q.get("expectation") or ""
    # 보고서 id = verify:<유니버스> (날짜 아님) — "이 유니버스가 이 날짜에 외부에서 재현됐다"는 기록이라 재검증 시 같은 글을 덮어쓴다.
    universe = arg("--universe", "fx-majors")
    data["verified_at"] = label
    data["universe"] = universe
    title = "환율 메이저 유니버스(17셀) 외부 재현 검증 — QuantConnect 독립 백테스트"
    st = data["stats"]
    subtitle = f"검증일 {label} · 백테스트 {data['meta']['start']}~{data['meta']['end']} · 순이익 {st['net_profit_pct']:+.0f}% · CAGR {st['cagr_pct']:.1f}% · 낙폭 {st['max_dd_pct']:.1f}% · 17셀 건수 ±10% 일치"
    payload = {"kind": "verify", "label": universe, "md": md, "title": title, "subtitle": subtitle, "data": data}
    tmp = os.path.join(tempfile.gettempdir(), f"verify_{label}.payload.json")
    json.dump(payload, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"payload {os.path.getsize(tmp)//1024} KB → {title}")
    if "--dry" in sys.argv:
        return
    subprocess.run(WSL + ["docker", "cp", to_wsl(tmp), "news-scrap:/tmp/verify_payload.json"], check=True)
    # 컨테이너 이미지의 report_store.py 가 'verify' kind 를 아직 모르면(재빌드 전) 로컬 최신본을 함께 넣는다
    local_rs = os.path.join(HERE, "..", "..", "..", "News_scrap", "app", "report_store.py")
    subprocess.run(WSL + ["docker", "cp", to_wsl(local_rs), "news-scrap:/app/report_store.py"], check=True)
    code = (
        "import json,report_store as rs; p=json.load(open('/tmp/verify_payload.json',encoding='utf-8')); "
        "rid=rs.publish_report(p['kind'],p['label'],p['md'],title=p['title'],subtitle=p['subtitle'],data=p['data']); "
        "print('published',rid); print([ (d.get('id'),d.get('title')) for d in rs.list_reports()[:5] ])"
    )
    # 컨테이너 WORKDIR 이 /app 이라 -w 불필요. 출력은 그대로 흘려보냄(capture 시 wsl 경유 출력이 None 으로 오는 경우 있음).
    r = subprocess.run(WSL + ["docker", "exec", "news-scrap", "python", "-c", code])
    if r.returncode != 0:
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()

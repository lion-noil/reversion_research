# -*- coding: utf-8 -*-
"""코스피 프로젝트의 백테스트를 웹 IDE 에서 'Make Public' → 'Backtest URL' 로 공개 링크 생성 (qc_ui_run.collect_logs_and_link 재사용).
api/v2 backtests/update {public:true} 는 success 만 주고 public 플래그가 안 바뀜(2026-10-25 확인) → UI 경로.
usage: .venv-quant/Scripts/python platforms/quantconnect/kospi/qc_make_public.py [--name kospi-align3-v2] [--project 38421655]"""
import os, sys, json, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
import qc_ui_run as U  # noqa: E402


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def main():
    project = arg("--project", open(os.path.join(HERE, ".project_id")).read().strip())
    name = arg("--name", "kospi-align3-v2")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(storage_state=U.STATE, viewport={"width": 1600, "height": 1000}, user_agent=U.UA)
        ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin="https://www.quantconnect.com")
        pg = ctx.new_page()
        pg.goto(f"https://www.quantconnect.com/project/{project}", wait_until="domcontentloaded"); pg.wait_for_timeout(12000)
        if "/login" in pg.url:
            raise SystemExit("세션 만료")
        cf = U.code_frame(pg)
        pg.wait_for_timeout(4000)
        out = U.collect_logs_and_link(pg, cf, name)
        pg.screenshot(path=os.path.join(HERE, "shots", "make_public.png"))
        fn = os.path.join(HERE, "results", f"share_{name}.json")
        json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("share_url:", out.get("share_url"), "| leak_guard:", out.get("leak_guard"), "| 저장", fn)
        ctx.storage_state(path=U.STATE)
        b.close()


if __name__ == "__main__":
    main()

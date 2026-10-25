# -*- coding: utf-8 -*-
"""QuantConnect 무료 계층에서 새 프로젝트(Python 알고리즘) 생성 — Playwright, 저장 세션(.qc_state.json) 재사용.
usage: .venv-quant\Scripts\python platforms/quantconnect/kospi/qc_new_project.py [--name KospiAlign3]
출력: 새 project id (stdout 마지막 줄 'PROJECT <id>'), kospi/.project_id 에 저장."""
import os, sys, re, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
STATE = os.path.join(PARENT, ".qc_state.json")
SHOTS = os.path.join(HERE, "shots"); os.makedirs(SHOTS, exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def text(pg):
    out = []
    for f in pg.frames:
        try:
            out.append(f.evaluate("() => document.body ? document.body.innerText : ''"))
        except Exception:
            pass
    return "\n".join(out)


def click_text(pg, txt):
    js = """(txt) => { const els = Array.from(document.querySelectorAll('a, button, span, div, li')).filter(e => (e.innerText||'').trim() === txt);
             if (!els.length) return 0; const e = els[els.length-1]; e.scrollIntoView(); e.click(); return 1; }"""
    for fr in pg.frames:
        try:
            if fr.evaluate(js, txt):
                return True
        except Exception:
            pass
    return False


def main():
    args = sys.argv[1:]
    name = args[args.index("--name") + 1] if "--name" in args else "KospiAlign3"
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(storage_state=STATE, viewport={"width": 1600, "height": 1000}, user_agent=UA)
        pg = ctx.new_page()
        pg.goto("https://www.quantconnect.com/project/", wait_until="domcontentloaded"); pg.wait_for_timeout(10000)
        if "/login" in pg.url:
            raise SystemExit("세션 만료 — qc_session_run.py --headed 로 재로그인")
        pg.screenshot(path=os.path.join(SHOTS, "projects.png"))
        before = pg.url
        # 다른 프로젝트가 코딩 세션을 잡고 있으면 Stop
        if "No Coding Session Available" in text(pg):
            click_text(pg, "Stop"); pg.wait_for_timeout(8000); log("기존 코딩 세션 정지")
        ok = False
        for label in ("Create New Algorithm", "New Algorithm", "Create Algorithm", "Create New Project"):
            if click_text(pg, label):
                log("클릭:", label); ok = True; break
        if not ok:
            # 메뉴 아이콘/버튼 탐색
            for sel in ("[title*='Create']", "[aria-label*='Create']", "button:has-text('Create')"):
                try:
                    pg.locator(sel).first.click(timeout=3000); log("클릭 셀렉터:", sel); ok = True; break
                except Exception:
                    pass
        if not ok:
            pg.screenshot(path=os.path.join(SHOTS, "no_create.png")); raise SystemExit("생성 버튼을 못 찾음")
        pg.wait_for_timeout(4000)
        # 이름/언어 입력 대화상자가 있으면 처리
        t = text(pg)
        try:
            inp = pg.locator("input[placeholder*='name' i], input[name*='name' i]").first
            if inp.count() and inp.is_visible():
                inp.fill(name); log("이름 입력:", name)
                for lab in ("Python", "Py"):
                    click_text(pg, lab)
                for lab in ("Create", "Create Project", "OK", "Confirm"):
                    if click_text(pg, lab):
                        break
        except Exception:
            pass
        pid = None
        for _ in range(40):
            pg.wait_for_timeout(1500)
            m = re.search(r"/project/(\d+)", pg.url)
            if m and pg.url != before:
                pid = m.group(1); break
        pg.screenshot(path=os.path.join(SHOTS, "after_create.png"))
        if not pid:
            raise SystemExit(f"프로젝트 URL 미확인: {pg.url}")
        pg.wait_for_timeout(8000)
        open(os.path.join(HERE, ".project_id"), "w").write(pid)
        ctx.storage_state(path=STATE)
        log("새 프로젝트", pid, pg.url)
        print("PROJECT", pid)
        b.close()


if __name__ == "__main__":
    main()

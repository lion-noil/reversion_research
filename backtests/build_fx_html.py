"""환율 4방향 스윕 결과 → 정적 HTML(자바스크립트 없음, 표를 파이썬이 직접 렌더).
어떤 뷰어/미리보기에서도 보이도록 JS 의존 제거."""
import json, os, html
os.chdir(os.path.dirname(os.path.abspath(__file__)))
RES = 'results'
SY = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
DIRS = [('역추세롱', 'long'), ('역추세숏', 'short'), ('추세숏', 'trend'), ('추세롱', 'trendlong')]
YR = 2.99

cells = {}      # (sym,dirname) -> row
detail = {}
for dn, dk in DIRS:
    D = json.load(open(f'{RES}/fx_14d_{dk}.json', encoding='utf-8'))
    rows = []
    for s in SY:
        if s in D and D[s]:
            r = dict(D[s][0]); r['sym'] = s; r['yr'] = round(r['n'] / YR)
            cells[(s, dn)] = r; rows.append(r)
    rows.sort(key=lambda x: -x['exp'])
    detail[dn] = rows


def cls(e):
    return 'c-ok' if e >= 0.20 else ('c-mid' if e > 0 else 'c-no')


def sgn(e):
    return f'+{e:.2f}' if e >= 0 else f'{e:.2f}'


# ── 매트릭스 ──
mhead = '<th class="l">심볼</th>' + ''.join(f'<th>{dn}</th>' for dn, _ in DIRS)
mrows = []
for s in SY:
    tds = [f'<td class="l"><b>{s}</b></td>']
    for dn, _ in DIRS:
        c = cells.get((s, dn))
        if not c:
            tds.append('<td>—</td>'); continue
        tds.append(f'<td><span class="cell {cls(c["exp"])}"><b>{sgn(c["exp"])}%</b>'
                   f'<span class="sm"> K{c["k1"]}·PF{c["pf"]}</span></span></td>')
    mrows.append('<tr>' + ''.join(tds) + '</tr>')
matrix = f'<table><thead><tr>{mhead}</tr></thead><tbody>{"".join(mrows)}</tbody></table>'

# ── 방향별 상세 ──
det_html = []
for dn, _ in DIRS:
    body = []
    for r in detail[dn]:
        ec = 'pos' if r['exp'] >= 0.20 else ('' if r['exp'] > 0 else 'neg')
        body.append(f'<tr><td class="l"><b>{r["sym"]}</b></td><td>{r["k1"]}</td><td>{r["b"]}</td>'
                    f'<td>{r["cd"]}</td><td>{r["n"]}</td><td>{r["yr"]}</td><td>{r["wr"]}%</td>'
                    f'<td class="{ec}">{sgn(r["exp"])[:1]}{abs(r["exp"]):.3f}%</td><td>{r["pf"]}</td><td>{r["maxc"]}</td></tr>')
    det_html.append(
        f'<h3>{dn}</h3><div class="wrap"><table><thead><tr><th class="l">심볼</th><th>K1</th><th>B</th>'
        f'<th>CD</th><th>n</th><th>회/년</th><th>승률</th><th>기대/건</th><th>PF</th><th>maxc</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>')

HTML = f'''<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>환율 7종 · 4방향 스윕</title>
<style>
 body{{margin:0;background:#0f1115;color:#e6e8ec;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:18px 20px;max-width:1000px}}
 h1{{font-size:18px;margin:0 0 2px}} h2{{font-size:15px;margin:22px 0 8px}} h3{{font-size:13.5px;margin:14px 0 6px;color:#4f8cff}}
 .sub{{color:#9aa3b2;font-size:13px;margin:0 0 14px}}
 .tag{{display:inline-block;background:#1d2530;border:1px solid #262b36;border-radius:6px;padding:2px 9px;margin-right:6px;color:#cdd6e4;font-size:12px}}
 .legend{{display:flex;gap:14px;color:#9aa3b2;font-size:12px;margin:0 0 12px;flex-wrap:wrap}}
 .legend i{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:4px;vertical-align:-1px}}
 .wrap{{overflow-x:auto;border:1px solid #262b36;border-radius:10px;margin-bottom:14px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}
 th,td{{padding:7px 10px;border-bottom:1px solid #262b36;text-align:right}}
 th{{background:#171a21;color:#9aa3b2;font-weight:500;font-size:12px}}
 td.l,th.l{{text-align:left}}
 .cell{{border-radius:6px;padding:4px 7px;display:inline-block;min-width:96px;text-align:center;font-size:12px}}
 .c-ok{{background:#16331f;color:#7fdca0}} .c-mid{{background:#332b15;color:#e6c878}} .c-no{{background:#331818;color:#eaa39b}}
 .sm{{font-size:11px;color:#9aa3b2}} .pos{{color:#4ec27a}} .neg{{color:#e2574b}}
 .note{{color:#9aa3b2;font-size:12px;margin:6px 2px 16px}} ul.note{{font-size:13px;line-height:1.7}}
</style></head><body>
<h1>환율 메이저 7종 · <b>4방향</b> σ-복귀/추세 스윕</h1>
<p class="sub"><span class="tag">K1 2.0~5.0</span><span class="tag">B −2~2</span><span class="tag">CD 0.5~3h</span>
 · n≥80(≈27/년) · maxc&lt;15 · 14일보유 · fee0.11% · 데이터 3년 · Dukascopy</p>
<div class="legend">
 <span><i style="background:#16331f"></i>유효 (기대 ≥+0.20%)</span>
 <span><i style="background:#332b15"></i>경계 (0~0.20%, 노이즈)</span>
 <span><i style="background:#331818"></i>제외 (음수)</span></div>
<h2>① 심볼 × 방향 매트릭스</h2>
<div class="wrap">{matrix}</div>
<p class="note">셀 = 심볼별 그 방향 top-1. 기대%/건 · K · PF. 초록=유효, 노랑=경계(본전), 빨강=음수.</p>
<h2>② 방향별 상세 (유효순)</h2>
{''.join(det_html)}
<h2>③ 요점</h2>
<ul class="note">
<li><b>상품/리스크 통화(AUD·NZD)=평균회귀형</b> — 역추세 롱·숏 둘 다 강함(NZD 양방향). 추세 약함.</li>
<li><b>EURUSD=추세롱만</b>(+0.78%/PF5.36), 역추세는 죽음. 잔잔한 메이저라 안 되돌리고 천천히 추세.</li>
<li><b>USDCHF=추세숏 최강</b>(+1.04%/PF6.54) — 안전자산 추세. 단 n81·고PF는 과적합 의심.</li>
<li><b>환율 기대값(+0.2~1.0%) &lt; 지수·크립토(+1~3%)</b> — 변동성 낮아 같은 수익에 더 많은 회전/레버리지 필요.</li>
<li>⚠️ 전부 in-sample, n 80~130(얕음), PF5~6은 소수 대박 의존 가능 → OOS 필수.</li></ul>
</body></html>'''

open('fx_results.html', 'w', encoding='utf-8').write(HTML)
print('wrote fx_results.html (static, no-JS)', len(HTML), 'bytes')

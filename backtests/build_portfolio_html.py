"""portfolio_sim_result.json -> 자체완결 HTML(자산곡선 로그스케일 + 방향별 기여 + 채택표 + 경고)."""
import json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open('portfolio_sim_result.json', encoding='utf-8'))
DATA = json.dumps(R, ensure_ascii=False)

HTML = r'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>4전략 통합 포트폴리오 3년 백테스트</title>
<style>
  :root{--bg:#0f1115;--surf:#171a21;--line:#262b36;--txt:#e6e8ec;--mut:#9aa3b2;
        --grn:#4ec27a;--red:#e2574b;--accent:#4f8cff;--gold:#e6c878;--warn:#e6a14e;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:18px 20px;max-width:1040px}
  h1{font-size:18px;font-weight:600;margin:0 0 2px}
  h2{font-size:15px;font-weight:600;margin:22px 0 8px}
  .sub{color:var(--mut);font-size:13px;margin:0 0 14px}
  .kpis{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 6px}
  .kpi{border:1px solid var(--line);border-radius:10px;padding:10px 14px;background:var(--surf);min-width:130px}
  .kpi .v{font-size:20px;font-weight:700}.kpi .l{font-size:11px;color:var(--mut)}
  .pos{color:var(--grn)}.neg{color:var(--red)}.gold{color:var(--gold)}
  .warn{border:1px solid var(--warn);background:#241c10;border-radius:10px;padding:12px 14px;margin:12px 0;font-size:13px;color:#f0d9b5}
  .warn b{color:var(--warn)}
  .wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin-bottom:6px}
  table{border-collapse:collapse;width:100%;font-size:12.5px;white-space:nowrap}
  th,td{padding:6px 9px;text-align:right;border-bottom:1px solid var(--line)}
  th{background:var(--surf);color:var(--mut);font-weight:500;font-size:11.5px}
  td.l{text-align:left}
  .card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px;background:var(--surf)}
  .bar{height:13px;border-radius:3px;background:var(--accent)}
  .note{color:var(--mut);font-size:12px;margin:6px 2px 16px}
</style></head><body>
<h1>4전략 통합 포트폴리오 · 3년 백테스트 <span class="gold">(in-sample)</span></h1>
<p class="sub" id="sub"></p>
<div class="kpis" id="kpis"></div>
<div class="warn" id="warn"></div>

<h2>자산곡선 (로그스케일, 월말)</h2>
<div class="card"><div id="chart"></div>
  <div class="note">세로축 로그스케일($10k→). 매끈한 우상향은 <b>in-sample 최적화</b>의 전형 — 실거래 기대 아님.</div></div>

<h2>방향별 실현손익 기여</h2>
<div class="card" id="bydir"></div>

<h2>채택 서브전략 (심볼×방향 top-1 양수기대)</h2>
<div class="wrap"><table>
  <thead><tr><th class="l">방향</th><th class="l">심볼</th><th>K1</th><th>B</th><th>CD</th><th>거래수</th><th>기대/건(단일)</th><th>PF</th></tr></thead>
  <tbody id="picks"></tbody></table></div>
<p class="note">기대/건·PF는 각 서브전략 단독 in-sample 성적(포트폴리오 합성 전). 전부 동일 3년 데이터에 최적화됨.</p>
<script>
const R=__DATA__, $=s=>document.querySelector(s);
const usd=v=>'$'+Math.round(v).toLocaleString();
$('#sub').textContent=`${R.span[0]} ~ ${R.span[1]} (${R.yrs}년) · 시작 ${usd(R.init)} · 한 랏=자산 ${R.lot_frac*100}% · 최대 ${R.max_lots}랏 · 피라미딩·14일보유·fee0.11%`;
const lev=(R.peak_lots*R.lot_frac).toFixed(1);
$('#kpis').innerHTML=[
  ['최종 자산',usd(R.final),'gold'],['배수',R.mult+'배','pos'],['CAGR',(R.cagr*100).toFixed(0)+'%','pos'],
  ['최대낙폭(실현)',(R.mdd*100).toFixed(1)+'%','neg'],['최대 동시랏',R.peak_lots+'랏 ('+lev+'배)','gold'],
  ['파산',R.ruin?'발생':'없음',R.ruin?'neg':'pos'],
].map(([l,v,c])=>`<div class="kpi"><div class="v ${c||''}">${v}</div><div class="l">${l}</div></div>`).join('');
$('#warn').innerHTML=`<b>⚠️ 해석 주의</b> — ① <b>전부 in-sample</b>: 파라미터를 바로 이 3년 데이터에 최적화 → ${R.mult}배는 과적합 상한이지 기대수익 아님. OOS 미검증. `+
  `② <b>레버리지 착시</b>: 최대 ${R.peak_lots}랏=명목 ${lev}배 노출. 실현곡선은 청산시점에만 손익을 반영해 장중 동시역행 위험을 과소평가 — 실제 MtM 낙폭은 −12%보다 훨씬 깊을 수 있음(지수 5종은 동반 움직임=분산 착시). `+
  `③ <b>피라미딩</b> 가정(한 과열장에 같은 방향 다수 진입). 단일포지션이면 결과 급감. ④ 200랏캡은 한 번도 안 걸림(peak ${R.peak_lots}) → 실효 한도는 ${lev}배.`;

// equity curve (log)
(function(){const m=R.monthly;const W=960,H=240,pl=58,pr=14,pt=12,pb=24,iw=W-pl-pr,ih=H-pt-pb;
  const vals=m.map(x=>x[1]);const lo=Math.log10(Math.min(...vals)),hi=Math.log10(Math.max(...vals));
  const x=i=>pl+iw*i/(m.length-1), y=v=>pt+ih*(1-(Math.log10(v)-lo)/(hi-lo));
  let path='';m.forEach((p,i)=>{path+=(i?'L':'M')+x(i).toFixed(1)+' '+y(p[1]).toFixed(1)+' ';});
  let grid='',ticks=[10000,30000,100000,300000];
  ticks.forEach(t=>{if(t<Math.min(...vals)||t>Math.max(...vals)*1.2)return;const yy=y(t);
    grid+=`<line x1="${pl}" y1="${yy}" x2="${pl+iw}" y2="${yy}" stroke="var(--line)"/><text x="${pl-6}" y="${yy+3}" fill="var(--mut)" font-size="10" text-anchor="end">$${(t/1000)}k</text>`;});
  let xt='';m.forEach((p,i)=>{if(i%6&&i!=m.length-1)return;xt+=`<text x="${x(i)}" y="${H-8}" fill="var(--mut)" font-size="9" text-anchor="middle">${p[0].slice(0,7)}</text>`;});
  const area=`${path} L ${x(m.length-1)} ${pt+ih} L ${pl} ${pt+ih} Z`;
  $('#chart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">
    <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--grn)" stop-opacity=".25"/><stop offset="1" stop-color="var(--grn)" stop-opacity="0"/></linearGradient></defs>
    ${grid}<path d="${area}" fill="url(#g)"/><path d="${path}" fill="none" stroke="var(--grn)" stroke-width="1.8"/>${xt}</svg>`;})();

// by-direction
(function(){const e=Object.entries(R.by_dir);const mx=Math.max(...e.map(x=>Math.abs(x[1])));
  $('#bydir').innerHTML=e.map(([k,v])=>`<div style="display:flex;align-items:center;gap:10px;margin:5px 0">
    <span style="width:90px;font-size:12px;color:var(--mut)">${k}</span>
    <div class="bar" style="width:${Math.max(2,100*Math.abs(v)/mx)}%;background:${v>=0?'var(--grn)':'var(--red)'}"></div>
    <span class="${v>=0?'pos':'neg'}" style="font-size:12px">${usd(v)}</span></div>`).join('');})();

// picks
$('#picks').innerHTML=R.picks.slice().sort((a,b)=>b.exp-a.exp).map(p=>`<tr>
  <td class="l">${p.dname}</td><td class="l">${p.sym}</td><td>${p.k1}</td><td>${p.b}</td><td>${p.cd}</td>
  <td>${p.ntr}</td><td class="pos">+${p.exp}%</td><td>${p.pf}</td></tr>`).join('');
</script></body></html>'''
open('portfolio_sim.html', 'w', encoding='utf-8').write(HTML.replace('__DATA__', DATA))
print('wrote portfolio_sim.html')

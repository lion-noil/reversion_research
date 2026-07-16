"""역추세 숏 고-K 스윕 → 자체완결 HTML 리포트(데이터 내장 + 심볼별 K-능선 SVG).
입력: _short_hik_report.json (build via persym_sweep + ridge scan). 출력: s1_best_short_hik.html
"""
import json, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
REP_JSON = sys.argv[1] if len(sys.argv) > 1 else '_short_hik_report.json'
OUT_HTML = sys.argv[2] if len(sys.argv) > 2 else 's1_best_short_hik.html'
MARKET = sys.argv[3] if len(sys.argv) > 3 else 'MT5'
rep = json.load(open(REP_JSON, encoding='utf-8'))
DATA = json.dumps(rep, ensure_ascii=False)

HTML = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S1 v2 - 역추세 숏 고-K 스윕</title>
<style>
  :root{--bg:#0f1115;--surf:#171a21;--line:#262b36;--txt:#e6e8ec;--mut:#9aa3b2;
        --grn:#4ec27a;--grnL:#9bd9af;--red:#e2574b;--redL:#eaa39b;--accent:#4f8cff;--gold:#e6c878;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:18px 20px;max-width:1060px}
  h1{font-size:18px;font-weight:600;margin:0 0 2px}
  h2{font-size:15px;font-weight:600;margin:24px 0 8px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .sub{color:var(--mut);font-size:13px;margin:0 0 14px}
  .tag{display:inline-block;background:#1d2530;border:1px solid var(--line);border-radius:6px;padding:2px 9px;margin-right:6px;color:#cdd6e4;font-size:12px}
  .wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin-bottom:6px}
  table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}
  th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
  th{background:var(--surf);color:var(--mut);font-weight:500;font-size:12px}
  tbody tr:hover{background:#1b1f28}
  td.sym,td.l{text-align:left}
  td.k{text-align:center;color:var(--mut)}
  .pos{color:var(--grn)}.neg{color:var(--red)}
  .big{font-size:13px}.sm{font-size:11px;color:var(--mut)}
  .pf{font-weight:600}
  .badge{font-size:11px;padding:1px 7px;border-radius:5px;font-weight:600}
  .b-ok{background:#16331f;color:#7fdca0}.b-no{background:#331818;color:#eaa39b}.b-mid{background:#332b15;color:#e6c878}.b-noise{background:#2a2030;color:#cba0e6}
  .nav{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 4px}
  .nav a{font-size:12px;color:var(--accent);text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:2px 8px}
  .card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px;background:var(--surf)}
  .legend{display:flex;gap:14px;color:var(--mut);font-size:12px;margin:0 0 12px;flex-wrap:wrap}
  .legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px;vertical-align:-1px}
  .row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
  .note{color:var(--mut);font-size:12px;margin:6px 2px 18px}
  .pick{color:var(--gold);font-weight:600}
</style>
</head>
<body>
  <h1>S1 v2 - __MARKET__ 역추세 <b>숏온리</b> - <span style="color:var(--gold)">고-K 스윕</span> (K1 3.0~5.0)</h1>
  <p class="sub">
    격자 <span class="tag">K1 3.0~5.0 (0.1)</span><span class="tag">B -2~2 (0.2)</span><span class="tag">쿨다운 0.5~3h (0.25)</span>
    - n 80~900 (약 27회/년 이상) - <span class="tag">최대중복 &lt;15</span> - 최대보유14일 - fee0.11% - 데이터 2.97년
  </p>
  <div class="legend">
    <span><i style="background:var(--grn)"></i>기대 &gt;0</span><span><i style="background:var(--red)"></i>기대 &lt;0</span>
    <span><i style="background:var(--accent)"></i>PF 선</span><span><i style="background:var(--gold)"></i>채택 K</span>
    <span style="margin-left:auto">판정: 견고(평탄대 양수,PF높음) / 보통 / <b style="color:#cba0e6">노이즈</b>(고K 스파이크뿐) / 제외(최적도 음수)</span>
  </div>

  <h2><i>(1) 요약 - 심볼별 채택값 (K&ge;3.5 흑자 우선, 없으면 전체 최적)</i></h2>
  <div class="wrap"><table>
    <thead><tr><th class="l">심볼</th><th>채택 K1</th><th>B</th><th>CD</th><th>n</th><th>회/년</th><th>최대중복</th><th>승률</th><th>기대/건</th><th>PF</th><th>능선양수율<br><span class="sm">채택K±0.6</span></th><th>판정</th></tr></thead>
    <tbody id="summary"></tbody>
  </table></div>
  <p class="note">능선양수율 = 채택K 좌우 ±0.6 구간에서 기대값이 양수인 비율. 1.0이면 그 부근 전부 흑자(견고한 평탄대), 낮으면 한두 점만 튀는 노이즈.</p>

  <h2><i>(2) 심볼별 K-능선</i> <span class="sm">기대값/PF가 K에 따라 어떻게 변하나 - 평탄대면 진짜 엣지, 스파이크면 과적합</span></h2>
  <div class="nav" id="nav"></div>
  <div id="sections"></div>
  <p class="note">주의: 전부 in-sample. 채택값은 약 27회/년(n80) 최저빈도 컷. 라이브 전 분기별 OOS 필수. 최대중복 14는 한 과열장에 숏이 무더기 진입 -> 동시포지션/자본배분 상한 필수.</p>
<script>
const RAW=__DATA__;
const $=s=>document.querySelector(s);
const KS=RAW.KS, S=RAW.syms;
function verdict(v){const p=v.pick,pr=v.posratio;
  if(p.exp<=0)return['제외','b-no'];
  if(pr>=0.85&&p.pf>=1.25)return['견고','b-ok'];
  if(pr<0.6)return['노이즈','b-noise'];
  return['보통','b-mid'];}
function pn(v){if(v==null)return '-';const c=v>=0?'pos':'neg';return '<span class="'+c+'">'+(v>=0?'+':'')+v+'%</span>';}
const ORD={'견고':0,'보통':1,'노이즈':2,'제외':3};
const syms=Object.keys(S);
syms.sort((a,b)=>{const va=verdict(S[a]),vb=verdict(S[b]);
  return (ORD[va[0]]-ORD[vb[0]])||((S[b].pick.exp)-(S[a].pick.exp));});

$('#summary').innerHTML=syms.map(s=>{const v=S[s],p=v.pick;const bv=verdict(v);
  const lowk=p.k1<3.5?' <span class="sm">(K&lt;3.5)</span>':'';
  return '<tr><td class="sym"><a href="#'+s+'" style="color:var(--txt);text-decoration:none">'+s+'</a></td>'+
   '<td class="k"><span class="pick">'+p.k1+'</span>'+lowk+'</td><td class="k">'+p.b+'</td><td class="k">'+p.cd+'</td>'+
   '<td>'+p.n+'</td><td>'+v.pick_yr+'</td><td>'+p.maxc+'</td><td>'+p.wr+'%</td>'+
   '<td class="big">'+pn(p.exp)+'</td><td class="pf">'+p.pf+'</td><td>'+v.posratio+'</td>'+
   '<td><span class="badge '+bv[1]+'">'+bv[0]+'</span></td></tr>';}).join('');
$('#nav').innerHTML=syms.map(s=>{const bv=verdict(S[s]);return '<a href="#'+s+'">'+s+' <span class="badge '+bv[1]+'" style="font-size:10px">'+bv[0]+'</span></a>';}).join('');

function chart(s){const v=S[s],R=v.ridge,p=v.pick;
  const W=560,Hh=170,padL=34,padR=34,padB=26,padT=14,iw=W-padL-padR,ih=Hh-padT-padB;
  const exps=R.map(r=>r.exp).filter(x=>x!=null);
  let emin=Math.min(...exps,0),emax=Math.max(...exps,0);const ep=(emax-emin)*0.12||0.1;emin-=ep;emax+=ep;
  const pfs=R.map(r=>r.pf).filter(x=>x!=null);let pmax=Math.max(...pfs,1.5);
  const x=i=>padL+iw*i/(KS.length-1);
  const ye=e=>padT+ih*(1-(e-emin)/(emax-emin));
  const yp=pf=>padT+ih*(1-(Math.min(pf,pmax))/(pmax));
  const bw=iw/KS.length*0.62;
  let bars='',line='',pfdots='';
  R.forEach((r,i)=>{if(r.exp==null)return;const cx=x(i);
    const y0=ye(0),y1=ye(r.exp);const top=Math.min(y0,y1),h=Math.abs(y1-y0);
    bars+='<rect x="'+(cx-bw/2)+'" y="'+top+'" width="'+bw+'" height="'+h+'" fill="'+(r.exp>=0?'var(--grn)':'var(--red)')+'" opacity=".55"><title>K'+r.k.toFixed(1)+'  exp '+r.exp+'%  PF '+r.pf+'  n'+r.n+'</title></rect>';});
  let prev=null;
  R.forEach((r,i)=>{if(r.pf==null){prev=null;return;}const cx=x(i),cy=yp(r.pf);
    if(prev)line+='<line x1="'+prev[0]+'" y1="'+prev[1]+'" x2="'+cx+'" y2="'+cy+'" stroke="var(--accent)" stroke-width="1.6"/>';
    pfdots+='<circle cx="'+cx+'" cy="'+cy+'" r="2" fill="var(--accent)"/>';prev=[cx,cy];});
  const pi=KS.findIndex(k=>Math.abs(k-p.k1)<1e-6);let pickline='';
  if(pi>=0)pickline='<line x1="'+x(pi)+'" y1="'+padT+'" x2="'+x(pi)+'" y2="'+(padT+ih)+'" stroke="var(--gold)" stroke-width="1" stroke-dasharray="3 3" opacity=".8"/>'+
     '<text x="'+x(pi)+'" y="'+(padT+8)+'" fill="var(--gold)" font-size="10" text-anchor="middle">채택</text>';
  const zy=ye(0);
  let xt='';KS.forEach((k,i)=>{if(i%2)return;xt+='<text x="'+x(i)+'" y="'+(Hh-7)+'" fill="var(--mut)" font-size="9" text-anchor="middle">'+k.toFixed(1)+'</text>';});
  const yl='<text x="3" y="'+(ye(emax)+8)+'" fill="var(--grn)" font-size="9">'+emax.toFixed(1)+'%</text>'+
    '<text x="3" y="'+(ye(emin)+2)+'" fill="var(--red)" font-size="9">'+emin.toFixed(1)+'%</text>'+
    '<text x="'+(W-30)+'" y="'+(padT+8)+'" fill="var(--accent)" font-size="9">PF '+pmax.toFixed(1)+'</text>'+
    '<text x="'+(W-30)+'" y="'+(padT+ih)+'" fill="var(--accent)" font-size="9">0</text>';
  const i35=KS.findIndex(k=>Math.abs(k-3.5)<1e-6);
  const g35=i35>=0?('<line x1="'+x(i35)+'" y1="'+padT+'" x2="'+x(i35)+'" y2="'+(padT+ih)+'" stroke="var(--mut)" stroke-width=".7" opacity=".4"/>'):'';
  return '<svg viewBox="0 0 '+W+' '+Hh+'" width="100%" style="max-width:'+W+'px">'+
    '<line x1="'+padL+'" y1="'+zy+'" x2="'+(padL+iw)+'" y2="'+zy+'" stroke="var(--line)" stroke-width="1"/>'+
    g35+pickline+bars+line+pfdots+xt+yl+'</svg>';}

function ntbl(s){const R=S[s].ridge;
  return '<div class="wrap"><table><thead><tr><th>K1</th>'+R.map(r=>'<th>'+r.k.toFixed(1)+'</th>').join('')+'</tr></thead><tbody>'+
    '<tr><td class="l">기대%</td>'+R.map(r=>'<td>'+(r.exp==null?'-':(r.exp>=0?'<span class=pos>+':'<span class=neg>')+r.exp.toFixed(2)+'</span>')+'</td>').join('')+'</tr>'+
    '<tr><td class="l">PF</td>'+R.map(r=>'<td>'+(r.pf==null?'-':r.pf)+'</td>').join('')+'</tr>'+
    '<tr><td class="l">n</td>'+R.map(r=>'<td class="sm">'+(r.n||'-')+'</td>').join('')+'</tr>'+
    '</tbody></table></div>';}

$('#sections').innerHTML=syms.map(s=>{const v=S[s],p=v.pick;const bv=verdict(v);
  return '<h2 id="'+s+'"><i>'+s+'</i> <span class="badge '+bv[1]+'">'+bv[0]+'</span>'+
     '<span class="sm">채택 <span class="pick">K'+p.k1+'</span> B'+p.b+' '+p.cd+' - n'+p.n+'('+v.pick_yr+'/yr) - 기대'+pn(p.exp)+' - PF'+p.pf+' - 능선양수율 '+v.posratio+'</span></h2>'+
   '<div class="card"><div class="row"><div style="flex:1;min-width:300px">'+chart(s)+
     '<div class="sm" style="margin-top:4px">막대=기대값/건(K별, n필터無, B/CD는 채택값 고정) - 파란선=PF - 금색=채택K - 회색=3.5</div></div></div>'+
     ntbl(s)+'</div>';}).join('');
</script>
</body>
</html>'''

HTML = HTML.replace('__DATA__', DATA).replace('__MARKET__', MARKET)
open(OUT_HTML, 'w', encoding='utf-8').write(HTML)
print('wrote', OUT_HTML, len(HTML), 'bytes')

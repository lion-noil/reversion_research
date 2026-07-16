# -*- coding: utf-8 -*-
"""MT5 한 계좌 합산(U2+U3) 최저증거금 — 신규 사이징(U2 ×3.5, U3 ×2.3) 기준. universe_scale 재사용."""
import sys, os
sys.argv = ['universe_scale.py', 'U2']  # 생성기 로드용 (아래서 재정의)
FX = {'USDJPY','EURUSD','GBPUSD','AUDUSD','USDCAD','USDCHF','NZDUSD'}
src = open('universe_scale.py', encoding='utf-8').read()
# want()를 비USDT 전체로, BOOK_FRAC에 U2/U3 배수 반영해 실행
src = src.replace("def want(uni, sym):\n    if uni == 'U1': return sym.endswith('USDT')\n    if uni == 'U2': return (not sym.endswith('USDT')) and (sym not in FX)\n    return sym in FX",
"def want(uni, sym):\n    return not sym.endswith('USDT')")
src = src.replace("}[UNI]\n\ntrades = []", "}[UNI]\n\ndef _fx_scale(sym): return 2.3 if sym in FX else 3.5\ntrades = []")
src = src.replace("fr = frac\n                if book == 's22m' and tag == 's13' and sym == 'WTI': fr = frac * 0.5",
"fr = frac * _fx_scale(sym)\n                if book == 's22m' and tag == 's13' and sym == 'WTI': fr = frac * 0.5 * 3.5")
src = src.replace("for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20):", "for k in (1,):")
src = src.replace("s11p = 1 * k; s33p = (2 if UNI == 'U2' else 5) * k",
"s11p='3.5/2.3'; s33p='7/11.5'")
src = src.replace("print(f'{k:>3} | {s11p:>4}% / {s33p:>4}% |", "print(f'{k:>3} | {s11p:>9} / {s33p:>7} |")
src = src.replace("'U2': {'s11m': 0.01, 's22m': 0.01, 's33m': 0.02},",
"'U2': {'s11m': 0.01, 's22m': 0.01, 's33m': 0.02, 's33f': 0.05},")
exec(compile(src, 'u23', 'exec'))

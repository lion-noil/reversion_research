"""1분봉 캐시 → 멀티 타임프레임 리샘플 (5m/1h/4h 등). 다운로드 없음.

기존 `data/{SYM}_1m.pkl`에서 UTC 경계로 버킷 집계 → `data/{SYM}_{N}m.pkl` 저장.
소스/세션이 1m과 100% 일치(휴장 갭 그대로, flats 없음). 일봉(Dm)을 1m서 리샘플한 것과 동일 방식.
저장 키: 5m→`_5m.pkl`, 1h→`_60m.pkl`, 4h→`_240m.pkl` (분 단위, 기존 `_1m`/`_Dm`과 일관).

⚠️ 4h/1h는 epoch(UTC 00:00) 경계 정렬. 브로커 세션 경계와 다를 수 있으나 동일심볼 내 일관.
정렬은 1m이 휴장 갭 가진 그대로라, 세션 안 도는 구간엔 봉이 안 생김(정상).

사용: python resample_tf.py                      # 전 1m캐시 심볼, 5/60/240분
      python resample_tf.py --tfs 5,15,60,240
      python resample_tf.py --symbols US100,BTCUSD,SOLUSDT
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
from reversion_calibrator import _load_cache, _save_cache, CACHE_DIR

MIN = 60_000  # ms


def resample(cs1m: list[dict], n_min: int) -> list[dict]:
    """1m(start 오름차순) → n_min봉. O=첫,H=max,L=min,C=마지막, start=버킷 UTC 경계."""
    if not cs1m:
        return []
    ts = np.fromiter((c["start"] for c in cs1m), np.int64, len(cs1m))
    o = np.fromiter((c["open"] for c in cs1m), np.float64, len(cs1m))
    h = np.fromiter((c["high"] for c in cs1m), np.float64, len(cs1m))
    l = np.fromiter((c["low"] for c in cs1m), np.float64, len(cs1m))
    c = np.fromiter((c["close"] for c in cs1m), np.float64, len(cs1m))
    bucket = n_min * MIN
    keys = (ts // bucket) * bucket
    uniq, start = np.unique(keys, return_index=True)   # sorted → start = 각 그룹 첫 인덱스
    end = np.append(start[1:] - 1, len(c) - 1)          # 각 그룹 마지막 인덱스
    O = o[start]; C = c[end]
    H = np.maximum.reduceat(h, start)
    L = np.minimum.reduceat(l, start)
    return [{"start": int(uniq[i]), "open": float(O[i]), "high": float(H[i]),
             "low": float(L[i]), "close": float(C[i])} for i in range(len(uniq))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="5,60,240", help="분 단위 콤마(기본 5,60,240=5m/1h/4h)")
    ap.add_argument("--symbols", help="콤마구분(기본: data/의 모든 *_1m 심볼)")
    args = ap.parse_args()
    tfs = [int(x) for x in args.tfs.split(",")]
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        syms = sorted(os.path.basename(f)[:-7] for f in glob.glob(os.path.join(CACHE_DIR, "*_1m.pkl")))

    def lbl(n):
        return f"{n}m" if n < 60 else (f"{n // 60}h" if n % 60 == 0 else f"{n}m")
    print(f"리샘플 {len(syms)}심볼 × TF {[lbl(n) for n in tfs]} (1m서 생성, 다운로드X)")
    for s in syms:
        cs1 = _load_cache(s, "1")
        if not cs1:
            print(f"  [{s}] 1m 캐시 없음 — 스킵"); continue
        line = f"  {s:10} 1m {len(cs1):>9,}봉"
        for n in tfs:
            r = resample(cs1, n)
            _save_cache(s, str(n), r)
            line += f"  {lbl(n)} {len(r):>8,}"
        print(line)
    print("완료. 로드: reversion_calibrator._load_cache(SYM, '5'|'60'|'240')")


if __name__ == "__main__":
    main()

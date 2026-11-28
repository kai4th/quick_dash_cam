"""
재생 화면 갱신 경로 테스트 (화면 필요). 스크립트로 실행: python tests/test_playback_perf.py

- photo_put_rgb(Tk_PhotoPutBlock 빠른 경로 / PPM 폴백)가 PPM 경로와 픽셀 단위로 같은 결과를 만드는지
- 빠른 경로가 PPM 경로보다 확실히 빠른지 (Windows에서만 엄격히 검사)
"""
from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.ui import theme


def main() -> int:
    root = tk.Tk()
    h, w = 360, 640
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    ref = theme.make_photo_image(a)
    ph = tk.PhotoImage(width=w, height=h)
    theme.photo_put_rgb(ph, a)
    for (x, y) in ((0, 0), (w - 1, h - 1), (123, 77), (w // 2, h // 2)):
        want = tuple(int(v) for v in a[y, x])
        assert ph.get(x, y) == want == ref.get(x, y), f"pixel mismatch at {(x, y)}: {ph.get(x, y)} vs {want}"
    theme.photo_put_rgb(ph, b)  # 같은 PhotoImage에 새 프레임을 덮어쓰기
    assert ph.get(5, 5) == tuple(int(v) for v in b[5, 5]), "second frame replaces the first"
    non_contig = np.ascontiguousarray(a)[:, ::1][::1]
    theme.photo_put_rgb(ph, np.asfortranarray(non_contig))
    assert ph.get(9, 9) == tuple(int(v) for v in a[9, 9]), "non-contiguous input handled"
    print("[PASS] photo_put_rgb matches PPM path pixel-for-pixel; fast path:", bool(theme._fast_put))

    big = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    bp = tk.PhotoImage(width=1280, height=720)
    theme.photo_put_rgb(bp, big)
    root.update()

    def bench(fn, n=15):
        t = time.perf_counter()
        for _ in range(n):
            fn()
            root.update()
        return (time.perf_counter() - t) / n * 1000.0

    fast = bench(lambda: theme.photo_put_rgb(bp, big))
    slow = bench(lambda: bp.put(b"P6 1280 720 255\n" + big.tobytes()))
    print(f"[INFO] 720p frame update: fast {fast:.1f} ms vs PPM {slow:.1f} ms")
    if theme._fast_put:
        assert fast < slow * 0.6, "fast path should be clearly faster than PPM put"
    print("[PASS] playback frame-update path")
    root.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())

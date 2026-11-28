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


def _play_all(worker, pieces, rate, box=(320, 180), limit=40.0):
    """재생 워커를 끝까지 돌리며 받은 (미디어 시각, 컷 순번) 목록을 반환."""
    sid = worker.start(pieces, rate=rate, box_w=box[0], box_h=box[1])
    got = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        item = worker.take_latest()
        if item is not None and item[0] == sid:
            got.append((item[1], item[3]))
        if worker.is_finished(sid) and not worker.has_pending():
            break
        time.sleep(0.004)
    return got


def test_playback_backends() -> None:
    """열어 둔 캡처 디코더(1x)·ffmpeg 디코더(2x)·캡처 실패 시 ffmpeg 전환이 모두 컷 이어재생을 끝까지 해내는지."""
    import tempfile
    from src.ui import media
    from src.ffmpeg_utils import probe_video_file
    from tests.test_pipeline import _create_synthetic_driving_video

    tmp = Path(tempfile.mkdtemp(prefix="qdc_play_"))
    vp = tmp / "VID_20261024_090000.mp4"
    _create_synthetic_driving_video(vp, 12, 20)
    info = probe_video_file(vp)
    clip = media.PlaybackClip(vp, 0.0, info.duration_sec, info.width, info.height, info.fps)
    pieces = lambda: [media.PlayPiece(clip, 1.0, 3.0, 0), media.PlayPiece(clip, 6.0, 8.0, 1)]  # noqa: E731

    w = media.PlaybackWorker()
    got = _play_all(w, pieces(), 1.0)
    assert w.stats["backend"] == "capture", w.stats
    assert 100 <= len(got) <= 130, f"expected ~120 frames for 4s at 30fps, got {len(got)}"
    assert {t for _m, t in got} == {0, 1}, "both cuts played in order"
    times = [m for m, _t in got]
    assert times == sorted(times), "media time never goes backwards"
    assert times[0] < 1.2 and times[-1] > 7.5, (times[0], times[-1])
    print(f"[PASS] capture backend: {len(got)} frames over 2 cuts")

    w2 = media.PlaybackWorker()
    got2 = _play_all(w2, pieces(), 2.0)
    assert w2.stats["backend"] == "ffmpeg", w2.stats
    assert len(got2) >= 40, len(got2)
    print(f"[PASS] 2x uses ffmpeg backend: {len(got2)} frames")

    orig = media.open_hw_capture
    media.open_hw_capture = lambda _p: None  # 캡처를 못 열 때
    try:
        w3 = media.PlaybackWorker()
        got3 = _play_all(w3, pieces(), 1.0)
    finally:
        media.open_hw_capture = orig
    assert not w3._cv_ok and len(got3) >= 100, (w3._cv_ok, len(got3))
    print(f"[PASS] capture failure falls back to ffmpeg: {len(got3)} frames")
    for x in (w, w2, w3):
        x.close()


def main() -> int:
    test_playback_backends()
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

"""정차 구간 앞뒤 STOP_MARGIN_SEC 이내는 컷 후보에서 빠지는지 확인."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np  # noqa: E402

import src.highlights as H  # noqa: E402
from src.models import HighlightConfig, Segment, SegmentType, VideoFileInfo  # noqa: E402


def _video() -> VideoFileInfo:
    fps, dur = 4.0, 80.0
    ts = np.arange(0, dur, 1 / fps, dtype=np.float32)
    rng = np.random.default_rng(0)
    motion = np.full(len(ts), 12.0, dtype=np.float32) + rng.normal(0, 0.2, len(ts)).astype(np.float32)
    motion[(ts >= 30) & (ts < 40)] = 0.5  # 정차 30~40초
    v = VideoFileInfo(path=Path("x.mp4"), filename="x.mp4", duration_sec=dur)
    v.feature_cache = {"timestamps": ts, "motion": motion, "signed_dx": np.zeros(len(ts), np.float32),
                       "scene": np.full(len(ts), 5.0, np.float32), "sample_fps": fps}
    v.segments = [Segment(0, 30, SegmentType.NORMAL), Segment(30, 40, SegmentType.STOP), Segment(40, dur, SegmentType.NORMAL)]
    return v


def _starts(margin: float) -> list:
    old = H.STOP_MARGIN_SEC
    H.STOP_MARGIN_SEC = margin
    try:
        v = _video()
        cfg = HighlightConfig()
        scored = H.compute_highlight_scores([v], cfg)
        cands = H._window_table([v], scored, 3.0)
        _vi, means, fps, _win = cands[0]
        return [i / fps for i in range(len(means)) if np.isfinite(means[i])]
    finally:
        H.STOP_MARGIN_SEC = old


def test_stop_margin() -> None:
    no_margin = _starts(0.0)
    assert any(40.0 <= s < 41.0 for s in no_margin), "baseline: window right after the stop is usable"
    assert any(26.5 <= s <= 27.0 for s in no_margin), "baseline: window ending right at the stop is usable"
    m = H.STOP_MARGIN_SEC
    starts = _starts(m)
    for s in starts:
        e = s + 3.0
        assert not (e > 30.0 - m and s < 40.0 + m), f"window {s:.2f}-{e:.2f} too close to the stop"
    assert any(s >= 40.0 + m for s in starts) and any(s + 3.0 <= 30.0 - m for s in starts), "far windows must remain"
    print(f"[PASS] Stop margin {m:.1f}s excludes cuts adjacent to a stop ({len(no_margin)} -> {len(starts)} windows)")


if __name__ == "__main__":
    test_stop_margin()
    print("ALL STOP MARGIN TESTS PASSED!")

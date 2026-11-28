"""실제 앱(QuickDashCamApp)을 기본 창 크기로 띄워 쇼츠 컷 이어보기 재생을 측정합니다. 원본 영상은 읽기만 합니다.
사용: python tools/bench_app_playback.py <영상> [시작초=120] [출력미리보기 0/1]
출력: 표시 fps, 프레임 간격 분포, 50ms 초과 수, 컷 경계 멈춤, UI 그리기 시간, GPU 디코딩 여부."""
from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["QUICK_DASH_CAM_HOME"] = tempfile.mkdtemp(prefix="qdc_bench_")

from src.ffmpeg_utils import probe_video_file
from src.models import HighlightClip
from src.ui.app import QuickDashCamApp


def main() -> None:
    path = Path(sys.argv[1])
    t0s = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    out_mode = len(sys.argv) > 3 and sys.argv[3] == "1"
    app = QuickDashCamApp()
    if os.environ.get('BENCH_ZOOMED'):
        app.state('zoomed')
    app.update()
    info = probe_video_file(path)
    info.ensure_segments_initialized()
    app.store.videos = [info]
    app.store.highlight_clips = [HighlightClip(info, t0s + o, t0s + o + 4.0, roll_adjust=1.5) for o in (0, 10, 20)]
    app.show_page("shorts")
    app.update()
    app.store.emit("videos", "highlights")
    app.update()
    sp = app.pages["shorts"]
    if out_mode:
        sp.output_preview_var.set(True)
    pl = sp.player
    app.update()
    cv = pl.canvas
    print(f"video {info.width}x{info.height}@{info.fps:.1f}  window {app.winfo_width()}x{app.winfo_height()}  canvas {cv.winfo_width()}x{cv.winfo_height()}  dpi-scaling={app.tk.call('tk','scaling')}")

    ev = []
    orig = pl.redraw

    def rd():
        a = time.perf_counter()
        orig()
        ev.append((a, pl.position, time.perf_counter() - a, pl._playback.session))
    pl.redraw = rd
    seg_changes = []
    pl.on_segment_start = lambda k: seg_changes.append((k, time.perf_counter()))
    pl._playback.trace = []
    sp.preview_all()
    start = time.perf_counter()
    # 실제 앱과 같이 mainloop로 돌림 (update() 바쁜 루프는 CPU 코어 하나를 점유해 측정을 왜곡함)
    def watchdog():
        if time.perf_counter() - start > 20 or (not pl.playing and time.perf_counter() - start > 1):
            app.quit()
        else:
            app.after(100, watchdog)
    app.after(100, watchdog)
    app.mainloop()
    wall = time.perf_counter() - start
    ev = [e for e in ev if e[0] >= start]
    ts = [e[0] for e in ev]
    dts = [(b - a) * 1000 for a, b in zip(ts, ts[1:])]
    draw = [e[2] * 1000 for e in ev]
    # 컷 경계 멈춤: media 시간이 컷 사이로 점프하는 지점의 벽시계 간격 - 정상 프레임 간격
    stalls = []
    for (a, m1, _d, _s), (b, m2, _d2, _s2) in zip(ev, ev[1:]):
        if m2 - m1 > 1.0:  # 컷 점프
            stalls.append((b - a) * 1000)
    media = 12.0
    st = pl._playback.stats
    dts_s = sorted(dts)
    print(f"hw={st['hw']} noref_forced={pl._playback._force_noref} stats={st}")
    print(f"frames shown={len(ev)} in wall={wall:.2f}s (media {media:.0f}s)  fps={len(ev)/media:.1f}")
    print(f"interval ms: median={statistics.median(dts):.1f} p95={dts_s[int(len(dts_s)*.95)]:.1f} p99={dts_s[int(len(dts_s)*.99)]:.1f} max={max(dts):.0f}  >40ms:{sum(d>40 for d in dts)}  >50ms:{sum(d>50 for d in dts)}  >100ms:{sum(d>100 for d in dts)}")
    print(f"UI draw ms: median={statistics.median(draw):.1f} p95={sorted(draw)[int(len(draw)*.95)]:.1f}")
    print(f"cut-boundary gaps ms: {[round(s) for s in stalls]}")
    tr = pl._playback.trace
    if tr:
        # 워커 쪽: 게시 간격(디코딩·페이싱), 그리기 쪽과 비교
        pub = [t for _a, t, _m, _q in tr]
        pdt = [(b - a) * 1000 for a, b in zip(pub, pub[1:])]
        pdt_s = sorted(pdt)
        print(f"worker publish interval ms: median={statistics.median(pdt):.1f} p95={pdt_s[int(len(pdt_s)*.95)]:.1f} max={max(pdt):.0f} >50ms:{sum(d>50 for d in pdt)}")
        import collections
        h = collections.Counter(int(d // 4) * 4 for d in pdt)
        print("publish interval histogram (ms bucket: count):", dict(sorted(h.items())))
        bad = [(i, round(pdt[i]), tr[i][3], tr[i + 1][3]) for i in range(len(pdt)) if pdt[i] > 60]
        print("stalls (idx, gap ms, queue depth before, after):", bad[:12])
        first_media = tr[0][2]
        print(f"first frame latency after play: {(pub[0]-start)*1000:.0f} ms")
        # 게시된 프레임 중 UI가 그리지 못하고 덮어쓴 수
        drawn = {round(e[1], 3) for e in ev}
        missed = sum(1 for _a, _t, m, _q in tr if round(m, 3) not in drawn)
        print(f"published={len(tr)} drawn={len(ev)} overwritten(dropped by UI)={missed}")
        # 게시 -> 그리기 지연
        import bisect
        lat = []
        evt = sorted((e[0], round(e[1], 3)) for e in ev)
        dm = {m: a for a, m in evt}
        for _a, t, m, _q in tr:
            if round(m, 3) in dm:
                lat.append((dm[round(m, 3)] - t) * 1000)
        if lat:
            print(f"publish->draw latency ms: median={statistics.median(lat):.1f} p95={sorted(lat)[int(len(lat)*.95)]:.1f} max={max(lat):.0f}")
    app._shutdown() if hasattr(app, "_shutdown") else app.destroy()


if __name__ == "__main__":
    main()

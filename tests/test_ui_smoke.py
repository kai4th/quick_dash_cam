"""
새 UI 전체 흐름 자동 조작 테스트 (화면이 필요 — 리눅스에서는 `xvfb-run python tests/test_ui_smoke.py`).

① 영상 불러오기 → ② 자동 구간 분석·리뷰 결정·되돌리기 → ③ 합치기 예상 길이 → ④ 하이라이트 찾기(알고리즘)
→ 컷 사용/제외 · 기울기 · 소실점 → 쇼츠 만들기. 파일 대화상자와 알림 창은 가짜로 대체합니다.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("[SKIP] tkinter가 없어 UI 테스트를 건너뜁니다.")
        return 0
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("[SKIP] 화면(DISPLAY)이 없어 UI 테스트를 건너뜁니다. xvfb-run으로 실행하세요.")
        return 0

    import tkinter.filedialog as fd
    import tkinter.messagebox as mb

    from src.ffmpeg_utils import probe_video_file
    from tests.test_pipeline import _create_synthetic_driving_video

    tmp = Path(tempfile.mkdtemp(prefix="qdc_ui_"))
    v1, v2 = tmp / "VID_20261024_090000.mp4", tmp / "VID_20261024_093000.mp4"
    _create_synthetic_driving_video(v1, 32, 20)
    _create_synthetic_driving_video(v2, 32, 20, palette=1)
    out = tmp / "shorts.mp4"
    fd.asksaveasfilename = lambda **k: str(out)
    msgs = []
    for name in ("showinfo", "showwarning", "showerror"):
        setattr(mb, name, (lambda n: lambda *a, **k: msgs.append((n, a)))(name))
    mb.askyesno = lambda *a, **k: True

    from src.ui.app import QuickDashCamApp

    app = QuickDashCamApp()
    app.geometry("1280x800+0+0")
    result = {"ok": False}

    def wait_idle(limit=180):
        t0 = time.time()
        while app.store.busy and time.time() - t0 < limit:
            yield 0.2

    def steps():
        yield 0.5
        app.pages["library"].load_paths([v2, v1])
        yield from wait_idle()
        assert [v.filename for v in app.store.videos] == [v1.name, v2.name], "chronological order"
        print("[PASS] ① 영상: 불러오기·촬영 시각 순 정렬")

        app.show_page("edit")
        ed = app.pages["edit"]
        ed.analyze()
        yield from wait_idle()
        assert app.store.review_items, "review items after analysis"
        ed.review_select(0, autoplay=False)
        yield 0.3
        before = [list(v.segments) for v in app.store.videos]
        ed.review_decide("cut")
        yield 0.3
        assert app.store.review_items and any(it.decision == "cut" for it in app.store.review_items)
        ed.undo()
        yield 0.3
        assert [len(v.segments) for v in app.store.videos] == [len(s) for s in before], "undo restores segments"
        print(f"[PASS] ② 편집: 자동 분석(검출 {len(app.store.review_items)}개)·리뷰 결정·되돌리기")

        app.show_page("export")
        yield 0.3
        assert "→" in app.pages["export"].big_label.cget("text")
        print(f"[PASS] ③ 합치기: 예상 길이 {app.pages['export'].big_label.cget('text')}")

        app.show_page("shorts")
        sp = app.pages["shorts"]
        sp.ai_var.set("알고리즘만 (빠름·오프라인)")
        sp.find_highlights()
        yield from wait_idle()
        yield 1.0
        clips = app.store.highlight_clips
        assert clips and sp.index is not None, "highlights found and first cut selected"
        sp.toggle_use()
        assert not sp.selected().included
        sp.toggle_use()
        sp.rotate(-1.5)
        geom = sp.player.image_geom
        assert geom is not None, "cut frame shown in the big preview"
        _x0, _y0, w, h = geom
        sp._on_player_click(w * 0.5, h * 0.4, w, h)
        c = sp.selected()
        assert c.roll_adjust == -1.5 and c.vp is not None, (c.roll_adjust, c.vp)
        sp.make_shorts()
        yield from wait_idle()
        p = probe_video_file(out)
        assert out.exists() and p.width == 1080 and p.height == 1920, (p.width, p.height)
        assert any("수동" in str(m[1]) for m in msgs if m[0] in ("showinfo", "showwarning")), msgs
        print(f"[PASS] ④ 쇼츠: 장면 {len(clips)}개 · 사용/제외 · 기울기·소실점 · 쇼츠 완성 {p.width}x{p.height}")
        result["ok"] = True

    gen = steps()

    def tick():
        try:
            app.after(int(next(gen) * 1000), tick)
        except StopIteration:
            app._on_close()
        except Exception:
            import traceback
            traceback.print_exc()
            app._on_close()

    app.after(300, tick)
    app.mainloop()
    if result["ok"]:
        print("ALL UI SMOKE TESTS PASSED!")
        return 0
    print("UI SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())

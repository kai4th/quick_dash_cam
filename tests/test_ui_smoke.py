"""
새 UI 전체 흐름 자동 조작 테스트 (화면이 필요 — 리눅스에서는 `xvfb-run python tests/test_ui_smoke.py`).

1차 실행: ① 영상 불러오기 → ② 자동 구간 분석·리뷰 결정·되돌리기 → ③ 합치기 예상 길이 → ④ 하이라이트 찾기
         → 사용/제외 · 컷 범위 옮기기·길이 · 기울기·전체 적용 · 소실점 · 결과 화면 미리보기 · 단축키 안내 → 쇼츠 만들기
2차 실행: 지난 작업 이어하기 → 영상·구간·쇼츠 컷·맞춤·설정·분석 캐시 복원 확인
파일 대화상자와 알림 창은 가짜로 대체하고, 설정 폴더는 임시 폴더를 씁니다.
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


def _run(app, steps) -> bool:
    result = {"ok": False}
    gen = steps(result)

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
    return result["ok"]


def main() -> int:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("[SKIP] tkinter가 없어 UI 테스트를 건너뜁니다.")
        return 0
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("[SKIP] 화면(DISPLAY)이 없어 UI 테스트를 건너뜁니다. xvfb-run으로 실행하세요.")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="qdc_ui_"))
    os.environ["QUICK_DASH_CAM_HOME"] = str(tmp / "config")  # 실제 사용자 설정을 건드리지 않음

    import tkinter.filedialog as fd
    import tkinter.messagebox as mb

    from src.ffmpeg_utils import probe_video_file
    from tests.test_pipeline import _create_synthetic_driving_video

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

    remembered = {}

    # ── 1차 실행 ──
    app = QuickDashCamApp()
    app.geometry("1280x800+0+0")

    def wait_idle(limit=180):
        t0 = time.time()
        while app.store.busy and time.time() - t0 < limit:
            yield 0.2

    def first(result):
        yield 0.8
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
        before = [len(v.segments) for v in app.store.videos]
        ed.review_decide("cut")
        yield 0.3
        assert any(it.decision == "cut" for it in app.store.review_items)
        ed.undo()
        yield 0.3
        assert [len(v.segments) for v in app.store.videos] == before, "undo restores segments"
        print(f"[PASS] ② 편집: 자동 분석(검출 {len(app.store.review_items)}개)·리뷰 결정·되돌리기")

        app.show_page("export")
        yield 0.3
        assert "→" in app.pages["export"].big_label.cget("text")
        print(f"[PASS] ③ 합치기: 예상 길이 {app.pages['export'].big_label.cget('text')}")

        app.show_page("shorts")
        sp = app.pages["shorts"]
        sp.ai_var.set("알고리즘만 (빠름·오프라인)")
        sp.len_var.set("15초")
        sp.find_highlights()
        yield from wait_idle()
        yield 1.0
        clips = app.store.highlight_clips
        assert len(clips) >= 2 and sp.index is not None, "highlights found and first cut selected"
        sp.toggle_use()
        assert not sp.selected().included
        sp.toggle_use()

        c = sp.selected()
        s0, e0 = c.start_sec, c.end_sec
        sp.shift_cut(-0.5)
        assert abs(c.start_sec - (s0 - 0.5)) < 1e-6 and abs(c.duration - (e0 - s0)) < 1e-6, "shift keeps length"
        sp.resize_cut(0.5)
        assert abs(c.duration - (e0 - s0 + 0.5)) < 1e-6, "resize grows by 0.5s"
        sp.resize_cut(-0.5)
        sp.shift_cut(0.5)
        yield 0.8
        print("[PASS] ④ 컷 범위: 앞뒤로 옮기기 · 길이 조절")

        sp.rotate(-1.5)
        sp.apply_roll_all()
        assert all(x.roll_adjust == -1.5 for x in clips if x.included), "roll applied to all used cuts"
        # 기울기 슬라이더: 오른쪽(+)으로 밀면 반시계(roll_adjust 음수), 저장값의 의미는 그대로
        sp._on_roll_slider(3.0)
        assert sp.selected().roll_adjust == -3.0, 'slider right = counter-clockwise (negative roll)'
        sp._on_roll_slider(-2.5)
        assert sp.selected().roll_adjust == 2.5, 'slider left = clockwise (positive roll)'
        sp._on_roll_slider(0.0)
        assert sp.selected().roll_adjust == 0.0 and str(sp.selected().roll_adjust) == '0.0', 'no -0.0'
        sp.selected().roll_adjust = -1.5
        sp._update_cut_panel()
        assert abs(sp.roll_slider.get() - 1.5) < 1e-6, 'panel shows stored roll mirrored on the slider'
        yield 0.5
        geom = sp.player.image_geom
        assert geom is not None, "cut frame shown in the big preview"
        _x0, _y0, w, h = geom
        sp._on_player_click(w * 0.5, h * 0.4, w, h)
        assert c.vp is not None
        print("[PASS] ④ 맞춤: 기울기 · 전체 적용 · 소실점")

        sp.title_entry.insert(0, "테스트 제목")
        sp.output_preview_var.set(True)
        sp.player.redraw()
        yield 0.3
        og = getattr(sp, "_out_geom", None)
        assert og and og["h"] > og["w"] and og["band"] > 0, f"vertical 9:16 preview: {og}"
        _x0, _y0, w2, h2 = sp.player.image_geom
        assert h2 > w2, "preview image is portrait"
        n_before = len(msgs)
        sp._on_player_click(10, 10, w2, h2)  # 결과 화면 미리보기에서는 소실점 지정 안 됨
        sp.output_preview_var.set(False)
        sp.player.redraw()
        print(f"[PASS] ④ 결과 화면 미리보기: {og['w']}×{og['h']} (세로 9:16, 흐림 배경 띠 {og['band']:.0f}px)")

        class Key:
            keysym, state = "F1", 0
        app._on_key(Key())
        assert len(msgs) > n_before and "단축키" in str(msgs[-1][1][0]), msgs[-1]
        print("[PASS] 단축키 안내 (F1)")

        sp.make_shorts()
        yield from wait_idle()
        p = probe_video_file(out)
        assert out.exists() and p.width == 1080 and p.height == 1920, (p.width, p.height)
        assert any("수동" in str(m[1]) for m in msgs if m[0] in ("showinfo", "showwarning")), msgs[-1]
        print(f"[PASS] ④ 쇼츠 완성 {p.width}x{p.height}")

        remembered.update(
            videos=[v.filename for v in app.store.videos],
            segments=[[(round(s.start_sec, 2), s.segment_type.value) for s in v.segments] for v in app.store.videos],
            clips=[(round(x.start_sec, 2), round(x.end_sec, 2), x.included, x.roll_adjust, x.vp) for x in clips],
            length=sp.len_var.get(),
        )
        result["ok"] = True

    if not _run(app, first):
        print("UI SMOKE TEST FAILED (1차)")
        return 1

    # ── 2차 실행: 지난 작업 이어하기 ──
    app = QuickDashCamApp()
    app.geometry("1280x800+0+0")

    def second(result):
        yield 1.5
        t0 = time.time()
        while app.store.busy and time.time() - t0 < 60:
            yield 0.2
        st = app.store
        assert [v.filename for v in st.videos] == remembered["videos"], "videos restored"
        segs = [[(round(s.start_sec, 2), s.segment_type.value) for s in v.segments] for v in st.videos]
        assert segs == remembered["segments"], "segments restored"
        clips = [(round(x.start_sec, 2), round(x.end_sec, 2), x.included, x.roll_adjust,
                  tuple(x.vp) if x.vp else None) for x in st.highlight_clips]
        want = [(a, b, c, d, tuple(e) if e else None) for a, b, c, d, e in remembered["clips"]]
        assert clips == want, (clips, want)
        assert all(v.feature_cache is not None for v in st.videos), "analysis caches reloaded"
        assert app.pages["shorts"].len_var.get() == remembered["length"], "settings remembered"
        assert app.pages["shorts"].title_entry.get() == "테스트 제목", "title remembered"
        assert app.current_page == "shorts", "opens the shorts page when cuts exist"
        print(f"[PASS] 이어하기: 영상 {len(st.videos)}개 · 쇼츠 컷 {len(st.highlight_clips)}개 · 맞춤 · 설정 · 분석 캐시 복원")
        result["ok"] = True

    if not _run(app, second):
        print("UI SMOKE TEST FAILED (2차)")
        return 1
    print("ALL UI SMOKE TESTS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

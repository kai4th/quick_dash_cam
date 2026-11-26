"""② 편집: 큰 플레이어 + 전체 타임라인 + 구간 편집 도구 + 자동 구간 분석 + 검출 구간 리뷰."""
from __future__ import annotations

from tkinter import messagebox
from typing import List, Optional

import customtkinter as ctk

from ..analyzer import analyze_video, reclassify_from_cache
from ..models import (
    AnalysisConfig,
    SegmentType,
    apply_global_timeline_override,
    apply_segment_override,
    map_global_time_to_clip,
    merge_preserving_manual_cuts,
)
from .player import VideoPlayer
from .state import DECISION_LABELS, decision_type
from .theme import (
    ACCENT, ACCENT_HOVER, BG_CARD, COLOR_MANUAL_CUT, COLOR_NORMAL, COLOR_STOP, COLOR_STRAIGHT, FG_MUTED,
    NEUTRAL, NEUTRAL_HOVER, font, format_duration,
)
from .timeline import GlobalTimelineCanvas
from .widgets import card, make_tree, tree_selected_index

PAD_OPTIONS = ["0.0초 (즉시 전환)", "0.5초", "1.0초 (자연스러운 전환)", "1.5초", "2.0초"]
REVIEW_FILTERS = ["전체 보기", "미검토만", "신호대기만", "단조 주행만"]


class EditPage(ctk.CTkFrame):
    SHORTCUTS = (
        "재생·이동\n  Space  재생 / 일시정지\n  ← →  1초 이동   (Shift: 5초)\n\n"
        "구간 편집\n  I / O  시작점 / 끝점 찍기\n  Del  선택 구간 삭제\n  Ctrl+Z  되돌리기\n\n"
        "검출 구간 리뷰\n  1 / 2 / 3  스킵 유지 / 일반 복원 / 삭제\n  N / P  다음 / 이전 구간\n  R  다시 재생\n"
        "  [ / ]  현재 위치를 구간 시작 / 끝으로\n\n"
        "타임라인\n  + / -  확대 / 축소   0  전체 보기\n  마우스 휠 확대·축소 · Shift+휠 이동 · 우클릭 드래그 구간 선택"
    )
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.store = app.store
        self.cut_in: Optional[float] = None
        self.cut_out: Optional[float] = None
        self.review_index: Optional[int] = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        right = ctk.CTkScrollableFrame(self, fg_color="transparent", width=380)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_columnconfigure(0, weight=1)

        self._build_player(left)
        self._build_timeline(left)
        self._build_analysis(right)
        self._build_review(right)

        self.store.subscribe("videos", self._on_videos)
        self.store.subscribe("segments", self._on_segments)
        self.store.subscribe("busy", self._on_busy)
        reg = app.settings
        reg.register_var("edit.detect_stop", self.chk_stop, [True, False])
        reg.register_var("edit.detect_straight", self.chk_straight, [True, False])
        reg.register("edit.sensitivity", self.sens.get, lambda v: self.sens.set(min(90, max(10, float(v)))))
        reg.register("edit.min_duration", self.mindur.get, lambda v: self.mindur.set(min(20, max(3, float(v)))))
        reg.register_var("edit.padding", self.pad_var, PAD_OPTIONS)
        reg.register_var("edit.gpu", self.gpu_var, [True, False])
        reg.register_var("edit.review_autoplay", self.autoplay_var, [True, False])
        self._sync_config()

    # ── 화면 구성 ──

    def _build_player(self, parent) -> None:
        self.player = VideoPlayer(parent, empty_text="① 영상 화면에서 주행 영상을 추가하면\n여기에서 재생·탐색할 수 있습니다")
        self.player.grid(row=0, column=0, sticky="nsew")
        self.player.on_position = self._on_player_position
        self.player.overlay = self._draw_badge

    def _build_timeline(self, parent) -> None:
        box = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10)
        box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        box.grid_columnconfigure(0, weight=1)

        row0 = ctk.CTkFrame(box, fg_color="transparent")
        row0.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        ctk.CTkLabel(row0, text="전체 타임라인", font=font(13, True)).pack(side="left", padx=(0, 10))
        for txt, cmd, w in (
            ("🔍+", lambda: self.timeline.zoom_by(0.5), 44), ("🔍−", lambda: self.timeline.zoom_by(2.0), 44),
            ("◀", lambda: self._pan(-0.5), 36), ("▶", lambda: self._pan(0.5), 36),
            ("1분 보기", lambda: self._zoom_around(60.0), 74), ("전체 보기", lambda: self.timeline.zoom_full(), 80),
        ):
            ctk.CTkButton(row0, text=txt, width=w, height=26, font=font(11), fg_color=NEUTRAL,
                          hover_color=NEUTRAL_HOVER, command=cmd).pack(side="left", padx=2)
        self.view_label = ctk.CTkLabel(row0, text="보기: 전체", font=font(11), text_color=FG_MUTED)
        self.view_label.pack(side="left", padx=8)
        self.clip_label = ctk.CTkLabel(row0, text="", font=font(11), text_color=FG_MUTED)
        self.clip_label.pack(side="right")

        self.timeline = GlobalTimelineCanvas(box, on_seek=self._on_timeline_seek, on_range_select=self._on_range_select,
                                             on_view_change=self._on_view_change, height=72)
        self.timeline.grid(row=1, column=0, sticky="ew", padx=10)

        legend = ctk.CTkFrame(box, fg_color="transparent")
        legend.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 0))
        for color, text in ((COLOR_NORMAL, "일반(1x)"), (COLOR_STOP, "정차(스킵)"), (COLOR_STRAIGHT, "직진(스킵)"),
                            (COLOR_MANUAL_CUT, "삭제")):
            ctk.CTkFrame(legend, width=10, height=10, fg_color=color, corner_radius=2).pack(side="left", padx=(0, 4))
            ctk.CTkLabel(legend, text=text, font=font(11)).pack(side="left", padx=(0, 12))
        ctk.CTkLabel(legend, text="휠: 확대·축소 · Shift+휠: 이동 · 우클릭 드래그: 구간 선택", font=font(11),
                     text_color="#38bdf8").pack(side="right")

        tools = ctk.CTkFrame(box, fg_color="transparent")
        tools.grid(row=3, column=0, sticky="ew", padx=10, pady=(8, 4))
        self.edit_buttons: List[ctk.CTkButton] = []
        for txt, cmd, fg, hv in (
            ("🚩 시작점 [I]", self.mark_in, "#0f766e", "#115e59"),
            ("🏁 끝점 [O]", self.mark_out, "#0f766e", "#115e59"),
            ("✂ 선택 구간 삭제 [Del]", self.cut_selection, "#dc2626", "#b91c1c"),
            ("♻ 일반(1x)으로 복원", self.restore_selection, "#15803d", "#166534"),
        ):
            b = ctk.CTkButton(tools, text=txt, height=32, font=font(12, True), fg_color=fg, hover_color=hv, command=cmd)
            b.pack(side="left", padx=(0, 6))
            self.edit_buttons.append(b)
        self.btn_undo = ctk.CTkButton(tools, text="↩ 되돌리기 [Ctrl+Z]", height=32, font=font(12), fg_color=NEUTRAL,
                                      hover_color=NEUTRAL_HOVER, state="disabled", command=self.undo)
        self.btn_undo.pack(side="left", padx=(0, 6))
        self.sel_label = ctk.CTkLabel(box, text="선택: 없음", font=font(12), text_color=FG_MUTED, anchor="w")
        self.sel_label.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))

    def _build_analysis(self, parent) -> None:
        outer, body = card(parent, "🔍 자동 구간 분석", "신호대기·고속도로 직진처럼 화면 변화가 적은 구간을 찾아 배속/삭제 대상으로 표시합니다.", wrap=330)
        outer.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.chk_stop = ctk.BooleanVar(value=True)
        self.chk_straight = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(body, text="신호대기 / 정차", variable=self.chk_stop, command=self._on_detect_setting).grid(row=0, column=0, sticky="w", pady=2)
        ctk.CTkCheckBox(body, text="고속도로 직진 (단조 주행)", variable=self.chk_straight, command=self._on_detect_setting).grid(row=1, column=0, sticky="w", pady=2)

        self.sens_label = ctk.CTkLabel(body, text="", font=font(12), anchor="w")
        self.sens_label.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.sens = ctk.CTkSlider(body, from_=10, to=90, number_of_steps=16, command=lambda _v: self._on_detect_setting())
        self.sens.set(50)
        self.sens.grid(row=3, column=0, sticky="ew")
        self.mindur_label = ctk.CTkLabel(body, text="", font=font(12), anchor="w")
        self.mindur_label.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.mindur = ctk.CTkSlider(body, from_=3, to=20, number_of_steps=17, command=lambda _v: self._on_detect_setting())
        self.mindur.set(5)
        self.mindur.grid(row=5, column=0, sticky="ew")
        ctk.CTkLabel(body, text="진입·출발 전후 여유", font=font(12), anchor="w").grid(row=6, column=0, sticky="ew", pady=(8, 0))
        self.pad_var = ctk.StringVar(value=PAD_OPTIONS[2])
        ctk.CTkOptionMenu(body, values=PAD_OPTIONS, variable=self.pad_var, command=lambda _v: self._on_detect_setting()).grid(row=7, column=0, sticky="ew")
        self.gpu_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(body, text="GPU로 빠르게 분석", variable=self.gpu_var, command=self._sync_config).grid(row=8, column=0, sticky="w", pady=(8, 2))
        self.btn_analyze = ctk.CTkButton(body, text="🔍 자동 구간 분석 시작", height=40, font=font(14, True),
                                         fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.analyze)
        self.btn_analyze.grid(row=9, column=0, sticky="ew", pady=(10, 0))
        ctk.CTkLabel(body, text="분석 후 민감도·최소 시간을 바꾸면 즉시 다시 계산됩니다.", font=font(11),
                     text_color=FG_MUTED, anchor="w").grid(row=10, column=0, sticky="ew", pady=(4, 0))

    def _build_review(self, parent) -> None:
        outer, body = card(parent, "🔎 검출 구간 리뷰", "구간을 고르면 그 부분이 재생됩니다. 스킵 유지 / 일반 복원 / 삭제를 정하세요.", wrap=330)
        outer.grid(row=1, column=0, sticky="nsew")
        head = ctk.CTkFrame(body, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        self.review_filter = ctk.StringVar(value=REVIEW_FILTERS[0])
        ctk.CTkOptionMenu(head, values=REVIEW_FILTERS, variable=self.review_filter, width=120, height=28,
                          command=lambda _v: self._refresh_review()).pack(side="left")
        self.autoplay_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(head, text="선택 시 재생", variable=self.autoplay_var, font=font(11)).pack(side="right")
        self.review_progress = ctk.CTkLabel(body, text="", font=font(11), text_color=FG_MUTED, anchor="w", justify="left")
        self.review_progress.grid(row=1, column=0, sticky="ew", pady=(6, 4))

        cols = [("type", "유형", 90, "w", False), ("start", "위치", 80, "center", False),
                ("dur", "길이", 60, "e", False), ("status", "결정", 100, "w", True)]
        self.review_tree, box = make_tree(body, cols, "Review2.Treeview", row_height=26)
        box.grid(row=2, column=0, sticky="ew")
        self.review_tree.configure(height=6)
        self.review_tree.bind("<<TreeviewSelect>>", self._on_review_select)
        for tag, color in (("pending", "#fbbf24"), ("keep", "#fdba74"), ("normal", "#4ade80"), ("cut", "#f87171")):
            self.review_tree.tag_configure(tag, foreground=color)

        dec = ctk.CTkFrame(body, fg_color="transparent")
        dec.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.review_buttons: List[ctk.CTkButton] = []
        for col, (txt, d, fg, hv) in enumerate((("✔ 스킵 [1]", "keep", "#b45309", "#92400e"),
                                                 ("♻ 일반 [2]", "normal", "#15803d", "#166534"),
                                                 ("✂ 삭제 [3]", "cut", "#dc2626", "#b91c1c"))):
            dec.grid_columnconfigure(col, weight=1)
            b = ctk.CTkButton(dec, text=txt, height=34, font=font(12, True), fg_color=fg, hover_color=hv,
                              command=lambda d=d: self.review_decide(d))
            b.grid(row=0, column=col, padx=2, sticky="ew")
            self.review_buttons.append(b)
        nav = ctk.CTkFrame(body, fg_color="transparent")
        nav.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        for col, (txt, cmd) in enumerate((("◀ 이전 [P]", lambda: self.review_step(-1)), ("▶ 다시 재생 [R]", self.review_replay),
                                          ("다음 [N] ▶", lambda: self.review_step(1)))):
            nav.grid_columnconfigure(col, weight=1)
            ctk.CTkButton(nav, text=txt, height=28, font=font(11), fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                          command=cmd).grid(row=0, column=col, padx=2, sticky="ew")
        bnd = ctk.CTkFrame(body, fg_color="transparent")
        bnd.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        for col, (txt, cmd) in enumerate((("⇤ 시작점을 현재 위치로 [ [ ]", lambda: self.review_boundary("start")),
                                          ("끝점을 현재 위치로 ⇥ [ ] ]", lambda: self.review_boundary("end")))):
            bnd.grid_columnconfigure(col, weight=1)
            ctk.CTkButton(bnd, text=txt, height=28, font=font(11), fg_color="#1e293b", hover_color=NEUTRAL,
                          border_width=1, border_color=NEUTRAL_HOVER, command=cmd).grid(row=0, column=col, padx=2, sticky="ew")
        self.btn_keep_all = ctk.CTkButton(body, text="미검토 구간 모두 스킵 유지", height=30, font=font(11),
                                          fg_color="#1e293b", hover_color=NEUTRAL, border_width=1,
                                          border_color=NEUTRAL_HOVER, command=self.review_keep_all)
        self.btn_keep_all.grid(row=6, column=0, sticky="ew", pady=(6, 0))

    # ── 화면 전환 ──

    def on_settings_loaded(self) -> None:
        self._sync_config()

    def on_show(self) -> None:
        self.timeline.redraw()
        self.player.request_frame()

    def on_hide(self) -> None:
        self.player.stop()

    def shutdown(self) -> None:
        self.player.close()

    # ── 상태 반영 ──

    def _on_videos(self) -> None:
        self.cut_in = self.cut_out = None
        self.review_index = None
        self.player.set_clips(self.store.playback_clips())
        self._on_segments()

    def _on_segments(self) -> None:
        self.timeline.set_state(self.store.videos, self.player.position, self.cut_in, self.cut_out)
        self.btn_undo.configure(state="normal" if self.store.can_undo() and not self.store.busy else "disabled")
        self._update_sel_label()
        self._refresh_review()
        self.player.redraw()

    def _on_busy(self) -> None:
        st = "disabled" if self.store.busy else "normal"
        self.btn_analyze.configure(state=st)
        for b in self.edit_buttons + self.review_buttons + [self.btn_keep_all]:
            b.configure(state=st)
        self.btn_undo.configure(state="normal" if self.store.can_undo() and not self.store.busy else "disabled")

    # ── 플레이어 · 타임라인 ──

    def _on_player_position(self, t: float) -> None:
        self.timeline.set_playhead(t)
        mapped = map_global_time_to_clip(self.store.videos, t)
        if mapped is not None:
            idx, v, local_t, _seg = mapped
            self.clip_label.configure(text=f"클립 #{idx + 1} {v.filename}  {format_duration(local_t, True)}")

    def _on_timeline_seek(self, t: float) -> None:
        self.player.seek(t)

    def _on_range_select(self, s: float, e: float) -> None:
        total = self.store.total_duration
        self.cut_in, self.cut_out = max(0.0, min(s, e)), min(total, max(s, e))
        self._on_segments()

    def _on_view_change(self, vs: float, ve: float, zoomed: bool) -> None:
        if not zoomed:
            self.view_label.configure(text="보기: 전체")
        else:
            ratio = self.store.total_duration / max(0.001, ve - vs)
            self.view_label.configure(text=f"보기: {format_duration(vs)} ~ {format_duration(ve)} (×{ratio:.1f})")

    def _pan(self, frac: float) -> None:
        vs, ve = self.timeline.view_range
        self.timeline.pan_by((ve - vs) * frac)

    def _zoom_around(self, span: float) -> None:
        t = self.player.position
        self.timeline.set_view(t - span / 2, t + span / 2)

    def _draw_badge(self, cv, x0: int, y0: int, w: int, h: int) -> None:
        mapped = map_global_time_to_clip(self.store.videos, self.player.position)
        if mapped is None:
            return
        _idx, v, _lt, seg = mapped
        if seg == SegmentType.MANUAL_CUT:
            txt, fill = "✂ 삭제 구간 (결과에서 빠짐)", "#991b1b"
        elif seg == SegmentType.STOP and v.analyzed:
            txt, fill = "🟠 신호대기·정차 (스킵)", "#92400e"
        elif seg == SegmentType.STRAIGHT and v.analyzed:
            txt, fill = "🟣 고속도로 직진 (스킵)", "#6b21a8"
        else:
            return
        cv.create_rectangle(x0 + 10, y0 + 10, x0 + 18 + len(txt) * 9, y0 + 38, fill=fill, outline="#ffffff")
        cv.create_text(x0 + 18, y0 + 24, text=txt, anchor="w", fill="#ffffff", font=("Malgun Gothic", 11, "bold"))

    # ── 구간 편집 ──

    def mark_in(self) -> None:
        if not self.store.videos:
            return
        self.cut_in = self.player.position
        if self.cut_out is not None and self.cut_out < self.cut_in:
            self.cut_out = None
        self._on_segments()

    def mark_out(self) -> None:
        if not self.store.videos:
            return
        self.cut_out = self.player.position
        if self.cut_in is None:
            self.cut_in = 0.0
        elif self.cut_out < self.cut_in:
            self.cut_in, self.cut_out = self.cut_out, self.cut_in
        self._on_segments()

    def _selection(self) -> Optional[tuple]:
        if self.cut_in is None and self.cut_out is None:
            return None
        s = self.cut_in if self.cut_in is not None else 0.0
        e = self.cut_out if self.cut_out is not None else self.store.total_duration
        return (min(s, e), max(s, e))

    def _update_sel_label(self) -> None:
        sel = self._selection()
        if sel is None:
            self.sel_label.configure(text="선택: 없음 (I/O 또는 타임라인 우클릭 드래그)", text_color=FG_MUTED)
        else:
            s, e = sel
            self.sel_label.configure(text=f"선택: {format_duration(s, True)} ~ {format_duration(e, True)} ({format_duration(e - s, True)})",
                                     text_color="#38bdf8")

    def _apply_selection(self, seg_type: SegmentType, verb: str) -> None:
        if self.store.busy or not self.store.videos:
            return
        sel = self._selection()
        if sel is None:
            messagebox.showinfo("구간 선택", "먼저 [🚩 시작점]과 [🏁 끝점]을 찍거나, 타임라인을 우클릭 드래그해 구간을 고르세요.")
            return
        s, e = sel
        if e - s < 0.1:
            return
        self.store.push_undo()
        if apply_global_timeline_override(self.store.videos, s, e, seg_type):
            self.app.set_status(f"{verb}: {format_duration(s, True)} ~ {format_duration(e, True)}")
            self.store.emit("segments")

    def cut_selection(self) -> None:
        self._apply_selection(SegmentType.MANUAL_CUT, "✂ 삭제 구간으로 지정")

    def restore_selection(self) -> None:
        self._apply_selection(SegmentType.NORMAL, "♻ 일반(1x)으로 복원")

    def undo(self) -> None:
        if self.store.busy or not self.store.undo():
            return
        self.app.set_status("↩ 이전 편집을 되돌렸습니다.")
        self.store.emit("segments")

    # ── 자동 구간 분석 ──

    def _sync_config(self) -> None:
        pad = float(self.pad_var.get().split("초")[0])
        sens = int(round(self.sens.get()))
        mind = int(round(self.mindur.get()))
        tag = "낮음" if sens < 35 else ("보통" if sens <= 65 else "높음")
        self.sens_label.configure(text=f"감지 민감도  {tag} ({sens}%)")
        self.mindur_label.configure(text=f"최소 지속 시간  {mind}초 이상만 스킵")
        self.store.analysis_config = AnalysisConfig(
            detect_stops=bool(self.chk_stop.get()), detect_highway_straight=bool(self.chk_straight.get()),
            sensitivity=sens / 100.0, min_skip_duration_sec=float(mind), transition_padding_sec=pad,
            use_gpu_accel=bool(self.gpu_var.get()),
        )

    def _on_detect_setting(self) -> None:
        self._sync_config()
        if self.store.busy or not any(v.feature_cache is not None for v in self.store.videos):
            return
        cfg = self.store.analysis_config
        for v in self.store.videos:
            if v.feature_cache is not None:
                new = reclassify_from_cache(v, cfg)
                if new is not None:
                    v.segments = merge_preserving_manual_cuts(list(v.segments), new, v.duration_sec)
                    v.analyzed = True
        self.store.emit("segments")

    def analyze(self) -> None:
        if self.store.busy:
            return
        if not self.store.videos:
            messagebox.showwarning("알림", "먼저 ① 영상 화면에서 주행 영상을 추가해 주세요.")
            return
        self._sync_config()
        cfg = self.store.analysis_config
        if not cfg.detect_stops and not cfg.detect_highway_straight:
            messagebox.showwarning("설정 확인", "신호대기 감지와 고속도로 직진 감지 중 하나 이상을 선택해 주세요.")
            return
        self.player.stop()
        self.store.push_undo()
        videos = list(self.store.videos)

        def job(report, cancelled):
            total = len(videos)
            for i, v in enumerate(videos):
                if cancelled():
                    raise RuntimeError("취소")

                def prog(frac, i=i, name=v.filename):
                    report((i + frac) / total, f"자동 구간 분석 [{i + 1}/{total}] {name} ({frac * 100:.0f}%)")
                old = list(v.segments)
                segs = analyze_video(v, cfg, progress_callback=prog, cancel_check=cancelled)
                if cancelled():
                    raise RuntimeError("취소")
                v.segments = merge_preserving_manual_cuts(old, segs, v.duration_sec)
                v.analyzed = True
                self.after(0, lambda: self.store.emit("segments"))
            return None

        def done(_r):
            n = len(self.store.review_items)
            self.app.set_status(f"✅ 자동 구간 분석 완료 — 검출 구간 {n}개. 오른쪽 리뷰 목록에서 하나씩 확인하거나 바로 ③ 합치기로 진행하세요.", 1.0)
            self.store.emit("segments")

        self.app.run_job("자동 구간 분석 중…", job, done, on_error=lambda _m: self.store.emit("segments"))

    # ── 리뷰 ──

    def _filtered(self) -> List[int]:
        f = self.review_filter.get()
        out = []
        for i, it in enumerate(self.store.review_items):
            if f == "미검토만" and it.decision is not None:
                continue
            if f == "신호대기만" and it.orig_type != SegmentType.STOP:
                continue
            if f == "단조 주행만" and it.orig_type != SegmentType.STRAIGHT:
                continue
            out.append(i)
        return out

    def _refresh_review(self) -> None:
        tree = self.review_tree
        tree.delete(*tree.get_children())
        items = self.store.review_items
        if self.review_index is not None and self.review_index >= len(items):
            self.review_index = None
        for i in self._filtered():
            it = items[i]
            g = self.store.review_global(it)
            if g is None:
                continue
            tree.insert("", "end", iid=str(i), tags=(it.decision or "pending",), values=(
                "🟠 신호대기" if it.orig_type == SegmentType.STOP else "🟣 단조 주행",
                format_duration(g[0]), f"{it.duration:.0f}초", DECISION_LABELS.get(it.decision, ""),
            ))
        if self.review_index is not None and tree.exists(str(self.review_index)):
            tree.selection_set(str(self.review_index))
            tree.see(str(self.review_index))
        n = len(items)
        if n == 0:
            self.review_progress.configure(text="검출된 구간이 없습니다. 위에서 [자동 구간 분석]을 먼저 실행하세요.")
            self.timeline.set_highlight(None)
        else:
            pending = [it for it in items if it.decision is None]
            cnt = {k: sum(1 for it in items if it.decision == k) for k in ("keep", "normal", "cut")}
            self.review_progress.configure(text=(
                f"검토 {n - len(pending)}/{n} 완료 · 미검토 {len(pending)}개\n"
                f"스킵 {cnt['keep']} · 복원 {cnt['normal']} · 삭제 {cnt['cut']}"))
            if self.review_index is not None:
                self.timeline.set_highlight(self.store.review_global(items[self.review_index]))

    def _on_review_select(self, _e=None) -> None:
        idx = tree_selected_index(self.review_tree)
        if idx is not None and idx != self.review_index:
            self.review_select(idx, autoplay=bool(self.autoplay_var.get()))

    def review_select(self, idx: int, autoplay: bool) -> None:
        items = self.store.review_items
        if not (0 <= idx < len(items)):
            return
        g = self.store.review_global(items[idx])
        if g is None:
            return
        self.review_index = idx
        ctx = max(4.0, (g[1] - g[0]) * 0.25)
        self.timeline.set_view(g[0] - ctx, g[1] + ctx)
        self.timeline.set_highlight(g)
        if autoplay:
            self.player.play_range(max(0.0, g[0] - 2.0), min(self.store.total_duration, g[1] + 2.0))
        else:
            self.player.stop()
            self.player.seek(max(0.0, g[0] - 2.0))

    def review_replay(self) -> None:
        if self.review_index is not None:
            self.review_select(self.review_index, autoplay=True)

    def review_step(self, delta: int) -> None:
        idxs = self._filtered()
        if not idxs:
            return
        if self.review_index in idxs:
            pos = idxs.index(self.review_index) + delta
        else:
            cur = self.review_index if self.review_index is not None else -1
            pos = next((k for k, i in enumerate(idxs) if i > cur), len(idxs) - 1) if delta > 0 else \
                max((k for k, i in enumerate(idxs) if i < cur), default=0)
        pos = max(0, min(len(idxs) - 1, pos))
        self.review_select(idxs[pos], autoplay=bool(self.autoplay_var.get()))
        self._refresh_review()

    def review_decide(self, decision: str) -> None:
        items = self.store.review_items
        if self.store.busy or self.review_index is None or not (0 <= self.review_index < len(items)):
            return
        it = items[self.review_index]
        self.store.push_undo()
        self.player.stop()
        it.video.segments = apply_segment_override(it.video.segments, it.video.duration_sec, it.start_sec, it.end_sec,
                                                   decision_type(it, decision), mark_reviewed=True)
        it.decision = decision
        ref = (it.video, it.start_sec)
        self.store.emit("segments")
        # 재구성된 목록에서 같은 항목을 다시 찾고, 다음 미검토 구간으로 이동
        items = self.store.review_items
        cur = next((i for i, x in enumerate(items) if x.video is ref[0] and abs(x.start_sec - ref[1]) < 0.05), None)
        nxt = next((i for i in self._filtered() if items[i].decision is None and (cur is None or i > cur)), None)
        if nxt is None:
            nxt = next((i for i in self._filtered() if items[i].decision is None), None)
        if nxt is not None:
            self.review_select(nxt, autoplay=bool(self.autoplay_var.get()))
            self._refresh_review()
            self.app.set_status("결정했습니다. 다음 미검토 구간으로 이동합니다.")
        else:
            self.review_index = cur
            self._refresh_review()
            self.app.set_status("🎉 모든 검출 구간을 검토했습니다! ③ 합치기로 진행하세요.")

    def review_boundary(self, which: str) -> None:
        items = self.store.review_items
        if self.store.busy or self.review_index is None or not (0 <= self.review_index < len(items)):
            return
        it = items[self.review_index]
        off = self.store.clip_offset(it.video)
        if off is None:
            return
        v = it.video
        local = self.player.position - off
        if local < -0.05 or local > v.duration_sec + 0.05:
            self.app.set_status("⚠ 경계 조정은 같은 클립 안의 위치에서만 가능합니다.")
            return
        local = max(0.0, min(v.duration_sec, local))
        new_s, new_e = (local, it.end_sec) if which == "start" else (it.start_sec, local)
        if new_e - new_s < 0.5:
            self.app.set_status("⚠ 구간이 너무 짧아집니다 (0.5초 이상).")
            return
        self.store.push_undo()
        decision = it.decision or "keep"
        v.segments = apply_segment_override(v.segments, v.duration_sec, it.start_sec, it.end_sec, SegmentType.NORMAL, mark_reviewed=True)
        v.segments = apply_segment_override(v.segments, v.duration_sec, new_s, new_e, decision_type(it, decision), mark_reviewed=True)
        it.start_sec, it.end_sec, it.decision = new_s, new_e, decision
        self.store.emit("segments")
        self.app.set_status(f"구간 {'시작점' if which == 'start' else '끝점'}을 {format_duration(self.player.position, True)}(으)로 조정했습니다.")

    def review_keep_all(self) -> None:
        pending = [it for it in self.store.review_items if it.decision is None]
        if self.store.busy or not pending:
            return
        if not messagebox.askyesno("일괄 확정", f"미검토 구간 {len(pending)}개를 모두 '스킵 유지'로 확정할까요?"):
            return
        self.store.push_undo()
        for it in pending:
            it.video.segments = apply_segment_override(it.video.segments, it.video.duration_sec, it.start_sec, it.end_sec,
                                                       it.orig_type, mark_reviewed=True)
            it.decision = "keep"
        self.store.emit("segments")
        self.app.set_status(f"✔ 미검토 구간 {len(pending)}개를 스킵 유지로 확정했습니다.")

    # ── 단축키 ──

    def handle_key(self, event) -> None:
        k = event.keysym
        shift = bool(event.state & 0x1)
        ctrl = bool(event.state & 0x4)
        if k == "space":
            self.player.toggle()
        elif k in ("Left", "Right"):
            self.player.step((-1 if k == "Left" else 1) * (5.0 if shift else 1.0))
        elif k in ("i", "I"):
            self.mark_in()
        elif k in ("o", "O"):
            self.mark_out()
        elif k == "Delete":
            self.cut_selection()
        elif ctrl and k in ("z", "Z"):
            self.undo()
        elif k in ("1", "2", "3"):
            self.review_decide({"1": "keep", "2": "normal", "3": "cut"}[k])
        elif k in ("n", "N"):
            self.review_step(1)
        elif k in ("p", "P"):
            self.review_step(-1)
        elif k in ("r", "R"):
            self.review_replay()
        elif k == "bracketleft":
            self.review_boundary("start")
        elif k == "bracketright":
            self.review_boundary("end")
        elif k in ("plus", "equal"):
            self.timeline.zoom_by(0.5)
        elif k == "minus":
            self.timeline.zoom_by(2.0)
        elif k == "0":
            self.timeline.zoom_full()

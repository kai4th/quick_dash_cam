"""
공용 영상 플레이어 위젯: 큰 화면 + (선택) 재생 버튼 줄.

- 여러 영상을 이어 붙인 '전체 타임라인' 시각으로 탐색·재생 (PlaybackWorker로 끊김 없이 스트리밍)
- 멈춘 상태에서는 PreviewFrameWorker로 해당 위치 한 장만 빠르게 디코딩
- frame_transform: 그리기 직전 프레임 변환 (쇼츠 화면의 기울기·소실점 맞춤 미리보기)
- overlay: 프레임 위에 기준선 등 덧그리기, on_click: 화면 클릭 위치 전달 (소실점 지정)
- play_segments: 여러 구간을 차례로 이어서 재생 (쇼츠 미리보기)
"""
from __future__ import annotations

import time
import tkinter as tk
from typing import Callable, List, Optional, Tuple

import customtkinter as ctk
import numpy as np

from .media import PlaybackClip, PlaybackWorker, PlayPiece, PreviewFrameWorker, build_pieces
from .theme import BG_DEEP, FG_MUTED, NEUTRAL, NEUTRAL_HOVER, ACCENT, ACCENT_HOVER, font, format_duration, make_photo_image, photo_put_rgb

RATES = ["0.5x", "1x", "2x", "4x", "8x"]


class VideoPlayer(ctk.CTkFrame):
    def __init__(self, master, show_transport: bool = True, empty_text: str = "영상을 불러오면 여기에 표시됩니다", **kw):
        super().__init__(master, fg_color=BG_DEEP, corner_radius=8, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg=BG_DEEP, highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.canvas.bind("<Configure>", lambda _e: self._on_resize())
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.clips: List[PlaybackClip] = []
        self.total: float = 0.0
        self.position: float = 0.0
        self.playing: bool = False
        self.empty_text = empty_text
        self._stop_at: Optional[float] = None
        self._segments: List[Tuple[float, float]] = []
        self._seg_index: int = -1
        self._session: int = -1
        self._restart_job: Optional[str] = None
        self._resize_job: Optional[str] = None
        self._poll_job: Optional[str] = None
        self._raw: Optional[np.ndarray] = None
        self._photo: Optional[tk.PhotoImage] = None
        self._photo_size: Tuple[int, int] = (0, 0)
        self._img_item: Optional[int] = None
        self.image_geom: Optional[Tuple[int, int, int, int]] = None  # (x0, y0, w, h)
        self._last_label = 0.0

        self.frame_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.overlay: Optional[Callable[[tk.Canvas, int, int, int, int], None]] = None
        self.on_click: Optional[Callable[[float, float, int, int], None]] = None
        self.on_position: Optional[Callable[[float], None]] = None
        self.on_play_state: Optional[Callable[[bool], None]] = None
        self.on_segment_start: Optional[Callable[[int], None]] = None

        self._preview = PreviewFrameWorker(self._on_still_frame)
        self._playback = PlaybackWorker()

        self.time_label: Optional[ctk.CTkLabel] = None
        self.btn_play: Optional[ctk.CTkButton] = None
        self.rate_var = ctk.StringVar(value="1x")
        if show_transport:
            self._build_transport()

    # ── 재생 버튼 줄 ──

    def _build_transport(self) -> None:
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        specs = [("⏮ -5s", -5.0), ("◀ -1s", -1.0), (None, None), ("+1s ▶", 1.0), ("+5s ⏭", 5.0)]
        for txt, d in specs:
            if txt is None:
                self.btn_play = ctk.CTkButton(bar, text="▶ 재생", width=96, height=32, font=font(13, True),
                                              fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.toggle)
                self.btn_play.pack(side="left", padx=3)
            else:
                ctk.CTkButton(bar, text=txt, width=70, height=32, font=font(12), fg_color=NEUTRAL,
                              hover_color=NEUTRAL_HOVER, command=lambda d=d: self.step(d)).pack(side="left", padx=2)
        ctk.CTkOptionMenu(bar, values=RATES, variable=self.rate_var, width=70, height=32,
                          command=lambda _v: self._restart_if_playing()).pack(side="left", padx=(8, 2))
        self.time_label = ctk.CTkLabel(bar, text="00:00.0 / 00:00.0", font=font(13, True), text_color="#38bdf8")
        self.time_label.pack(side="right", padx=6)

    # ── 소스 ──

    def set_clips(self, clips: List[PlaybackClip]) -> None:
        self.stop()
        self.clips = list(clips)
        self.total = sum(c.duration for c in self.clips)
        self.position = min(self.position, self.total)
        self._raw = None
        self.request_frame()

    def clip_at(self, t: float) -> Optional[Tuple[PlaybackClip, float]]:
        if not self.clips:
            return None
        for i, c in enumerate(self.clips):
            if t < c.global_start + c.duration or i == len(self.clips) - 1:
                return c, max(0.0, min(c.duration, t - c.global_start))
        return None

    # ── 탐색 ──

    def seek(self, t: float) -> None:
        self.position = max(0.0, min(self.total, t))
        if self._segments:  # 컷 이어보기 중 다른 곳으로 이동하면 일반 재생으로 전환
            self._segments = []
            self._stop_at = None
        if self.playing:
            self._playback.stop()
            self._schedule_restart()
        else:
            self.request_frame()
        self._update_time_label(force=True)
        if self.on_position:
            self.on_position(self.position)

    def step(self, delta: float) -> None:
        self._segments = []
        self.seek(self.position + delta)

    def request_frame(self) -> None:
        if self.playing or not self.clips:
            if not self.clips:
                self._raw = None
                self.redraw()
            return
        found = self.clip_at(self.position)
        if found is None:
            return
        clip, local = found
        self._preview.request_frame(clip.path, local, max(160, self.canvas.winfo_width()),
                                    max(90, self.canvas.winfo_height()))

    def _on_still_frame(self, arr: np.ndarray, _req_id: int) -> None:
        def apply():
            if not self.playing:
                self._raw = arr
                self.redraw()
        try:
            self.after(0, apply)
        except Exception:
            pass

    # ── 재생 ──

    def rate(self) -> float:
        try:
            return float(self.rate_var.get().rstrip("x"))
        except ValueError:
            return 1.0

    def toggle(self) -> None:
        if self.playing:
            self.stop()
        else:
            if self.position >= self.total - 0.1:
                self.position = 0.0
            self.play()

    def play(self, stop_at: Optional[float] = None) -> None:
        if not self.clips:
            return
        self._stop_at = stop_at
        self.playing = True
        self._set_play_button()
        self._launch()
        self._start_poll()

    def play_range(self, start: float, end: float) -> None:
        self._segments = []
        self.stop()
        self.position = max(0.0, min(self.total, start))
        self.play(stop_at=end)

    def play_segments(self, segments: List[Tuple[float, float]]) -> None:
        """여러 구간을 차례로 이어서 재생 (쇼츠 미리보기). 다음 컷은 재생 워커가 미리 열어 두어 바로 이어집니다."""
        self.stop()
        segs = [(max(0.0, a), min(self.total, b)) for a, b in segments if b - a > 0.05]
        if not segs or not self.clips:
            return
        self._segments = segs
        self._seg_index = 0
        self.position, self._stop_at = segs[0]
        if self.on_segment_start:
            self.on_segment_start(0)
        self.playing = True
        self._set_play_button()
        self._launch()
        self._start_poll()

    def stop(self) -> None:
        self._segments = []
        if self._restart_job is not None:
            try:
                self.after_cancel(self._restart_job)
            except Exception:
                pass
            self._restart_job = None
        if self.playing:
            self.playing = False
            self._playback.stop()
            self._set_play_button()
            self.request_frame()

    def _pieces(self) -> List[PlayPiece]:
        if not self._segments:
            return build_pieces(self.clips, self.position, self._stop_at)
        pieces: List[PlayPiece] = []
        for k in range(max(0, self._seg_index), len(self._segments)):
            g0, g1 = self._segments[k]
            if k == self._seg_index:
                g0 = max(g0, min(self.position, g1 - 0.05))
            pieces += build_pieces(self.clips, g0, g1, tag=k)
        return pieces

    def _launch(self) -> None:
        self._session = self._playback.start(
            self._pieces(), rate=self.rate(),
            box_w=max(160, self.canvas.winfo_width()), box_h=max(90, self.canvas.winfo_height()),
        )

    def _schedule_restart(self) -> None:
        if self._restart_job is not None:
            try:
                self.after_cancel(self._restart_job)
            except Exception:
                pass

        def restart():
            self._restart_job = None
            if self.playing:
                if self._stop_at is not None and self.position >= self._stop_at:
                    self._stop_at = None
                self._launch()
        self._restart_job = self.after(250, restart)

    def _restart_if_playing(self) -> None:
        if self.playing:
            self._playback.stop()
            self._schedule_restart()

    def _start_poll(self) -> None:
        """화면 갱신 루프는 항상 하나만 돌도록 (겹치면 그리기가 두 배로 늘어 끊김)."""
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except Exception:
                pass
        self._poll_job = self.after(1, self._poll)

    def _poll(self) -> None:
        self._poll_job = None
        if not self.playing:
            return
        item = self._playback.take_latest()
        if item is not None:
            sid, media_t, arr, tag = item
            if sid == self._session:
                if self._segments and tag != self._seg_index and 0 <= tag < len(self._segments):
                    self._seg_index = tag
                    self._stop_at = self._segments[tag][1]
                    if self.on_segment_start:
                        self.on_segment_start(tag)
                self.position = media_t
                self._raw = arr
                self.redraw()
                self._update_time_label()
                if self.on_position:
                    self.on_position(media_t)
        if self._playback.is_finished(self._session) and self._restart_job is None \
                and not self._playback.has_pending():
            if self._stop_at is not None:
                self.position = min(self._stop_at, self.total)
            self._segments = []
            self.playing = False
            self._set_play_button()
            self.request_frame()
            if self.on_position:
                self.on_position(self.position)
            return
        self._poll_job = self.after(4, self._poll)

    def _set_play_button(self) -> None:
        if self.btn_play is not None:
            if self.playing:
                self.btn_play.configure(text="⏸ 일시정지", fg_color="#d97706")
            else:
                self.btn_play.configure(text="▶ 재생", fg_color=ACCENT)
        if self.on_play_state:
            self.on_play_state(self.playing)

    def _update_time_label(self, force: bool = False) -> None:
        if self.time_label is None:
            return
        now = time.perf_counter()
        if not force and now - self._last_label < 0.1:
            return
        self._last_label = now
        self.time_label.configure(text=f"{format_duration(self.position, True)} / {format_duration(self.total, True)}")

    # ── 그리기 ──

    def redraw(self) -> None:
        cv = self.canvas
        cw, ch = max(10, cv.winfo_width()), max(10, cv.winfo_height())
        if self._raw is None:
            cv.delete("all")
            self._img_item = None
            self.image_geom = None
            cv.create_text(cw // 2, ch // 2, text=self.empty_text, fill=FG_MUTED, font=("Malgun Gothic", 13),
                           justify="center")
            return
        arr = self._raw
        if self.frame_transform is not None:
            try:
                arr = self.frame_transform(arr)
            except Exception:
                arr = self._raw
        h, w = arr.shape[:2]
        # 영상 그림 하나를 계속 재사용 (매 프레임 새로 만들면 그리기 비용이 약 두 배)
        if self._img_item is not None:
            cv.addtag_all("old")
            cv.dtag(self._img_item, "old")
            cv.delete("old")
        else:
            cv.delete("all")
        if self._photo is not None and self._photo_size == (w, h) and self._img_item is not None:
            photo_put_rgb(self._photo, arr)
            cv.coords(self._img_item, cw // 2, ch // 2)
        else:
            if self._img_item is not None:
                cv.delete(self._img_item)
            self._photo = make_photo_image(arr)
            self._photo_size = (w, h)
            self._img_item = cv.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")
        x0, y0 = cw // 2 - w // 2, ch // 2 - h // 2
        self.image_geom = (x0, y0, w, h)
        if self.overlay is not None:
            try:
                self.overlay(cv, x0, y0, w, h)
            except Exception:
                import traceback
                traceback.print_exc()
        if self.playing and abs(self.rate() - 1.0) > 0.01:
            cv.create_text(cw - 10, 12, text=f"▶ {self.rate():g}x", anchor="ne", fill="#facc15",
                           font=("Consolas", 12, "bold"))

    def raw_frame(self) -> Optional[np.ndarray]:
        return self._raw

    def _on_resize(self) -> None:
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.after(150, self._after_resize)

    def _after_resize(self) -> None:
        self._resize_job = None
        if self.playing:
            self._restart_if_playing()
        else:
            self.request_frame()
            self.redraw()

    def _on_canvas_click(self, event) -> None:
        if self.on_click is None or self.image_geom is None:
            return
        x0, y0, w, h = self.image_geom
        qx, qy = event.x - x0, event.y - y0
        if 0 <= qx < w and 0 <= qy < h:
            self.on_click(qx, qy, w, h)

    def close(self) -> None:
        self.playing = False
        for job in (self._poll_job, self._restart_job, self._resize_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        self._playback.stop()
        self._preview.close()

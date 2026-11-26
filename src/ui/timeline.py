"""전체 연결 타임라인 위젯 (확대/이동, 미니맵, 구간 색, 재생 위치, 구간 선택)."""
from __future__ import annotations

import tkinter as tk
from typing import List, Optional, Tuple

from ..models import Segment, SegmentType, VideoFileInfo
from .theme import (
    COLOR_HIGHLIGHT,
    COLOR_MANUAL_CUT,
    COLOR_NORMAL,
    COLOR_STOP,
    COLOR_STRAIGHT,
    COLOR_UNANALYZED,
    format_duration,
)


class GlobalTimelineCanvas(tk.Canvas):
    """
    모든 영상 클립을 시간순으로 이어붙인 대화형 전체 타임라인 위젯.
    - 좌클릭 / 좌클릭 드래그: 현재 재생 위치(Playhead) 탐색
    - 우클릭 드래그 (또는 Shift + 좌클릭 드래그): 잘라낼 [시작점(In) ~ 끝점(Out)] 구간 직접 드래그 선택
    - 마우스 휠: 커서 위치 기준 확대/축소 / Shift + 휠 또는 휠 버튼 드래그: 좌우 이동
    - 상단 미니맵: 전체 타임라인과 현재 확대 보기 영역 표시 (클릭 시 해당 위치로 이동)
    """

    MIN_VIEW_SPAN = 4.0
    MINI_Y1 = 2
    MINI_Y2 = 10
    RULER_H = 28

    def __init__(
        self,
        master,
        on_seek,
        on_range_select,
        on_view_change=None,
        height: int = 76,
        **kwargs,
    ):
        super().__init__(
            master,
            height=height,
            bg="#0f172a",
            highlightthickness=1,
            highlightbackground="#334155",
            bd=0,
            cursor="hand2",
            **kwargs,
        )
        self._on_seek = on_seek
        self._on_range_select = on_range_select
        self._on_view_change = on_view_change
        self._videos: List[VideoFileInfo] = []
        self._total_duration: float = 0.0
        self._playhead_sec: float = 0.0
        self._in_sec: Optional[float] = None
        self._out_sec: Optional[float] = None
        self._drag_mode: Optional[str] = None
        self._range_anchor_sec: float = 0.0
        self._view_start: float = 0.0
        self._view_end: Optional[float] = None  # None = 전체 보기
        self._highlight: Optional[Tuple[float, float]] = None
        self._marks: List[Tuple[float, float]] = []
        self._pan_anchor: Optional[Tuple[int, float, float]] = None

        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self._on_left_down)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<ButtonPress-3>", self._on_right_down)
        self.bind("<B3-Motion>", self._on_right_drag)
        self.bind("<ButtonRelease-3>", self._on_mouse_up)
        self.bind("<ButtonPress-2>", self._on_middle_down)
        self.bind("<B2-Motion>", self._on_middle_drag)
        self.bind("<ButtonRelease-2>", self._on_mouse_up)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", lambda e: self._on_wheel_button(e, 1.0))
        self.bind("<Button-5>", lambda e: self._on_wheel_button(e, -1.0))

    # ── 상태 / 보기 영역 ──

    def set_state(
        self,
        videos: List[VideoFileInfo],
        playhead_sec: float,
        in_sec: Optional[float],
        out_sec: Optional[float],
    ) -> None:
        self._videos = videos
        self._total_duration = sum(v.duration_sec for v in videos)
        self._playhead_sec = max(0.0, min(self._total_duration, playhead_sec))
        self._in_sec = in_sec
        self._out_sec = out_sec
        self._clamp_view()
        self.redraw()

    def set_highlight(self, rng: Optional[Tuple[float, float]]) -> None:
        self._highlight = rng
        self.redraw()

    def set_marks(self, ranges: List[Tuple[float, float]]) -> None:
        """하이라이트 쇼츠 장면 위치를 타임라인 바 위쪽에 하늘색 표식으로 표시합니다."""
        self._marks = list(ranges)
        self.redraw()

    @property
    def view_range(self) -> Tuple[float, float]:
        total = max(0.001, self._total_duration)
        if self._view_end is None:
            return 0.0, total
        return self._view_start, self._view_end

    @property
    def is_zoomed(self) -> bool:
        vs, ve = self.view_range
        return (ve - vs) < self._total_duration - 0.01

    def _clamp_view(self) -> None:
        total = self._total_duration
        if self._view_end is None or total <= 0:
            self._view_start, self._view_end = 0.0, None
            return
        span = self._view_end - self._view_start
        span = max(min(self.MIN_VIEW_SPAN, total), min(total, span))
        if span >= total - 0.01:
            self._view_start, self._view_end = 0.0, None
            return
        s = max(0.0, min(total - span, self._view_start))
        self._view_start, self._view_end = s, s + span

    def _notify_view(self) -> None:
        if self._on_view_change:
            vs, ve = self.view_range
            self._on_view_change(vs, ve, self.is_zoomed)

    def set_view(self, start: float, end: float) -> None:
        self._view_start = min(start, end)
        self._view_end = max(start, end)
        self._clamp_view()
        self.redraw()
        self._notify_view()

    def zoom_full(self) -> None:
        self._view_start, self._view_end = 0.0, None
        self.redraw()
        self._notify_view()

    def zoom_by(self, factor: float, center_sec: Optional[float] = None) -> None:
        if self._total_duration <= 0:
            return
        vs, ve = self.view_range
        c = self._playhead_sec if center_sec is None else center_sec
        c = max(vs, min(ve, c)) if center_sec is not None else c
        frac = (c - vs) / max(1e-6, ve - vs)
        new_span = (ve - vs) * factor
        new_s = c - frac * new_span
        self.set_view(new_s, new_s + new_span)

    def pan_by(self, delta_sec: float) -> None:
        if not self.is_zoomed:
            return
        vs, ve = self.view_range
        self.set_view(vs + delta_sec, ve + delta_sec)

    def center_on(self, sec: float) -> None:
        if not self.is_zoomed:
            return
        vs, ve = self.view_range
        span = ve - vs
        self.set_view(sec - span / 2, sec + span / 2)

    def set_playhead(self, sec: float, follow: bool = True) -> None:
        """재생 중 호출되는 가벼운 Playhead 갱신 (전체 다시 그리기 최소화)."""
        self._playhead_sec = max(0.0, min(self._total_duration, sec))
        vs, ve = self.view_range
        if follow and self.is_zoomed and not (vs <= self._playhead_sec <= ve):
            span = ve - vs
            self.set_view(self._playhead_sec - span * 0.1, self._playhead_sec + span * 0.9)
            return
        px = self._sec_to_x(self._playhead_sec)
        rh = self.RULER_H
        h = max(30, self.winfo_height())
        self.coords("playhead_line", px, rh - 6, px, h)
        self.coords("playhead_tri", px - 6, rh - 6, px + 6, rh - 6, px, rh + 2)
        mx = self._mini_x(self._playhead_sec)
        self.coords("playhead_mini", mx, self.MINI_Y1 - 1, mx, self.MINI_Y2 + 1)

    # ── 좌표 변환 ──

    def _x_to_sec(self, x: int) -> float:
        w = max(1, self.winfo_width())
        vs, ve = self.view_range
        frac = max(0.0, min(1.0, x / w))
        return max(0.0, min(self._total_duration, vs + frac * (ve - vs)))

    def _sec_to_x(self, sec: float) -> float:
        if self._total_duration <= 0:
            return 0
        w = max(1, self.winfo_width())
        vs, ve = self.view_range
        return (sec - vs) / max(1e-6, ve - vs) * w

    def _mini_x(self, sec: float) -> float:
        w = max(1, self.winfo_width())
        return (sec / max(1e-6, self._total_duration)) * w

    # ── 마우스 이벤트 ──

    def _on_wheel(self, event) -> None:
        if self._total_duration <= 0:
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        # Windows: +-120 단위, macOS: +-1~5 단위
        notches = delta / 120.0 if abs(delta) >= 10 else float(delta)
        self._apply_wheel_scroll(event, notches)

    def _on_wheel_button(self, event, direction: float) -> None:
        if self._total_duration <= 0:
            return
        self._apply_wheel_scroll(event, direction)

    def _apply_wheel_scroll(self, event, notches: float) -> None:
        if event.state & 0x0001:  # Shift + 휠: 좌우 이동
            vs, ve = self.view_range
            self.pan_by(-notches * (ve - vs) * 0.15)
            return
        factor = 0.8 ** notches
        self.zoom_by(factor, center_sec=self._x_to_sec(event.x))

    def _on_middle_down(self, event) -> None:
        vs, ve = self.view_range
        self._pan_anchor = (event.x, vs, ve)
        self._drag_mode = "pan"

    def _on_middle_drag(self, event) -> None:
        if self._drag_mode != "pan" or self._pan_anchor is None:
            return
        x0, vs, ve = self._pan_anchor
        w = max(1, self.winfo_width())
        d = -(event.x - x0) / w * (ve - vs)
        if self.is_zoomed:
            self.set_view(vs + d, ve + d)

    def _on_left_down(self, event) -> None:
        if self._total_duration <= 0:
            return
        # 상단 미니맵 클릭: 해당 위치로 보기 영역 이동 + 탐색
        if event.y <= self.MINI_Y2 + 2:
            sec = (event.x / max(1, self.winfo_width())) * self._total_duration
            self._drag_mode = "mini"
            self.center_on(sec)
            self._on_seek(sec)
            return

        sec = self._x_to_sec(event.x)
        # Shift + 좌클릭이면 구간 선택 시작
        if event.state & 0x0001:
            self._drag_mode = "range"
            self._range_anchor_sec = sec
            self._on_range_select(sec, sec)
            self._on_seek(sec)
            return

        # In 또는 Out 핸들 근처(7px 이내) 클릭 시 핸들 드래그
        if self._in_sec is not None and abs(event.x - self._sec_to_x(self._in_sec)) <= 7:
            self._drag_mode = "in_handle"
            return
        if self._out_sec is not None and abs(event.x - self._sec_to_x(self._out_sec)) <= 7:
            self._drag_mode = "out_handle"
            return

        self._drag_mode = "seek"
        self._on_seek(sec)

    def _on_left_drag(self, event) -> None:
        if self._total_duration <= 0 or not self._drag_mode:
            return
        if self._drag_mode == "mini":
            sec = max(0.0, min(1.0, event.x / max(1, self.winfo_width()))) * self._total_duration
            self.center_on(sec)
            self._on_seek(sec)
            return
        sec = self._x_to_sec(event.x)
        if self._drag_mode == "seek":
            self._on_seek(sec)
        elif self._drag_mode == "range":
            s = min(self._range_anchor_sec, sec)
            e = max(self._range_anchor_sec, sec)
            self._on_range_select(s, e)
            self._on_seek(sec)
        elif self._drag_mode == "in_handle":
            other = self._out_sec if self._out_sec is not None else self._total_duration
            self._on_range_select(min(sec, other), max(sec, other))
            self._on_seek(sec)
        elif self._drag_mode == "out_handle":
            other = self._in_sec if self._in_sec is not None else 0.0
            self._on_range_select(min(other, sec), max(other, sec))
            self._on_seek(sec)

    def _on_right_down(self, event) -> None:
        if self._total_duration <= 0:
            return
        sec = self._x_to_sec(event.x)
        self._drag_mode = "range"
        self._range_anchor_sec = sec
        self._on_range_select(sec, sec)
        self._on_seek(sec)

    def _on_right_drag(self, event) -> None:
        if self._total_duration <= 0 or self._drag_mode != "range":
            return
        sec = self._x_to_sec(event.x)
        s = min(self._range_anchor_sec, sec)
        e = max(self._range_anchor_sec, sec)
        self._on_range_select(s, e)
        self._on_seek(sec)

    def _on_mouse_up(self, _event) -> None:
        self._drag_mode = None
        self._pan_anchor = None

    # ── 그리기 ──

    @staticmethod
    def _seg_color(seg_type: SegmentType) -> str:
        return {
            SegmentType.NORMAL: COLOR_NORMAL,
            SegmentType.STOP: COLOR_STOP,
            SegmentType.STRAIGHT: COLOR_STRAIGHT,
            SegmentType.MANUAL_CUT: COLOR_MANUAL_CUT,
        }.get(seg_type, COLOR_NORMAL)

    def redraw(self) -> None:
        self.delete("all")
        w = max(10, self.winfo_width())
        h = max(30, self.winfo_height())

        ruler_h = self.RULER_H
        bar_y1 = ruler_h
        bar_y2 = h - 4

        if not self._videos or self._total_duration <= 0:
            self.create_rectangle(0, bar_y1, w, bar_y2, fill="#1e293b", outline="")
            self.create_text(
                w // 2,
                (bar_y1 + bar_y2) // 2,
                text="영상을 추가하면 전체 타임라인 바가 여기에 표시됩니다 (좌클릭: 위치 탐색 / 우클릭 드래그: 잘라낼 구간 지정 / 휠: 확대·축소)",
                fill="#64748b",
                font=("Malgun Gothic", 9),
            )
            return

        vs, ve = self.view_range
        span = max(1e-6, ve - vs)

        # 0. 상단 미니맵 (전체 타임라인 + 현재 보기 영역)
        cursor = 0.0
        for v in self._videos:
            segs = v.segments if v.segments else [Segment(0.0, v.duration_sec, SegmentType.NORMAL)]
            for seg in segs:
                x1 = self._mini_x(cursor + seg.start_sec)
                x2 = max(x1 + 1, self._mini_x(cursor + seg.end_sec))
                self.create_rectangle(
                    x1, self.MINI_Y1, x2, self.MINI_Y2, fill=self._seg_color(seg.segment_type), outline=""
                )
            cursor += v.duration_sec
        if self.is_zoomed:
            self.create_rectangle(
                self._mini_x(vs), self.MINI_Y1 - 1, self._mini_x(ve), self.MINI_Y2 + 1,
                outline="#ffffff", width=1,
            )
        mx = self._mini_x(self._playhead_sec)
        self.create_line(mx, self.MINI_Y1 - 1, mx, self.MINI_Y2 + 1, fill="#ffffff", width=1, tags="playhead_mini")

        # 1. 시간 눈금(Ruler) 그리기: 확대 배율에 맞춰 눈금 간격 자동 조정
        steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200]
        min_px = 80.0
        step = steps[-1]
        for st in steps:
            if st / span * w >= min_px:
                step = st
                break
        t = (int(vs // step)) * step
        while t <= ve + 1e-6:
            if t >= vs - 1e-6:
                tx = self._sec_to_x(t)
                self.create_line(tx, ruler_h - 5, tx, ruler_h, fill="#475569")
                anchor = "nw" if tx < 20 else ("ne" if tx > w - 20 else "n")
                self.create_text(
                    tx, self.MINI_Y2 + 3, text=format_duration(t), anchor=anchor,
                    fill="#94a3b8", font=("Consolas", 8),
                )
            t += step

        # 2. 각 영상의 구간(Segment) 컬러 블록 및 클립 경계선 그리기
        cursor = 0.0
        for idx, v in enumerate(self._videos):
            c_start = cursor
            c_end = cursor + v.duration_sec
            cursor = c_end
            if c_end < vs or c_start > ve:
                continue

            segs = v.segments if v.segments else [Segment(0.0, v.duration_sec, SegmentType.NORMAL)]
            has_meaningful = v.analyzed or v.has_manual_cuts or len(segs) > 1

            for seg in segs:
                g_s = c_start + seg.start_sec
                g_e = c_start + seg.end_sec
                if g_e < vs or g_s > ve:
                    continue
                x1 = max(-2.0, self._sec_to_x(g_s))
                x2 = min(w + 2.0, max(x1 + 1, self._sec_to_x(g_e)))
                if not has_meaningful and seg.segment_type == SegmentType.NORMAL:
                    color = "#16a34a"
                else:
                    color = self._seg_color(seg.segment_type)
                self.create_rectangle(x1, bar_y1, x2, bar_y2, fill=color, outline="")

                # 수동 잘라내기(MANUAL_CUT) 구간에는 빗금 표시로 시각적 구분 강화
                if seg.segment_type == SegmentType.MANUAL_CUT and (x2 - x1) >= 6:
                    for sx in range(int(x1), int(x2), 8):
                        self.create_line(
                            sx, bar_y2, min(x2, sx + 6), bar_y1, fill="#7f1d1d", width=1
                        )
                # 리뷰에서 '스킵 유지'로 확정된 정차/직진 구간은 하단에 흰색 확인선 표시
                if seg.reviewed and seg.segment_type in (SegmentType.STOP, SegmentType.STRAIGHT):
                    self.create_line(x1, bar_y2 - 2, x2, bar_y2 - 2, fill="#f8fafc", width=2)

            # 클립 경계선 및 클립 번호 표시
            bx = self._sec_to_x(c_start)
            if idx > 0 and 0 <= bx <= w:
                self.create_line(bx, bar_y1, bx, bar_y2, fill="#0f172a", width=2)
            label_x = max(0.0, bx)
            if min(w, self._sec_to_x(c_end)) - label_x >= 36:
                self.create_text(
                    label_x + 5,
                    bar_y1 + 3,
                    text=f"#{idx + 1}",
                    anchor="nw",
                    fill="#0f172a",
                    font=("Consolas", 8, "bold"),
                )

        # 2-A. 하이라이트 쇼츠 장면 표식 (바 상단 하늘색 띠 + 미니맵 점)
        for ms, me in self._marks:
            mx1 = self._mini_x(ms)
            self.create_rectangle(
                mx1, self.MINI_Y1, max(mx1 + 2, self._mini_x(me)), self.MINI_Y2, fill=COLOR_HIGHLIGHT, outline=""
            )
            if me < vs or ms > ve:
                continue
            hx1 = max(-2.0, self._sec_to_x(ms))
            hx2 = min(w + 2.0, max(hx1 + 3, self._sec_to_x(me)))
            self.create_rectangle(hx1, bar_y1, hx2, bar_y1 + 6, fill=COLOR_HIGHLIGHT, outline="#0e7490")

        # 2-B. 현재 리뷰 중인 구간 강조 (노란 테두리)
        if self._highlight is not None:
            hs, he = self._highlight
            hx1 = max(-3.0, self._sec_to_x(hs))
            hx2 = min(w + 3.0, self._sec_to_x(he))
            if hx2 >= 0 and hx1 <= w:
                self.create_rectangle(
                    hx1, bar_y1 - 1, max(hx1 + 2, hx2), bar_y2 + 1, outline="#facc15", width=3
                )

        # 3. 선택된 [시작점(In) ~ 끝점(Out)] 구간 브래킷 표시
        if self._in_sec is not None or self._out_sec is not None:
            s_sec = self._in_sec if self._in_sec is not None else 0.0
            e_sec = self._out_sec if self._out_sec is not None else self._total_duration
            if e_sec < s_sec:
                s_sec, e_sec = e_sec, s_sec
            ix1 = self._sec_to_x(s_sec)
            ix2 = self._sec_to_x(e_sec)

            # 선택 영역 테두리 박스
            self.create_rectangle(
                ix1,
                bar_y1 + 1,
                ix2,
                bar_y2 - 1,
                outline="#38bdf8",
                width=2,
                stipple="gray25",
                fill="#38bdf8",
            )
            # In 마커 핸들
            if self._in_sec is not None:
                self.create_line(ix1, bar_y1 - 2, ix1, bar_y2 + 2, fill="#38bdf8", width=3)
                self.create_polygon(
                    ix1, bar_y1, ix1 + 8, bar_y1, ix1, bar_y1 + 8, fill="#38bdf8", outline=""
                )
            # Out 마커 핸들
            if self._out_sec is not None:
                self.create_line(ix2, bar_y1 - 2, ix2, bar_y2 + 2, fill="#38bdf8", width=3)
                self.create_polygon(
                    ix2, bar_y2, ix2 - 8, bar_y2, ix2, bar_y2 - 8, fill="#38bdf8", outline=""
                )

        # 4. 현재 재생 위치(Playhead) 포인터 표시
        px = self._sec_to_x(self._playhead_sec)
        self.create_line(px, ruler_h - 6, px, h, fill="#ffffff", width=2, tags="playhead_line")
        self.create_polygon(
            px - 6,
            ruler_h - 6,
            px + 6,
            ruler_h - 6,
            px,
            ruler_h + 2,
            fill="#ffffff",
            outline="#0f172a",
            tags="playhead_tri",
        )

from __future__ import annotations

import copy
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np

from .analyzer import analyze_video, reclassify_from_cache
from .ffmpeg_utils import (
    VIDEO_EXTENSIONS,
    detect_best_gpu_decoder,
    detect_best_h264_encoder,
    get_CreationFlags,
    get_ffmpeg_exe,
    probe_video_file,
)
from .models import (
    AnalysisConfig,
    AudioMode,
    ExportConfig,
    Segment,
    SegmentType,
    SkipMode,
    SortMode,
    VideoCodec,
    VideoFileInfo,
    VideoQualityPreset,
    apply_global_timeline_override,
    apply_segment_override,
    map_global_time_to_clip,
    merge_preserving_manual_cuts,
)
from .processor import render_merged_video
from .sorter import scan_folder_for_videos, sort_video_files

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

COLOR_NORMAL = "#22c55e"      # 초록: 일반 주행 (1x)
COLOR_STOP = "#f59e0b"        # 주황: 신호대기 / 정차
COLOR_STRAIGHT = "#a855f7"    # 보라: 장시간 고속도로 직진
COLOR_MANUAL_CUT = "#ef4444"  # 빨강: 수동 잘라내기 (삭제 구간)
COLOR_UNANALYZED = "#475569"


def format_duration(seconds: float, include_tenths: bool = False) -> str:
    """초 단위 시간을 MM:SS 또는 HH:MM:SS (옵션: .d) 문자열로 변환합니다."""
    seconds = max(0.0, float(seconds))
    total_int = int(seconds)
    tenths = int((seconds - total_int) * 10)
    h = total_int // 3600
    m = (total_int % 3600) // 60
    s = total_int % 60
    if h > 0:
        base = f"{h:d}:{m:02d}:{s:02d}"
    else:
        base = f"{m:02d}:{s:02d}"
    return f"{base}.{tenths}" if include_tenths else base


def make_photo_image(img_or_arr: Union[np.ndarray, object]) -> tk.PhotoImage:
    """
    RGB uint8 NumPy 배열((H, W, 3) 또는 (H, W))을 Tkinter 네이티브 PhotoImage로 고속 변환합니다.
    외부 C 확장 DLL(_imagingtk.pyd 등)에 의존하지 않고 Netpbm(P6/P5) 바이너리 포맷을
    Tcl/Tk 코어에 직접 전달하므로, Windows Application Control(WDAC, Smart App Control)이나
    파이썬 버전(3.14+)에 구애받지 않고 100% 안전하고 빠르게 동작합니다.
    """
    if not isinstance(img_or_arr, np.ndarray):
        arr = np.asarray(img_or_arr)
    else:
        arr = img_or_arr

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    if arr.ndim == 2:
        h, w = arr.shape
        ppm = f"P5 {w} {h} 255\n".encode("ascii") + arr.tobytes()
    else:
        h, w = arr.shape[:2]
        ppm = f"P6 {w} {h} 255\n".encode("ascii") + arr.tobytes()
    return tk.PhotoImage(data=ppm)


class PreviewFrameWorker:
    """
    마우스 드래그로 전체 타임라인을 빠르게 탐색(Scrubbing)할 때
    UI 멈춤 없이 가장 최신 위치의 프레임만 비동기로 디코딩하여 전달하는 워커.
    """

    def __init__(self, on_frame_ready):
        self._on_frame_ready = on_frame_ready
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest_req: Optional[Tuple[Path, float, int, int, int]] = None
        self._req_id = 0
        self._cap_path: Optional[Path] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_frame(self, video_path: Path, local_sec: float, max_w: int, max_h: int) -> None:
        with self._lock:
            self._req_id += 1
            self._latest_req = (video_path, max(0.0, local_sec), max(64, max_w), max(64, max_h), self._req_id)
            self._event.set()

    def close(self) -> None:
        self._stop = True
        self._event.set()

    def _run(self) -> None:
        while not self._stop:
            self._event.wait(timeout=0.2)
            self._event.clear()
            if self._stop:
                break

            with self._lock:
                req = self._latest_req
                self._latest_req = None

            if req is None:
                continue

            v_path, local_sec, max_w, max_h, req_id = req
            bgr: Optional[np.ndarray] = None
            try:
                if self._cap is None or self._cap_path != v_path:
                    if self._cap is not None:
                        self._cap.release()
                    self._cap = cv2.VideoCapture(str(v_path))
                    self._cap_path = v_path

                if self._cap and self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_POS_MSEC, local_sec * 1000.0)
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        self._cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, local_sec - 0.15) * 1000.0)
                        ret, frame = self._cap.read()
                    if ret and frame is not None:
                        bgr = frame
            except Exception:
                bgr = None

            # OpenCV 디코딩 실패 시 FFmpeg 1프레임 즉시 추출 폴백
            if bgr is None:
                try:
                    ff = get_ffmpeg_exe()
                    cmd = [
                        ff,
                        "-hide_banner",
                        "-loglevel", "error",
                        "-nostdin",
                        "-ss", f"{max(0.0, local_sec):.3f}",
                        "-i", str(v_path),
                        "-frames:v", "1",
                        "-vf", f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease",
                        "-f", "image2pipe",
                        "-vcodec", "png",
                        "-",
                    ]
                    res = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        timeout=2.0,
                        creationflags=get_CreationFlags(),
                    )
                    if res.returncode == 0 and res.stdout:
                        nparr = np.frombuffer(res.stdout, np.uint8)
                        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    bgr = None

            if bgr is None:
                continue

            try:
                h, w = bgr.shape[:2]
                scale = min(max_w / max(1, w), max_h / max(1, h))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                self._on_frame_ready(rgb, req_id)
            except Exception:
                continue


@dataclass
class PlaybackClip:
    path: Path
    global_start: float
    duration: float
    width: int
    height: int


class PlaybackWorker:
    """
    실제 동영상 플레이어처럼 끊김 없이 재생하기 위한 스트리밍 재생 워커.
    - 매 프레임마다 탐색(seek)하던 기존 방식 대신, FFmpeg(GPU NVDEC + scale_cuda 우선)로
      재생 위치부터 미리보기 해상도 프레임을 연속 디코딩하여 실제 시간(벽시계)에 맞춰 내보냅니다.
    - 다음 클립 디코더를 미리 띄워두어 클립 경계에서도 멈춤 없이 이어서 재생합니다.
    - 디코딩이 배속을 따라가지 못하면 프레임을 건너뛰며(드롭) 재생 시간을 맞춥니다.
    """

    DISPLAY_FPS = 30.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session = 0
        self._stop_evt: Optional[threading.Event] = None
        self._latest: Optional[Tuple[int, float, np.ndarray]] = None
        self._finished_session: int = -1
        self._use_cuda = detect_best_gpu_decoder() == "cuda_scale"

    @property
    def session(self) -> int:
        return self._session

    def start(
        self,
        clips: List[PlaybackClip],
        global_start: float,
        stop_at: Optional[float],
        rate: float,
        box_w: int,
        box_h: int,
    ) -> int:
        self.stop()
        with self._lock:
            self._session += 1
            sid = self._session
            self._latest = None
        evt = threading.Event()
        self._stop_evt = evt
        threading.Thread(
            target=self._run,
            args=(sid, evt, clips, global_start, stop_at, max(0.25, rate), box_w, box_h),
            daemon=True,
        ).start()
        return sid

    def stop(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()
            self._stop_evt = None

    def take_latest(self) -> Optional[Tuple[int, float, np.ndarray]]:
        with self._lock:
            item = self._latest
            self._latest = None
        return item

    def is_finished(self, sid: int) -> bool:
        return self._finished_session == sid

    def _spawn(self, clip: PlaybackClip, local_start: float, out_fps: float, rate: float,
               box_w: int, box_h: int, use_cuda: bool) -> Tuple[Optional[subprocess.Popen], int, int]:
        scale = min(box_w / max(1, clip.width), box_h / max(1, clip.height))
        w = max(2, int(clip.width * scale) // 2 * 2)
        h = max(2, int(clip.height * scale) // 2 * 2)
        ff = get_ffmpeg_exe()
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if rate >= 3.0:
            cmd += ["-skip_frame", "noref"]
        if use_cuda:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        cmd += ["-ss", f"{max(0.0, local_start):.3f}", "-i", str(clip.path), "-an", "-sn"]
        if use_cuda:
            vf = f"fps={out_fps:.4f},scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12"
        else:
            vf = f"fps={out_fps:.4f},scale={w}:{h}:flags=bilinear"
        cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=w * h * 3 * 4,
                creationflags=get_CreationFlags(),
            )
        except Exception:
            return None, w, h
        return proc, w, h

    @staticmethod
    def _kill(proc: Optional[subprocess.Popen]) -> None:
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass

    def _run(self, sid: int, evt: threading.Event, clips: List[PlaybackClip], g_start: float,
             stop_at: Optional[float], rate: float, box_w: int, box_h: int) -> None:
        out_fps = self.DISPLAY_FPS / rate  # 원본 1초당 디코딩할 프레임 수
        frame_dt = 1.0 / self.DISPLAY_FPS
        todo = [c for c in clips if c.global_start + c.duration > g_start + 0.01]
        if stop_at is not None:
            todo = [c for c in todo if c.global_start < stop_at]

        wall0: Optional[float] = None
        media0 = g_start
        use_cuda = self._use_cuda
        prefetched: Optional[Tuple[Optional[subprocess.Popen], int, int]] = None

        try:
            for ci, clip in enumerate(todo):
                if evt.is_set():
                    return
                local0 = max(0.0, g_start - clip.global_start) if ci == 0 else 0.0
                if prefetched is not None:
                    proc, w, h = prefetched
                    prefetched = None
                else:
                    proc, w, h = self._spawn(clip, local0, out_fps, rate, box_w, box_h, use_cuda)
                # 다음 클립 디코더를 미리 띄워 클립 경계 끊김 제거
                if ci + 1 < len(todo):
                    prefetched = self._spawn(todo[ci + 1], 0.0, out_fps, rate, box_w, box_h, use_cuda)

                frame_bytes = w * h * 3
                idx = 0
                got_any = False
                while proc is not None and not evt.is_set():
                    raw = proc.stdout.read(frame_bytes) if proc.stdout else b""
                    if not raw or len(raw) < frame_bytes:
                        break
                    got_any = True
                    media_t = clip.global_start + local0 + idx / out_fps
                    idx += 1
                    if stop_at is not None and media_t >= stop_at:
                        self._kill(proc)
                        return
                    now = time.perf_counter()
                    if wall0 is None:
                        wall0, media0 = now, media_t
                    target = wall0 + (media_t - media0) / rate
                    if now < target:
                        if evt.wait(target - now):
                            break
                    elif now - target > 0.30:
                        # 디코딩이 크게 밀리면 기준 시각을 재설정 (멈춤 없이 이어서 재생)
                        wall0, media0 = now, media_t
                    elif now - target > 1.5 * frame_dt:
                        continue  # 늦은 프레임은 화면 표시 생략(드롭)
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                    with self._lock:
                        if self._session == sid:
                            self._latest = (sid, media_t, arr)
                self._kill(proc)

                if not got_any and use_cuda and not evt.is_set():
                    # CUDA 디코딩 실패 시 CPU 디코딩으로 폴백하여 같은 클립을 다시 재생
                    use_cuda = False
                    self._use_cuda = False
                    if prefetched is not None:
                        self._kill(prefetched[0])
                        prefetched = None
                    return self._run(sid, evt, clips, clip.global_start + local0, stop_at,
                                     rate, box_w, box_h)
        finally:
            if prefetched is not None:
                self._kill(prefetched[0])
            if not evt.is_set():
                self._finished_session = sid


@dataclass
class ReviewItem:
    """자동 검출된 신호대기/단조 주행 구간 하나에 대한 리뷰 항목 (클립 로컬 시간 기준)."""
    video: VideoFileInfo
    start_sec: float
    end_sec: float
    orig_type: SegmentType
    decision: Optional[str] = None  # None(미검토) / "keep"(스킵 유지) / "normal"(일반 복원) / "cut"(삭제)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def clone(self) -> "ReviewItem":
        return ReviewItem(self.video, self.start_sec, self.end_sec, self.orig_type, self.decision)


DECISION_LABELS = {
    None: "⏳ 미검토",
    "keep": "✔ 스킵 유지",
    "normal": "♻ 일반 복원",
    "cut": "✂ 삭제",
}


class TimelineCanvas(tk.Canvas):
    """단일 비디오의 NORMAL / STOP / STRAIGHT / MANUAL_CUT 구간 비율을 보여주는 컬러 바."""

    def __init__(self, master, height: int = 12, **kwargs):
        super().__init__(
            master,
            height=height,
            bg="#1e293b",
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._segments: List[Segment] = []
        self._duration: float = 0.0
        self._has_data: bool = False
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_data(self, duration: float, segments: List[Segment], has_data: bool) -> None:
        self._duration = max(0.0, duration)
        self._segments = list(segments)
        self._has_data = has_data
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        w = max(10, self.winfo_width())
        h = max(6, self.winfo_height())

        if not self._has_data or self._duration <= 0 or not self._segments:
            self.create_rectangle(0, 0, w, h, fill=COLOR_UNANALYZED, outline="")
            return

        for seg in self._segments:
            x1 = int(round((seg.start_sec / self._duration) * w))
            x2 = int(round((seg.end_sec / self._duration) * w))
            if x2 <= x1:
                x2 = x1 + 1
            color = {
                SegmentType.NORMAL: COLOR_NORMAL,
                SegmentType.STOP: COLOR_STOP,
                SegmentType.STRAIGHT: COLOR_STRAIGHT,
                SegmentType.MANUAL_CUT: COLOR_MANUAL_CUT,
            }.get(seg.segment_type, COLOR_NORMAL)
            self.create_rectangle(x1, 0, x2, h, fill=color, outline="")


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


class QuickDashCamApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("QuickDashCam - 액션캠 주행 영상 시간순 병합 & 스마트 타임라인 편집기")
        self.geometry("1180x760")
        self.minsize(860, 520)

        self.videos: List[VideoFileInfo] = []
        self.selected_index: Optional[int] = None
        self.is_busy: bool = False
        self.cancel_requested: bool = False
        self.last_output_path: Optional[Path] = None

        # 타임라인 탐색 및 구간 편집 상태
        self.playhead_sec: float = 0.0
        self.cut_in_sec: Optional[float] = None
        self.cut_out_sec: Optional[float] = None
        self.is_playing_preview: bool = False
        self._last_play_tick: float = 0.0
        self._preview_photo: Optional[tk.PhotoImage] = None
        self._undo_stack: List[Tuple[List[List[Segment]], List[ReviewItem]]] = []

        # 스트리밍 재생 상태
        self._playback = PlaybackWorker()
        self._play_session: int = -1
        self._play_stop_at: Optional[float] = None
        self._play_restart_job: Optional[str] = None
        self._last_label_update: float = 0.0

        # 검출 구간 리뷰 상태
        self.review_items: List[ReviewItem] = []
        self.review_index: Optional[int] = None
        self._suppress_tree_select: bool = False

        self._preview_worker = PreviewFrameWorker(self._on_worker_frame_ready)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._bind_shortcuts()
        self.after(200, self._detect_encoder_async)

    def _on_close(self) -> None:
        self.is_playing_preview = False
        self._playback.stop()
        self._preview_worker.close()
        self.destroy()

    def _bind_shortcuts(self) -> None:
        self.bind("<Left>", lambda _e: self._step_playhead(-1.0))
        self.bind("<Right>", lambda _e: self._step_playhead(1.0))
        self.bind("<Shift-Left>", lambda _e: self._step_playhead(-5.0))
        self.bind("<Shift-Right>", lambda _e: self._step_playhead(5.0))
        self.bind("<space>", lambda _e: self._toggle_preview_play())
        self.bind("i", lambda _e: self._on_mark_in())
        self.bind("I", lambda _e: self._on_mark_in())
        self.bind("o", lambda _e: self._on_mark_out())
        self.bind("O", lambda _e: self._on_mark_out())
        self.bind("<Delete>", lambda _e: self._on_apply_manual_cut())
        # 타임라인 확대/축소
        self.bind("<plus>", lambda _e: self.global_timeline.zoom_by(0.5))
        self.bind("<equal>", lambda _e: self.global_timeline.zoom_by(0.5))
        self.bind("<minus>", lambda _e: self.global_timeline.zoom_by(2.0))
        self.bind("<Key-0>", lambda _e: self.global_timeline.zoom_full())
        # 검출 구간 리뷰
        self.bind("<Key-1>", lambda _e: self._review_decide("keep"))
        self.bind("<Key-2>", lambda _e: self._review_decide("normal"))
        self.bind("<Key-3>", lambda _e: self._review_decide("cut"))
        for key in ("n", "N"):
            self.bind(key, lambda _e: self._review_step(1))
        for key in ("p", "P"):
            self.bind(key, lambda _e: self._review_step(-1))
        for key in ("r", "R"):
            self.bind(key, lambda _e: self._review_play_current())
        self.bind("<bracketleft>", lambda _e: self._review_set_boundary("start"))
        self.bind("<bracketright>", lambda _e: self._review_set_boundary("end"))

    def _detect_encoder_async(self) -> None:
        def worker():
            try:
                enc_name, _ = detect_best_h264_encoder(prefer_hw=True)
                dec_mode = detect_best_gpu_decoder()
                enc_map = {
                    "h264_nvenc": "NVIDIA NVENC",
                    "h264_qsv": "Intel QuickSync",
                    "h264_amf": "AMD AMF",
                    "libx264": "CPU (libx264)",
                }
                dec_map = {
                    "cuda_scale": "CUDA NVDEC + VRAM 리사이저",
                    "d3d11va": "D3D11VA 하드웨어 디코더",
                    "cpu": "CPU 디코더",
                }
                enc_desc = enc_map.get(enc_name, enc_name)
                dec_desc = dec_map.get(dec_mode, dec_mode)
                self.after(
                    0,
                    lambda: self.encoder_badge.configure(
                        text=f"⚡ 분석: {dec_desc}  |  저장: {enc_desc}"
                    ),
                )
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=7)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(1, weight=1)

        # ── 1. 상단 헤더 ──
        header = ctk.CTkFrame(self, corner_radius=0, fg_color="#0f172a")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        header.grid_columnconfigure(0, weight=1)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w", padx=14, pady=6)

        ctk.CTkLabel(
            title_box,
            text="🚗 QuickDashCam",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left", padx=(0, 10))

        ctk.CTkLabel(
            title_box,
            text="시간순 자동 정렬  |  스마트 스킵  |  타임라인 구간 편집",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
        ).pack(side="left")

        self.encoder_badge = ctk.CTkLabel(
            header,
            text="⚡ 인코더 확인 중...",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            corner_radius=6,
            padx=8,
            pady=3,
        )
        self.encoder_badge.grid(row=0, column=1, sticky="e", padx=14, pady=6)

        # ── 2. 좌측 메인 영역 (상단: 전체 타임라인 탐색 & 잘라내기 편집기 / 하단: 영상 목록) ──
        left_panel = ctk.CTkFrame(self, corner_radius=10)
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(10, 4), pady=(4, 4))
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(2, weight=1)

        # ── 2-A. 전체 타임라인 미리보기 & 구간 잘라내기 섹션 ──
        editor_frame = ctk.CTkFrame(left_panel, fg_color="#0f172a", corner_radius=8)
        editor_frame.grid(row=0, column=0, sticky="ew", padx=6, pady=3)
        editor_frame.grid_columnconfigure(0, weight=5)
        editor_frame.grid_columnconfigure(1, weight=5)

        # 좌측: 비디오 프리뷰 캔버스 (16:9)
        preview_container = ctk.CTkFrame(editor_frame, fg_color="#020617", corner_radius=6)
        preview_container.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=4)
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(
            preview_container,
            width=340,
            height=190,
            bg="#020617",
            highlightthickness=0,
            bd=0,
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.preview_canvas.bind("<Configure>", lambda _e: self._request_current_preview_frame())

        # 우측: 타임라인 탐색 컨트롤 & 구간 잘라내기 도구함
        cut_panel = ctk.CTkFrame(editor_frame, fg_color="#1e293b", corner_radius=6)
        cut_panel.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=4)
        cut_panel.grid_columnconfigure(0, weight=1)

        # Row 0: 타이틀 + 전체 위치 라벨
        info_row0 = ctk.CTkFrame(cut_panel, fg_color="transparent")
        info_row0.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 1))
        info_row0.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            info_row0,
            text="✂️ 타임라인 탐색 & 구간 편집",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")

        self.playhead_info_label = ctk.CTkLabel(
            info_row0,
            text="전체: 00:00.0 / 00:00.0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8",
            anchor="e",
        )
        self.playhead_info_label.pack(side="right")

        # Row 1: 현재 클립 + 선택 구간 라벨
        info_row1 = ctk.CTkFrame(cut_panel, fg_color="transparent")
        info_row1.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 3))
        info_row1.grid_columnconfigure(0, weight=1)

        self.clip_info_label = ctk.CTkLabel(
            info_row1,
            text="클립: 없음",
            font=ctk.CTkFont(size=11),
            text_color="#cbd5e1",
            anchor="w",
        )
        self.clip_info_label.pack(side="left")

        self.selection_range_label = ctk.CTkLabel(
            info_row1,
            text="선택: 미지정",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
            anchor="e",
        )
        self.selection_range_label.pack(side="right")

        # Row 2: 재생 및 미세 이동 버튼 줄
        nav_row = ctk.CTkFrame(cut_panel, fg_color="transparent")
        nav_row.grid(row=2, column=0, sticky="ew", padx=6, pady=2)
        for col_i in range(6):
            nav_row.grid_columnconfigure(col_i, weight=1)

        ctk.CTkButton(
            nav_row,
            text="⏮ -5s",
            height=26,
            width=48,
            font=ctk.CTkFont(size=11),
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self._step_playhead(-5.0),
        ).grid(row=0, column=0, padx=1, sticky="ew")

        ctk.CTkButton(
            nav_row,
            text="◀ -1s",
            height=26,
            width=48,
            font=ctk.CTkFont(size=11),
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self._step_playhead(-1.0),
        ).grid(row=0, column=1, padx=1, sticky="ew")

        self.btn_play_preview = ctk.CTkButton(
            nav_row,
            text="▶ 재생",
            height=26,
            width=64,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._toggle_preview_play,
        )
        self.btn_play_preview.grid(row=0, column=2, padx=1, sticky="ew")

        ctk.CTkButton(
            nav_row,
            text="+1s ▶",
            height=26,
            width=48,
            font=ctk.CTkFont(size=11),
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self._step_playhead(1.0),
        ).grid(row=0, column=3, padx=1, sticky="ew")

        ctk.CTkButton(
            nav_row,
            text="+5s ⏭",
            height=26,
            width=48,
            font=ctk.CTkFont(size=11),
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self._step_playhead(5.0),
        ).grid(row=0, column=4, padx=1, sticky="ew")

        self.play_rate_var = ctk.StringVar(value="1x")
        ctk.CTkOptionMenu(
            nav_row,
            values=["0.5x", "1x", "2x", "4x", "8x"],
            variable=self.play_rate_var,
            width=54,
            height=26,
            font=ctk.CTkFont(size=11),
            command=lambda _v: self._restart_playback_if_playing(),
        ).grid(row=0, column=5, padx=1, sticky="ew")

        # Row 3: 시작점(In) / 끝점(Out) 지정 버튼 줄
        inout_row = ctk.CTkFrame(cut_panel, fg_color="transparent")
        inout_row.grid(row=3, column=0, sticky="ew", padx=6, pady=2)
        inout_row.grid_columnconfigure(0, weight=1)
        inout_row.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            inout_row,
            text="🚩 시작점(In) [I]",
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#0f766e",
            hover_color="#115e59",
            command=self._on_mark_in,
        ).grid(row=0, column=0, padx=(1, 3), sticky="ew")

        ctk.CTkButton(
            inout_row,
            text="🏁 끝점(Out) [O]",
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#0f766e",
            hover_color="#115e59",
            command=self._on_mark_out,
        ).grid(row=0, column=1, padx=(3, 1), sticky="ew")

        # Row 4: 잘라내기 실행 / 복원 / 되돌리기 버튼 줄
        cut_action_row = ctk.CTkFrame(cut_panel, fg_color="transparent")
        cut_action_row.grid(row=4, column=0, sticky="ew", padx=6, pady=(2, 5))
        cut_action_row.grid_columnconfigure(0, weight=2)
        cut_action_row.grid_columnconfigure(1, weight=2)
        cut_action_row.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(
            cut_action_row,
            text="✂️ 선택 잘라내기 [Del]",
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            command=self._on_apply_manual_cut,
        ).grid(row=0, column=0, padx=1, sticky="ew")

        ctk.CTkButton(
            cut_action_row,
            text="♻️ 일반(1x) 복원",
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#15803d",
            hover_color="#166534",
            command=self._on_restore_selection_normal,
        ).grid(row=0, column=1, padx=1, sticky="ew")

        self.btn_undo_cut = ctk.CTkButton(
            cut_action_row,
            text="↩ 되돌리기",
            height=28,
            width=64,
            font=ctk.CTkFont(size=11),
            fg_color="#475569",
            hover_color="#334155",
            state="disabled",
            command=self._on_undo_cut,
        )
        self.btn_undo_cut.grid(row=0, column=2, padx=1, sticky="ew")

        # 하단: 전체 연결 타임라인 캔버스 + 범례
        tl_box = ctk.CTkFrame(editor_frame, fg_color="transparent")
        tl_box.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(1, 5))
        tl_box.grid_columnconfigure(0, weight=1)

        # 타임라인 확대/축소 컨트롤 줄
        zoom_row = ctk.CTkFrame(tl_box, fg_color="transparent")
        zoom_row.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        for txt, cmd, wdt in [
            ("🔍+", lambda: self.global_timeline.zoom_by(0.5), 38),
            ("🔍-", lambda: self.global_timeline.zoom_by(2.0), 38),
            ("◀", lambda: self._pan_timeline(-0.5), 32),
            ("▶", lambda: self._pan_timeline(0.5), 32),
            ("⌖ 1분", lambda: self._zoom_around_playhead(60.0), 56),
            ("⤢ 전체 [0]", lambda: self.global_timeline.zoom_full(), 72),
        ]:
            ctk.CTkButton(
                zoom_row, text=txt, width=wdt, height=22, font=ctk.CTkFont(size=10),
                fg_color="#334155", hover_color="#475569", command=cmd,
            ).pack(side="left", padx=(0, 2))

        self.view_range_label = ctk.CTkLabel(
            zoom_row,
            text="보기: 전체",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
        )
        self.view_range_label.pack(side="left", padx=(6, 0))

        self.global_timeline = GlobalTimelineCanvas(
            tl_box,
            on_seek=self._on_seek_global_timeline,
            on_range_select=self._on_range_selected_from_timeline,
            on_view_change=self._on_timeline_view_changed,
            height=60,
        )
        self.global_timeline.grid(row=1, column=0, sticky="ew")

        legend_row = ctk.CTkFrame(tl_box, fg_color="transparent")
        legend_row.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        for color, text in [
            (COLOR_NORMAL, "일반(1x)"),
            (COLOR_STOP, "정차(스킵)"),
            (COLOR_STRAIGHT, "직진(스킵)"),
            (COLOR_MANUAL_CUT, "삭제(컷)"),
        ]:
            dot = ctk.CTkFrame(legend_row, width=9, height=9, fg_color=color, corner_radius=2)
            dot.pack(side="left", padx=(0, 3))
            ctk.CTkLabel(legend_row, text=text, font=ctk.CTkFont(size=10)).pack(
                side="left", padx=(0, 8)
            )

        ctk.CTkLabel(
            legend_row,
            text="💡 휠: 줌 · Shift+휠: 이동 · 우클릭 드래그: 구간 지정",
            font=ctk.CTkFont(size=10),
            text_color="#38bdf8",
        ).pack(side="right")

        # ── 2-B. 파일 추가 및 정렬 툴바 ──
        toolbar = ctk.CTkFrame(left_panel, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 2))

        self.btn_add_folder = ctk.CTkButton(
            toolbar,
            text="📁 폴더 추가",
            width=88,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._on_add_folder,
        )
        self.btn_add_folder.pack(side="left", padx=(0, 4))

        self.btn_add_files = ctk.CTkButton(
            toolbar,
            text="➕ 파일 추가",
            width=84,
            height=26,
            font=ctk.CTkFont(size=11),
            command=self._on_add_files,
        )
        self.btn_add_files.pack(side="left", padx=(0, 6))

        ctk.CTkLabel(toolbar, text="정렬:", font=ctk.CTkFont(size=11)).pack(
            side="left", padx=(2, 4)
        )
        self.sort_var = ctk.StringVar(value="촬영 시각 순 (권장)")
        self.sort_menu = ctk.CTkOptionMenu(
            toolbar,
            values=["촬영 시각 순 (권장)", "파일 수정일 순", "파일명 순"],
            variable=self.sort_var,
            width=130,
            height=26,
            font=ctk.CTkFont(size=11),
            command=lambda _v: self._resort_videos(),
        )
        self.sort_menu.pack(side="left", padx=(0, 6))

        self.btn_clear = ctk.CTkButton(
            toolbar,
            text="전체 비우기",
            width=70,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#475569",
            hover_color="#334155",
            command=self._on_clear_all,
        )
        self.btn_clear.pack(side="right", padx=(2, 0))

        self.btn_remove = ctk.CTkButton(
            toolbar,
            text="✕ 제외",
            width=58,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            command=self._on_remove_selected,
        )
        self.btn_remove.pack(side="right", padx=2)

        self.btn_down = ctk.CTkButton(
            toolbar,
            text="▼",
            width=28,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self._move_selected(1),
        )
        self.btn_down.pack(side="right", padx=2)

        self.btn_up = ctk.CTkButton(
            toolbar,
            text="▲",
            width=28,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self._move_selected(-1),
        )
        self.btn_up.pack(side="right", padx=2)

        # ── 2-C. 탭: [클립 목록] / [검출 구간 리뷰] ──
        self.left_tabs = ctk.CTkTabview(left_panel)
        self.left_tabs.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 2))
        tab_clips = self.left_tabs.add("📋 클립 목록")
        tab_review = self.left_tabs.add("🔎 검출 구간 리뷰")
        for tab in (tab_clips, tab_review):
            tab.grid_columnconfigure(0, weight=1)
        tab_clips.grid_rowconfigure(0, weight=1)

        self.list_scroll = ctk.CTkScrollableFrame(
            tab_clips,
            label_text="개별 주행 영상 클립 목록 (클릭하여 타임라인 이동)",
            label_font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.list_scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.list_scroll.grid_columnconfigure(0, weight=1)

        self._build_review_tab(tab_review)

        # ── 2-D. 하단 요약 통계 바 ──
        self.summary_label = ctk.CTkLabel(
            left_panel,
            text="영상을 추가하면 전체 주행 시간과 예상 단축 시간이 여기에 표시됩니다.",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            corner_radius=6,
            anchor="w",
            padx=10,
            pady=4,
        )
        self.summary_label.grid(row=3, column=0, sticky="ew", padx=8, pady=(2, 4))

        # ── 3. 우측 패널: 자동 감지 및 출력 설정 ──
        right_panel = ctk.CTkScrollableFrame(self, corner_radius=10, width=300)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(4, 10), pady=(4, 4))
        right_panel.grid_columnconfigure(0, weight=1)

        # 섹션 A: 건너뛰기 방식
        sec_skip = ctk.CTkFrame(right_panel, fg_color="#1e293b", corner_radius=8)
        sec_skip.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 10))
        sec_skip.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            sec_skip,
            text="⏩ 1. 정차·직진 구간 단축 방식",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        self.skip_mode_var = ctk.StringVar(value="배속 재생 (타임랩스)")
        self.skip_mode_seg = ctk.CTkSegmentedButton(
            sec_skip,
            values=["배속 재생 (타임랩스)", "완전 잘라내기 (점프컷)"],
            variable=self.skip_mode_var,
            command=self._on_skip_mode_changed,
        )
        self.skip_mode_seg.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.speed_frame = ctk.CTkFrame(sec_skip, fg_color="transparent")
        self.speed_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        self.speed_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.speed_frame, text="배속 배율:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.speed_var = ctk.StringVar(value="8x (권장)")
        self.speed_menu = ctk.CTkOptionMenu(
            self.speed_frame,
            values=["4x (부드러움)", "8x (권장)", "12x (빠름)", "16x (초고속)"],
            variable=self.speed_var,
            command=lambda _v: self._update_summary(),
        )
        self.speed_menu.grid(row=0, column=1, sticky="ew")

        # 섹션 B: 정차 & 고속도로 직진 감지 설정
        sec_detect = ctk.CTkFrame(right_panel, fg_color="#1e293b", corner_radius=8)
        sec_detect.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 10))
        sec_detect.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            sec_detect,
            text="🎯 2. 자동 화면 변화 감지 조건",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        self.chk_stop_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            sec_detect,
            text="신호대기 / 정차 구간 감지",
            variable=self.chk_stop_var,
            command=self._reapply_analysis_from_cache,
        ).grid(row=1, column=0, sticky="w", padx=12, pady=4)

        self.chk_straight_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            sec_detect,
            text="장시간 고속도로 직진(단조 주행) 감지",
            variable=self.chk_straight_var,
            command=self._reapply_analysis_from_cache,
        ).grid(row=2, column=0, sticky="w", padx=12, pady=(4, 10))

        sens_header = ctk.CTkFrame(sec_detect, fg_color="transparent")
        sens_header.grid(row=3, column=0, sticky="ew", padx=12)
        ctk.CTkLabel(sens_header, text="감지 민감도:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.sens_val_label = ctk.CTkLabel(
            sens_header,
            text="보통 (50%)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8",
        )
        self.sens_val_label.pack(side="right")

        self.sens_slider = ctk.CTkSlider(
            sec_detect,
            from_=10,
            to=90,
            number_of_steps=16,
            command=self._on_sens_changed,
        )
        self.sens_slider.set(50)
        self.sens_slider.grid(row=4, column=0, sticky="ew", padx=12, pady=(2, 8))

        min_dur_header = ctk.CTkFrame(sec_detect, fg_color="transparent")
        min_dur_header.grid(row=5, column=0, sticky="ew", padx=12)
        ctk.CTkLabel(
            min_dur_header,
            text="최소 지속 시간 (이 시간 이상만 스킵):",
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        self.min_dur_val_label = ctk.CTkLabel(
            min_dur_header,
            text="5초",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8",
        )
        self.min_dur_val_label.pack(side="right")

        self.min_dur_slider = ctk.CTkSlider(
            sec_detect,
            from_=3,
            to=20,
            number_of_steps=17,
            command=self._on_min_dur_changed,
        )
        self.min_dur_slider.set(5)
        self.min_dur_slider.grid(row=6, column=0, sticky="ew", padx=12, pady=(2, 8))

        pad_row = ctk.CTkFrame(sec_detect, fg_color="transparent")
        pad_row.grid(row=7, column=0, sticky="ew", padx=12, pady=(2, 6))
        pad_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(pad_row, text="진입/출발 전후 여유:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.pad_var = ctk.StringVar(value="1.0초 (자연스러운 전환)")
        ctk.CTkOptionMenu(
            pad_row,
            values=["0.0초 (즉시 전환)", "0.5초", "1.0초 (자연스러운 전환)", "1.5초", "2.0초"],
            variable=self.pad_var,
            command=lambda _v: self._reapply_analysis_from_cache(),
        ).grid(row=0, column=1, sticky="ew")

        self.gpu_analyze_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            sec_detect,
            text="GPU 가속으로 구간 분석 (CUDA NVDEC + VRAM 축소)",
            variable=self.gpu_analyze_var,
        ).grid(row=8, column=0, sticky="w", padx=12, pady=(4, 12))

        # 섹션 C: 화질·코덱·오디오 등 저장 설정
        sec_out = ctk.CTkFrame(right_panel, fg_color="#1e293b", corner_radius=8)
        sec_out.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 10))
        sec_out.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            sec_out,
            text="🎬 3. 화질·코덱·오디오 저장 설정",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        ctk.CTkLabel(sec_out, text="화질 등급:", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, sticky="w", padx=12
        )
        self.quality_var = ctk.StringVar(value="원본 유지급 최고화질 (무손실에 가까움 · CRF 14)")
        ctk.CTkOptionMenu(
            sec_out,
            values=[
                "원본 유지급 최고화질 (무손실에 가까움 · CRF 14)",
                "고화질 (원본 수준 권장 · CRF 18)",
                "표준 화질 (화질·용량 균형 · CRF 22)",
                "저용량·고속 (빠른 저장 · CRF 27)",
            ],
            variable=self.quality_var,
        ).grid(row=2, column=0, sticky="ew", padx=12, pady=(2, 8))

        ctk.CTkLabel(sec_out, text="출력 해상도:", font=ctk.CTkFont(size=12)).grid(
            row=3, column=0, sticky="w", padx=12
        )
        self.res_var = ctk.StringVar(value="원본 해상도 유지 (4K/QHD/FHD 그대로)")
        ctk.CTkOptionMenu(
            sec_out,
            values=[
                "원본 해상도 유지 (4K/QHD/FHD 그대로)",
                "4K UHD (3840x2160)",
                "2.7K QHD (2560x1440)",
                "1080p FHD (1920x1080)",
                "720p HD (1280x720 - 빠른 인코딩)",
            ],
            variable=self.res_var,
        ).grid(row=4, column=0, sticky="ew", padx=12, pady=(2, 8))

        # 프레임 레이트 & 비디오 코덱 (2열 배치)
        fps_codec_row = ctk.CTkFrame(sec_out, fg_color="transparent")
        fps_codec_row.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 8))
        fps_codec_row.grid_columnconfigure(0, weight=1)
        fps_codec_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(fps_codec_row, text="프레임 레이트(FPS):", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=(0, 4)
        )
        ctk.CTkLabel(fps_codec_row, text="비디오 코덱:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=1, sticky="w", padx=(4, 0)
        )

        self.fps_var = ctk.StringVar(value="원본 프레임 유지")
        ctk.CTkOptionMenu(
            fps_codec_row,
            values=["원본 프레임 유지", "60fps (부드러움)", "30fps (표준 호환)", "24fps (시네마틱)"],
            variable=self.fps_var,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(2, 0))

        self.codec_var = ctk.StringVar(value="H.264 (최고 호환성)")
        ctk.CTkOptionMenu(
            fps_codec_row,
            values=["H.264 (최고 호환성)", "H.265 / HEVC (고효율·4K)"],
            variable=self.codec_var,
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(2, 0))

        ctk.CTkLabel(sec_out, text="오디오 처리:", font=ctk.CTkFont(size=12)).grid(
            row=6, column=0, sticky="w", padx=12
        )
        self.audio_var = ctk.StringVar(value="정상 구간 원본 유지 / 스킵 구간 음소거")
        ctk.CTkOptionMenu(
            sec_out,
            values=[
                "정상 구간 원본 유지 / 스킵 구간 음소거",
                "배속 구간도 오디오 속도 높여서 함께 재생",
                "결과 영상 전체 음소거",
            ],
            variable=self.audio_var,
        ).grid(row=7, column=0, sticky="ew", padx=12, pady=(2, 8))

        self.badge_overlay_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            sec_out,
            text="배속 재생 구간에 화면 배지 표시 (예: >> 8x)",
            variable=self.badge_overlay_var,
        ).grid(row=8, column=0, sticky="w", padx=12, pady=(4, 4))

        self.hw_accel_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            sec_out,
            text="GPU 하드웨어 가속 인코딩 사용 (가능 시)",
            variable=self.hw_accel_var,
        ).grid(row=9, column=0, sticky="w", padx=12, pady=(4, 12))

        # ── 4. 하단 실행 및 진행 상태 바 ──
        bottom_bar = ctk.CTkFrame(self, corner_radius=8, fg_color="#0f172a")
        bottom_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 6))
        bottom_bar.grid_columnconfigure(0, weight=1)

        prog_row = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        prog_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(4, 2))
        prog_row.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            prog_row,
            text="대기 중 — 주행 영상 파일 또는 폴더를 추가해 주세요.",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(prog_row, height=8)
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(2, 2))
        self.progress_bar.set(0.0)

        btn_row = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 6))
        btn_row.grid_columnconfigure(0, weight=3)
        btn_row.grid_columnconfigure(1, weight=4)
        btn_row.grid_columnconfigure(2, weight=1)
        btn_row.grid_columnconfigure(3, weight=2)

        self.btn_analyze = ctk.CTkButton(
            btn_row,
            text="🔍 1. 자동 구간 분석",
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._on_start_analysis_only,
        )
        self.btn_analyze.grid(row=0, column=0, padx=2, sticky="ew")

        self.btn_export = ctk.CTkButton(
            btn_row,
            text="🎬 2. 영상 합치기 (MP4 저장)",
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#16a34a",
            hover_color="#15803d",
            command=self._on_start_merge_export,
        )
        self.btn_export.grid(row=0, column=1, padx=2, sticky="ew")

        self.btn_cancel = ctk.CTkButton(
            btn_row,
            text="⏹ 취소",
            height=32,
            font=ctk.CTkFont(size=11),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            state="disabled",
            command=self._on_cancel_task,
        )
        self.btn_cancel.grid(row=0, column=2, padx=2, sticky="ew")

        self.btn_open_folder = ctk.CTkButton(
            btn_row,
            text="📂 결과 폴더",
            height=32,
            font=ctk.CTkFont(size=11),
            fg_color="#334155",
            hover_color="#475569",
            state="disabled",
            command=self._on_open_output_folder,
        )
        self.btn_open_folder.grid(row=0, column=3, padx=2, sticky="ew")

        self._refresh_video_list_ui()
        self._update_timeline_and_preview()

    # ── 전체 타임라인 탐색 & 프리뷰 플레이어 로직 ──

    @property
    def total_timeline_duration(self) -> float:
        return sum(v.duration_sec for v in self.videos)

    def _on_seek_global_timeline(self, target_sec: float) -> None:
        if not self.videos:
            return
        total = self.total_timeline_duration
        self.playhead_sec = max(0.0, min(total, target_sec))
        if self.is_playing_preview:
            # 재생 중 탐색: 디코더를 잠시 멈추고 탐색이 끝나면(0.25초 후) 새 위치에서 이어서 재생
            self._playback.stop()
            self._schedule_playback_restart()
        self._update_timeline_and_preview()

    def _step_playhead(self, delta_sec: float) -> None:
        if not self.videos:
            return
        self._on_seek_global_timeline(self.playhead_sec + delta_sec)

    # ── 스트리밍 재생 ──

    def _get_play_rate(self) -> float:
        try:
            return float(self.play_rate_var.get().rstrip("x"))
        except ValueError:
            return 1.0

    def _build_playback_clips(self) -> List[PlaybackClip]:
        clips: List[PlaybackClip] = []
        cursor = 0.0
        for v in self.videos:
            clips.append(PlaybackClip(v.path, cursor, v.duration_sec, v.width, v.height))
            cursor += v.duration_sec
        return clips

    def _toggle_preview_play(self) -> None:
        if not self.videos:
            return
        if self.is_playing_preview:
            self._stop_preview_play()
        else:
            if self.playhead_sec >= self.total_timeline_duration - 0.1:
                self.playhead_sec = 0.0
            self._start_preview_play(stop_at=None)

    def _start_preview_play(self, stop_at: Optional[float]) -> None:
        if not self.videos:
            return
        self._play_stop_at = stop_at
        self.is_playing_preview = True
        self.btn_play_preview.configure(text="⏸ 일시정지", fg_color="#d97706")
        self._launch_playback()
        self._playback_poll()

    def _launch_playback(self) -> None:
        cw = max(160, self.preview_canvas.winfo_width())
        ch = max(90, self.preview_canvas.winfo_height())
        self._play_session = self._playback.start(
            clips=self._build_playback_clips(),
            global_start=self.playhead_sec,
            stop_at=self._play_stop_at,
            rate=self._get_play_rate(),
            box_w=cw,
            box_h=ch,
        )

    def _schedule_playback_restart(self) -> None:
        if self._play_restart_job is not None:
            try:
                self.after_cancel(self._play_restart_job)
            except Exception:
                pass

        def restart():
            self._play_restart_job = None
            if self.is_playing_preview:
                # 탐색으로 정지 지점을 지나쳤으면 정지 지점 해제
                if self._play_stop_at is not None and self.playhead_sec >= self._play_stop_at:
                    self._play_stop_at = None
                self._launch_playback()

        self._play_restart_job = self.after(250, restart)

    def _restart_playback_if_playing(self) -> None:
        if self.is_playing_preview:
            self._playback.stop()
            self._schedule_playback_restart()

    def _stop_preview_play(self) -> None:
        if self.is_playing_preview:
            self.is_playing_preview = False
            self._playback.stop()
            if self._play_restart_job is not None:
                try:
                    self.after_cancel(self._play_restart_job)
                except Exception:
                    pass
                self._play_restart_job = None
            self.btn_play_preview.configure(text="▶ 재생", fg_color="#0284c7")
            self._update_timeline_and_preview()

    def _playback_poll(self) -> None:
        if not self.is_playing_preview:
            return
        item = self._playback.take_latest()
        if item is not None:
            sid, media_t, arr = item
            if sid == self._play_session:
                self.playhead_sec = media_t
                self._draw_preview_image(arr)
                self.global_timeline.set_playhead(media_t)
                now = time.perf_counter()
                if now - self._last_label_update > 0.12:
                    self._last_label_update = now
                    self._update_playhead_labels()
        if self._playback.is_finished(self._play_session) and self._play_restart_job is None:
            if self._play_stop_at is not None:
                self.playhead_sec = min(self._play_stop_at, self.total_timeline_duration)
            elif self.playhead_sec >= self.total_timeline_duration - 0.5:
                self.playhead_sec = self.total_timeline_duration
            self._stop_preview_play()
            return
        self.after(12, self._playback_poll)

    # ── 타임라인 확대/축소 ──

    def _on_timeline_view_changed(self, vs: float, ve: float, zoomed: bool) -> None:
        if not zoomed:
            self.view_range_label.configure(text="보기: 전체")
        else:
            span = ve - vs
            ratio = self.total_timeline_duration / max(0.001, span)
            self.view_range_label.configure(
                text=f"보기: {format_duration(vs)} ~ {format_duration(ve)} (×{ratio:.1f} 확대)"
            )

    def _pan_timeline(self, frac: float) -> None:
        vs, ve = self.global_timeline.view_range
        self.global_timeline.pan_by((ve - vs) * frac)

    def _zoom_around_playhead(self, span: float) -> None:
        if not self.videos:
            return
        self.global_timeline.set_view(self.playhead_sec - span / 2, self.playhead_sec + span / 2)

    def _on_range_selected_from_timeline(self, start_sec: float, end_sec: float) -> None:
        total = self.total_timeline_duration
        self.cut_in_sec = max(0.0, min(total, min(start_sec, end_sec)))
        self.cut_out_sec = max(0.0, min(total, max(start_sec, end_sec)))
        self._update_selection_label()
        self.global_timeline.set_state(
            self.videos, self.playhead_sec, self.cut_in_sec, self.cut_out_sec
        )

    def _on_mark_in(self) -> None:
        if not self.videos:
            return
        self.cut_in_sec = self.playhead_sec
        if self.cut_out_sec is not None and self.cut_out_sec < self.cut_in_sec:
            self.cut_out_sec = None
        self._update_selection_label()
        self.global_timeline.set_state(
            self.videos, self.playhead_sec, self.cut_in_sec, self.cut_out_sec
        )

    def _on_mark_out(self) -> None:
        if not self.videos:
            return
        self.cut_out_sec = self.playhead_sec
        if self.cut_in_sec is None:
            self.cut_in_sec = 0.0
        elif self.cut_out_sec < self.cut_in_sec:
            self.cut_in_sec, self.cut_out_sec = self.cut_out_sec, self.cut_in_sec
        self._update_selection_label()
        self.global_timeline.set_state(
            self.videos, self.playhead_sec, self.cut_in_sec, self.cut_out_sec
        )

    def _update_selection_label(self) -> None:
        if self.cut_in_sec is None and self.cut_out_sec is None:
            self.selection_range_label.configure(
                text="선택: 미지정",
                text_color="#94a3b8",
            )
            return
        s = self.cut_in_sec if self.cut_in_sec is not None else 0.0
        e = self.cut_out_sec if self.cut_out_sec is not None else self.total_timeline_duration
        if e < s:
            s, e = e, s
        dur = max(0.0, e - s)
        self.selection_range_label.configure(
            text=f"선택: {format_duration(s, True)}~{format_duration(e, True)} ({format_duration(dur, True)})",
            text_color="#38bdf8",
        )

    def _push_undo_snapshot(self) -> None:
        snapshot = (
            [copy.deepcopy(v.segments) for v in self.videos],
            [it.clone() for it in self.review_items],
        )
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 40:
            self._undo_stack.pop(0)
        self.btn_undo_cut.configure(state="normal")

    def _on_undo_cut(self) -> None:
        if self.is_busy or not self._undo_stack:
            return
        seg_snapshot, review_snapshot = self._undo_stack.pop()
        for v, old_segs in zip(self.videos, seg_snapshot):
            v.segments = old_segs
        self.review_items = review_snapshot
        if not self._undo_stack:
            self.btn_undo_cut.configure(state="disabled")
        self._refresh_video_list_ui()
        self._update_timeline_and_preview()
        self.status_label.configure(text="이전 구간 편집을 되돌렸습니다.")

    def _on_apply_manual_cut(self) -> None:
        """선택된 [In ~ Out] 구간을 수동 잘라내기(MANUAL_CUT) 구간으로 지정합니다."""
        if self.is_busy or not self.videos:
            return
        if self.cut_in_sec is None and self.cut_out_sec is None:
            messagebox.showinfo(
                "구간 선택 필요",
                "먼저 타임라인 바에서 우클릭 드래그하거나, [시작점(In) 찍기]와 [끝점(Out) 찍기] 버튼으로 잘라낼 구간을 지정해 주세요.",
            )
            return

        s = self.cut_in_sec if self.cut_in_sec is not None else 0.0
        e = self.cut_out_sec if self.cut_out_sec is not None else self.total_timeline_duration
        if abs(e - s) < 0.1:
            messagebox.showwarning("알림", "잘라낼 구간 길이가 너무 짧습니다 (최소 0.1초 이상 선택해 주세요).")
            return

        self._push_undo_snapshot()
        count = apply_global_timeline_override(self.videos, s, e, SegmentType.MANUAL_CUT)
        if count > 0:
            self.status_label.configure(
                text=f"✂️ 전체 타임라인 {format_duration(min(s, e), True)} ~ {format_duration(max(s, e), True)} 구간을 잘라내기(삭제) 처리했습니다."
            )
            self._refresh_video_list_ui()
            self._update_timeline_and_preview()

    def _on_restore_selection_normal(self) -> None:
        """선택된 [In ~ Out] 구간을 일반 주행(1x 정상 재생)으로 복원합니다."""
        if self.is_busy or not self.videos:
            return
        if self.cut_in_sec is None and self.cut_out_sec is None:
            messagebox.showinfo(
                "구간 선택 필요",
                "복원할 구간을 타임라인에서 우클릭 드래그하거나 시작점/끝점으로 먼저 선택해 주세요.",
            )
            return

        s = self.cut_in_sec if self.cut_in_sec is not None else 0.0
        e = self.cut_out_sec if self.cut_out_sec is not None else self.total_timeline_duration
        if abs(e - s) < 0.1:
            return

        self._push_undo_snapshot()
        count = apply_global_timeline_override(self.videos, s, e, SegmentType.NORMAL)
        if count > 0:
            self.status_label.configure(
                text=f"♻️ 전체 타임라인 {format_duration(min(s, e), True)} ~ {format_duration(max(s, e), True)} 구간을 일반 주행(1x)으로 복원했습니다."
            )
            self._refresh_video_list_ui()
            self._update_timeline_and_preview()

    def _update_timeline_and_preview(self) -> None:
        total = self.total_timeline_duration
        self.global_timeline.set_state(
            self.videos, self.playhead_sec, self.cut_in_sec, self.cut_out_sec
        )

        if not self.videos or total <= 0:
            self.playhead_info_label.configure(text="전체: 00:00.0 / 00:00.0")
            self.clip_info_label.configure(text="클립: 없음")
            self.selection_range_label.configure(text="선택: 미지정", text_color="#94a3b8")
            self.preview_canvas.delete("all")
            cw = max(100, self.preview_canvas.winfo_width())
            ch = max(80, self.preview_canvas.winfo_height())
            self.preview_canvas.create_text(
                cw // 2,
                ch // 2,
                text="주행 영상을 추가하면\n타임라인 탐색 미리보기 화면이 표시됩니다.",
                fill="#64748b",
                font=("Malgun Gothic", 10),
                justify="center",
            )
            return

        mapped = map_global_time_to_clip(self.videos, self.playhead_sec)
        if mapped is None:
            return
        self._update_playhead_labels()
        if not self.is_playing_preview:
            self._request_current_preview_frame()

    def _update_playhead_labels(self) -> None:
        mapped = map_global_time_to_clip(self.videos, self.playhead_sec)
        if mapped is None:
            return
        clip_idx, video, local_t, seg_type = mapped
        total = self.total_timeline_duration

        seg_short = {
            SegmentType.NORMAL: "1x",
            SegmentType.STOP: "정차",
            SegmentType.STRAIGHT: "직진",
            SegmentType.MANUAL_CUT: "삭제",
        }.get(seg_type, "1x")

        self.playhead_info_label.configure(
            text=f"전체: {format_duration(self.playhead_sec, True)} / {format_duration(total, True)} [{seg_short}]"
        )
        self.clip_info_label.configure(
            text=f"클립: [#{clip_idx + 1}] {video.filename[:22]}"
        )

    def _request_current_preview_frame(self) -> None:
        if not self.videos or self.is_playing_preview:
            return
        mapped = map_global_time_to_clip(self.videos, self.playhead_sec)
        if mapped is None:
            return
        _clip_idx, video, local_t, _seg_type = mapped
        cw = max(160, self.preview_canvas.winfo_width())
        ch = max(90, self.preview_canvas.winfo_height())
        self._preview_worker.request_frame(video.path, local_t, cw, ch)

    def _on_worker_frame_ready(self, frame_arr: np.ndarray, _req_id: int) -> None:
        def draw():
            if not self.videos or self.is_playing_preview:
                return
            self._draw_preview_image(frame_arr)

        try:
            self.after(0, draw)
        except Exception:
            pass

    def _draw_preview_image(self, img_or_arr: Union[np.ndarray, object]) -> None:
        self._preview_photo = make_photo_image(img_or_arr)
        cw = self.preview_canvas.winfo_width()
        ch = self.preview_canvas.winfo_height()
        if cw <= 10:
            cw = int(self.preview_canvas.cget("width") or 340)
        if ch <= 10:
            ch = int(self.preview_canvas.cget("height") or 190)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            cw // 2, ch // 2, image=self._preview_photo, anchor="center"
        )

        # 현재 위치가 수동 잘라내기(삭제) 또는 스킵 구간인 경우 프리뷰 상단에 배지 오버레이 표시
        mapped = map_global_time_to_clip(self.videos, self.playhead_sec)
        if mapped is not None:
            clip_idx, v, local_t, seg_type = mapped
            if seg_type == SegmentType.MANUAL_CUT:
                self.preview_canvas.create_rectangle(
                    8, 8, 225, 34, fill="#991b1b", outline="#ef4444", width=1
                )
                self.preview_canvas.create_text(
                    16,
                    21,
                    text="✂️ 수동 잘라내기 구간 (출력 시 삭제됨)",
                    anchor="w",
                    fill="#ffffff",
                    font=("Malgun Gothic", 9, "bold"),
                )
            elif seg_type in (SegmentType.STOP, SegmentType.STRAIGHT) and v.analyzed:
                tag_txt = (
                    "🟠 신호대기/정차 구간"
                    if seg_type == SegmentType.STOP
                    else "🟣 고속도로 직진 구간"
                )
                bg_col = "#92400e" if seg_type == SegmentType.STOP else "#6b21a8"
                self.preview_canvas.create_rectangle(
                    8, 8, 175, 34, fill=bg_col, outline="#ffffff", width=1
                )
                self.preview_canvas.create_text(
                    16,
                    21,
                    text=tag_txt,
                    anchor="w",
                    fill="#ffffff",
                    font=("Malgun Gothic", 9, "bold"),
                )
        if self.is_playing_preview:
            rate = self._get_play_rate()
            if abs(rate - 1.0) > 0.01:
                self.preview_canvas.create_text(
                    cw - 10, 12, text=f"▶ {rate:g}x", anchor="ne",
                    fill="#facc15", font=("Consolas", 11, "bold"),
                )

    # ── 검출 구간 리뷰 (신호대기 / 단조 주행 하나씩 확인 후 결정) ──

    def _build_review_tab(self, parent) -> None:
        parent.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(parent, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.review_progress_label = ctk.CTkLabel(
            head,
            text="검출 구간 목록 (클릭하여 확인 및 결정)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#cbd5e1",
        )
        self.review_progress_label.pack(side="left")

        self.review_filter_var = ctk.StringVar(value="전체 보기")
        ctk.CTkOptionMenu(
            head,
            values=["전체 보기", "미검토만", "신호대기만", "단조 주행만"],
            variable=self.review_filter_var,
            width=96,
            height=24,
            font=ctk.CTkFont(size=11),
            command=lambda _v: self._refresh_review_ui(),
        ).pack(side="right", padx=(4, 0))
        self.review_autoplay_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            head,
            text="선택 시 자동 재생",
            variable=self.review_autoplay_var,
            font=ctk.CTkFont(size=10),
            checkbox_width=16,
            checkbox_height=16,
        ).pack(side="right", padx=(4, 0))
        ctk.CTkButton(
            head,
            text="▶ 리뷰 시작",
            width=78,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._review_start,
        ).pack(side="right", padx=(4, 0))

        card = ctk.CTkFrame(parent, fg_color="#0f172a", corner_radius=6)
        card.grid(row=1, column=0, sticky="ew", pady=(2, 2))
        card.grid_columnconfigure(0, weight=1)

        self.review_current_label = ctk.CTkLabel(
            card,
            text="현재 검토 구간: 없음",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#facc15",
            anchor="w",
        )
        self.review_current_label.grid(row=0, column=0, sticky="ew", padx=8, pady=(3, 1))

        btn_row1 = ctk.CTkFrame(card, fg_color="transparent")
        btn_row1.grid(row=1, column=0, sticky="ew", padx=4, pady=1)
        row1_specs = [
            ("◀ [P]", lambda: self._review_step(-1), "#334155", "#475569"),
            ("▶ 재생 [R]", self._review_play_current, "#0284c7", "#0369a1"),
            ("✔ 스킵 [1]", lambda: self._review_decide("keep"), "#b45309", "#92400e"),
            ("♻ 일반 [2]", lambda: self._review_decide("normal"), "#15803d", "#166534"),
            ("✂ 삭제 [3]", lambda: self._review_decide("cut"), "#dc2626", "#b91c1c"),
            ("[N] ▶", lambda: self._review_step(1), "#334155", "#475569"),
        ]
        for col_i, (txt, cmd, fg, hv) in enumerate(row1_specs):
            btn_row1.grid_columnconfigure(col_i, weight=1)
            ctk.CTkButton(
                btn_row1, text=txt, height=24, font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=fg, hover_color=hv, command=cmd,
            ).grid(row=0, column=col_i, padx=1, sticky="ew")

        btn_row2 = ctk.CTkFrame(card, fg_color="transparent")
        btn_row2.grid(row=2, column=0, sticky="ew", padx=4, pady=(1, 3))
        row2_specs = [
            ("⇤ 시작점 [ [ ]", lambda: self._review_set_boundary("start")),
            ("끝점 ⇥ [ ] ]", lambda: self._review_set_boundary("end")),
            ("미검토 모두 스킵 유지", self._review_keep_all_pending),
        ]
        for col_i, (txt, cmd) in enumerate(row2_specs):
            btn_row2.grid_columnconfigure(col_i, weight=1)
            ctk.CTkButton(
                btn_row2, text=txt, height=24, font=ctk.CTkFont(size=11),
                fg_color="#1e293b", hover_color="#334155", border_width=1,
                border_color="#475569", command=cmd,
            ).grid(row=0, column=col_i, padx=1, sticky="ew")

        tree_box = ctk.CTkFrame(parent, fg_color="#1e293b", corner_radius=6)
        tree_box.grid(row=2, column=0, sticky="nsew")
        tree_box.grid_columnconfigure(0, weight=1)
        tree_box.grid_rowconfigure(0, weight=1)

        scaling = ctk.ScalingTracker.get_widget_scaling(self)
        body_px = int(round(-12 * scaling))
        head_px = int(round(-12 * scaling))
        row_height = max(24, int(round(24 * scaling)))
        head_pad_y = max(3, int(round(4 * scaling)))

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Review.Treeview",
            background="#1e293b",
            fieldbackground="#1e293b",
            foreground="#e2e8f0",
            rowheight=row_height,
            borderwidth=0,
            font=("Malgun Gothic", body_px),
        )
        style.configure(
            "Review.Treeview.Heading",
            background="#334155",
            foreground="#e2e8f0",
            relief="flat",
            padding=(0, head_pad_y),
            font=("Malgun Gothic", head_px, "bold"),
        )
        style.map("Review.Treeview", background=[("selected", "#0369a1")], foreground=[("selected", "#ffffff")])
        style.map("Review.Treeview.Heading", background=[("active", "#475569")])

        columns = ("no", "type", "start", "end", "dur", "clip", "status")
        self.review_tree = ttk.Treeview(
            tree_box, columns=columns, show="headings", height=4,
            style="Review.Treeview", selectmode="browse",
        )
        for col, txt, base_wdt, anchor in [
            ("no", "No", 44, "center"),
            ("type", "유형", 85, "w"),
            ("start", "시작", 75, "center"),
            ("end", "끝", 75, "center"),
            ("dur", "길이", 62, "e"),
            ("clip", "클립", 48, "center"),
            ("status", "결정", 96, "w"),
        ]:
            scaled_wdt = int(round(base_wdt * scaling))
            self.review_tree.heading(col, text=txt)
            self.review_tree.column(col, width=scaled_wdt, anchor=anchor, stretch=(col == "status"))
        self.review_tree.tag_configure("pending", foreground="#fbbf24")
        self.review_tree.tag_configure("keep", foreground="#fdba74")
        self.review_tree.tag_configure("normal", foreground="#4ade80")
        self.review_tree.tag_configure("cut", foreground="#f87171")
        self.review_tree.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
        sb = ctk.CTkScrollbar(tree_box, command=self.review_tree.yview)
        sb.grid(row=0, column=1, sticky="ns", pady=4)
        self.review_tree.configure(yscrollcommand=sb.set)
        self.review_tree.bind("<<TreeviewSelect>>", self._on_review_tree_select)
        self.review_tree.bind("<Double-1>", lambda _e: self._review_play_current())

    def _clip_offset(self, video: VideoFileInfo) -> Optional[float]:
        cursor = 0.0
        for v in self.videos:
            if v is video:
                return cursor
            cursor += v.duration_sec
        return None

    def _item_global(self, it: ReviewItem) -> Optional[Tuple[float, float]]:
        off = self._clip_offset(it.video)
        if off is None:
            return None
        return off + it.start_sec, off + it.end_sec

    @staticmethod
    def _type_coverage(v: VideoFileInfo, s: float, e: float, seg_type: SegmentType) -> float:
        dur = max(1e-6, e - s)
        cov = 0.0
        for seg in v.segments:
            if seg.segment_type == seg_type:
                cov += max(0.0, min(e, seg.end_sec) - max(s, seg.start_sec))
        return cov / dur

    def _find_matching_item(self, ref: Optional[ReviewItem]) -> Optional[int]:
        if ref is None:
            return None
        best_i, best_ov = None, 0.0
        for i, it in enumerate(self.review_items):
            if it.video is not ref.video:
                continue
            ov = min(it.end_sec, ref.end_sec) - max(it.start_sec, ref.start_sec)
            if ov > best_ov:
                best_i, best_ov = i, ov
        return best_i

    def _rebuild_review_items(self) -> None:
        """현재 구간 상태로부터 리뷰 항목 목록을 다시 만들고, 사용자의 기존 결정을 유지합니다."""
        cur = None
        if self.review_index is not None and 0 <= self.review_index < len(self.review_items):
            cur = self.review_items[self.review_index]

        present = {id(v) for v in self.videos}
        kept: List[ReviewItem] = []
        for it in self.review_items:
            if id(it.video) not in present or it.decision not in ("normal", "cut"):
                continue
            want = SegmentType.NORMAL if it.decision == "normal" else SegmentType.MANUAL_CUT
            if self._type_coverage(it.video, it.start_sec, it.end_sec, want) >= 0.5:
                kept.append(it)

        fresh: List[ReviewItem] = []
        for v in self.videos:
            for seg in v.segments:
                if seg.segment_type in (SegmentType.STOP, SegmentType.STRAIGHT) and seg.duration >= 0.05:
                    fresh.append(
                        ReviewItem(v, seg.start_sec, seg.end_sec, seg.segment_type,
                                   "keep" if seg.reviewed else None)
                    )

        order = {id(v): i for i, v in enumerate(self.videos)}
        items = kept + fresh
        items.sort(key=lambda it: (order[id(it.video)], it.start_sec))
        self.review_items = items
        self.review_index = self._find_matching_item(cur)

    def _filtered_review_indices(self) -> List[int]:
        f = self.review_filter_var.get()
        res: List[int] = []
        for i, it in enumerate(self.review_items):
            if f == "미검토만" and it.decision is not None:
                continue
            if f == "신호대기만" and it.orig_type != SegmentType.STOP:
                continue
            if f == "단조 주행만" and it.orig_type != SegmentType.STRAIGHT:
                continue
            res.append(i)
        return res

    def _refresh_review_ui(self) -> None:
        tree = self.review_tree
        tree.delete(*tree.get_children())
        clip_no = {id(v): i + 1 for i, v in enumerate(self.videos)}
        for i in self._filtered_review_indices():
            it = self.review_items[i]
            g = self._item_global(it)
            if g is None:
                continue
            type_txt = "🟠 신호대기" if it.orig_type == SegmentType.STOP else "🟣 단조 주행"
            tree.insert(
                "", "end", iid=str(i),
                values=(
                    i + 1, type_txt, format_duration(g[0], True), format_duration(g[1], True),
                    f"{it.duration:.1f}초", f"#{clip_no.get(id(it.video), '?')}",
                    DECISION_LABELS.get(it.decision, ""),
                ),
                tags=(it.decision or "pending",),
            )
        if self.review_index is not None and tree.exists(str(self.review_index)):
            tree.selection_set(str(self.review_index))
            tree.see(str(self.review_index))

        n = len(self.review_items)
        if n == 0:
            self.review_progress_label.configure(
                text="검출된 구간 없음 — [자동 구간 분석]을 먼저 실행해 주세요."
            )
        else:
            pending = [it for it in self.review_items if it.decision is None]
            cnt = {k: sum(1 for it in self.review_items if it.decision == k) for k in ("keep", "normal", "cut")}
            pend_dur = sum(it.duration for it in pending)
            self.review_progress_label.configure(
                text=(
                    f"검토 {n - len(pending)}/{n} 완료  ·  미검토 {len(pending)}개 ({format_duration(pend_dur)})  ·  "
                    f"유지 {cnt['keep']} / 복원 {cnt['normal']} / 삭제 {cnt['cut']}"
                )
            )

        g = None
        if self.review_index is not None:
            it = self.review_items[self.review_index]
            g = self._item_global(it)
            if g is not None:
                type_txt = "🟠 신호대기/정차" if it.orig_type == SegmentType.STOP else "🟣 단조 주행(고속도로 직진)"
                self.review_current_label.configure(
                    text=(
                        f"[{self.review_index + 1}/{n}] {type_txt}   {format_duration(g[0], True)} ~ "
                        f"{format_duration(g[1], True)}  (길이 {it.duration:.1f}초)   ·   "
                        f"현재 결정: {DECISION_LABELS.get(it.decision, '')}"
                    )
                )
        if g is None:
            self.review_current_label.configure(text="현재 검토 구간: 없음 (목록에서 구간을 선택하거나 [리뷰 시작]을 누르세요)")
        self.global_timeline.set_highlight(g)

    def _on_review_tree_select(self, _event=None) -> None:
        sel = self.review_tree.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
        except ValueError:
            return
        if idx == self.review_index:
            return
        self._review_select(idx, autoplay=bool(self.review_autoplay_var.get()))

    def _review_select(self, idx: int, autoplay: bool = False) -> None:
        if not (0 <= idx < len(self.review_items)) or not self.videos:
            return
        it = self.review_items[idx]
        g = self._item_global(it)
        if g is None:
            return
        gs, ge = g
        self.review_index = idx
        self._stop_preview_play()

        # 검토 구간 앞뒤 문맥이 보이도록 타임라인 자동 확대
        ctx = max(4.0, it.duration * 0.25)
        self.global_timeline.set_view(gs - ctx, ge + ctx)
        self.playhead_sec = max(0.0, gs - 2.0)  # 진입 2초 전부터 확인
        self._refresh_review_ui()
        self._update_timeline_and_preview()
        if autoplay:
            self._start_preview_play(stop_at=min(self.total_timeline_duration, ge + 2.0))

    def _review_play_current(self) -> None:
        if self.review_index is None or not (0 <= self.review_index < len(self.review_items)):
            return
        g = self._item_global(self.review_items[self.review_index])
        if g is None:
            return
        self._stop_preview_play()
        self.playhead_sec = max(0.0, g[0] - 2.0)
        self._update_timeline_and_preview()
        self._start_preview_play(stop_at=min(self.total_timeline_duration, g[1] + 2.0))

    def _review_step(self, delta: int) -> None:
        indices = self._filtered_review_indices()
        if not indices:
            return
        if self.review_index in indices:
            pos = indices.index(self.review_index) + delta
        else:
            # 현재 항목이 필터에서 빠졌으면 현재 위치 기준으로 가장 가까운 다음/이전 항목
            cur = self.review_index if self.review_index is not None else -1
            if delta > 0:
                pos = next((k for k, i in enumerate(indices) if i > cur), len(indices) - 1)
            else:
                pos = max((k for k, i in enumerate(indices) if i < cur), default=0)
        pos = max(0, min(len(indices) - 1, pos))
        self._review_select(indices[pos], autoplay=bool(self.review_autoplay_var.get()))

    def _next_pending_index(self, after: Optional[int]) -> Optional[int]:
        indices = [i for i in self._filtered_review_indices() if self.review_items[i].decision is None]
        if not indices:
            return None
        start = after if after is not None else -1
        for i in indices:
            if i > start:
                return i
        return indices[0]

    def _review_start(self) -> None:
        self.left_tabs.set("🔎 검출 구간 리뷰")
        if not self.review_items:
            messagebox.showinfo(
                "리뷰할 구간 없음",
                "검출된 신호대기·단조 주행 구간이 없습니다.\n먼저 [자동 구간 분석]을 실행해 주세요.",
            )
            return
        nxt = self._next_pending_index(None)
        if nxt is None:
            nxt = self._filtered_review_indices()[0] if self._filtered_review_indices() else 0
            self.status_label.configure(text="모든 검출 구간을 이미 검토했습니다. 필요하면 다시 확인할 수 있습니다.")
        self._review_select(nxt, autoplay=bool(self.review_autoplay_var.get()))

    @staticmethod
    def _decision_type(it: ReviewItem, decision: str) -> SegmentType:
        return {
            "keep": it.orig_type,
            "normal": SegmentType.NORMAL,
            "cut": SegmentType.MANUAL_CUT,
        }[decision]

    def _review_decide(self, decision: str) -> None:
        if self.is_busy or self.review_index is None or not (0 <= self.review_index < len(self.review_items)):
            return
        it = self.review_items[self.review_index]
        self._push_undo_snapshot()
        self._stop_preview_play()

        v = it.video
        v.segments = apply_segment_override(
            v.segments, v.duration_sec, it.start_sec, it.end_sec,
            self._decision_type(it, decision), mark_reviewed=True,
        )
        it.decision = decision
        self._refresh_video_list_ui()  # 리뷰 목록 재구성 (현재 항목 위치 유지)
        self._update_timeline_and_preview()

        msg = {
            "keep": "✔ 스킵 유지로 확정했습니다.",
            "normal": "♻ 일반 주행(1x)으로 복원했습니다.",
            "cut": "✂ 완전 삭제(수동 잘라내기)로 지정했습니다.",
        }[decision]
        nxt = self._next_pending_index(self.review_index)
        if nxt is not None:
            self.status_label.configure(text=f"{msg} 다음 미검토 구간으로 이동합니다.")
            self._review_select(nxt, autoplay=bool(self.review_autoplay_var.get()))
        else:
            self.status_label.configure(text=f"{msg} 🎉 모든 검출 구간 검토를 완료했습니다!")

    def _review_set_boundary(self, which: str) -> None:
        if self.is_busy or self.review_index is None or not (0 <= self.review_index < len(self.review_items)):
            return
        it = self.review_items[self.review_index]
        off = self._clip_offset(it.video)
        if off is None:
            return
        v = it.video
        local = self.playhead_sec - off
        if local < -0.05 or local > v.duration_sec + 0.05:
            self.status_label.configure(text="⚠ 경계 조정은 같은 클립 안의 위치에서만 가능합니다.")
            return
        local = max(0.0, min(v.duration_sec, local))
        new_s, new_e = (local, it.end_sec) if which == "start" else (it.start_sec, local)
        if new_e - new_s < 0.5:
            self.status_label.configure(text="⚠ 구간 길이가 너무 짧아집니다 (시작점은 끝점보다 0.5초 이상 앞이어야 합니다).")
            return

        self._push_undo_snapshot()
        was_playing = self.is_playing_preview
        self._stop_preview_play()
        decision = it.decision or "keep"
        # 기존 범위는 일반 주행으로 되돌린 뒤 새 범위에 결정 적용
        v.segments = apply_segment_override(
            v.segments, v.duration_sec, it.start_sec, it.end_sec, SegmentType.NORMAL, mark_reviewed=True
        )
        v.segments = apply_segment_override(
            v.segments, v.duration_sec, new_s, new_e, self._decision_type(it, decision), mark_reviewed=True
        )
        it.start_sec, it.end_sec, it.decision = new_s, new_e, decision
        self._refresh_video_list_ui()
        self._update_timeline_and_preview()
        label = "시작점" if which == "start" else "끝점"
        self.status_label.configure(
            text=f"구간 {label}을 {format_duration(self.playhead_sec, True)}(으)로 조정했습니다."
        )
        if was_playing:
            self._start_preview_play(stop_at=self._play_stop_at)

    def _review_keep_all_pending(self) -> None:
        if self.is_busy:
            return
        pending = [it for it in self.review_items if it.decision is None]
        if not pending:
            self.status_label.configure(text="미검토 구간이 없습니다.")
            return
        if not messagebox.askyesno(
            "일괄 확정", f"남은 미검토 구간 {len(pending)}개를 모두 '스킵 유지'로 확정할까요?"
        ):
            return
        self._push_undo_snapshot()
        self._stop_preview_play()
        for it in pending:
            it.video.segments = apply_segment_override(
                it.video.segments, it.video.duration_sec, it.start_sec, it.end_sec,
                it.orig_type, mark_reviewed=True,
            )
            it.decision = "keep"
        self._refresh_video_list_ui()
        self._update_timeline_and_preview()
        self.status_label.configure(text=f"✔ 미검토 구간 {len(pending)}개를 모두 스킵 유지로 확정했습니다.")

    # ── UI 설정 이벤트 핸들러 ──

    def _on_skip_mode_changed(self, value: str) -> None:
        if "점프컷" in value:
            self.speed_menu.configure(state="disabled")
        else:
            self.speed_menu.configure(state="normal")
        self._refresh_video_list_ui()

    def _on_sens_changed(self, value: float) -> None:
        v = int(round(value))
        if v < 35:
            tag = "낮음 (둔감)"
        elif v <= 65:
            tag = "보통"
        else:
            tag = "높음 (민감)"
        self.sens_val_label.configure(text=f"{tag} ({v}%)")
        self._reapply_analysis_from_cache()

    def _on_min_dur_changed(self, value: float) -> None:
        v = int(round(value))
        self.min_dur_val_label.configure(text=f"{v}초")
        self._reapply_analysis_from_cache()

    def _reapply_analysis_from_cache(self) -> None:
        """이미 특징이 추출된 영상들에 대해 디코딩 없이 즉시(0.01초 내) 새 민감도/최소시간을 반영합니다."""
        if self.is_busy or not self.videos:
            return
        if not any(v.feature_cache is not None for v in self.videos):
            return

        cfg = self._get_current_analysis_config()
        for v in self.videos:
            if v.feature_cache is not None:
                old_segs = list(v.segments)
                new_segs = reclassify_from_cache(v, cfg)
                if new_segs is not None:
                    v.segments = merge_preserving_manual_cuts(old_segs, new_segs, v.duration_sec)
                    v.analyzed = True

        self._refresh_video_list_ui()
        self._update_timeline_and_preview()

    def _get_current_sort_mode(self) -> SortMode:
        val = self.sort_var.get()
        if "파일명" in val:
            return SortMode.FILENAME
        if "수정일" in val:
            return SortMode.FILE_MTIME
        return SortMode.CHRONOLOGICAL

    def _get_current_skip_mode(self) -> SkipMode:
        if "점프컷" in self.skip_mode_var.get():
            return SkipMode.JUMPCUT
        return SkipMode.TIMELAPSE

    def _get_current_speed_multiplier(self) -> float:
        raw = self.speed_var.get().split("x")[0].strip()
        try:
            return float(raw)
        except ValueError:
            return 8.0

    def _get_current_analysis_config(self) -> AnalysisConfig:
        pad_str = self.pad_var.get().split("초")[0].strip()
        try:
            pad_sec = float(pad_str)
        except ValueError:
            pad_sec = 1.0

        return AnalysisConfig(
            detect_stops=bool(self.chk_stop_var.get()),
            detect_highway_straight=bool(self.chk_straight_var.get()),
            sensitivity=float(self.sens_slider.get()) / 100.0,
            min_skip_duration_sec=float(round(self.min_dur_slider.get())),
            transition_padding_sec=pad_sec,
            use_gpu_accel=bool(self.gpu_analyze_var.get()),
        )

    def _get_current_export_config(self, output_path: Path) -> ExportConfig:
        res_str = self.res_var.get()
        if "4K" in res_str:
            target_res = "4k"
        elif "2.7K" in res_str:
            target_res = "1440p"
        elif "1080p" in res_str:
            target_res = "1080p"
        elif "720p" in res_str:
            target_res = "720p"
        else:
            target_res = "original"

        fps_str = self.fps_var.get()
        if "60fps" in fps_str:
            fps_mode = "60"
        elif "30fps" in fps_str:
            fps_mode = "30"
        elif "24fps" in fps_str:
            fps_mode = "24"
        else:
            fps_mode = "original"

        qual_str = self.quality_var.get()
        if "최고화질" in qual_str:
            qual_preset = VideoQualityPreset.LOSSLESS_LIKE
        elif "표준" in qual_str:
            qual_preset = VideoQualityPreset.BALANCED
        elif "저용량" in qual_str:
            qual_preset = VideoQualityPreset.COMPACT
        else:
            qual_preset = VideoQualityPreset.HIGH

        codec_str = self.codec_var.get()
        video_codec = VideoCodec.HEVC if "H.265" in codec_str else VideoCodec.H264

        audio_str = self.audio_var.get()
        if "전체 음소거" in audio_str:
            audio_mode = AudioMode.MUTE_ALL
        elif "배속 구간도 오디오" in audio_str:
            audio_mode = AudioMode.KEEP_AND_SPEED_UP
        else:
            audio_mode = AudioMode.KEEP_NORMAL_MUTE_SKIP

        return ExportConfig(
            output_path=output_path,
            skip_mode=self._get_current_skip_mode(),
            speed_multiplier=self._get_current_speed_multiplier(),
            audio_mode=audio_mode,
            target_resolution=target_res,
            fps_mode=fps_mode,
            quality_preset=qual_preset,
            video_codec=video_codec,
            overlay_speed_badge=bool(self.badge_overlay_var.get()),
            use_hw_accel=bool(self.hw_accel_var.get()),
        )

    # ── 파일 추가 / 정렬 / 관리 ──

    def _on_add_folder(self) -> None:
        if self.is_busy:
            return
        folder = filedialog.askdirectory(title="액션캠 주행 영상이 들어있는 폴더 선택")
        if not folder:
            return
        paths = scan_folder_for_videos(folder, recursive=False)
        if not paths:
            messagebox.showinfo("알림", "선택한 폴더에서 비디오 파일을 찾지 못했습니다.")
            return
        self._load_video_paths_async(paths)

    def _on_add_files(self) -> None:
        if self.is_busy:
            return
        exts = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
        files = filedialog.askopenfilenames(
            title="추가할 주행 영상 파일 선택",
            filetypes=[("비디오 파일", exts), ("모든 파일", "*.*")],
        )
        if not files:
            return
        paths = [Path(f).resolve() for f in files]
        self._load_video_paths_async(paths)

    def _load_video_paths_async(self, paths: List[Path]) -> None:
        existing = {v.path for v in self.videos}
        new_paths = [p for p in paths if p not in existing]
        if not new_paths:
            messagebox.showinfo("알림", "선택한 영상들이 이미 목록에 추가되어 있습니다.")
            return

        self._stop_preview_play()
        self._set_busy_state(True, "영상 메타데이터 및 촬영 시각을 읽는 중...")

        def worker():
            loaded: List[VideoFileInfo] = []
            total = len(new_paths)
            for idx, p in enumerate(new_paths):
                if self.cancel_requested:
                    break
                self.after(
                    0,
                    lambda i=idx, name=p.name: (
                        self.progress_bar.set((i + 1) / total),
                        self.status_label.configure(
                            text=f"파일 정보 읽는 중 ({i + 1}/{total}): {name}"
                        ),
                    ),
                )
                try:
                    info = probe_video_file(p)
                    if info.duration_sec > 0.1:
                        info.ensure_segments_initialized()
                        loaded.append(info)
                except Exception:
                    continue

            def finish():
                self.videos.extend(loaded)
                self.videos = sort_video_files(self.videos, self._get_current_sort_mode())
                self._undo_stack.clear()
                self.btn_undo_cut.configure(state="disabled")
                if self.videos and self.selected_index is None:
                    self.selected_index = 0
                self._set_busy_state(
                    False,
                    f"총 {len(self.videos)}개 영상이 시간순으로 정렬되었습니다. 상단 타임라인 바를 클릭/드래그하여 탐색하거나 구간을 잘라낼 수 있습니다.",
                )
                self._refresh_video_list_ui()
                self._update_timeline_and_preview()

            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _resort_videos(self) -> None:
        if self.is_busy or not self.videos:
            return
        self.videos = sort_video_files(self.videos, self._get_current_sort_mode())
        self._undo_stack.clear()
        self.btn_undo_cut.configure(state="disabled")
        self._refresh_video_list_ui()
        self._update_timeline_and_preview()

    def _move_selected(self, delta: int) -> None:
        if self.is_busy or self.selected_index is None:
            return
        idx = self.selected_index
        new_idx = idx + delta
        if 0 <= new_idx < len(self.videos):
            self.videos[idx], self.videos[new_idx] = self.videos[new_idx], self.videos[idx]
            self.selected_index = new_idx
            self._undo_stack.clear()
            self.btn_undo_cut.configure(state="disabled")
            self._refresh_video_list_ui()
            self._update_timeline_and_preview()

    def _on_remove_selected(self) -> None:
        if self.is_busy or self.selected_index is None:
            return
        if 0 <= self.selected_index < len(self.videos):
            self.videos.pop(self.selected_index)
            self._undo_stack.clear()
            self.btn_undo_cut.configure(state="disabled")
            if not self.videos:
                self.selected_index = None
                self.playhead_sec = 0.0
                self.cut_in_sec = None
                self.cut_out_sec = None
            else:
                self.selected_index = min(self.selected_index, len(self.videos) - 1)
                self.playhead_sec = min(self.playhead_sec, self.total_timeline_duration)
            self._refresh_video_list_ui()
            self._update_timeline_and_preview()

    def _on_clear_all(self) -> None:
        if self.is_busy or not self.videos:
            return
        self._stop_preview_play()
        self.videos.clear()
        self.selected_index = None
        self.playhead_sec = 0.0
        self.cut_in_sec = None
        self.cut_out_sec = None
        self._undo_stack.clear()
        self.btn_undo_cut.configure(state="disabled")
        self._refresh_video_list_ui()
        self._update_selection_label()
        self._update_timeline_and_preview()
        self.status_label.configure(text="목록을 비웠습니다.")
        self.progress_bar.set(0.0)

    def _select_row(self, index: int) -> None:
        if self.is_busy:
            return
        self.selected_index = index
        # 선택한 클립의 시작 지점으로 전체 타임라인 Playhead 이동
        if 0 <= index < len(self.videos):
            clip_start = sum(v.duration_sec for v in self.videos[:index])
            self.playhead_sec = clip_start
            self._update_timeline_and_preview()
        self._refresh_video_list_ui()

    def _refresh_video_list_ui(self) -> None:
        # 구간이 바뀌었을 수 있으므로 리뷰 목록도 함께 동기화
        self._rebuild_review_items()
        self._refresh_review_ui()

        for child in self.list_scroll.winfo_children():
            child.destroy()

        if not self.videos:
            empty_lbl = ctk.CTkLabel(
                self.list_scroll,
                text=(
                    "\n📂 아직 추가된 주행 영상이 없습니다.\n"
                    "상단의 [폴더 통째로 추가] 또는 [영상 파일 추가] 버튼을 눌러 액션캠 영상을 불러오세요.\n"
                ),
                font=ctk.CTkFont(size=13),
                text_color="#94a3b8",
                justify="center",
            )
            empty_lbl.grid(row=0, column=0, padx=20, pady=24)
            self._update_summary()
            return

        skip_mode = self._get_current_skip_mode()
        speed_mult = self._get_current_speed_multiplier()

        for idx, v in enumerate(self.videos):
            is_sel = idx == self.selected_index
            card_bg = "#1e3a5f" if is_sel else "#1e293b"
            border_col = "#38bdf8" if is_sel else "#334155"

            card = ctk.CTkFrame(
                self.list_scroll,
                fg_color=card_bg,
                border_width=2 if is_sel else 1,
                border_color=border_col,
                corner_radius=8,
            )
            card.grid(row=idx, column=0, sticky="ew", padx=4, pady=3)
            card.grid_columnconfigure(1, weight=1)

            card.bind("<Button-1>", lambda _e, i=idx: self._select_row(i))

            num_lbl = ctk.CTkLabel(
                card,
                text=f"#{idx + 1}",
                width=38,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#38bdf8" if is_sel else "#cbd5e1",
            )
            num_lbl.grid(row=0, column=0, rowspan=2, padx=(8, 4), pady=6)
            num_lbl.bind("<Button-1>", lambda _e, i=idx: self._select_row(i))

            top_row = ctk.CTkFrame(card, fg_color="transparent")
            top_row.grid(row=0, column=1, sticky="ew", padx=(4, 10), pady=(6, 2))
            top_row.bind("<Button-1>", lambda _e, i=idx: self._select_row(i))

            fn_lbl = ctk.CTkLabel(
                top_row,
                text=v.filename,
                font=ctk.CTkFont(size=13, weight="bold"),
            )
            fn_lbl.pack(side="left", padx=(0, 10))
            fn_lbl.bind("<Button-1>", lambda _e, i=idx: self._select_row(i))

            dt_str = (
                v.recorded_at.strftime("%Y-%m-%d %H:%M:%S")
                if v.recorded_at
                else "시각 미상"
            )
            meta_lbl = ctk.CTkLabel(
                top_row,
                text=f"🕒 {dt_str} ({v.timestamp_source})  |  {v.width}x{v.height} @ {v.fps:.0f}fps",
                font=ctk.CTkFont(size=11),
                text_color="#94a3b8",
            )
            meta_lbl.pack(side="left")
            meta_lbl.bind("<Button-1>", lambda _e, i=idx: self._select_row(i))

            has_edits_or_analysis = v.analyzed or v.has_manual_cuts or len(v.segments) > 1
            if has_edits_or_analysis:
                est_dur = v.estimated_output_duration(skip_mode, speed_mult)
                parts = []
                if v.stop_duration > 0:
                    parts.append(f"정차 {format_duration(v.stop_duration)}")
                if v.straight_duration > 0:
                    parts.append(f"직진 {format_duration(v.straight_duration)}")
                if v.manual_cut_duration > 0:
                    parts.append(f"수동컷 {format_duration(v.manual_cut_duration)}")
                detail_str = f" ({' / '.join(parts)})" if parts else ""
                dur_text = (
                    f"원본 {format_duration(v.duration_sec)} → 결과 {format_duration(est_dur)}{detail_str}"
                )
                dur_color = "#4ade80"
            else:
                dur_text = f"원본 길이: {format_duration(v.duration_sec)}"
                dur_color = "#cbd5e1"

            dur_lbl = ctk.CTkLabel(
                top_row,
                text=dur_text,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=dur_color,
            )
            dur_lbl.pack(side="right")
            dur_lbl.bind("<Button-1>", lambda _e, i=idx: self._select_row(i))

            tl_canvas = TimelineCanvas(card, height=10)
            tl_canvas.grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=(2, 6))
            tl_canvas.set_data(v.duration_sec, v.segments, has_edits_or_analysis)
            tl_canvas.bind("<Button-1>", lambda _e, i=idx: self._select_row(i))

        self._update_summary()

    def _update_summary(self) -> None:
        if not self.videos:
            self.summary_label.configure(
                text="영상을 추가하면 전체 주행 시간과 예상 단축 시간이 여기에 표시됩니다."
            )
            return

        total_orig = sum(v.duration_sec for v in self.videos)
        skip_mode = self._get_current_skip_mode()
        speed_mult = self._get_current_speed_multiplier()

        total_est = sum(v.estimated_output_duration(skip_mode, speed_mult) for v in self.videos)
        total_stop = sum(v.stop_duration for v in self.videos)
        total_straight = sum(v.straight_duration for v in self.videos)
        total_mcut = sum(v.manual_cut_duration for v in self.videos)
        saved = max(0.0, total_orig - total_est)
        pct = (saved / total_orig * 100.0) if total_orig > 0 else 0.0

        self.summary_label.configure(
            text=(
                f"📊 총 {len(self.videos)}개 영상  |  원본: {format_duration(total_orig)}  →  "
                f"예상 완성 길이: {format_duration(total_est)} ({pct:.0f}% 단축)   "
                f"[정차 {format_duration(total_stop)} | 직진 {format_duration(total_straight)} | 수동 잘라내기 {format_duration(total_mcut)}]"
            )
        )

    # ── 작업 상태 제어 ──

    def _set_busy_state(self, busy: bool, status_msg: Optional[str] = None) -> None:
        self.is_busy = busy
        if not busy:
            self.cancel_requested = False

        state_normal = "disabled" if busy else "normal"
        self.btn_add_folder.configure(state=state_normal)
        self.btn_add_files.configure(state=state_normal)
        self.btn_clear.configure(state=state_normal)
        self.btn_remove.configure(state=state_normal)
        self.btn_up.configure(state=state_normal)
        self.btn_down.configure(state=state_normal)
        self.btn_analyze.configure(state=state_normal)
        self.btn_export.configure(state=state_normal)
        self.btn_cancel.configure(state="normal" if busy else "disabled")

        if status_msg is not None:
            self.status_label.configure(text=status_msg)

    def _on_cancel_task(self) -> None:
        if self.is_busy:
            self.cancel_requested = True
            self.status_label.configure(text="작업 취소 요청 중... 잠시만 기다려 주세요.")

    def _on_open_output_folder(self) -> None:
        if self.last_output_path and self.last_output_path.exists():
            folder = self.last_output_path.parent
            try:
                if os.name == "nt":
                    os.startfile(str(folder))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(folder)])
                else:
                    subprocess.Popen(["xdg-open", str(folder)])
            except Exception:
                pass

    # ── 1단계: 자동 구간 분석 실행 ──

    def _on_start_analysis_only(self) -> None:
        if self.is_busy:
            return
        if not self.videos:
            messagebox.showwarning("알림", "먼저 분석할 주행 영상 파일을 추가해 주세요.")
            return

        cfg = self._get_current_analysis_config()
        if not cfg.detect_stops and not cfg.detect_highway_straight:
            messagebox.showwarning(
                "설정 확인",
                "신호대기 감지와 고속도로 직진 감지 중 최소 하나 이상을 체크해 주세요.",
            )
            return

        self._stop_preview_play()
        self._push_undo_snapshot()
        self._set_busy_state(True, "주행 영상 변화 구간을 분석하는 중...")
        self.progress_bar.set(0.0)

        def worker():
            total = len(self.videos)
            for idx, v in enumerate(self.videos):
                if self.cancel_requested:
                    break

                def on_prog(frac: float, i=idx, name=v.filename):
                    overall = (i + frac) / total
                    self.after(
                        0,
                        lambda o=overall, f=frac, n=name, cur=i + 1: (
                            self.progress_bar.set(o),
                            self.status_label.configure(
                                text=f"분석 중 [{cur}/{total}] {n} ({f * 100:.0f}%)"
                            ),
                        ),
                    )

                old_segs = list(v.segments)
                segs = analyze_video(
                    video_info=v,
                    config=cfg,
                    progress_callback=on_prog,
                    cancel_check=lambda: self.cancel_requested,
                )
                if self.cancel_requested:
                    break
                # 기존에 사용자가 수동으로 지정한 MANUAL_CUT 구간은 보존!
                v.segments = merge_preserving_manual_cuts(old_segs, segs, v.duration_sec)
                v.analyzed = True
                self.after(0, lambda: (self._refresh_video_list_ui(), self._update_timeline_and_preview()))

            def finish():
                if self.cancel_requested:
                    self._set_busy_state(False, "구간 분석이 취소되었습니다.")
                    self.progress_bar.set(0.0)
                else:
                    self.progress_bar.set(1.0)
                    n_items = len(self.review_items)
                    self._set_busy_state(
                        False,
                        f"자동 구간 분석 완료! 검출 구간 {n_items}개 — [🔎 검출 구간 리뷰] 탭의 [▶ 리뷰 시작]으로 하나씩 확인·결정할 수 있습니다.",
                    )
                    if n_items:
                        self.left_tabs.set("🔎 검출 구간 리뷰")
                self._refresh_video_list_ui()
                self._update_timeline_and_preview()

            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    # ── 2단계: 병합 MP4 내보내기 ──

    def _on_start_merge_export(self) -> None:
        if self.is_busy:
            return
        if not self.videos:
            messagebox.showwarning("알림", "먼저 합칠 주행 영상 파일을 추가해 주세요.")
            return

        self._stop_preview_play()
        default_name = f"merged_drive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        initial_dir = str(self.videos[0].path.parent)
        save_path_str = filedialog.asksaveasfilename(
            title="합쳐진 주행 영상을 저장할 위치 선택",
            initialdir=initial_dir,
            initialfile=default_name,
            defaultextension=".mp4",
            filetypes=[("MP4 비디오", "*.mp4")],
        )
        if not save_path_str:
            return

        output_path = Path(save_path_str).resolve()
        if any(v.path.resolve() == output_path for v in self.videos):
            messagebox.showerror(
                "오류",
                "원본 영상 파일과 동일한 이름으로 덮어쓸 수 없습니다. 다른 파일명을 지정해 주세요.",
            )
            return

        analysis_cfg = self._get_current_analysis_config()
        export_cfg = self._get_current_export_config(output_path)

        self._set_busy_state(True, "주행 영상 분석 및 병합 작업을 시작합니다...")
        self.progress_bar.set(0.0)

        def worker():
            try:
                # 1. 아직 자동 분석되지 않은 영상만 자동 분석 수행 (수동 컷은 항상 보존)
                un_analyzed = [
                    (idx, v)
                    for idx, v in enumerate(self.videos)
                    if not v.analyzed and (analysis_cfg.detect_stops or analysis_cfg.detect_highway_straight)
                ]
                total_v = len(self.videos)
                for idx, v in un_analyzed:
                    if self.cancel_requested:
                        raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")

                    def on_ana_prog(frac: float, i=idx, name=v.filename):
                        overall = ((i + frac) / total_v) * 0.35
                        self.after(
                            0,
                            lambda o=overall, f=frac, n=name, cur=i + 1: (
                                self.progress_bar.set(o),
                                self.status_label.configure(
                                    text=f"[1/2단계: 구간 분석 {cur}/{total_v}] {n} ({f * 100:.0f}%)"
                                ),
                            ),
                        )

                    old_segs = list(v.segments)
                    segs = analyze_video(
                        video_info=v,
                        config=analysis_cfg,
                        progress_callback=on_ana_prog,
                        cancel_check=lambda: self.cancel_requested,
                    )
                    if self.cancel_requested:
                        raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")
                    v.segments = merge_preserving_manual_cuts(old_segs, segs, v.duration_sec)
                    v.analyzed = True
                    self.after(0, lambda: (self._refresh_video_list_ui(), self._update_timeline_and_preview()))

                # 2. 분석 및 수동 편집된 구간을 바탕으로 렌더링 및 병합
                def on_render_prog(frac: float, msg: str):
                    overall = 0.35 + frac * 0.65
                    self.after(
                        0,
                        lambda o=overall, m=msg: (
                            self.progress_bar.set(o),
                            self.status_label.configure(text=f"[2/2단계: 영상 병합] {m}"),
                        ),
                    )

                final_path = render_merged_video(
                    videos=self.videos,
                    export_config=export_cfg,
                    progress_callback=on_render_prog,
                    cancel_check=lambda: self.cancel_requested,
                )

                def on_success():
                    self.last_output_path = final_path
                    self.progress_bar.set(1.0)
                    self.btn_open_folder.configure(state="normal")
                    self._set_busy_state(
                        False,
                        f"✅ 병합 완료! 저장된 파일: {final_path}",
                    )
                    messagebox.showinfo(
                        "병합 완료",
                        f"주행 영상 병합이 완료되었습니다!\n\n저장 경로:\n{final_path}",
                    )

                self.after(0, on_success)

            except Exception as exc:
                err_msg = str(exc)

                def on_error():
                    self.progress_bar.set(0.0)
                    self._set_busy_state(False, f"❌ 작업 중단: {err_msg}")
                    if "취소" not in err_msg:
                        messagebox.showerror("오류 발생", err_msg)

                self.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

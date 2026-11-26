"""미리보기 프레임·연속 재생·썸네일을 백그라운드에서 디코딩하는 워커들."""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..ffmpeg_utils import detect_best_gpu_decoder, get_CreationFlags, get_ffmpeg_exe


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


class ThumbnailLoader:
    """
    목록용 썸네일을 백그라운드 스레드(3개)에서 FFmpeg로 뽑아 메모리에 캐시합니다.
    request(...)의 콜백은 작업 스레드에서 호출되므로, 화면 갱신은 콜백 안에서 widget.after(0, ...)로 넘기세요.
    """

    def __init__(self, workers: int = 3) -> None:
        self._cache: Dict[Tuple[str, int, int, int], np.ndarray] = {}
        self._pending: Dict[Tuple[str, int, int, int], List[Callable[[np.ndarray], None]]] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Tuple[Tuple[str, int, int, int], Path, float, int, int]]" = queue.Queue()
        self._stop = False
        for _ in range(workers):
            threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _key(path: Path, t: float, w: int, h: int) -> Tuple[str, int, int, int]:
        return str(path), int(round(t * 10)), w, h

    def get_cached(self, path: Path, t: float, w: int, h: int) -> Optional[np.ndarray]:
        return self._cache.get(self._key(path, t, w, h))

    def request(self, path: Path, t: float, w: int, h: int, callback: Callable[[np.ndarray], None]) -> None:
        key = self._key(path, t, w, h)
        with self._lock:
            if key in self._cache:
                arr = self._cache[key]
                callback(arr)
                return
            if key in self._pending:
                self._pending[key].append(callback)
                return
            self._pending[key] = [callback]
        self._queue.put((key, path, t, w, h))

    def close(self) -> None:
        self._stop = True

    def _run(self) -> None:
        ff = get_ffmpeg_exe()
        while not self._stop:
            try:
                key, path, t, w, h = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            arr: Optional[np.ndarray] = None
            try:
                res = subprocess.run(
                    [ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{max(0.0, t):.3f}",
                     "-i", str(path), "-frames:v", "1",
                     "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
                     "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, creationflags=get_CreationFlags(),
                )
                if len(res.stdout) >= w * h * 3:
                    arr = np.frombuffer(res.stdout[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()
            except Exception:
                arr = None
            if arr is None:
                arr = np.full((h, w, 3), 30, dtype=np.uint8)
            with self._lock:
                self._cache[key] = arr
                callbacks = self._pending.pop(key, [])
            for cb in callbacks:
                try:
                    cb(arr)
                except Exception:
                    pass

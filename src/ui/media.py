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

    def close(self, wait: float = 1.5) -> None:
        """작업 스레드를 멈추고 OpenCV 디코더를 그 스레드에서 해제 (종료 시 비정상 종료 방지)."""
        self._stop = True
        self._event.set()
        if wait > 0 and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=wait)

    def _run(self) -> None:
        try:
            self._loop()
        finally:
            cap, self._cap = self._cap, None
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _loop(self) -> None:
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
    fps: float = 0.0


@dataclass
class PlayPiece:
    """연속 재생할 한 조각: 한 영상 파일의 [local_start, local_end) 구간. tag는 쇼츠 미리보기의 컷 순번."""
    clip: PlaybackClip
    local_start: float
    local_end: Optional[float] = None  # None = 파일 끝까지
    tag: int = 0

    @property
    def length(self) -> float:
        end = self.clip.duration if self.local_end is None else min(self.clip.duration, self.local_end)
        return max(0.0, end - self.local_start)


def build_pieces(clips: List[PlaybackClip], g_start: float, stop_at: Optional[float], tag: int = 0) -> List[PlayPiece]:
    """전체 타임라인의 [g_start, stop_at) 구간을 영상 파일별 조각으로 나눕니다."""
    pieces: List[PlayPiece] = []
    for c in clips:
        c0, c1 = c.global_start, c.global_start + c.duration
        if c1 <= g_start + 0.01:
            continue
        if stop_at is not None and c0 >= stop_at - 0.01:
            break
        local_end = None if stop_at is None or stop_at >= c1 else stop_at - c0
        pieces.append(PlayPiece(c, max(0.0, g_start - c0), local_end, tag))
    return pieces


class _Decoder:
    """
    FFmpeg 디코더 1개 + 읽기 스레드. 재생 시각보다 앞서 프레임을 디코딩해 큐(약 0.5초 분량)에 쌓아 두므로
    디코딩이 순간적으로 느려져도(키프레임·디스크 지연 등) 화면이 멈추지 않습니다.
    """

    QUEUE_BYTES = 48 * 1024 * 1024

    def __init__(self, piece: PlayPiece, out_fps: float, box_w: int, box_h: int, hw: str, skip_noref: bool) -> None:
        clip = piece.clip
        scale = min(box_w / max(1, clip.width), box_h / max(1, clip.height))
        self.w = max(2, int(clip.width * scale) // 2 * 2)
        self.h = max(2, int(clip.height * scale) // 2 * 2)
        self.piece = piece
        self.noref = skip_noref
        self.got_any = False
        self._closed = False
        frame_bytes = self.w * self.h * 3
        self.q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(
            maxsize=max(3, min(16, self.QUEUE_BYTES // max(1, frame_bytes))))

        cmd = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if skip_noref:
            # 60fps 원본은 화면 표시(30fps)에 필요 없는 비참조 프레임 디코딩을 건너뜀 → 디코딩 부하 약 절반
            cmd += ["-skip_frame", "noref"]
        if hw == "cuda":
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif hw != "none":
            cmd += ["-hwaccel", hw]  # d3d11va / videotoolbox / auto: GPU 디코딩 후 자동으로 메모리로 가져옴
        cmd += ["-ss", f"{piece.local_start:.3f}"]
        if piece.local_end is not None:
            cmd += ["-t", f"{max(0.1, piece.local_end - piece.local_start + 0.25):.3f}"]
        cmd += ["-i", str(clip.path), "-an", "-sn", "-dn"]
        if hw == "cuda":
            vf = f"fps={out_fps:.4f},scale_cuda={self.w}:{self.h}:format=nv12,hwdownload,format=nv12"
        else:
            vf = f"fps={out_fps:.4f},scale={self.w}:{self.h}:flags=area"
        cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        try:
            self.proc: Optional[subprocess.Popen] = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 2,
                creationflags=get_CreationFlags(),
            )
        except Exception:
            self.proc = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        frame_bytes = self.w * self.h * 3
        try:
            out = self.proc.stdout if self.proc is not None else None
            while out is not None and not self._closed:
                raw = out.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                self.got_any = True
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3))
                if not self._put(arr):
                    return
        except Exception:
            pass
        self._put(None)  # 끝 표시

    def _put(self, item: Optional[np.ndarray]) -> bool:
        while not self._closed:
            try:
                self.q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def failed(self) -> bool:
        """프레임 없이 오류로 끝났는지 (하드웨어 디코딩 실패 판단용)."""
        if self.got_any or self.proc is None:
            return self.proc is None
        try:
            return self.proc.wait(timeout=1.0) != 0
        except Exception:
            return True

    def close(self) -> None:
        self._closed = True
        proc, self.proc = self.proc, None
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


class PlaybackWorker:
    """
    실제 동영상 플레이어처럼 끊김 없이 재생하기 위한 스트리밍 재생 워커.
    - 재생 위치부터 미리보기 해상도 프레임을 FFmpeg로 연속 디코딩(GPU 디코딩 우선)하여 실제 시간에 맞춰 내보냅니다.
    - 읽기 스레드가 약 0.5초 분량을 미리 디코딩해 두고, 다음 조각(다음 영상·다음 컷)도 끝나기 전에 미리 엽니다.
    - 프레임이 늦게 도착해도 버리지 않고 바로 보여준 뒤 재생 기준 시각을 미룹니다(뚝뚝 끊김 대신 잠깐 느려짐).
      계속 밀리면 비참조 프레임 건너뛰기를 켜고 그 위치부터 다시 디코딩합니다.
    """

    DISPLAY_FPS = 30.0
    PREFETCH_SEC = 1.5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session = 0
        self._stop_evt: Optional[threading.Event] = None
        self._latest: Optional[Tuple[int, float, np.ndarray, int]] = None
        self._finished_session: int = -1
        self._hw: Optional[str] = None      # 첫 재생 때 감지: cuda / d3d11va / videotoolbox / auto / none
        self._force_noref = False           # 디코딩이 실시간을 못 따라간 적이 있으면 이후 계속 사용
        self.stats = {"shown": 0, "late": 0, "restarts": 0, "hw": ""}

    @property
    def session(self) -> int:
        return self._session

    def start(self, pieces: List[PlayPiece], rate: float, box_w: int, box_h: int) -> int:
        self.stop()
        with self._lock:
            self._session += 1
            sid = self._session
            self._latest = None
        evt = threading.Event()
        self._stop_evt = evt
        threading.Thread(target=self._run, args=(sid, evt, list(pieces), max(0.25, rate), box_w, box_h),
                         daemon=True).start()
        return sid

    def stop(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()
            self._stop_evt = None

    def take_latest(self) -> Optional[Tuple[int, float, np.ndarray, int]]:
        with self._lock:
            item = self._latest
            self._latest = None
        return item

    def has_pending(self) -> bool:
        with self._lock:
            return self._latest is not None

    def is_finished(self, sid: int) -> bool:
        return self._finished_session == sid

    def _detect_hw(self) -> str:
        if self._hw is None:
            try:
                mode = detect_best_gpu_decoder()
            except Exception:
                mode = "cpu"
            self._hw = {"cuda_scale": "cuda", "d3d11va": "d3d11va", "videotoolbox": "videotoolbox",
                        "vaapi": "auto"}.get(mode, "none")
        self.stats["hw"] = self._hw
        return self._hw

    def _open(self, piece: PlayPiece, rate: float, box_w: int, box_h: int) -> _Decoder:
        noref = self._force_noref or rate >= 2.0 or piece.clip.fps >= 48.0
        return _Decoder(piece, self.DISPLAY_FPS / rate, box_w, box_h, self._detect_hw(), noref)

    def _run(self, sid: int, evt: threading.Event, pieces: List[PlayPiece], rate: float, box_w: int, box_h: int) -> None:
        out_fps = self.DISPLAY_FPS / rate  # 원본 1초당 디코딩할 프레임 수
        frame_dt = 1.0 / self.DISPLAY_FPS
        nxt: Optional[_Decoder] = None
        dec: Optional[_Decoder] = None
        i = 0
        try:
            while i < len(pieces) and not evt.is_set():
                piece = pieces[i]
                dec, nxt = (nxt if nxt is not None else self._open(piece, rate, box_w, box_h)), None
                wall0: Optional[float] = None
                anchor = 0.0
                idx = 0
                late_window: List[Tuple[float, float]] = []  # (벽시계, 밀린 시간)
                restart_at: Optional[float] = None
                while not evt.is_set():
                    if nxt is None and i + 1 < len(pieces) and piece.length - idx / out_fps < self.PREFETCH_SEC:
                        nxt = self._open(pieces[i + 1], rate, box_w, box_h)
                    try:
                        arr = dec.q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if arr is None:
                        break
                    local = piece.local_start + idx / out_fps
                    idx += 1
                    if piece.local_end is not None and local >= piece.local_end - 1e-3:
                        break
                    media_t = piece.clip.global_start + local
                    now = time.perf_counter()
                    if wall0 is None:
                        wall0, anchor = now, media_t
                    target = wall0 + (media_t - anchor) / rate
                    if now < target:
                        if evt.wait(target - now):
                            break
                    elif now - target > frame_dt:
                        # 늦게 도착한 프레임도 버리지 않고 바로 표시, 기준 시각을 그만큼 미룸
                        lag = now - target
                        wall0 += lag
                        self.stats["late"] += 1
                        late_window = [(t, l) for t, l in late_window if now - t < 2.0] + [(now, lag)]
                        remaining = piece.length - (local - piece.local_start)
                        if not dec.noref and remaining > 1.0 and sum(l for _t, l in late_window) > 0.5:
                            restart_at = local
                    with self._lock:
                        if self._session == sid:
                            self._latest = (sid, media_t, arr, piece.tag)
                    self.stats["shown"] += 1
                    if restart_at is not None:
                        break

                if evt.is_set():
                    break
                if restart_at is not None:
                    # 실시간을 계속 못 따라감 → 비참조 프레임 건너뛰기를 켜고 현재 위치부터 다시 디코딩
                    self._force_noref = True
                    self.stats["restarts"] += 1
                    dec.close()
                    if nxt is not None:
                        nxt.close()
                        nxt = None
                    pieces[i] = PlayPiece(piece.clip, restart_at + 1.0 / out_fps, piece.local_end, piece.tag)
                    continue
                if not dec.got_any and self._hw not in (None, "none") and piece.length > 0.2 and dec.failed():
                    # GPU 디코딩 실패 → CPU 디코딩으로 같은 조각을 다시 재생
                    self._hw = "none"
                    self.stats["restarts"] += 1
                    dec.close()
                    if nxt is not None:
                        nxt.close()
                        nxt = None
                    continue
                dec.close()
                dec = None
                i += 1
        finally:
            for d in (dec, nxt):
                if d is not None:
                    d.close()
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

"""공통 색상·글꼴·시간 표시·이미지 변환 도우미."""
from __future__ import annotations

import tkinter as tk
from typing import Union

import customtkinter as ctk
import numpy as np

# 구간 색상
COLOR_NORMAL = "#22c55e"      # 초록: 일반 주행 (1x)
COLOR_STOP = "#f59e0b"        # 주황: 신호대기 / 정차
COLOR_STRAIGHT = "#a855f7"    # 보라: 장시간 고속도로 직진
COLOR_MANUAL_CUT = "#ef4444"  # 빨강: 수동 잘라내기 (삭제 구간)
COLOR_UNANALYZED = "#475569"
COLOR_HIGHLIGHT = "#22d3ee"   # 하늘색: 하이라이트 쇼츠 장면

# 화면 색상
BG_APP = "#0b1120"
BG_PANEL = "#111827"
BG_CARD = "#1e293b"
BG_DEEP = "#020617"
FG_MUTED = "#94a3b8"
FG_SOFT = "#cbd5e1"
ACCENT = "#0ea5e9"
ACCENT_HOVER = "#0284c7"
PRIMARY = "#16a34a"
PRIMARY_HOVER = "#15803d"
SHORTS = "#db2777"
SHORTS_HOVER = "#be185d"
NEUTRAL = "#334155"
NEUTRAL_HOVER = "#475569"


def font(size: int = 12, bold: bool = False) -> ctk.CTkFont:
    return ctk.CTkFont(size=size, weight="bold" if bold else "normal")


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


# ── 프레임 고속 갱신 ──
# PhotoImage.put(P6 데이터)는 Tk가 픽셀마다 PPM을 해석해서 메인(UI) 스레드에서 720p 한 장에 약 17ms, 1080p는 약 40ms+가 듭니다.
# 재생 중 프레임마다 이 비용을 내면 30fps(33ms 예산)를 못 맞추고 화면이 뚝뚝 끊깁니다.
# 같은 Tk가 제공하는 Tk_PhotoPutBlock(C 함수)을 ctypes로 직접 불러 RGB 버퍼를 그대로 복사하면 약 4배 빠릅니다.
# 별도 확장 DLL(_imagingtk 등)이 아니라 파이썬이 이미 로드한 tk DLL의 함수만 쓰므로 WDAC 환경에서도 동작하며,
# 어떤 이유로든 실패하면 기존 PPM 방식으로 자동 전환합니다.
import ctypes as _ct
import glob as _glob
import os as _os
import sys as _sys


class _TkImageBlock(_ct.Structure):
    _fields_ = [("pixelPtr", _ct.c_void_p), ("width", _ct.c_int), ("height", _ct.c_int),
                ("pitch", _ct.c_int), ("pixelSize", _ct.c_int), ("offset", _ct.c_int * 4)]


_fast_put = None      # (find_photo, put_block) 또는 False(사용 불가)


def _load_fast_put():
    global _fast_put
    if _fast_put is not None:
        return _fast_put
    _fast_put = False
    try:
        names = ["tk86t.dll", "tk86.dll", "tk90.dll"] if _sys.platform == "win32" else []
        dll = None
        base = _os.path.dirname(_sys.executable)
        for n in names:
            for d in (_os.path.join(base, "DLLs"), base):
                p = _os.path.join(d, n)
                if _os.path.exists(p):
                    dll = _ct.CDLL(p)
                    break
            if dll is not None:
                break
        if dll is None:  # macOS/Linux: 이미 로드된 tk 라이브러리를 찾음
            import ctypes.util
            lib = ctypes.util.find_library("tk8.6") or ctypes.util.find_library("tk")
            dll = _ct.CDLL(lib) if lib else None
        if dll is None:
            return _fast_put
        find = dll.Tk_FindPhoto
        find.restype = _ct.c_void_p
        find.argtypes = [_ct.c_void_p, _ct.c_char_p]
        put = dll.Tk_PhotoPutBlock
        put.restype = _ct.c_int
        put.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.POINTER(_TkImageBlock),
                        _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int]
        _fast_put = (find, put)
    except Exception:
        _fast_put = False
    return _fast_put


def photo_put_rgb(photo: tk.PhotoImage, arr: np.ndarray) -> None:
    """PhotoImage(photo)의 내용을 (H, W, 3) uint8 RGB 배열로 교체합니다. 크기는 photo와 같아야 합니다."""
    fast = _load_fast_put()
    if fast and arr.ndim == 3 and arr.shape[2] == 3:
        try:
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            h, w = arr.shape[:2]
            tk_app = photo.tk
            interp = tk_app.interpaddr()
            handle = fast[0](interp, str(photo).encode("ascii"))
            if handle:
                blk = _TkImageBlock(arr.ctypes.data, w, h, w * 3, 3, (0, 1, 2, 0))
                # compRule 1 = TK_PHOTO_COMPOSITE_SET (알파 합성 없이 그대로 덮어쓰기)
                if fast[1](interp, handle, _ct.byref(blk), 0, 0, w, h, 1) == 0:
                    return
        except Exception:
            pass
        globals()["_fast_put"] = False  # 한 번 실패하면 계속 시도하지 않음
    h, w = arr.shape[:2]
    photo.put(f"P6 {w} {h} 255\n".encode("ascii") + np.ascontiguousarray(arr).tobytes())


def enable_high_res_timer() -> None:
    """Windows 기본 타이머 해상도는 약 15.6ms라 30fps 재생 프레임 표시 시각이 ±15ms 흔들립니다(31·31·46ms 식으로 불규칙).
    1ms 해상도로 올리면 프레임 간격이 고르게 유지됩니다. 프로세스가 끝나면 자동으로 원복됩니다."""
    if _sys.platform != "win32":
        return
    try:
        _ct.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass
    try:
        # Windows 11은 창이 가려지거나 포커스가 없는 프로세스의 타이머 해상도 요청을 무시해 15.6ms 단위로 되돌립니다.
        # 이 무시 동작을 끄면 재생 중 프레임 표시 시각이 15.6ms 배수(31·46ms)로 튀지 않습니다.
        class _PowerThrottling(_ct.Structure):
            _fields_ = [("Version", _ct.c_ulong), ("ControlMask", _ct.c_ulong), ("StateMask", _ct.c_ulong)]
        st = _PowerThrottling(1, 0x4, 0x0)  # PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION 해제
        k32 = _ct.windll.kernel32
        k32.GetCurrentProcess.restype = _ct.c_void_p
        k32.SetProcessInformation(_ct.c_void_p(k32.GetCurrentProcess()), 4, _ct.byref(st), _ct.sizeof(st))  # ProcessPowerThrottling = 4
    except Exception:
        pass

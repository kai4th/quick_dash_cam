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

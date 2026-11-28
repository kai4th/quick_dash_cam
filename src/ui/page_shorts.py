"""
④ 쇼츠: 하이라이트 장면 찾기(알고리즘 / Claude) → 컷 확인·사용/제외 → 컷별 기울기·소실점 맞춤 → 쇼츠 만들기.

화면 구성: 왼쪽 장면 목록 · 가운데 큰 미리보기(맞춤 결과, 클릭 = 소실점) · 오른쪽 설정 + [쇼츠 만들기]
"""
from __future__ import annotations

import os
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np

from ..ai_director import AiDirectorError, ai_select_highlights, find_claude_cli
from ..analyzer import analyze_video
from ..highlights import (
    clip_timestamp_text,
    ensure_appearance,
    reframe_affine,
    reframe_geometry,
    reframe_inverse_point,
    render_highlight_shorts,
    select_highlights,
    vp_target_for,
)
from ..models import (
    AiDirectorMode,
    HighlightClip,
    HighlightConfig,
    SegmentType,
    ShortsAudioMode,
    ShortsExportConfig,
    ShortsLayout,
    ShortsTransition,
    VideoQualityPreset,
    merge_preserving_manual_cuts,
)
from .player import VideoPlayer
from .theme import (
    ACCENT, ACCENT_HOVER, BG_CARD, FG_MUTED, FG_SOFT, NEUTRAL, NEUTRAL_HOVER, SHORTS, SHORTS_HOVER,
    font, format_duration, make_photo_image,
)
from .widgets import Section, card, field_label, make_tree, stylize_entry_isolation, tree_selected_index

THUMB_W, THUMB_H = 80, 45
ROLL_LIMIT = 10.0


def slider_to_roll(value: float) -> float:
    """기울기 슬라이더 위치 → 컷에 저장되는 기울기(roll_adjust, +=시계). 슬라이더 방향만 뒤집고 저장 의미는 그대로."""
    return round(-float(value), 1) + 0.0  # + 0.0: -0.0을 0.0으로


def fmt_roll(roll: float) -> str:
    """화면에 보이는 기울기 숫자: 슬라이더 방향 기준(오른쪽 = +). 저장값 roll_adjust와는 부호가 반대."""
    return f"{-float(roll) + 0.0:+.1f}°"


def roll_to_slider(roll: float) -> float:
    return -float(roll) + 0.0

AI_MODES = {"Claude Code (구독 계정)": AiDirectorMode.CLAUDE_CODE, "Claude API (API 키)": AiDirectorMode.CLAUDE_API,
            "알고리즘만 (빠름·오프라인)": AiDirectorMode.OFF}
LENGTHS = {"15초": 15.0, "30초": 30.0, "45초": 45.0, "60초": 60.0, "90초": 90.0}
CLIP_LENGTHS = {"2초": 2.0, "3초": 3.0, "4초": 4.0, "5초": 5.0}
LAYOUTS = {"세로 9:16 · 흐림 배경": ShortsLayout.VERTICAL_BLUR, "세로 9:16 · 꽉 채우기": ShortsLayout.VERTICAL_CROP,
           "가로 16:9": ShortsLayout.LANDSCAPE}
RESOLUTIONS = {"자동 (원본 화질)": "auto", "4K (2160×3840)": "2160p", "1440p": "1440p", "1080p": "1080p", "720p": "720p"}
FPS = {"원본 fps": 0.0, "60fps": 60.0, "30fps": 30.0}
QUALITY = {"최고화질 (CRF 14)": VideoQualityPreset.LOSSLESS_LIKE, "고화질 (CRF 18)": VideoQualityPreset.HIGH,
           "표준 (CRF 22)": VideoQualityPreset.BALANCED}
AUDIO = {"원본 주행 소리": ShortsAudioMode.ORIGINAL, "원본 + 배경음악": ShortsAudioMode.BGM_MIX,
         "배경음악만": ShortsAudioMode.BGM, "무음": ShortsAudioMode.MUTE}
TRANSITIONS = {"크로스 디졸브": ShortsTransition.DISSOLVE, "검은 화면 페이드": ShortsTransition.FADE_BLACK,
               "바로 컷": ShortsTransition.CUT}
TRANS_LENGTHS = {"0.3초": 0.3, "0.5초": 0.5, "0.8초": 0.8, "1.0초": 1.0}


def compute_highlights(store, acfg, hcfg: HighlightConfig, mode: AiDirectorMode, api_key: Optional[str], report, cancelled
                       ) -> Tuple[List[HighlightClip], Optional[str]]:
    """(작업 스레드) 미분석 영상 분석 → 장면 모습 썸네일 → 알고리즘/Claude 선정. 반환: (컷 목록, AI 실패 메시지)."""
    videos = list(store.videos)
    n = max(1, len(videos))
    use_ai = mode != AiDirectorMode.OFF
    share = 0.5 if use_ai else 1.0
    for i, v in enumerate(videos):
        if cancelled():
            raise RuntimeError("취소")
        if v.feature_cache is None:
            old = list(v.segments)
            segs = analyze_video(v, acfg, progress_callback=lambda f, i=i, nm=v.filename: report(
                (i + f * 0.9) / n * share, f"[하이라이트] 주행 장면 분석 {i + 1}/{n}: {nm} ({f * 100:.0f}%)"),
                cancel_check=cancelled)
            if cancelled():
                raise RuntimeError("취소")
            if not v.analyzed:
                v.segments = merge_preserving_manual_cuts(old, segs, v.duration_sec)
                v.analyzed = True
        report((i + 0.95) / n * share, f"[하이라이트] 장면 모습 확인 {i + 1}/{n}: {v.filename}")
        ensure_appearance(v, cancel_check=cancelled)
    if cancelled():
        raise RuntimeError("취소")
    if use_ai:
        try:
            clips = ai_select_highlights(videos, hcfg, mode, api_key=api_key,
                                         progress=lambda f, m: report(share + (1 - share) * f, m), cancel_check=cancelled)
            return clips, None
        except AiDirectorError as exc:
            if cancelled():
                raise RuntimeError("취소") from exc
            err = str(exc)
        except Exception as exc:  # noqa: BLE001
            if cancelled():
                raise RuntimeError("취소") from exc
            err = f"{type(exc).__name__}: {exc}"
        report(1.0, "[하이라이트] AI 선정 실패 — 알고리즘으로 고르는 중…")
        return select_highlights(videos, hcfg), err
    report(1.0, "[하이라이트] 장면을 고르는 중…")
    return select_highlights(videos, hcfg), None


class ShortsPage(ctk.CTkFrame):
    SHORTCUTS = (
        "컷\n  Space  이 컷 재생 / 멈춤\n  X  사용 / 제외\n  PgUp / PgDn  이전 / 다음 컷\n  , / .  컷을 0.5초 앞 / 뒤로\n\n"
        "맞춤\n  ← →  기울기 0.1°   (Shift: 1°)\n  미리보기 클릭  소실점 지정\n\n"
        "목록\n  ↑ ↓  컷 선택 · 더블클릭  사용 / 제외"
    )
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.store = app.store
        self.index: Optional[int] = None
        self.bgm_path: Optional[Path] = None
        self._api_key: Optional[str] = None
        self._consent = False
        self._thumbs: Dict[Tuple[str, int], tk.PhotoImage] = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_list()
        self._build_center()
        self._build_settings()

        self.store.subscribe("videos", self._on_videos)
        self.store.subscribe("highlights", self.refresh_list)
        self.store.subscribe("busy", self._on_busy)
        self._register_settings()
        self.refresh_list()

    # ── 왼쪽: 장면 목록 ──

    def _build_list(self) -> None:
        outer, body = card(self, "장면", "✔ 사용 컷이 쇼츠에 들어갑니다. X 키로 사용/제외.", wrap=300)
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        outer.configure(width=330)
        body.grid_rowconfigure(2, weight=1)
        self.btn_find = ctk.CTkButton(body, text="✨ 하이라이트 찾기", height=42, font=font(15, True),
                                      fg_color="#7c3aed", hover_color="#6d28d9", command=self.find_highlights)
        self.btn_find.grid(row=0, column=0, sticky="ew")
        self.list_info = ctk.CTkLabel(body, text="", font=font(12), text_color=FG_SOFT, anchor="w", justify="left")
        self.list_info.grid(row=1, column=0, sticky="ew", pady=(8, 6))
        cols = [("use", "사용", 40, "center", False), ("time", "위치", 56, "center", False),
                ("desc", "장면", 120, "w", True), ("score", "점수", 50, "center", False), ("adj", "맞춤", 56, "center", False)]
        self.tree, box = make_tree(body, cols, "Shorts.Treeview", row_height=THUMB_H + 8, with_image=True,
                                   image_width=THUMB_W + 10)
        box.grid(row=2, column=0, sticky="nsew")
        self.tree.tag_configure("on", foreground="#67e8f9")
        self.tree.tag_configure("off", foreground="#64748b")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", lambda _e: self.toggle_use())
        foot = ctk.CTkFrame(body, fg_color="transparent")
        foot.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        foot.grid_columnconfigure(0, weight=1)
        foot.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(foot, text="⊘ 사용 / 제외 [X]", height=34, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                      command=self.toggle_use).grid(row=0, column=0, padx=(0, 3), sticky="ew")
        ctk.CTkButton(foot, text="▶▶ 쇼츠 미리보기", height=34, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self.preview_all).grid(row=0, column=1, padx=(3, 0), sticky="ew")

    # ── 가운데: 큰 미리보기 + 컷 맞춤 ──

    def _build_center(self) -> None:
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=6)
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(1, weight=1)
        self.cut_title = ctk.CTkLabel(center, text="", font=font(14, True), anchor="w")
        self.cut_title.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.player = VideoPlayer(center, show_transport=False,
                                  empty_text="왼쪽에서 [✨ 하이라이트 찾기]를 누르면\n고른 장면이 여기에 크게 표시됩니다")
        self.player.grid(row=1, column=0, sticky="nsew")
        self.player.frame_transform = self._transform_frame
        self.player.overlay = self._draw_overlay
        self.player.on_click = self._on_player_click
        self.player.on_segment_start = self._on_preview_segment
        self.player.on_play_state = self._on_play_state

        ctl = ctk.CTkFrame(center, fg_color=BG_CARD, corner_radius=10)
        ctl.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ctl.grid_columnconfigure(0, weight=1)

        # 1줄: 재생 · 컷 이동
        r1 = ctk.CTkFrame(ctl, fg_color="transparent")
        r1.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        for col in range(3):
            r1.grid_columnconfigure(col, weight=1)
        self.btn_play = ctk.CTkButton(r1, text="▶ 이 컷 재생 [Space]", height=34, font=font(13, True),
                                      fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.play_cut)
        self.btn_play.grid(row=0, column=0, padx=(0, 3), sticky="ew")
        ctk.CTkButton(r1, text="◀ 이전 컷 [PgUp]", height=34, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                      command=lambda: self.step_cut(-1)).grid(row=0, column=1, padx=3, sticky="ew")
        ctk.CTkButton(r1, text="다음 컷 [PgDn] ▶", height=34, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                      command=lambda: self.step_cut(1)).grid(row=0, column=2, padx=(3, 0), sticky="ew")

        # 2줄: 컷 범위 (앞뒤로 옮기기 · 길이 조절)
        rr = ctk.CTkFrame(ctl, fg_color="transparent")
        rr.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        rr.grid_columnconfigure(5, weight=1)
        ctk.CTkLabel(rr, text="컷 범위", font=font(13, True), anchor="w").grid(row=0, column=0, padx=(0, 6))
        for col, (txt, cmd) in enumerate((("◀ 0.5초 [,]", lambda: self.shift_cut(-0.5)), ("0.5초 ▶ [.]", lambda: self.shift_cut(0.5)),
                                          ("길이 −0.5초", lambda: self.resize_cut(-0.5)), ("길이 +0.5초", lambda: self.resize_cut(0.5))), start=1):
            ctk.CTkButton(rr, text=txt, width=84, height=30, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                          command=cmd).grid(row=0, column=col, padx=1)
        self.range_label = ctk.CTkLabel(rr, text="", font=font(12), text_color=FG_SOFT, anchor="e")
        self.range_label.grid(row=0, column=5, sticky="ew", padx=(6, 0))

        # 3줄: 기울기
        r2 = ctk.CTkFrame(ctl, fg_color="transparent")
        r2.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        r2.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(r2, text="기울기", font=font(13, True), anchor="w").grid(row=0, column=0, padx=(0, 6))
        # 슬라이더는 오른쪽으로 밀수록 반시계(↺, 값 -), 왼쪽이 시계(↻, 값 +): 저장값(roll_adjust)의 부호는 그대로이고 화면 방향만 반대.
        # 슬라이더 양옆 버튼도 같은 방향이 되도록 왼쪽 = ↻, 오른쪽 = ↺.
        for col, (txt, d) in ((1, ("↻1°", 1.0)), (2, ("↻.1°", 0.1))):
            ctk.CTkButton(r2, text=txt, width=48, height=30, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                          command=lambda d=d: self.rotate(d)).grid(row=0, column=col, padx=1)
        self.roll_slider = ctk.CTkSlider(r2, from_=-ROLL_LIMIT, to=ROLL_LIMIT, number_of_steps=int(ROLL_LIMIT * 20),
                                         command=self._on_roll_slider)
        self.roll_slider.set(0.0)
        self.roll_slider.grid(row=0, column=3, sticky="ew", padx=6)
        for col, (txt, d) in ((4, ("↺.1°", -0.1)), (5, ("↺1°", -1.0))):
            ctk.CTkButton(r2, text=txt, width=48, height=30, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                          command=lambda d=d: self.rotate(d)).grid(row=0, column=col, padx=1)
        self.roll_value = ctk.CTkLabel(r2, text="+0.0°", font=font(18, True), text_color="#facc15", width=70)
        self.roll_value.grid(row=0, column=6, padx=(6, 0))
        ctk.CTkButton(r2, text="전체 적용", width=72, height=30, fg_color="#1e293b", hover_color=NEUTRAL,
                      border_width=1, border_color=NEUTRAL_HOVER, command=self.apply_roll_all).grid(row=0, column=7, padx=(4, 0))

        # 4줄: 소실점
        r3 = ctk.CTkFrame(ctl, fg_color="transparent")
        r3.grid(row=3, column=0, sticky="ew", padx=10, pady=4)
        r3.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(r3, text="소실점", font=font(13, True), anchor="w").grid(row=0, column=0, padx=(0, 6))
        self.vp_label = ctk.CTkLabel(r3, text="", font=font(12), anchor="w")
        self.vp_label.grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(r3, text="지우기", width=64, height=30, fg_color="#1e293b", hover_color=NEUTRAL,
                      border_width=1, border_color=NEUTRAL_HOVER, command=self.clear_vp).grid(row=0, column=2, padx=2)
        ctk.CTkButton(r3, text="이 컷 초기화", width=96, height=30, fg_color="#1e293b", hover_color=NEUTRAL,
                      border_width=1, border_color=NEUTRAL_HOVER, command=self.reset_adjust).grid(row=0, column=3, padx=2)

        # 5줄: 보기 옵션 · 확대 정보
        r4 = ctk.CTkFrame(ctl, fg_color="transparent")
        r4.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 2))
        self.output_preview_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(r4, text="📱 결과 화면 미리보기", variable=self.output_preview_var, command=self.player.redraw,
                        checkbox_width=18, checkbox_height=18).pack(side="left", padx=(0, 10))
        self.show_result_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(r4, text="맞춤 결과 보기", variable=self.show_result_var, command=self.player.redraw,
                        checkbox_width=18, checkbox_height=18).pack(side="left", padx=(0, 10))
        self.grid_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(r4, text="격자", variable=self.grid_var, command=self.player.redraw,
                        checkbox_width=18, checkbox_height=18).pack(side="left")
        self.zoom_label = ctk.CTkLabel(r4, text="", font=font(12), text_color=FG_MUTED, anchor="e")
        self.zoom_label.pack(side="right")
        ctk.CTkLabel(ctl, text="💡 화면 클릭 = 소실점(도로가 모이는 곳) · 하늘색 원 = 모든 컷이 맞춰질 위치 · ← → 기울기",
                     font=font(11), text_color=FG_MUTED, anchor="w").grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 8))

    # ── 오른쪽: 설정 + 만들기 ──

    def _build_settings(self) -> None:
        right = ctk.CTkFrame(self, fg_color="transparent", width=300)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        scroll = ctk.CTkScrollableFrame(right, fg_color="transparent", width=284)
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        def menu(parent, values, default, row, col=0, colspan=2, cmd=None):
            var = ctk.StringVar(value=default)
            ctk.CTkOptionMenu(parent, values=list(values), variable=var, height=32,
                              command=(lambda _v: cmd()) if cmd else None).grid(row=row, column=col, columnspan=colspan,
                                                                                 sticky="ew", pady=(2, 0), padx=(0, 2))
            return var

        s1 = Section(scroll, "장면 선정")
        s1.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        field_label(s1.body, "고르는 방법", 0)
        self.ai_var = menu(s1.body, AI_MODES, "Claude Code (구독 계정)", 1, cmd=self._on_selection_setting)
        field_label(s1.body, "쇼츠 길이", 2, 0, 1)
        field_label(s1.body, "컷 하나 길이", 2, 1, 1)
        self.len_var = menu(s1.body, LENGTHS, "30초", 3, 0, 1, cmd=self._on_selection_setting)
        self.clip_var = menu(s1.body, CLIP_LENGTHS, "3초", 3, 1, 1, cmd=self._on_selection_setting)
        ctk.CTkLabel(s1.body, text="컷 안에서는 차분하고 컷끼리는 서로 다른 풍경을 고릅니다.\nClaude는 후보 사진을 보고 최종 장면을 정합니다.",
                     font=font(11), text_color=FG_MUTED, justify="left", anchor="w", wraplength=250).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        s2 = Section(scroll, "화면")
        s2.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        field_label(s2.body, "레이아웃", 0)
        self.layout_var = menu(s2.body, LAYOUTS, "세로 9:16 · 흐림 배경", 1, cmd=self.player.redraw)
        field_label(s2.body, "해상도", 2, 0, 1)
        field_label(s2.body, "프레임레이트", 2, 1, 1)
        self.res_var = menu(s2.body, RESOLUTIONS, "자동 (원본 화질)", 3, 0, 1)
        self.fps_var = menu(s2.body, FPS, "원본 fps", 3, 1, 1)
        field_label(s2.body, "화질", 4)
        self.quality_var = menu(s2.body, QUALITY, "최고화질 (CRF 14)", 5)
        self.hw_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(s2.body, text="GPU 인코딩 사용", variable=self.hw_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.auto_level_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(s2.body, text="수평 자동 보정 (실험적)", variable=self.auto_level_var).grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))

        s3 = Section(scroll, "글자")
        s3.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        field_label(s3.body, "제목 (비우면 넣지 않음)", 0)
        self.title_entry = ctk.CTkEntry(s3.body, height=32, placeholder_text="예: 가을 산길 드라이브")
        self.title_entry.grid(row=1, column=0, columnspan=2, sticky="ew")
        stylize_entry_isolation(self.title_entry)
        self.title_entry.bind("<KeyRelease>", lambda _e: self.output_preview_var.get() and self.player.redraw())
        self.caption_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(s3.body, text="AI 장면 자막 (작게, 왼쪽 아래)", variable=self.caption_var,
                        command=self.player.redraw).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.ts_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(s3.body, text="촬영 시각 표시", variable=self.ts_var, command=self.player.redraw).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        s4 = Section(scroll, "소리", opened=False)
        s4.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.audio_var = menu(s4.body, AUDIO, "원본 주행 소리", 0)
        self.bgm_btn = ctk.CTkButton(s4.body, text="🎵 배경음악 선택…", height=32, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                                     command=self.pick_bgm)
        self.bgm_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.bgm_label = ctk.CTkLabel(s4.body, text="배경음악 없음", font=font(11), text_color=FG_MUTED, anchor="w")
        self.bgm_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        s5 = Section(scroll, "전환", opened=False)
        s5.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        self.trans_var = menu(s5.body, TRANSITIONS, "크로스 디졸브", 0, 0, 1)
        self.trans_len_var = menu(s5.body, TRANS_LENGTHS, "0.5초", 0, 1, 1)

        bottom = ctk.CTkFrame(right, fg_color=BG_CARD, corner_radius=10)
        bottom.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        bottom.grid_columnconfigure(0, weight=1)
        self.make_info = ctk.CTkLabel(bottom, text="", font=font(12), text_color=FG_SOFT, justify="left", anchor="w")
        self.make_info.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        self.btn_make = ctk.CTkButton(bottom, text="🎬 쇼츠 만들기 (MP4)", height=48, font=font(16, True),
                                      fg_color=SHORTS, hover_color=SHORTS_HOVER, command=self.make_shorts)
        self.btn_make.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))

    def _register_settings(self) -> None:
        reg = self.app.settings
        for key, var, allowed in (
            ("shorts.ai", self.ai_var, list(AI_MODES)), ("shorts.length", self.len_var, list(LENGTHS)),
            ("shorts.clip", self.clip_var, list(CLIP_LENGTHS)), ("shorts.layout", self.layout_var, list(LAYOUTS)),
            ("shorts.resolution", self.res_var, list(RESOLUTIONS)), ("shorts.fps", self.fps_var, list(FPS)),
            ("shorts.quality", self.quality_var, list(QUALITY)), ("shorts.audio", self.audio_var, list(AUDIO)),
            ("shorts.transition", self.trans_var, list(TRANSITIONS)), ("shorts.transition_len", self.trans_len_var, list(TRANS_LENGTHS)),
        ):
            reg.register_var(key, var, allowed)
        for key, var in (("shorts.hw", self.hw_var), ("shorts.auto_level", self.auto_level_var),
                         ("shorts.captions", self.caption_var), ("shorts.timestamp", self.ts_var),
                         ("shorts.show_result", self.show_result_var), ("shorts.grid", self.grid_var),
                         ("shorts.output_preview", self.output_preview_var)):
            reg.register_var(key, var, [True, False])

        def set_title(v):
            if v:  # 빈 값이면 건드리지 않아야 안내 문구(placeholder)가 유지됨
                self.title_entry.delete(0, "end")
                self.title_entry.insert(0, str(v))
        reg.register("shorts.title", lambda: self.title_entry.get(), set_title)

        def set_bgm(v):
            if v and Path(v).exists():
                self.bgm_path = Path(v)
                self.bgm_label.configure(text=f"🎵 {self.bgm_path.name}", text_color="#f9a8d4")
        reg.register("shorts.bgm", lambda: str(self.bgm_path) if self.bgm_path else "", set_bgm)

    # ── 화면 전환 ──

    def on_show(self) -> None:
        self.player.request_frame()
        self.refresh_list()

    def on_hide(self) -> None:
        self.player.stop()

    def shutdown(self) -> None:
        self.player.close()

    # ── 상태 ──

    def _on_videos(self) -> None:
        self.player.set_clips(self.store.playback_clips())
        self.refresh_list()

    def _on_busy(self) -> None:
        st = "disabled" if self.store.busy else "normal"
        self.btn_find.configure(state=st)
        self.btn_make.configure(state=st)

    def selected(self) -> Optional[HighlightClip]:
        clips = self.store.highlight_clips
        if self.index is None or not (0 <= self.index < len(clips)):
            return None
        return clips[self.index]

    def _target(self) -> Optional[Tuple[float, float]]:
        return vp_target_for(self.store.highlight_clips)

    def refresh_list(self) -> None:
        clips = self.store.highlight_clips
        present = {id(v) for v in self.store.videos}
        if any(id(c.video) not in present for c in clips):
            self.store.highlight_clips = clips = [c for c in clips if id(c.video) in present]
        if self.index is not None and self.index >= len(clips):
            self.index = None
        tree = self.tree
        tree.delete(*tree.get_children())
        max_score = max((c.score for c in clips), default=0.0)
        for i, c in enumerate(clips):
            g = self.store.highlight_global(c)
            if g is None:
                continue
            if c.ai_score is not None:
                score = f"AI {c.ai_score:.0f}"
            else:
                score = f"{(c.score / max_score * 100) if max_score > 0 else 0:.0f}"
            adj = []
            if c.roll_adjust:
                adj.append(f"{fmt_roll(c.roll_adjust)}")
            if c.vp is not None:
                adj.append("⌖")
            desc = c.reason or "-"
            if c.caption:
                desc += f" · 「{c.caption}」"
            values = ("✔" if c.included else "✕", format_duration(g[0]), desc, score, " ".join(adj) or "-")
            key = (str(c.video.path), int(round(c.start_sec * 10)))
            photo = self._thumbs.get(key)
            kw = {"image": photo} if photo is not None else {}
            tree.insert("", "end", iid=str(i), values=values, tags=("on" if c.included else "off",), **kw)
            if photo is None:
                self._request_thumb(c, key)
        if self.index is not None and tree.exists(str(self.index)):
            tree.selection_set(str(self.index))
            tree.see(str(self.index))
        used = [c for c in clips if c.included]
        if not clips:
            self.list_info.configure(text="아직 장면이 없습니다. 위 버튼을 누르면 주행 영상에서\n볼 만한 장면을 골라 줍니다.")
        else:
            n_adj = sum(1 for c in used if c.vp is not None or c.roll_adjust)
            self.list_info.configure(text=f"사용 {len(used)}개 / 후보 {len(clips)}개 · 완성 약 {sum(c.duration for c in used):.0f}초\n"
                                          f"맞춤한 컷 {n_adj}개")
        self.make_info.configure(text=f"✔ 사용 컷 {len(used)}개 · 약 {sum(c.duration for c in used):.0f}초" if used
                                 else "장면을 먼저 찾거나, 바로 만들면 자동으로 찾습니다.")
        self._update_cut_panel()

    def _request_thumb(self, c: HighlightClip, key) -> None:
        def ready(arr, key=key):
            def apply():
                self._thumbs[key] = make_photo_image(arr)
                for iid in self.tree.get_children():
                    i = int(iid)
                    if i < len(self.store.highlight_clips):
                        cc = self.store.highlight_clips[i]
                        if (str(cc.video.path), int(round(cc.start_sec * 10))) == key:
                            self.tree.item(iid, image=self._thumbs[key])
            self.after(0, apply)
        self.app.thumbs.request(c.video.path, c.start_sec + c.duration / 2, THUMB_W, THUMB_H, ready)

    # ── 컷 선택 · 재생 ──

    def _on_tree_select(self, _e=None) -> None:
        idx = tree_selected_index(self.tree)
        if idx is not None and idx != self.index:
            self.select_cut(idx)

    def select_cut(self, idx: int, seek: bool = True) -> None:
        clips = self.store.highlight_clips
        if not (0 <= idx < len(clips)):
            return
        self.index = idx
        if self.tree.exists(str(idx)) and self.tree.selection() != (str(idx),):
            self.tree.selection_set(str(idx))
            self.tree.see(str(idx))
        if seek:
            g = self.store.highlight_global(clips[idx])
            if g is not None:
                self.player.stop()
                self.player.seek((g[0] + g[1]) / 2)
        self._update_cut_panel()

    def step_cut(self, delta: int) -> None:
        clips = self.store.highlight_clips
        order = [i for i, c in enumerate(clips) if c.included] or list(range(len(clips)))
        if not order:
            return
        if self.index in order:
            k = order.index(self.index) + delta
        else:
            cur = self.index if self.index is not None else -1
            k = next((j for j, i in enumerate(order) if i > cur), len(order) - 1) if delta > 0 else \
                max((j for j, i in enumerate(order) if i < cur), default=0)
        self.select_cut(order[max(0, min(len(order) - 1, k))])

    def play_cut(self) -> None:
        if self.player.playing:
            self.player.stop()
            return
        c = self.selected()
        g = self.store.highlight_global(c) if c is not None else None
        if g is not None:
            self.player.play_range(g[0], g[1])

    def preview_all(self) -> None:
        order = [i for i, c in enumerate(self.store.highlight_clips) if c.included]
        segs = [self.store.highlight_global(self.store.highlight_clips[i]) for i in order]
        pairs = [(i, g) for i, g in zip(order, segs) if g is not None]
        if not pairs:
            messagebox.showinfo("미리보기", "사용(✔) 중인 컷이 없습니다.")
            return
        self._preview_order = [i for i, _g in pairs]
        self.player.play_segments([g for _i, g in pairs])

    def _on_preview_segment(self, k: int) -> None:
        order = getattr(self, "_preview_order", [])
        if 0 <= k < len(order):
            self.select_cut(order[k], seek=False)

    def _on_play_state(self, playing: bool) -> None:
        self.btn_play.configure(text="⏸ 멈춤 [Space]" if playing else "▶ 이 컷 재생 [Space]",
                                fg_color="#d97706" if playing else ACCENT)

    def toggle_use(self) -> None:
        c = self.selected()
        if c is None:
            return
        c.included = not c.included
        self.store.emit("highlights")

    # ── 컷 맞춤 (기울기 · 소실점) ──

    def _cut_on_screen(self) -> Optional[HighlightClip]:
        c = self.selected()
        if c is None:
            return None
        g = self.store.highlight_global(c)
        if g is None or not (g[0] - 0.3 <= self.player.position <= g[1] + 0.6):
            return None
        return c

    def _transform_frame(self, arr: np.ndarray) -> np.ndarray:
        self._out_geom = None
        c = self._cut_on_screen()
        if c is None:
            return arr
        output_mode = bool(self.output_preview_var.get())
        if self.show_result_var.get() or output_mode:
            roll, target = c.roll_adjust or 0.0, self._target()
            if abs(roll) >= 0.05 or (c.vp is not None and target is not None):
                h, w = arr.shape[:2]
                arr = cv2.warpAffine(arr, reframe_affine(w, h, roll, c.vp, target), (w, h), flags=cv2.INTER_LINEAR)
        if output_mode:
            arr = self._compose_output(arr)
        return arr

    def _compose_output(self, arr: np.ndarray) -> np.ndarray:
        """현재 레이아웃 설정대로 실제 결과 화면(세로 9:16 흐림 배경 / 꽉 채우기 / 가로)을 합성해 미리 보여줌."""
        layout = LAYOUTS[self.layout_var.get()]
        h, w = arr.shape[:2]
        if layout == ShortsLayout.LANDSCAPE:
            self._out_geom = {"layout": layout, "w": w, "h": h, "fg_h": h, "band": 0.0}
            return arr
        oh = max(160, self.player.canvas.winfo_height() - 8)
        ow = int(round(oh * 9 / 16))
        avail = max(90, self.player.canvas.winfo_width() - 8)
        if ow > avail:
            ow, oh = avail, int(round(avail * 16 / 9))
        if layout == ShortsLayout.VERTICAL_CROP:
            scale = max(ow / w, oh / h)
            big = cv2.resize(arr, (int(w * scale) + 2, int(h * scale) + 2), interpolation=cv2.INTER_LINEAR)
            y0, x0 = (big.shape[0] - oh) // 2, (big.shape[1] - ow) // 2
            out = np.ascontiguousarray(big[y0:y0 + oh, x0:x0 + ow])
            self._out_geom = {"layout": layout, "w": ow, "h": oh, "fg_h": oh, "band": 0.0}
            return out
        # 흐림 배경: 화면을 꽉 채우게 키워 흐리고 어둡게 한 위에, 원본 화면을 폭에 맞춰 가운데 배치
        sw, sh = max(2, ow // 4), max(2, oh // 4)
        scale = max(sw / w, sh / h)
        small = cv2.resize(arr, (int(w * scale) + 2, int(h * scale) + 2), interpolation=cv2.INTER_AREA)
        y0, x0 = (small.shape[0] - sh) // 2, (small.shape[1] - sw) // 2
        bg = cv2.GaussianBlur(small[y0:y0 + sh, x0:x0 + sw], (0, 0), 4)
        out = cv2.resize(cv2.convertScaleAbs(bg, alpha=0.82), (ow, oh), interpolation=cv2.INTER_LINEAR)  # 어둡게는 작은 그림에서 먼저(float 변환보다 약 10배 빠름)
        fh = min(oh, int(round(h * ow / w)))
        fg = cv2.resize(arr, (ow, fh), interpolation=cv2.INTER_AREA)
        fy = (oh - fh) // 2
        out[fy:fy + fh, :] = fg
        self._out_geom = {"layout": layout, "w": ow, "h": oh, "fg_h": fh, "band": (oh - fh) / 2.0}
        return np.ascontiguousarray(out)

    def _draw_output_texts(self, cv: tk.Canvas, c: HighlightClip, x0: int, y0: int, w: int, h: int) -> None:
        """렌더러와 같은 비율 위치에 제목·자막·촬영 시각을 미리 표시."""
        geom = getattr(self, "_out_geom", None) or {}
        layout = geom.get("layout", LAYOUTS[self.layout_var.get()])
        vertical = layout != ShortsLayout.LANDSCAPE
        band = float(geom.get("band", 0.0))
        fg_h = float(geom.get("fg_h", h))
        use_band = layout == ShortsLayout.VERTICAL_BLUR and band >= h * 0.12

        def units(t: str) -> float:
            return sum(1.0 if ord(ch) >= 0x2E80 else 0.6 for ch in t) or 1.0

        def text(xc, y, t, size, anchor="n", shadow=True, color="white"):
            if shadow:
                cv.create_text(xc + 2, y + 2, text=t, anchor=anchor, fill="black", font=("Malgun Gothic", -size, "bold"))
            cv.create_text(xc, y, text=t, anchor=anchor, fill=color, font=("Malgun Gothic", -size, "bold"))

        title = self.title_entry.get().strip()
        if title:
            fs = int(max(9, min(w * (0.075 if vertical else 0.042), 0.88 * w / units(title))))
            ty = max(h * 0.04, band * 0.5 - fs * 0.6) if use_band else h * 0.06
            text(x0 + w / 2, y0 + ty, title, fs)
        if self.caption_var.get() and c.caption:
            fs = int(max(8, w * (0.036 if vertical else 0.020)))
            bottom = (h + fg_h) / 2.0 if layout == ShortsLayout.VERTICAL_BLUR else h * (0.86 if vertical else 0.88)
            text(x0 + w * 0.05, y0 + bottom - fs - h * (0.018 if vertical else 0.03), c.caption, fs, anchor="nw", color="#f1f5f9")
        if self.ts_var.get():
            ts = clip_timestamp_text(c)
            if ts:
                fs = int(max(8, w * (0.042 if vertical else 0.022)))
                yy = h - band + band * 0.25 if use_band else h * 0.90
                text(x0 + w / 2, y0 + yy, ts, fs)

    def _draw_overlay(self, cv: tk.Canvas, x0: int, y0: int, w: int, h: int) -> None:
        c = self._cut_on_screen()
        if c is None:
            return
        if self.output_preview_var.get():
            self._draw_output_texts(cv, c, x0, y0, w, h)
            label = {ShortsLayout.VERTICAL_BLUR: "세로 9:16 · 흐림 배경", ShortsLayout.VERTICAL_CROP: "세로 9:16 · 꽉 채우기",
                     ShortsLayout.LANDSCAPE: "가로 16:9"}[LAYOUTS[self.layout_var.get()]]
            t_id = cv.create_text(x0 + 12, y0 + 18, text=f"📱 결과 화면 · {label}", anchor="w", fill="#f9a8d4",
                                  font=("Malgun Gothic", 10, "bold"))
            bx = cv.bbox(t_id)
            if bx:
                bg = cv.create_rectangle(bx[0] - 6, bx[1] - 4, bx[2] + 6, bx[3] + 4, fill="#0f172a", outline="")
                cv.tag_lower(bg, t_id)
            return
        if self.grid_var.get():
            for k in range(1, 12):
                y = y0 + h * k / 12.0
                cv.create_line(x0, y, x0 + w, y, fill="#f8fafc" if k % 3 == 0 else "#94a3b8", dash=(4, 6))
            for k in (1, 2):
                cv.create_line(x0 + w * k / 3.0, y0, x0 + w * k / 3.0, y0 + h, fill="#94a3b8", dash=(4, 6))
            cv.create_line(x0 + w / 2, y0, x0 + w / 2, y0 + h, fill="#f8fafc", dash=(6, 4))
        target = self._target()
        if target is not None:
            tx, ty = x0 + target[0] * w, y0 + target[1] * h
            cv.create_line(x0, ty, x0 + w, ty, fill="#22d3ee", width=2)
            cv.create_oval(tx - 14, ty - 14, tx + 14, ty + 14, outline="#22d3ee", width=3)
        if c.vp is not None:
            roll = c.roll_adjust or 0.0
            if self.show_result_var.get():
                s, R, p, t = reframe_geometry(w, h, roll, c.vp, target)
                m = s * (R @ (np.array([c.vp[0] * w, c.vp[1] * h]) - p)) + t
                px, py = x0 + m[0], y0 + m[1]
            else:
                px, py = x0 + c.vp[0] * w, y0 + c.vp[1] * h
            cv.create_line(px - 18, py, px + 18, py, fill="#facc15", width=3)
            cv.create_line(px, py - 18, px, py + 18, fill="#facc15", width=3)
        mode = "맞춤 결과" if self.show_result_var.get() else "원본"
        cv.create_rectangle(x0 + 8, y0 + 8, x0 + 150, y0 + 34, fill="#0f172a", outline="")
        cv.create_text(x0 + 16, y0 + 21, text=f"🎯 {mode}", anchor="w", fill="#facc15", font=("Malgun Gothic", 11, "bold"))

    def _on_player_click(self, qx: float, qy: float, w: int, h: int) -> None:
        c = self._cut_on_screen()
        if c is None:
            return
        if self.output_preview_var.get():
            self.app.set_status("📍 소실점은 '결과 화면 미리보기'를 끄고 맞춤 화면에서 클릭해 지정하세요.")
            return
        if self.show_result_var.get():
            sx, sy = reframe_inverse_point(w, h, c.roll_adjust or 0.0, c.vp, self._target(), qx, qy)
        else:
            sx, sy = qx, qy
        c.vp = (min(0.98, max(0.02, sx / w)), min(0.98, max(0.02, sy / h)))
        self._after_adjust()

    def rotate(self, delta: float) -> None:
        c = self.selected()
        if c is None:
            return
        c.roll_adjust = round(max(-ROLL_LIMIT, min(ROLL_LIMIT, (c.roll_adjust or 0.0) + delta)), 2)
        self._after_adjust()

    def _on_roll_slider(self, value: float) -> None:
        if getattr(self, "_suppress_slider", False):
            return
        c = self.selected()
        if c is None:
            return
        c.roll_adjust = slider_to_roll(value)
        self._after_adjust()

    def clear_vp(self) -> None:
        c = self.selected()
        if c is not None:
            c.vp = None
            self._after_adjust()

    def reset_adjust(self) -> None:
        c = self.selected()
        if c is not None:
            c.vp = None
            c.roll_adjust = None
            self._after_adjust()

    def shift_cut(self, delta: float) -> None:
        """컷을 앞뒤로 옮김 (길이 유지, 영상 범위 안에서)."""
        c = self.selected()
        if c is None:
            return
        dur = c.duration
        start = min(max(0.0, c.start_sec + delta), max(0.0, c.video.duration_sec - dur))
        if abs(start - c.start_sec) < 1e-3:
            return
        c.start_sec, c.end_sec = start, start + dur
        self._after_trim()

    def resize_cut(self, delta: float) -> None:
        """컷 길이를 늘이거나 줄임 (1~8초, 시작점 고정)."""
        c = self.selected()
        if c is None:
            return
        end = min(c.video.duration_sec, c.start_sec + min(8.0, max(1.0, c.duration + delta)))
        if end - c.start_sec < 1.0 - 1e-6 or abs(end - c.end_sec) < 1e-3:
            return
        c.end_sec = end
        self._after_trim()

    def _after_trim(self) -> None:
        c = self.selected()
        g = self.store.highlight_global(c) if c is not None else None
        if g is not None:
            self.player.stop()
            self.player.seek((g[0] + g[1]) / 2)
        self.store.emit("highlights")

    def apply_roll_all(self) -> None:
        c = self.selected()
        if c is None:
            return
        roll = c.roll_adjust or 0.0
        others = [x for x in self.store.highlight_clips if x.included and x is not c]
        if not others:
            return
        differing = [x for x in others if x.roll_adjust not in (None, roll) and abs((x.roll_adjust or 0.0) - roll) > 0.05]
        if differing and not messagebox.askyesno(
                "기울기 전체 적용", f"사용 중인 다른 컷 {len(others)}개의 기울기를 모두 {fmt_roll(roll)}로 바꿀까요?\n"
                                   f"(따로 맞춰 둔 컷 {len(differing)}개도 바뀝니다)"):
            return
        for x in others:
            x.roll_adjust = roll
        self.app.set_status(f"↻ 사용 중인 컷 {len(others) + 1}개 모두 기울기 {fmt_roll(roll)}로 맞췄습니다.")
        self.store.emit("highlights")

    def _after_adjust(self) -> None:
        c = self.selected()
        g = self.store.highlight_global(c) if c is not None else None
        if g is not None and not (g[0] <= self.player.position <= g[1]) and not self.player.playing:
            self.player.seek((g[0] + g[1]) / 2)
        self.player.redraw()
        self.store.emit("highlights")

    def _update_cut_panel(self) -> None:
        c = self.selected()
        self._suppress_slider = True
        self.roll_slider.set(roll_to_slider(c.roll_adjust or 0.0) if c is not None else 0.0)
        self._suppress_slider = False
        if c is None:
            self.cut_title.configure(text="장면을 고르면 여기에서 확인하고 기울기·소실점을 맞출 수 있습니다")
            self.roll_value.configure(text="+0.0°")
            self.vp_label.configure(text="-")
            self.zoom_label.configure(text="")
            self.range_label.configure(text="")
            return
        used = [i for i, x in enumerate(self.store.highlight_clips) if x.included]
        pos = f"사용 컷 {used.index(self.index) + 1}/{len(used)}" if self.index in used else "제외된 후보"
        g = self.store.highlight_global(c)
        self.cut_title.configure(text=f"컷 #{(self.index or 0) + 1} · {pos} · {c.video.filename} {format_duration(c.start_sec)}"
                                      f"{' · ' + c.reason if c.reason else ''}")
        roll = c.roll_adjust or 0.0
        self.roll_value.configure(text=fmt_roll(roll))
        self.range_label.configure(text=f"{format_duration(c.start_sec, True)} ~ {format_duration(c.end_sec, True)}  ({c.duration:.1f}초)")
        target = self._target()
        self.vp_label.configure(text=("✔ 지정됨" if c.vp is not None else "✕ 미지정 — 화면을 클릭해 지정"),
                                text_color="#4ade80" if c.vp is not None else "#f87171")
        raw = self.player.raw_frame()
        if raw is not None:
            h, w = raw.shape[:2]
            zoom = reframe_geometry(w, h, roll, c.vp, target)[0]
        else:
            zoom = 1.0
        tgt = f"목표 높이 {target[1] * 100:.0f}%" if target else "목표 미정"
        self.zoom_label.configure(text=f"{tgt} · 확대 {zoom:.2f}×" + ("  ⚠ 많이 확대됨" if zoom > 1.35 else ""))

    # ── 설정 ──

    def hconfig(self) -> HighlightConfig:
        return HighlightConfig(target_duration_sec=LENGTHS[self.len_var.get()], clip_duration_sec=CLIP_LENGTHS[self.clip_var.get()])

    def export_config(self, out: Path) -> ShortsExportConfig:
        return ShortsExportConfig(
            output_path=out, layout=LAYOUTS[self.layout_var.get()], resolution=RESOLUTIONS[self.res_var.get()],
            fps=FPS[self.fps_var.get()], audio_mode=AUDIO[self.audio_var.get()], bgm_path=self.bgm_path,
            title_text=self.title_entry.get().strip(), show_timestamp=bool(self.ts_var.get()),
            show_captions=bool(self.caption_var.get()), auto_level=bool(self.auto_level_var.get()),
            transition=TRANSITIONS[self.trans_var.get()], transition_sec=TRANS_LENGTHS[self.trans_len_var.get()],
            quality_preset=QUALITY[self.quality_var.get()], use_hw_accel=bool(self.hw_var.get()),
        )

    def _on_selection_setting(self) -> None:
        if self.store.busy or not self.store.highlight_clips:
            return
        if AI_MODES[self.ai_var.get()] != AiDirectorMode.OFF:
            self.app.set_status("⚙ 장면 선정 설정이 바뀌었습니다 — [✨ 하이라이트 찾기]를 다시 누르면 새 설정으로 고릅니다.")
            return
        if all(v.feature_cache is not None and "appear" in v.feature_cache for v in self.store.videos):
            self.store.highlight_clips = select_highlights(self.store.videos, self.hconfig())
            self.index = None
            self.store.emit("highlights")

    def pick_bgm(self) -> None:
        p = filedialog.askopenfilename(title="배경음악 선택",
                                       filetypes=[("오디오", "*.mp3 *.m4a *.aac *.wav *.flac *.ogg"), ("모든 파일", "*.*")])
        if not p:
            return
        self.bgm_path = Path(p)
        self.bgm_label.configure(text=f"🎵 {self.bgm_path.name}", text_color="#f9a8d4")
        if self.audio_var.get() in ("원본 주행 소리", "무음"):
            self.audio_var.set("원본 + 배경음악")

    # ── 하이라이트 찾기 / 만들기 ──

    def _prepare_ai(self) -> Optional[Tuple[AiDirectorMode, Optional[str]]]:
        mode = AI_MODES[self.ai_var.get()]
        if mode == AiDirectorMode.OFF:
            return mode, None
        if not self._consent:
            if not messagebox.askyesno("AI 장면 선정 안내",
                                       "Claude가 장면을 고르도록 후보 장면 사진(장면당 3장, 저해상도)을 Anthropic 서버로 보냅니다.\n"
                                       "사진에 번호판·사람·장소가 보일 수 있습니다.\n\n계속할까요? ('아니요' = 이번엔 알고리즘으로만)"):
                return AiDirectorMode.OFF, None
            self._consent = True
        if mode == AiDirectorMode.CLAUDE_CODE:
            if find_claude_cli() is None:
                if messagebox.askyesno("Claude Code 없음", "이 PC에서 Claude Code(claude 명령)를 찾지 못했습니다.\n"
                                                           "설치 후 터미널에서 claude를 한 번 실행해 로그인해 주세요.\n\n이번에는 알고리즘으로만 고를까요?"):
                    return AiDirectorMode.OFF, None
                return None
            return mode, None
        key = os.environ.get("ANTHROPIC_API_KEY") or self._api_key
        if not key:
            key = simpledialog.askstring("Claude API 키", "Claude API 키를 입력하세요 (sk-ant-...).\n이번 실행 동안만 기억하고 저장하지 않습니다.",
                                         show="*", parent=self)
            if not key:
                return None
            self._api_key = key.strip()
            key = self._api_key
        return mode, key

    def find_highlights(self, then_make: Optional[Path] = None) -> None:
        if self.store.busy:
            return
        if not self.store.videos:
            messagebox.showwarning("알림", "먼저 ① 영상 화면에서 주행 영상을 추가해 주세요.")
            return
        prepared = self._prepare_ai()
        if prepared is None:
            return
        mode, key = prepared
        acfg, hcfg = self.store.analysis_config, self.hconfig()
        self.player.stop()
        cfg = self.export_config(then_make) if then_make is not None else None

        def job(report, cancelled):
            clips, err = compute_highlights(self.store, acfg, hcfg, mode, key,
                                            (lambda f, m: report(f * (0.3 if cfg else 1.0), m)), cancelled)
            if cfg is None:
                return clips, err, None, [], []
            warns, notes = [], []
            self.store.highlight_clips = clips
            use = [c for c in clips if c.included and self.store.type_coverage(c.video, c.start_sec, c.end_sec, SegmentType.MANUAL_CUT) < 0.3]
            path = render_highlight_shorts(use, cfg, progress_callback=lambda f, m: report(0.3 + 0.7 * f, f"[쇼츠] {m}"),
                                           cancel_check=cancelled, warnings=warns, notes=notes)
            return clips, err, path, warns, notes

        def done(result):
            clips, err, path, warns, notes = result
            self.store.highlight_clips = clips
            self.index = next((i for i, c in enumerate(clips) if c.included), None)
            self.store.emit("highlights", "segments")
            if self.index is not None:
                self.select_cut(self.index)
            if err:
                messagebox.showwarning("AI 장면 선정 실패", f"{err}\n\n대신 알고리즘이 고른 장면으로 표시합니다.")
            used = [c for c in clips if c.included]
            if path is not None:
                self._finish_render(path, used, warns, notes)
            elif not clips:
                self.app.set_status("하이라이트로 쓸 만한 장면을 찾지 못했습니다 (정차·회전·주차장 장면은 제외됩니다).", 1.0)
            else:
                ai = any(c.ai_score is not None for c in clips)
                self.app.set_status(
                    f"{'🤖 Claude가' if ai else '✨'} 후보 {len(clips)}개 중 {len(used)}개 선정 (약 {sum(c.duration for c in used):.0f}초). "
                    "가운데 화면에서 컷을 확인·맞춤한 뒤 [🎬 쇼츠 만들기]를 누르세요.", 1.0)

        self.app.run_job("하이라이트 장면을 찾는 중…", job, done)

    def make_shorts(self) -> None:
        if self.store.busy or not self.store.videos:
            if not self.store.videos:
                messagebox.showwarning("알림", "먼저 ① 영상 화면에서 주행 영상을 추가해 주세요.")
            return
        if AUDIO[self.audio_var.get()] in (ShortsAudioMode.BGM, ShortsAudioMode.BGM_MIX) and not (self.bgm_path and self.bgm_path.exists()):
            messagebox.showwarning("배경음악", "배경음악을 쓰려면 [소리] 설정에서 음악 파일을 먼저 골라 주세요.")
            return
        default = f"highlight_shorts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        p = filedialog.asksaveasfilename(title="쇼츠를 저장할 위치", initialdir=str(self.store.videos[0].path.parent),
                                         initialfile=default, defaultextension=".mp4", filetypes=[("MP4 비디오", "*.mp4")])
        if not p:
            return
        out = Path(p).resolve()
        if any(v.path.resolve() == out for v in self.store.videos):
            messagebox.showerror("오류", "원본 영상과 같은 파일명으로 저장할 수 없습니다.")
            return
        if not any(c.included for c in self.store.highlight_clips):
            self.find_highlights(then_make=out)
            return
        cfg = self.export_config(out)
        use = [c for c in self.store.highlight_clips
               if c.included and self.store.type_coverage(c.video, c.start_sec, c.end_sec, SegmentType.MANUAL_CUT) < 0.3]
        self.player.stop()

        def job(report, cancelled):
            warns, notes = [], []
            path = render_highlight_shorts(use, cfg, progress_callback=lambda f, m: report(f, f"[쇼츠] {m}"),
                                           cancel_check=cancelled, warnings=warns, notes=notes)
            return path, warns, notes

        self.app.run_job("쇼츠를 만드는 중…", job, lambda r: self._finish_render(r[0], use, r[1], r[2]))

    def _finish_render(self, path: Path, used: List[HighlightClip], warns: List[str], notes: List[str]) -> None:
        self.app.set_output(path)
        self.app.set_status(f"✅ 쇼츠 완성: {path}", 1.0)
        msg = f"쇼츠를 만들었습니다!\n\n컷 {len(used)}개 · 약 {sum(c.duration for c in used):.0f}초\n\n{path}"
        if notes:
            msg += "\n\n" + "\n".join(notes)
        if warns:
            msg += "\n\n⚠ " + "\n⚠ ".join(warns)
        (messagebox.showwarning if warns else messagebox.showinfo)("쇼츠 완성", msg)

    # ── 단축키 ──

    def handle_key(self, event) -> None:
        k = event.keysym
        shift = bool(event.state & 0x1)
        if k == "space":
            self.play_cut()
        elif k in ("x", "X"):
            self.toggle_use()
        elif k in ("Left", "Right"):
            self.rotate((-1 if k == "Left" else 1) * (1.0 if shift else 0.1))
        elif k == "comma":
            self.shift_cut(-0.5)
        elif k == "period":
            self.shift_cut(0.5)
        elif k == "Prior":
            self.step_cut(-1)
        elif k == "Next":
            self.step_cut(1)

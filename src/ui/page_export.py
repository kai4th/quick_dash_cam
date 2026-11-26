"""③ 합치기: 편집한 주행 영상을 하나의 MP4로 저장 (정차·직진 구간 배속/삭제, 화질·코덱·오디오 설정)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from ..analyzer import analyze_video
from ..models import (
    AudioMode,
    ExportConfig,
    SkipMode,
    VideoCodec,
    VideoQualityPreset,
    merge_preserving_manual_cuts,
)
from ..processor import render_merged_video
from .theme import BG_CARD, FG_MUTED, FG_SOFT, PRIMARY, PRIMARY_HOVER, font, format_duration
from .widgets import card, field_label

QUALITY = {
    "원본 유지급 최고화질 (CRF 14)": VideoQualityPreset.LOSSLESS_LIKE,
    "고화질 (CRF 18, 권장)": VideoQualityPreset.HIGH,
    "표준 (CRF 22, 용량 절약)": VideoQualityPreset.BALANCED,
    "저용량·빠름 (CRF 27)": VideoQualityPreset.COMPACT,
}
RESOLUTION = {"원본 그대로": "original", "4K (3840×2160)": "4k", "2.7K (2560×1440)": "1440p",
              "1080p (1920×1080)": "1080p", "720p (1280×720)": "720p"}
FPS = {"원본 그대로": "original", "60fps": "60", "30fps": "30", "24fps": "24"}
CODEC = {"H.264 (어디서나 재생)": VideoCodec.H264, "H.265 / HEVC (고효율)": VideoCodec.HEVC}
AUDIO = {"일반 구간 소리 유지 · 스킵 구간 무음": AudioMode.KEEP_NORMAL_MUTE_SKIP,
         "배속 구간도 빠르게 재생": AudioMode.KEEP_AND_SPEED_UP, "전체 무음": AudioMode.MUTE_ALL}
SPEEDS = ["4x", "8x", "12x", "16x"]


class ExportPage(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.store = app.store
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_columnconfigure(0, weight=1)
        self._build_settings(left)
        self._build_summary()

        self.store.subscribe("videos", self.refresh)
        self.store.subscribe("segments", self.refresh)
        self.store.subscribe("busy", self._on_busy)
        reg = app.settings
        reg.register_var("export.skip", self.skip_var, ["배속 재생 (타임랩스)", "완전히 잘라내기 (점프컷)"])
        reg.register_var("export.speed", self.speed_var, SPEEDS)
        reg.register_var("export.badge", self.badge_var, [True, False])
        reg.register_var("export.quality", self.quality_var, list(QUALITY))
        reg.register_var("export.resolution", self.res_var, list(RESOLUTION))
        reg.register_var("export.fps", self.fps_var, list(FPS))
        reg.register_var("export.codec", self.codec_var, list(CODEC))
        reg.register_var("export.hw", self.hw_var, [True, False])
        reg.register_var("export.audio", self.audio_var, list(AUDIO))
        self.refresh()

    def _menu(self, parent, values, row, default, col=0, colspan=2):
        var = ctk.StringVar(value=default)
        ctk.CTkOptionMenu(parent, values=list(values), variable=var, height=34,
                          command=lambda _v: self.refresh()).grid(row=row, column=col, columnspan=colspan, sticky="ew", pady=(2, 0))
        return var

    def _build_settings(self, parent) -> None:
        o1, b1 = card(parent, "⏩ 정차·직진 구간 처리", "② 편집에서 찾은 신호대기·고속도로 직진 구간을 어떻게 줄일지 정합니다.")
        o1.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.skip_var = ctk.StringVar(value="배속 재생 (타임랩스)")
        ctk.CTkSegmentedButton(b1, values=["배속 재생 (타임랩스)", "완전히 잘라내기 (점프컷)"], variable=self.skip_var,
                               height=34, command=lambda _v: self.refresh()).grid(row=0, column=0, columnspan=2, sticky="ew")
        field_label(b1, "배속 배율", 1)
        self.speed_var = ctk.StringVar(value="8x")
        self.speed_seg = ctk.CTkSegmentedButton(b1, values=SPEEDS, variable=self.speed_var, height=32,
                                                command=lambda _v: self.refresh())
        self.speed_seg.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.badge_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(b1, text="배속 구간에 '>> 8x' 배지 표시", variable=self.badge_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        o2, b2 = card(parent, "🎞 화질 · 해상도 · 코덱")
        o2.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        b2.grid_columnconfigure(1, weight=1)
        field_label(b2, "화질", 0)
        self.quality_var = self._menu(b2, QUALITY, 1, "원본 유지급 최고화질 (CRF 14)")
        field_label(b2, "해상도", 2, 0, 1)
        field_label(b2, "프레임레이트", 2, 1, 1)
        self.res_var = self._menu(b2, RESOLUTION, 3, "원본 그대로", 0, 1)
        self.fps_var = self._menu(b2, FPS, 3, "원본 그대로", 1, 1)
        field_label(b2, "코덱", 4)
        self.codec_var = self._menu(b2, CODEC, 5, "H.264 (어디서나 재생)")
        self.hw_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(b2, text="GPU 하드웨어 인코딩 사용 (가능할 때)", variable=self.hw_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

        o3, b3 = card(parent, "🔊 소리")
        o3.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.audio_var = self._menu(b3, AUDIO, 0, "일반 구간 소리 유지 · 스킵 구간 무음")

    def _build_summary(self) -> None:
        outer, body = card(self, "결과 미리 보기")
        outer.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.big_label = ctk.CTkLabel(body, text="", font=font(26, True), justify="center")
        self.big_label.grid(row=0, column=0, sticky="ew", pady=(20, 6))
        self.sub_label = ctk.CTkLabel(body, text="", font=font(13), text_color=FG_SOFT, justify="center")
        self.sub_label.grid(row=1, column=0, sticky="ew")
        self.stats_label = ctk.CTkLabel(body, text="", font=font(12), text_color=FG_MUTED, justify="left", anchor="w")
        self.stats_label.grid(row=2, column=0, sticky="ew", pady=(24, 0))
        self.btn_start = ctk.CTkButton(body, text="🎬 영상 합치기 시작 (MP4 저장)", height=50, font=font(16, True),
                                       fg_color=PRIMARY, hover_color=PRIMARY_HOVER, command=self.start)
        self.btn_start.grid(row=3, column=0, sticky="ew", pady=(30, 6))
        ctk.CTkLabel(body, text="아직 분석하지 않은 영상은 시작할 때 자동으로 분석합니다.\n진행 상황은 맨 아래 상태 줄에 표시됩니다.",
                     font=font(11), text_color=FG_MUTED, justify="center").grid(row=4, column=0, sticky="ew")

    # ── 설정 ──

    def skip_mode(self) -> SkipMode:
        return SkipMode.JUMPCUT if "점프컷" in self.skip_var.get() else SkipMode.TIMELAPSE

    def speed(self) -> float:
        return float(self.speed_var.get().rstrip("x"))

    def export_config(self, output_path: Path) -> ExportConfig:
        return ExportConfig(
            output_path=output_path, skip_mode=self.skip_mode(), speed_multiplier=self.speed(),
            audio_mode=AUDIO[self.audio_var.get()], target_resolution=RESOLUTION[self.res_var.get()],
            fps_mode=FPS[self.fps_var.get()], quality_preset=QUALITY[self.quality_var.get()],
            video_codec=CODEC[self.codec_var.get()], overlay_speed_badge=bool(self.badge_var.get()),
            use_hw_accel=bool(self.hw_var.get()),
        )

    # ── 표시 ──

    def refresh(self) -> None:
        vids = self.store.videos
        self.speed_seg.configure(state="disabled" if self.skip_mode() == SkipMode.JUMPCUT else "normal")
        if not vids:
            self.big_label.configure(text="영상 없음")
            self.sub_label.configure(text="① 영상 화면에서 주행 영상을 추가하세요.")
            self.stats_label.configure(text="")
            self._on_busy()
            return
        orig = self.store.total_duration
        est = sum(v.estimated_output_duration(self.skip_mode(), self.speed()) for v in vids)
        saved = max(0.0, orig - est)
        self.big_label.configure(text=f"{format_duration(orig)}  →  {format_duration(est)}")
        self.sub_label.configure(text=f"원본 → 예상 완성 길이  ({saved / orig * 100:.0f}% 단축)" if orig > 0 else "")
        analyzed = sum(1 for v in vids if v.analyzed)
        lines = [
            f"영상 {len(vids)}개 (분석 완료 {analyzed}개)",
            f"🟠 정차 {format_duration(sum(v.stop_duration for v in vids))}",
            f"🟣 직진 {format_duration(sum(v.straight_duration for v in vids))}",
            f"✂ 삭제 {format_duration(sum(v.manual_cut_duration for v in vids))}",
        ]
        if analyzed < len(vids):
            lines.append("⚠ 분석 전 영상이 있어 실제 결과는 더 짧아질 수 있습니다.")
        self.stats_label.configure(text="\n".join(lines))
        self._on_busy()

    def _on_busy(self) -> None:
        self.btn_start.configure(state="disabled" if self.store.busy or not self.store.videos else "normal")

    def on_show(self) -> None:
        self.refresh()

    def on_settings_loaded(self) -> None:
        self.refresh()

    # ── 실행 ──

    def start(self) -> None:
        if self.store.busy or not self.store.videos:
            return
        default = f"merged_drive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        path = filedialog.asksaveasfilename(title="합친 영상을 저장할 위치", initialdir=str(self.store.videos[0].path.parent),
                                            initialfile=default, defaultextension=".mp4", filetypes=[("MP4 비디오", "*.mp4")])
        if not path:
            return
        out = Path(path).resolve()
        if any(v.path.resolve() == out for v in self.store.videos):
            messagebox.showerror("오류", "원본 영상과 같은 파일명으로 저장할 수 없습니다.")
            return
        cfg = self.export_config(out)
        acfg = self.store.analysis_config
        videos = list(self.store.videos)

        def job(report, cancelled):
            pending = [v for v in videos if not v.analyzed and (acfg.detect_stops or acfg.detect_highway_straight)]
            for k, v in enumerate(pending):
                if cancelled():
                    raise RuntimeError("취소")
                old = list(v.segments)
                segs = analyze_video(v, acfg, progress_callback=lambda f, k=k, n=v.filename: report(
                    (k + f) / len(pending) * 0.3, f"[1/2 분석] {n} ({f * 100:.0f}%)"), cancel_check=cancelled)
                if cancelled():
                    raise RuntimeError("취소")
                v.segments = merge_preserving_manual_cuts(old, segs, v.duration_sec)
                v.analyzed = True
                self.after(0, lambda: self.store.emit("segments"))
            base = 0.3 if pending else 0.0
            return render_merged_video(videos, cfg, progress_callback=lambda f, m: report(base + f * (1 - base), f"[합치기] {m}"),
                                       cancel_check=cancelled)

        def done(final_path):
            self.app.set_output(final_path)
            self.app.set_status(f"✅ 합치기 완료: {final_path}", 1.0)
            messagebox.showinfo("완료", f"주행 영상을 하나로 합쳤습니다.\n\n{final_path}")

        self.app.run_job("영상 합치기를 시작합니다…", job, done)

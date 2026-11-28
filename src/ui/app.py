"""
QuickDashCam 메인 창: 상단 단계 탭(① 영상 → ② 편집 → ③ 합치기 / ④ 쇼츠) + 화면 + 공통 상태 줄.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Callable, Dict, Optional

import customtkinter as ctk

from ..ffmpeg_utils import detect_best_gpu_decoder, detect_best_h264_encoder
from . import persist
from .media import ThumbnailLoader
from .state import AppState
from .theme import BG_APP, BG_PANEL, FG_MUTED, NEUTRAL, NEUTRAL_HOVER, font, format_duration, enable_high_res_timer

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

PAGE_TITLES = {
    "library": "① 영상",
    "edit": "② 편집",
    "export": "③ 합치기",
    "shorts": "④ 쇼츠",
}


class JobCancelled(Exception):
    pass


class QuickDashCamApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        enable_high_res_timer()
        self.title("QuickDashCam — 주행 영상 편집 · 하이라이트 쇼츠")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1600, int(sw * 0.92)), min(1000, int(sh * 0.88))
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_APP)

        self.store = AppState()
        self.thumbs = ThumbnailLoader()
        self.settings = persist.SettingsRegistry()
        self._autosave_job: Optional[str] = None
        self._restoring = False
        self._resume_checked = False  # 시작 시 '이어하기' 확인 전에는 저장 파일을 건드리지 않음
        self.pages: Dict[str, ctk.CTkFrame] = {}
        self.current_page: Optional[str] = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self.page_host = ctk.CTkFrame(self, fg_color="transparent")
        self.page_host.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6, 4))
        self.page_host.grid_columnconfigure(0, weight=1)
        self.page_host.grid_rowconfigure(0, weight=1)
        self._build_status_bar()

        # 화면은 뼈대가 준비된 뒤 생성 (서로 상태를 공유)
        from .page_edit import EditPage
        from .page_export import ExportPage
        from .page_library import LibraryPage
        from .page_shorts import ShortsPage

        for key, cls in (("library", LibraryPage), ("edit", EditPage), ("export", ExportPage), ("shorts", ShortsPage)):
            page = cls(self.page_host, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page

        self.store.subscribe("videos", self._update_summary)
        self.store.subscribe("segments", self._update_summary)
        for topic in ("videos", "segments", "highlights"):
            self.store.subscribe(topic, self._schedule_autosave)
        self.bind("<KeyPress>", self._on_key)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # 지난번 설정 복원
        self.settings.apply(persist.load_settings())
        for page in self.pages.values():
            if hasattr(page, "on_settings_loaded"):
                page.on_settings_loaded()

        self.show_page("library")
        self._update_summary()
        self.after(300, self._detect_encoder_async)
        self.after(500, self._offer_resume)

    # ── 상단 ──

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="🚗 QuickDashCam", font=font(19, True)).grid(row=0, column=0, padx=(16, 18), pady=10)
        self.step_var = ctk.StringVar(value=PAGE_TITLES["library"])
        self.step_tabs = ctk.CTkSegmentedButton(
            header, values=list(PAGE_TITLES.values()), variable=self.step_var, height=36, font=font(14, True),
            command=self._on_step_clicked,
        )
        self.step_tabs.grid(row=0, column=1, sticky="w", pady=10)
        self.summary_label = ctk.CTkLabel(header, text="", font=font(12), text_color=FG_MUTED)
        self.summary_label.grid(row=0, column=2, padx=10)
        self.encoder_badge = ctk.CTkLabel(header, text="⚡ 장치 확인 중…", font=font(11, True), fg_color="#1e293b",
                                          corner_radius=6, padx=8, pady=3)
        self.encoder_badge.grid(row=0, column=3, padx=(4, 6))
        ctk.CTkButton(header, text="⌨ 단축키 (F1)", width=110, height=30, font=font(12), fg_color=NEUTRAL,
                      hover_color=NEUTRAL_HOVER, command=self.show_shortcuts).grid(row=0, column=4, padx=(0, 16))

    def _on_step_clicked(self, value: str) -> None:
        for key, title in PAGE_TITLES.items():
            if title == value:
                self.show_page(key)
                return

    def show_page(self, key: str) -> None:
        if key == self.current_page:
            return
        prev = self.pages.get(self.current_page) if self.current_page else None
        if prev is not None and hasattr(prev, "on_hide"):
            prev.on_hide()
        self.current_page = key
        self.step_var.set(PAGE_TITLES[key])
        page = self.pages[key]
        page.tkraise()
        if hasattr(page, "on_show"):
            page.on_show()

    def _update_summary(self) -> None:
        vids = self.store.videos
        if not vids:
            self.summary_label.configure(text="영상 없음")
            return
        analyzed = sum(1 for v in vids if v.analyzed)
        top = max(vids, key=lambda v: v.width * v.height)
        res = "4K" if min(top.width, top.height) >= 2000 else f"{min(top.width, top.height)}p"
        self.summary_label.configure(
            text=f"영상 {len(vids)}개 · {format_duration(self.store.total_duration)} · {res} · 분석 {analyzed}/{len(vids)}"
        )

    def _detect_encoder_async(self) -> None:
        def worker():
            try:
                enc, _ = detect_best_h264_encoder(prefer_hw=True)
                dec = detect_best_gpu_decoder()
                enc_txt = {"h264_nvenc": "NVIDIA NVENC", "h264_qsv": "Intel QSV", "h264_amf": "AMD AMF",
                           "h264_videotoolbox": "VideoToolbox", "h264_vaapi": "VAAPI", "libx264": "CPU"}.get(enc, enc)
                dec_txt = {"cuda_scale": "CUDA", "d3d11va": "D3D11VA", "videotoolbox": "VideoToolbox",
                           "vaapi": "VAAPI", "cpu": "CPU"}.get(dec, dec)
                self.after(0, lambda: self.encoder_badge.configure(text=f"⚡ 디코딩 {dec_txt} · 인코딩 {enc_txt}"))
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()

    # ── 상태 줄 · 작업 실행 ──

    def _build_status_bar(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(bar, text="① 영상 화면에서 주행 영상 폴더나 파일을 추가하세요.", font=font(12), anchor="w")
        self.status_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(6, 0))
        self.progress = ctk.CTkProgressBar(bar, height=8)
        self.progress.grid(row=1, column=0, sticky="ew", padx=14, pady=(4, 8))
        self.progress.set(0.0)
        self.btn_cancel = ctk.CTkButton(bar, text="⏹ 취소", width=90, height=30, fg_color="#dc2626",
                                        hover_color="#b91c1c", state="disabled", command=self.cancel_job)
        self.btn_cancel.grid(row=0, column=1, rowspan=2, padx=4, pady=6)
        self.btn_open = ctk.CTkButton(bar, text="📂 결과 폴더", width=110, height=30, fg_color=NEUTRAL,
                                      hover_color=NEUTRAL_HOVER, state="disabled", command=self.open_output_folder)
        self.btn_open.grid(row=0, column=2, rowspan=2, padx=(4, 14), pady=6)

    def set_status(self, text: str, progress: Optional[float] = None) -> None:
        self.status_label.configure(text=text)
        if progress is not None:
            self.progress.set(max(0.0, min(1.0, progress)))

    def run_job(
        self,
        title: str,
        fn: Callable[[Callable[[float, str], None], Callable[[], bool]], object],
        on_done: Optional[Callable[[object], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        오래 걸리는 작업을 백그라운드에서 실행. fn(report, cancelled)의 report(진행률 0~1, 메시지)는 어느 스레드에서나 호출 가능.
        작업 중에는 state.busy=True와 "busy" 알림으로 각 화면이 실행 버튼을 잠급니다.
        """
        if self.store.busy:
            return False
        self.store.busy = True
        self.store.cancel_requested = False
        self.btn_cancel.configure(state="normal")
        self.set_status(title, 0.0)
        self.store.emit("busy")

        t0 = time.monotonic()

        def report(frac: float, msg: str) -> None:
            # 경과 시간과, 진행률이 어느 정도 쌓이면 남은 시간 추정치를 함께 표시
            elapsed = time.monotonic() - t0
            extra = ""
            if elapsed >= 3.0:
                extra = f"   ·   경과 {format_duration(elapsed)}"
                if 0.05 <= frac < 0.999:
                    extra += f" · 남은 약 {format_duration(elapsed * (1.0 - frac) / frac)}"
            self.after(0, lambda f=frac, m=msg + extra: self.set_status(m, f))

        def cancelled() -> bool:
            return self.store.cancel_requested

        def worker():
            try:
                result = fn(report, cancelled)
                err: Optional[str] = None
            except Exception as exc:  # noqa: BLE001
                result, err = None, str(exc) or type(exc).__name__

            def finish():
                was_cancelled = self.store.cancel_requested
                self.store.busy = False
                self.store.cancel_requested = False
                self.btn_cancel.configure(state="disabled")
                self.store.emit("busy")
                if err is not None or was_cancelled:
                    msg = "작업이 취소되었습니다." if (was_cancelled or "취소" in (err or "")) else (err or "")
                    self.set_status(f"❌ {msg}" if not was_cancelled else f"⏹ {msg}", 0.0)
                    if on_error:
                        on_error(msg)
                    elif not was_cancelled and "취소" not in msg:
                        messagebox.showerror("오류", msg)
                    return
                if on_done:
                    on_done(result)
            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def cancel_job(self) -> None:
        if self.store.busy:
            self.store.cancel_requested = True
            self.set_status("취소하는 중… 잠시만 기다려 주세요.")

    def set_output(self, path) -> None:
        self.store.last_output_path = path
        self.btn_open.configure(state="normal")

    def open_output_folder(self) -> None:
        p = self.store.last_output_path
        if not p or not p.exists():
            return
        folder = p.parent
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:
            pass

    # ── 단축키 ──

    def show_shortcuts(self) -> None:
        page = self.pages.get(self.current_page or "")
        title = PAGE_TITLES.get(self.current_page or "", "")
        body = getattr(page, "SHORTCUTS", "이 화면에는 단축키가 없습니다.")
        messagebox.showinfo(f"단축키 — {title}", body + "\n\n공통: F1 = 이 안내")

    def _on_key(self, event) -> None:
        if event.keysym == "F1":
            self.show_shortcuts()
            return
        focus = self.focus_get()
        if isinstance(focus, (tk.Entry, tk.Text)):
            return
        page = self.pages.get(self.current_page or "")
        if page is not None and hasattr(page, "handle_key"):
            page.handle_key(event)

    # ── 작업 저장 · 이어하기 ──

    def _schedule_autosave(self) -> None:
        if self._restoring:
            return
        if self._autosave_job is not None:
            try:
                self.after_cancel(self._autosave_job)
            except Exception:
                pass
        self._autosave_job = self.after(1500, self._autosave)

    def _autosave(self) -> None:
        self._autosave_job = None
        if self.store.videos:
            persist.save_session(self.store)
        elif self._resume_checked:
            persist.clear_session()  # 목록을 모두 비웠으면 다음 실행 때 이어하기를 묻지 않음

    def _offer_resume(self) -> None:
        if self.store.videos:
            self._resume_checked = True
            return
        data = persist.load_session()
        self._resume_checked = True
        if data is None:
            return
        n_v, n_c = len(data.get("videos", [])), sum(1 for c in data.get("highlights", []) if c.get("included"))
        when = str(data.get("saved_at", "")).replace("T", " ")
        if not messagebox.askyesno(
            "지난 작업 이어하기",
            f"지난번 작업을 이어서 할까요?\n\n저장 시각: {when}\n영상 {n_v}개 · 쇼츠 컷 {n_c}개\n"
            "(구간 편집·리뷰 결정·컷 맞춤이 그대로 복원됩니다)",
        ):
            # 실수로 '아니요'를 눌러도 되살릴 수 있게 지우지 않고 백업으로 옮김
            try:
                src = persist.config_dir() / "session.json"
                src.replace(persist.config_dir() / "session.previous.json")
            except OSError:
                pass
            return
        self._restoring = True
        try:
            n, missing = persist.restore_session(self.store, data)
            self.store.emit("videos", "segments", "highlights")
        finally:
            self._restoring = False
        if missing:
            messagebox.showwarning("일부 영상을 찾지 못함",
                                   "다음 영상은 옮겨졌거나 지워져서 빼고 복원했습니다:\n\n" + "\n".join(missing[:15]))
        self._load_caches_after_restore()
        self.show_page("shorts" if self.store.highlight_clips else ("edit" if any(v.analyzed for v in self.store.videos) else "library"))
        self.set_status(f"✅ 지난 작업을 이어서 열었습니다 — 영상 {n}개, 쇼츠 컷 {len(self.store.highlight_clips)}개", 1.0)

    def _load_caches_after_restore(self) -> None:
        from ..analyzer import load_cached_features
        from ..highlights import _appear_cache_path, ensure_appearance

        videos = list(self.store.videos)
        cfg = self.store.analysis_config

        def job(report, cancelled):
            for i, v in enumerate(videos):
                if cancelled():
                    break
                report((i + 1) / max(1, len(videos)), f"저장된 분석 결과 불러오는 중 ({i + 1}/{len(videos)}): {v.filename}")
                load_cached_features(v, cfg.sample_fps)
                cache = _appear_cache_path(v.path)
                if v.feature_cache is not None and cache is not None and cache.exists():
                    ensure_appearance(v)
            return None

        self.run_job("저장된 분석 결과 불러오는 중…", job, lambda _r: self.set_status("✅ 지난 작업을 이어서 열었습니다.", 1.0))

    def _on_close(self) -> None:
        if self.store.busy:
            if not messagebox.askyesno("작업 중", "작업이 진행 중입니다. 취소하고 종료할까요?"):
                return
            self.cancel_job()
            self._close_when_idle(0)
            return
        self._shutdown()

    def _close_when_idle(self, waited_ms: int) -> None:
        # 진행 중인 FFmpeg가 정리될 때까지 잠시 기다린 뒤 종료 (최대 약 5초)
        if self.store.busy and waited_ms < 5000:
            self.after(100, lambda: self._close_when_idle(waited_ms + 100))
            return
        self._shutdown()

    def _shutdown(self) -> None:
        if self._autosave_job is not None:
            try:
                self.after_cancel(self._autosave_job)
            except Exception:
                pass
        if self.store.videos:
            persist.save_session(self.store)
        elif self._resume_checked:
            persist.clear_session()
        persist.save_settings(self.settings.snapshot())
        for page in self.pages.values():
            if hasattr(page, "shutdown"):
                try:
                    page.shutdown()
                except Exception:
                    pass
        self.thumbs.close()
        self.destroy()

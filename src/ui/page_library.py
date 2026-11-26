"""① 영상: 주행 영상 불러오기, 시간순 정렬·순서 정리, 영상 정보 확인."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

import customtkinter as ctk
import numpy as np

from ..ffmpeg_utils import VIDEO_EXTENSIONS, probe_video_file
from ..models import SortMode, VideoFileInfo
from ..sorter import scan_folder_for_videos, sort_video_files
from .theme import (
    ACCENT, ACCENT_HOVER, BG_DEEP, FG_MUTED, FG_SOFT, NEUTRAL, NEUTRAL_HOVER, PRIMARY, PRIMARY_HOVER,
    font, format_duration, make_photo_image,
)
from .widgets import card, make_tree, tree_selected_index

THUMB_W, THUMB_H = 96, 54
DETAIL_W, DETAIL_H = 352, 198
SORT_LABELS = {"촬영 시각 순 (권장)": SortMode.CHRONOLOGICAL, "파일 수정일 순": SortMode.FILE_MTIME, "파일명 순": SortMode.FILENAME}


def _thumb_time(v: VideoFileInfo) -> float:
    return min(max(1.0, v.duration_sec * 0.1), max(0.0, v.duration_sec - 0.5))


class LibraryPage(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.store = app.store
        self._thumb_photos: Dict[str, tk.PhotoImage] = {}
        self._detail_photo: Optional[tk.PhotoImage] = None
        self._selected: Optional[int] = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_list()
        self._build_detail()
        self.store.subscribe("videos", self.refresh)
        self.store.subscribe("segments", self.refresh)
        self.store.subscribe("busy", self._on_busy)
        self.refresh()

    # ── 화면 구성 ──

    def _build_list(self) -> None:
        outer, body = card(self, "주행 영상", "폴더나 파일을 추가하면 촬영 시각 순으로 자동 정렬됩니다. 목록 순서대로 이어 붙여집니다.")
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        body.grid_rowconfigure(1, weight=1)

        bar = ctk.CTkFrame(body, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.btn_folder = ctk.CTkButton(bar, text="📁 폴더 추가", width=120, height=36, font=font(13, True),
                                        fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.add_folder)
        self.btn_folder.pack(side="left", padx=(0, 6))
        self.btn_files = ctk.CTkButton(bar, text="➕ 파일 추가", width=120, height=36, font=font(13, True),
                                       fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.add_files)
        self.btn_files.pack(side="left", padx=(0, 16))
        ctk.CTkLabel(bar, text="정렬", font=font(12), text_color=FG_MUTED).pack(side="left", padx=(0, 6))
        self.sort_var = ctk.StringVar(value="촬영 시각 순 (권장)")
        ctk.CTkOptionMenu(bar, values=list(SORT_LABELS), variable=self.sort_var, width=170, height=34,
                          command=lambda _v: self.resort()).pack(side="left")
        self.btn_clear = ctk.CTkButton(bar, text="전체 비우기", width=100, height=34, fg_color=NEUTRAL,
                                       hover_color=NEUTRAL_HOVER, command=self.clear_all)
        self.btn_clear.pack(side="right")

        cols = [
            ("no", "순서", 50, "center", False),
            ("name", "파일명", 280, "w", True),
            ("time", "촬영 시각", 160, "center", False),
            ("dur", "길이", 90, "e", False),
            ("video", "영상", 150, "center", False),
            ("audio", "소리", 60, "center", False),
            ("status", "분석", 170, "w", False),
        ]
        self.tree, box = make_tree(body, cols, "Library.Treeview", row_height=THUMB_H + 8, with_image=True,
                                   image_width=THUMB_W + 12)
        box.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        foot = ctk.CTkFrame(body, fg_color="transparent")
        foot.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        foot.grid_columnconfigure(0, weight=1)
        self.total_label = ctk.CTkLabel(foot, text="", font=font(13, True), anchor="w")
        self.total_label.grid(row=0, column=0, sticky="ew")
        self.btn_next = ctk.CTkButton(foot, text="다음: ② 편집 →", width=170, height=38, font=font(14, True),
                                      fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                                      command=lambda: self.app.show_page("edit"))
        self.btn_next.grid(row=0, column=1, padx=(6, 0))

    def _build_detail(self) -> None:
        outer, body = card(self, "선택한 영상")
        outer.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        outer.configure(width=400)
        self.detail_canvas = tk.Canvas(body, width=DETAIL_W, height=DETAIL_H, bg=BG_DEEP, highlightthickness=0)
        self.detail_canvas.grid(row=0, column=0, pady=(0, 10))
        self.detail_info = ctk.CTkLabel(body, text="", font=font(12), text_color=FG_SOFT, justify="left", anchor="nw")
        self.detail_info.grid(row=1, column=0, sticky="new")
        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        for col in range(3):
            btns.grid_columnconfigure(col, weight=1)
        self.btn_up = ctk.CTkButton(btns, text="▲ 위로", height=34, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                                    command=lambda: self.move(-1))
        self.btn_up.grid(row=0, column=0, padx=2, sticky="ew")
        self.btn_down = ctk.CTkButton(btns, text="▼ 아래로", height=34, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
                                      command=lambda: self.move(1))
        self.btn_down.grid(row=0, column=1, padx=2, sticky="ew")
        self.btn_remove = ctk.CTkButton(btns, text="✕ 제외 (Del)", height=34, fg_color="#dc2626", hover_color="#b91c1c",
                                        command=self.remove_selected)
        self.btn_remove.grid(row=0, column=2, padx=2, sticky="ew")
        ctk.CTkLabel(body, text="💡 순서가 틀렸다면 ▲▼로 바로잡을 수 있습니다.\n    목록 순서대로 이어 붙여 편집·합치기·쇼츠에 쓰입니다.",
                     font=font(11), text_color=FG_MUTED, justify="left", anchor="w").grid(row=3, column=0, sticky="ew", pady=(14, 0))

    # ── 표시 ──

    def refresh(self) -> None:
        vids = self.store.videos
        tree = self.tree
        tree.delete(*tree.get_children())
        for i, v in enumerate(vids):
            key = str(v.path)
            photo = self._thumb_photos.get(key)
            if photo is None:
                self._request_thumb(v)
            status = "⏳ 분석 전"
            if v.analyzed:
                stop, straight = v.stop_duration, v.straight_duration
                status = f"✔ 정차 {format_duration(stop)} · 직진 {format_duration(straight)}"
            elif v.has_manual_cuts:
                status = "✂ 수동 편집됨"
            values = (
                i + 1, v.filename,
                v.recorded_at.strftime("%Y-%m-%d %H:%M:%S") if v.recorded_at else "-",
                format_duration(v.duration_sec),
                f"{v.width}×{v.height} · {v.fps:.0f}fps",
                "있음" if v.has_audio else "없음",
                status,
            )
            if photo is not None:
                tree.insert("", "end", iid=str(i), image=photo, values=values)
            else:
                tree.insert("", "end", iid=str(i), values=values)
        if vids:
            sel = self._selected if self._selected is not None and self._selected < len(vids) else 0
            tree.selection_set(str(sel))
            tree.see(str(sel))
        else:
            self._selected = None
        total = self.store.total_duration
        self.total_label.configure(
            text=f"총 {len(vids)}개 · {format_duration(total)}" if vids else "아직 추가된 영상이 없습니다 — 위의 [📁 폴더 추가] 또는 [➕ 파일 추가]를 눌러 주세요."
        )
        self._show_detail()
        self._on_busy()

    def _request_thumb(self, v: VideoFileInfo) -> None:
        def ready(arr: np.ndarray, key=str(v.path)):
            def apply():
                try:
                    self._thumb_photos[key] = make_photo_image(arr)
                    for iid in self.tree.get_children():
                        idx = int(iid)
                        if idx < len(self.store.videos) and str(self.store.videos[idx].path) == key:
                            self.tree.item(iid, image=self._thumb_photos[key])
                except Exception:
                    pass
            self.after(0, apply)
        self.app.thumbs.request(v.path, _thumb_time(v), THUMB_W, THUMB_H, ready)

    def _on_select(self, _e=None) -> None:
        idx = tree_selected_index(self.tree)
        if idx is not None:
            self._selected = idx
            self._show_detail()

    def _show_detail(self) -> None:
        cv = self.detail_canvas
        cv.delete("all")
        vids = self.store.videos
        if self._selected is None or self._selected >= len(vids):
            cv.create_text(DETAIL_W // 2, DETAIL_H // 2, text="영상을 선택하세요", fill=FG_MUTED, font=("Malgun Gothic", 12))
            self.detail_info.configure(text="")
            return
        v = vids[self._selected]

        def ready(arr: np.ndarray, key=str(v.path)):
            def apply():
                cur = self.store.videos[self._selected] if self._selected is not None and self._selected < len(self.store.videos) else None
                if cur is not None and str(cur.path) == key:
                    self._detail_photo = make_photo_image(arr)
                    cv.delete("all")
                    cv.create_image(DETAIL_W // 2, DETAIL_H // 2, image=self._detail_photo)
            self.after(0, apply)
        self.app.thumbs.request(v.path, _thumb_time(v), DETAIL_W, DETAIL_H, ready)

        lines = [
            f"📄 {v.filename}",
            f"🕒 {v.recorded_at.strftime('%Y-%m-%d %H:%M:%S') if v.recorded_at else '-'}  ({v.timestamp_source} 기준)",
            f"⏱ 길이 {format_duration(v.duration_sec)}",
            f"🎞 {v.width}×{v.height} · {v.fps:.2f}fps · 소리 {'있음' if v.has_audio else '없음'}",
            f"💾 {v.file_size_bytes / (1024 ** 3):.2f} GB" if v.file_size_bytes else "",
        ]
        if v.analyzed:
            lines.append(f"🔍 정차 {format_duration(v.stop_duration)} · 직진 {format_duration(v.straight_duration)} · "
                         f"삭제 {format_duration(v.manual_cut_duration)}")
        else:
            lines.append("🔍 아직 분석 전 (② 편집 화면에서 자동 구간 분석)")
        self.detail_info.configure(text="\n".join(x for x in lines if x))

    def _on_busy(self) -> None:
        busy = self.store.busy
        st = "disabled" if busy else "normal"
        for b in (self.btn_folder, self.btn_files, self.btn_clear, self.btn_up, self.btn_down, self.btn_remove):
            b.configure(state=st)
        self.btn_next.configure(state="normal" if self.store.videos else "disabled")

    # ── 동작 ──

    def add_folder(self) -> None:
        if self.store.busy:
            return
        folder = filedialog.askdirectory(title="주행 영상이 들어 있는 폴더 선택")
        if not folder:
            return
        paths = scan_folder_for_videos(folder, recursive=False)
        if not paths:
            messagebox.showinfo("알림", "선택한 폴더에서 영상 파일을 찾지 못했습니다.")
            return
        self.load_paths(paths)

    def add_files(self) -> None:
        if self.store.busy:
            return
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS)) + " " + " ".join(f"*{e.upper()}" for e in sorted(VIDEO_EXTENSIONS))
        files = filedialog.askopenfilenames(title="주행 영상 파일 선택", filetypes=[("영상 파일", exts), ("모든 파일", "*.*")])
        if files:
            self.load_paths([Path(f) for f in files])

    def load_paths(self, paths: List[Path]) -> None:
        existing = {v.path for v in self.store.videos}
        new_paths = [Path(p).resolve() for p in paths if Path(p).resolve() not in existing]
        if not new_paths:
            messagebox.showinfo("알림", "선택한 영상들은 이미 목록에 있습니다.")
            return

        def job(report, cancelled):
            loaded: List[VideoFileInfo] = []
            for i, p in enumerate(new_paths):
                if cancelled():
                    break
                report((i + 1) / len(new_paths), f"영상 정보 읽는 중 ({i + 1}/{len(new_paths)}): {p.name}")
                try:
                    info = probe_video_file(p)
                    if info.duration_sec > 0.1:
                        info.ensure_segments_initialized()
                        loaded.append(info)
                except Exception:
                    continue
            return loaded

        def done(loaded):
            self.store.videos = sort_video_files(self.store.videos + list(loaded), self.store.sort_mode)
            self.store.clear_undo()
            self.app.set_status(f"✅ 영상 {len(loaded)}개를 추가했습니다 (총 {len(self.store.videos)}개, 촬영 시각 순 정렬). "
                                "다음 단계: ② 편집에서 자동 구간 분석", 1.0)
            self.store.emit("videos")

        self.app.run_job("영상 정보를 읽는 중…", job, done)

    def resort(self) -> None:
        self.store.sort_mode = SORT_LABELS.get(self.sort_var.get(), SortMode.CHRONOLOGICAL)
        if self.store.busy or not self.store.videos:
            return
        self.store.videos = sort_video_files(self.store.videos, self.store.sort_mode)
        self.store.clear_undo()
        self.store.emit("videos")

    def move(self, delta: int) -> None:
        idx = self._selected
        vids = self.store.videos
        if self.store.busy or idx is None or not (0 <= idx + delta < len(vids)):
            return
        vids[idx], vids[idx + delta] = vids[idx + delta], vids[idx]
        self._selected = idx + delta
        self.store.clear_undo()
        self.store.emit("videos")

    def remove_selected(self) -> None:
        idx = self._selected
        if self.store.busy or idx is None or idx >= len(self.store.videos):
            return
        removed = self.store.videos.pop(idx)
        self.store.highlight_clips = [c for c in self.store.highlight_clips if c.video is not removed]
        self._selected = min(idx, len(self.store.videos) - 1) if self.store.videos else None
        self.store.clear_undo()
        self.store.emit("videos", "highlights")

    def clear_all(self) -> None:
        if self.store.busy or not self.store.videos:
            return
        if not messagebox.askyesno("전체 비우기", "목록의 영상을 모두 비울까요? (원본 파일은 지워지지 않습니다)"):
            return
        self.store.videos = []
        self.store.highlight_clips = []
        self._selected = None
        self.store.clear_undo()
        self.store.emit("videos", "highlights")
        self.app.set_status("목록을 비웠습니다.", 0.0)

    def handle_key(self, event) -> None:
        if event.keysym == "Delete":
            self.remove_selected()

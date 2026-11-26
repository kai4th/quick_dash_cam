"""화면 공통 UI 부품: 카드 패널, 접이식 섹션, 목록 표(썸네일 지원)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Sequence, Tuple

import customtkinter as ctk

from .theme import BG_CARD, FG_MUTED, font


def card(master, title: Optional[str] = None, subtitle: Optional[str] = None, wrap: int = 420, **kw) -> Tuple[ctk.CTkFrame, ctk.CTkFrame]:
    """제목이 있는 카드. 반환: (카드 프레임, 내용을 넣을 프레임)."""
    outer = ctk.CTkFrame(master, fg_color=BG_CARD, corner_radius=10, **kw)
    outer.grid_columnconfigure(0, weight=1)
    r = 0
    if title:
        ctk.CTkLabel(outer, text=title, font=font(14, True), anchor="w").grid(row=r, column=0, sticky="ew", padx=14, pady=(12, 0))
        r += 1
    if subtitle:
        ctk.CTkLabel(outer, text=subtitle, font=font(11), text_color=FG_MUTED, anchor="w", justify="left",
                     wraplength=wrap).grid(row=r, column=0, sticky="ew", padx=14, pady=(2, 0))
        r += 1
    body = ctk.CTkFrame(outer, fg_color="transparent")
    body.grid(row=r, column=0, sticky="nsew", padx=12, pady=(8, 12))
    body.grid_columnconfigure(0, weight=1)
    outer.grid_rowconfigure(r, weight=1)
    return outer, body


class Section(ctk.CTkFrame):
    """제목을 눌러 접고 펼 수 있는 설정 묶음."""

    def __init__(self, master, title: str, opened: bool = True, **kw):
        super().__init__(master, fg_color=BG_CARD, corner_radius=8, **kw)
        self.grid_columnconfigure(0, weight=1)
        self._title = title
        self._open = opened
        self.header = ctk.CTkButton(self, text="", anchor="w", height=34, font=font(13, True),
                                    fg_color="transparent", hover_color="#273449", command=self.toggle)
        self.header.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_columnconfigure(1, weight=1)
        self._apply()

    def toggle(self) -> None:
        self._open = not self._open
        self._apply()

    def _apply(self) -> None:
        self.header.configure(text=("▾  " if self._open else "▸  ") + self._title)
        if self._open:
            self.body.grid(row=1, column=0, sticky="ew", padx=12, pady=(2, 10))
        else:
            self.body.grid_remove()


def field_label(master, text: str, row: int, col: int = 0, colspan: int = 2) -> ctk.CTkLabel:
    lbl = ctk.CTkLabel(master, text=text, font=font(12), text_color=FG_MUTED, anchor="w")
    lbl.grid(row=row, column=col, columnspan=colspan, sticky="ew", pady=(6, 0))
    return lbl


_style_done: set = set()


def make_tree(
    master,
    columns: Sequence[Tuple[str, str, int, str, bool]],
    style_name: str,
    row_height: int = 28,
    with_image: bool = False,
    image_width: int = 0,
    selectmode: str = "browse",
) -> Tuple[ttk.Treeview, ctk.CTkFrame]:
    """
    어두운 테마의 목록 표. columns: (id, 제목, 너비, 정렬(w/e/center), 늘어남 여부).
    with_image=True면 맨 앞 칸(#0)에 썸네일 이미지를 넣을 수 있습니다.
    반환: (Treeview, 감싼 프레임)
    """
    box = ctk.CTkFrame(master, fg_color=BG_CARD, corner_radius=8)
    box.grid_columnconfigure(0, weight=1)
    box.grid_rowconfigure(0, weight=1)
    scaling = ctk.ScalingTracker.get_widget_scaling(master)
    style = ttk.Style(master)
    if style_name not in _style_done:
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(style_name, background=BG_CARD, fieldbackground=BG_CARD, foreground="#e2e8f0",
                        rowheight=int(round(row_height * scaling)), borderwidth=0,
                        font=("Malgun Gothic", int(round(-12 * scaling))))
        style.configure(f"{style_name}.Heading", background="#334155", foreground="#e2e8f0", relief="flat",
                        padding=(0, int(round(5 * scaling))), font=("Malgun Gothic", int(round(-12 * scaling)), "bold"))
        style.map(style_name, background=[("selected", "#0369a1")], foreground=[("selected", "#ffffff")])
        style.map(f"{style_name}.Heading", background=[("active", "#475569")])
        _style_done.add(style_name)
    tree = ttk.Treeview(box, columns=[c[0] for c in columns], show="tree headings" if with_image else "headings",
                        style=style_name, selectmode=selectmode)
    if with_image:
        tree.column("#0", width=int(round(image_width * scaling)), stretch=False, anchor="center")
        tree.heading("#0", text="")
    for cid, title, width, anchor, stretch in columns:
        tree.heading(cid, text=title)
        tree.column(cid, width=int(round(width * scaling)), anchor=anchor, stretch=stretch)
    tree.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
    sb = ctk.CTkScrollbar(box, command=tree.yview)
    sb.grid(row=0, column=1, sticky="ns", pady=4)
    tree.configure(yscrollcommand=sb.set)
    return tree, box


def tree_selected_index(tree: ttk.Treeview) -> Optional[int]:
    sel = tree.selection()
    if not sel:
        return None
    try:
        return int(sel[0])
    except ValueError:
        return None


def stylize_entry_isolation(entry: ctk.CTkEntry) -> None:
    """입력 칸에서 글자를 칠 때 최상위 창 단축키가 동작하지 않도록 바인딩 분리."""
    inner = getattr(entry, "_entry", None)
    if inner is not None:
        top_tag = str(inner.winfo_toplevel())
        inner.bindtags(tuple(t for t in inner.bindtags() if t != top_tag))



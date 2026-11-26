"""
화면들이 함께 쓰는 앱 상태.

- videos: 정렬된 영상 목록 (구간 정보 포함), highlight_clips: 쇼츠 후보·선정 컷, review_items: 검출 구간 리뷰
- subscribe/emit: 한 화면에서 바꾼 내용을 다른 화면이 자동으로 다시 그리도록 알림
  주제: "videos"(목록·순서), "segments"(구간 편집·분석), "highlights"(쇼츠 컷), "busy"(작업 중 여부)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..models import (
    AnalysisConfig,
    HighlightClip,
    SegmentType,
    SortMode,
    VideoFileInfo,
)
from .media import PlaybackClip


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


def decision_type(it: ReviewItem, decision: str) -> SegmentType:
    return {"keep": it.orig_type, "normal": SegmentType.NORMAL, "cut": SegmentType.MANUAL_CUT}[decision]


class AppState:
    def __init__(self) -> None:
        self.videos: List[VideoFileInfo] = []
        self.highlight_clips: List[HighlightClip] = []
        self.review_items: List[ReviewItem] = []
        self.sort_mode: SortMode = SortMode.CHRONOLOGICAL
        self.analysis_config: AnalysisConfig = AnalysisConfig()
        self.last_output_path: Optional[Path] = None
        self.busy: bool = False
        self.cancel_requested: bool = False
        self._undo: List[Tuple[List[list], List[ReviewItem]]] = []
        self._listeners: Dict[str, List[Callable[[], None]]] = {}

    # ── 알림 ──

    def subscribe(self, topic: str, callback: Callable[[], None]) -> None:
        self._listeners.setdefault(topic, []).append(callback)

    def emit(self, *topics: str) -> None:
        if "videos" in topics or "segments" in topics:
            self.rebuild_review_items()
        for topic in topics:
            for cb in list(self._listeners.get(topic, [])):
                try:
                    cb()
                except Exception:
                    import traceback
                    traceback.print_exc()

    # ── 타임라인 ──

    @property
    def total_duration(self) -> float:
        return sum(v.duration_sec for v in self.videos)

    def clip_offset(self, video: VideoFileInfo) -> Optional[float]:
        cursor = 0.0
        for v in self.videos:
            if v is video:
                return cursor
            cursor += v.duration_sec
        return None

    def playback_clips(self) -> List[PlaybackClip]:
        clips: List[PlaybackClip] = []
        cursor = 0.0
        for v in self.videos:
            clips.append(PlaybackClip(v.path, cursor, v.duration_sec, v.width, v.height, float(v.fps or 0.0)))
            cursor += v.duration_sec
        return clips

    def highlight_global(self, clip: HighlightClip) -> Optional[Tuple[float, float]]:
        off = self.clip_offset(clip.video)
        if off is None:
            return None
        return off + clip.start_sec, off + clip.end_sec

    # ── 되돌리기 ──

    def push_undo(self) -> None:
        self._undo.append(([copy.deepcopy(v.segments) for v in self.videos], [it.clone() for it in self.review_items]))
        if len(self._undo) > 40:
            self._undo.pop(0)

    def can_undo(self) -> bool:
        return bool(self._undo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        seg_snapshot, review_snapshot = self._undo.pop()
        for v, old in zip(self.videos, seg_snapshot):
            v.segments = old
        self.review_items = review_snapshot
        return True

    def clear_undo(self) -> None:
        self._undo.clear()

    # ── 리뷰 항목 ──

    @staticmethod
    def type_coverage(v: VideoFileInfo, s: float, e: float, seg_type: SegmentType) -> float:
        dur = max(1e-6, e - s)
        cov = 0.0
        for seg in v.segments:
            if seg.segment_type == seg_type:
                cov += max(0.0, min(e, seg.end_sec) - max(s, seg.start_sec))
        return cov / dur

    def rebuild_review_items(self) -> None:
        """현재 구간 상태로부터 리뷰 항목 목록을 다시 만들고, 사용자의 기존 결정을 유지합니다."""
        present = {id(v) for v in self.videos}
        kept: List[ReviewItem] = []
        for it in self.review_items:
            if id(it.video) not in present or it.decision not in ("normal", "cut"):
                continue
            want = SegmentType.NORMAL if it.decision == "normal" else SegmentType.MANUAL_CUT
            if self.type_coverage(it.video, it.start_sec, it.end_sec, want) >= 0.5:
                kept.append(it)
        fresh: List[ReviewItem] = []
        for v in self.videos:
            for seg in v.segments:
                if seg.segment_type in (SegmentType.STOP, SegmentType.STRAIGHT) and seg.duration >= 0.05:
                    fresh.append(ReviewItem(v, seg.start_sec, seg.end_sec, seg.segment_type,
                                            "keep" if seg.reviewed else None))
        order = {id(v): i for i, v in enumerate(self.videos)}
        items = kept + fresh
        items.sort(key=lambda it: (order[id(it.video)], it.start_sec))
        self.review_items = items

    def review_global(self, it: ReviewItem) -> Optional[Tuple[float, float]]:
        off = self.clip_offset(it.video)
        if off is None:
            return None
        return off + it.start_sec, off + it.end_sec

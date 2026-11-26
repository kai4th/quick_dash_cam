"""
작업(세션)과 설정을 사용자 폴더(~/.quick_dash_cam)에 저장·복원합니다.

- session.json: 영상 목록(읽어 둔 영상 정보 + 구간 + 분석 여부), 리뷰 결정, 쇼츠 컷(맞춤 포함), 정렬 방식
- settings.json: 각 화면 설정값, 마지막 폴더, 배경음악 경로
영상 원본이 옮겨졌거나 지워졌으면 그 영상(과 그 영상의 쇼츠 컷)만 빼고 복원합니다.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import HighlightClip, Segment, SegmentType, SortMode, VideoFileInfo
from .state import AppState, ReviewItem

SESSION_VERSION = 1


def config_dir() -> Path:
    base = os.environ.get("QUICK_DASH_CAM_HOME")
    d = Path(base) if base else Path.home() / ".quick_dash_cam"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)  # 저장 도중 꺼져도 이전 파일이 깨지지 않게


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── 세션 ──

def _video_to_dict(v: VideoFileInfo) -> Dict[str, Any]:
    try:
        mtime = int(v.path.stat().st_mtime)
    except OSError:
        mtime = 0
    return {
        "path": str(v.path), "filename": v.filename, "duration_sec": v.duration_sec, "width": v.width,
        "height": v.height, "fps": v.fps, "has_audio": v.has_audio,
        "recorded_at": v.recorded_at.isoformat() if v.recorded_at else None,
        "timestamp_source": v.timestamp_source, "file_size_bytes": v.file_size_bytes,
        "gopro_clip_key": list(v.gopro_clip_key) if v.gopro_clip_key else None,
        "analyzed": v.analyzed, "mtime": mtime,
        "segments": [[s.start_sec, s.end_sec, s.segment_type.value, s.reviewed] for s in v.segments],
    }


def _video_from_dict(d: Dict[str, Any]) -> Optional[VideoFileInfo]:
    p = Path(d["path"])
    try:
        st = p.stat()
    except OSError:
        return None
    if d.get("file_size_bytes") and st.st_size != d["file_size_bytes"]:
        return None  # 같은 이름의 다른 파일로 바뀜
    v = VideoFileInfo(
        path=p, filename=d["filename"], duration_sec=float(d["duration_sec"]), width=int(d["width"]),
        height=int(d["height"]), fps=float(d["fps"]), has_audio=bool(d["has_audio"]),
        recorded_at=datetime.fromisoformat(d["recorded_at"]) if d.get("recorded_at") else None,
        timestamp_source=d.get("timestamp_source", "파일 수정일"), file_size_bytes=int(d.get("file_size_bytes") or 0),
        gopro_clip_key=tuple(d["gopro_clip_key"]) if d.get("gopro_clip_key") else None,
        analyzed=bool(d.get("analyzed")),
    )
    v.segments = [Segment(float(s), float(e), SegmentType(t), bool(r)) for s, e, t, r in d.get("segments", [])]
    v.ensure_segments_initialized()
    return v


def session_snapshot(store: AppState) -> Dict[str, Any]:
    index = {id(v): i for i, v in enumerate(store.videos)}
    reviews = [
        [index[id(it.video)], it.start_sec, it.end_sec, it.orig_type.value, it.decision]
        for it in store.review_items if it.decision in ("normal", "cut") and id(it.video) in index
    ]
    clips = [
        {"video": index[id(c.video)], "start": c.start_sec, "end": c.end_sec, "score": c.score, "reason": c.reason,
         "included": c.included, "ai_score": c.ai_score, "caption": c.caption, "roll_adjust": c.roll_adjust,
         "vp": list(c.vp) if c.vp else None}
        for c in store.highlight_clips if id(c.video) in index
    ]
    return {"version": SESSION_VERSION, "saved_at": datetime.now().isoformat(timespec="seconds"),
            "sort_mode": store.sort_mode.value, "videos": [_video_to_dict(v) for v in store.videos],
            "reviews": reviews, "highlights": clips}


def save_session(store: AppState) -> None:
    try:
        _write_json(config_dir() / "session.json", session_snapshot(store))
    except Exception:
        pass


def load_session() -> Optional[Dict[str, Any]]:
    data = _read_json(config_dir() / "session.json")
    if not data or data.get("version") != SESSION_VERSION or not data.get("videos"):
        return None
    return data


def clear_session() -> None:
    try:
        (config_dir() / "session.json").unlink()
    except OSError:
        pass


def restore_session(store: AppState, data: Dict[str, Any]) -> Tuple[int, List[str]]:
    """세션을 store에 복원. 반환: (복원한 영상 수, 찾지 못한 파일 이름 목록)."""
    videos: List[Optional[VideoFileInfo]] = [_video_from_dict(d) for d in data.get("videos", [])]
    missing = [d["filename"] for d, v in zip(data.get("videos", []), videos) if v is None]
    store.videos = [v for v in videos if v is not None]
    try:
        store.sort_mode = SortMode(data.get("sort_mode", SortMode.CHRONOLOGICAL.value))
    except ValueError:
        pass
    store.review_items = []
    for vi, s, e, t, dec in data.get("reviews", []):
        v = videos[vi] if 0 <= vi < len(videos) else None
        if v is not None:
            store.review_items.append(ReviewItem(v, float(s), float(e), SegmentType(t), dec))
    clips: List[HighlightClip] = []
    for c in data.get("highlights", []):
        v = videos[c["video"]] if 0 <= c["video"] < len(videos) else None
        if v is None:
            continue
        clips.append(HighlightClip(
            video=v, start_sec=float(c["start"]), end_sec=float(c["end"]), score=float(c.get("score") or 0.0),
            reason=c.get("reason") or "", included=bool(c.get("included", True)), ai_score=c.get("ai_score"),
            caption=c.get("caption") or "", roll_adjust=c.get("roll_adjust"),
            vp=tuple(c["vp"]) if c.get("vp") else None,
        ))
    store.highlight_clips = clips
    store.clear_undo()
    return len(store.videos), missing


# ── 설정 ──

class SettingsRegistry:
    """화면마다 설정 항목(읽기·쓰기 함수, 허용 값)을 등록해 두고 한꺼번에 저장·복원."""

    def __init__(self) -> None:
        self._items: Dict[str, Tuple[Callable[[], Any], Callable[[Any], None], Optional[List[Any]]]] = {}

    def register(self, key: str, getter: Callable[[], Any], setter: Callable[[Any], None],
                 allowed: Optional[List[Any]] = None) -> None:
        self._items[key] = (getter, setter, allowed)

    def register_var(self, key: str, var, allowed: Optional[List[Any]] = None) -> None:
        self.register(key, var.get, var.set, allowed)

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, (getter, _setter, _allowed) in self._items.items():
            try:
                out[key] = getter()
            except Exception:
                pass
        return out

    def apply(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            item = self._items.get(key)
            if item is None:
                continue
            _getter, setter, allowed = item
            if allowed is not None and value not in allowed:
                continue  # 이전 버전의 선택지 등 지금은 없는 값은 무시
            try:
                setter(value)
            except Exception:
                pass


def save_settings(values: Dict[str, Any]) -> None:
    try:
        _write_json(config_dir() / "settings.json", values)
    except Exception:
        pass


def load_settings() -> Dict[str, Any]:
    return _read_json(config_dir() / "settings.json") or {}

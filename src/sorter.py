from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from .ffmpeg_utils import VIDEO_EXTENSIONS, probe_video_file
from .models import SortMode, VideoFileInfo


def natural_sort_key(text: str) -> list:
    """숫자가 포함된 파일명을 자연스러운 순서(1, 2, 10)로 정렬하기 위한 키를 생성합니다."""
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", text)
    ]


def sort_video_files(
    videos: Iterable[VideoFileInfo],
    mode: SortMode = SortMode.CHRONOLOGICAL,
) -> List[VideoFileInfo]:
    """주어진 정렬 모드에 따라 액션캠 영상 목록을 시간순으로 정렬합니다."""
    items = list(videos)
    if not items:
        return []

    if mode == SortMode.FILENAME:
        # 파일명 정렬 시에도 GoPro 챕터(GX010042, GX020042)가 감지되면 클립 번호 우선 적용
        def filename_key(v: VideoFileInfo):
            if v.gopro_clip_key is not None:
                return (0, v.gopro_clip_key, natural_sort_key(v.filename))
            return (1, (0, 0), natural_sort_key(v.filename))

        return sorted(items, key=filename_key)

    if mode == SortMode.FILE_MTIME:
        def mtime_key(v: VideoFileInfo):
            try:
                mt = v.path.stat().st_mtime
            except Exception:
                mt = 0.0
            return (mt, v.gopro_clip_key or (0, 0), natural_sort_key(v.filename))

        return sorted(items, key=mtime_key)

    # 기본: SortMode.CHRONOLOGICAL (촬영 시각 + GoPro 챕터 보정 + 자연 파일명)
    # 동일 클립의 분할 챕터(예: GX010042, GX020042)가 파일 복사 과정에서 타임스탬프가 뒤집히더라도
    # 같은 클립 번호 내에서는 반드시 챕터 순서(01 -> 02 -> 03)를 유지하도록 보정합니다.
    clip_min_time: dict[int, datetime] = {}
    for v in items:
        if v.gopro_clip_key is not None and v.recorded_at is not None:
            clip_num = v.gopro_clip_key[0]
            if clip_num not in clip_min_time or v.recorded_at < clip_min_time[clip_num]:
                clip_min_time[clip_num] = v.recorded_at

    def chrono_key(v: VideoFileInfo):
        dt = v.recorded_at or datetime.min
        if v.gopro_clip_key is not None:
            clip_num, chapter_num = v.gopro_clip_key
            base_dt = clip_min_time.get(clip_num, dt)
            return (base_dt, clip_num, chapter_num, dt, natural_sort_key(v.filename))
        return (dt, 0, 0, dt, natural_sort_key(v.filename))

    return sorted(items, key=chrono_key)


def scan_folder_for_videos(folder_path: Path | str, recursive: bool = False) -> List[Path]:
    """폴더 내의 지원되는 모든 비디오 파일 경로를 찾습니다."""
    p = Path(folder_path)
    if not p.is_dir():
        return []
    iterator = p.rglob("*") if recursive else p.iterdir()
    found = [
        f.resolve()
        for f in iterator
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(found, key=lambda x: natural_sort_key(x.name))

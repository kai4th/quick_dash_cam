from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class SegmentType(str, Enum):
    NORMAL = "normal"           # 일반 주행 (회전, 시내 주행, 차선 변경 등 변화가 있는 구간)
    STOP = "stop"               # 신호대기 / 정차 구간 (화면 변화가 거의 없음)
    STRAIGHT = "straight"       # 장시간 고속도로 직진 / 단조로운 순항 구간
    MANUAL_CUT = "manual_cut"   # 사용자가 타임라인에서 직접 지정한 잘라내기(삭제) 구간


class SkipMode(str, Enum):
    TIMELAPSE = "timelapse"  # 배속 재생 (예: 4x ~ 16x 빠르게 감기)
    JUMPCUT = "jumpcut"      # 구간 잘라내기 (점프컷)


class AudioMode(str, Enum):
    KEEP_NORMAL_MUTE_SKIP = "keep_normal_mute_skip"  # 정상 구간 원본 오디오 유지, 배속/스킵 구간 음소거
    KEEP_AND_SPEED_UP = "keep_and_speed_up"          # 배속 구간도 오디오 속도를 함께 높여서 유지
    MUTE_ALL = "mute_all"                            # 전체 음소거


class VideoQualityPreset(str, Enum):
    LOSSLESS_LIKE = "lossless_like"  # 원본 유지급 최고화질 (시각적 무손실 수준, CRF 14 / CQ 15)
    HIGH = "high"                    # 고화질 권장 (원본에 가까운 수준, CRF 18 / CQ 19)
    BALANCED = "balanced"            # 표준 화질 (화질·용량 균형, CRF 22 / CQ 23)
    COMPACT = "compact"              # 저용량·고속 (빠른 인코딩 및 용량 절감, CRF 27 / CQ 28)


class VideoCodec(str, Enum):
    H264 = "h264"  # H.264 / AVC (최고 호환성)
    HEVC = "hevc"  # H.265 / HEVC (고효율·4K 고화질 권장)


class SortMode(str, Enum):
    CHRONOLOGICAL = "chronological"  # 메타데이터 생성시각 / 파일 수정시각 + 액션캠 챕터 규칙
    FILENAME = "filename"            # 파일 이름 오름차순
    FILE_MTIME = "file_mtime"        # 파일 수정 시각 오름차순


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    segment_type: SegmentType
    reviewed: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def output_duration(self, skip_mode: SkipMode, speed_multiplier: float) -> float:
        """이 구간이 최종 결과 영상에서 차지할 예상 시간(초)을 반환합니다."""
        dur = self.duration
        if dur <= 0:
            return 0.0
        if self.segment_type == SegmentType.MANUAL_CUT:
            # 수동 잘라내기 구간은 배속/점프컷 모드와 무관하게 항상 완전히 제거됨
            return 0.0
        if self.segment_type == SegmentType.NORMAL:
            return dur
        if skip_mode == SkipMode.TIMELAPSE:
            return dur / max(1.0, speed_multiplier)
        else:
            # JUMPCUT 모드: 전환 여유 시간(transition_padding_sec)은 인접 NORMAL 구간에 이미 반영되어 있으므로 0초
            return 0.0


def apply_segment_override(
    segments: List[Segment],
    total_duration: float,
    start_sec: float,
    end_sec: float,
    new_type: SegmentType,
    mark_reviewed: bool = True,
) -> List[Segment]:
    """
    기존 구간 리스트에서 [start_sec, end_sec] 범위를 new_type으로 덮어쓰고(분할/병합),
    정리된 새 Segment 리스트를 반환합니다.
    """
    total_duration = max(0.0, total_duration)
    if total_duration <= 0:
        return []

    s_cut = max(0.0, min(total_duration, min(start_sec, end_sec)))
    e_cut = max(0.0, min(total_duration, max(start_sec, end_sec)))

    base_segs = list(segments) if segments else [Segment(0.0, total_duration, SegmentType.NORMAL)]
    if e_cut - s_cut < 0.05:
        return base_segs

    updated: List[Segment] = []
    for seg in base_segs:
        if seg.end_sec <= s_cut + 1e-4 or seg.start_sec >= e_cut - 1e-4:
            updated.append(Segment(seg.start_sec, seg.end_sec, seg.segment_type, seg.reviewed))
            continue
        if seg.start_sec < s_cut - 1e-4:
            updated.append(Segment(seg.start_sec, s_cut, seg.segment_type, seg.reviewed))
        if seg.end_sec > e_cut + 1e-4:
            updated.append(Segment(e_cut, seg.end_sec, seg.segment_type, seg.reviewed))

    updated.append(Segment(s_cut, e_cut, new_type, mark_reviewed))
    updated.sort(key=lambda x: (x.start_sec, x.end_sec))

    merged: List[Segment] = []
    for seg in updated:
        if seg.duration <= 0.02:
            continue
        if merged and merged[-1].segment_type == seg.segment_type and abs(merged[-1].end_sec - seg.start_sec) < 0.05:
            merged[-1].end_sec = max(merged[-1].end_sec, seg.end_sec)
            merged[-1].reviewed = merged[-1].reviewed and seg.reviewed
        else:
            merged.append(Segment(seg.start_sec, seg.end_sec, seg.segment_type, seg.reviewed))

    return merged


def merge_preserving_manual_cuts(
    old_segments: List[Segment],
    analyzed_segments: List[Segment],
    total_duration: float,
) -> List[Segment]:
    """
    자동 분석 결과 위에 사용자가 기존에 직접 결정한 구간들을 그대로 보존하여 덧씌웁니다.
    - MANUAL_CUT(수동 잘라내기) 구간
    - 리뷰에서 확정(reviewed=True)한 구간: 스킵 유지 / 일반 주행 복원 / 경계 조정 결과
    """
    result = list(analyzed_segments)
    for seg in old_segments:
        if (seg.segment_type == SegmentType.MANUAL_CUT or seg.reviewed) and seg.duration >= 0.05:
            result = apply_segment_override(
                result, total_duration, seg.start_sec, seg.end_sec, seg.segment_type,
                mark_reviewed=True,
            )
    return result


@dataclass
class VideoFileInfo:
    path: Path
    filename: str
    duration_sec: float = 0.0
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    has_audio: bool = True
    recorded_at: Optional[datetime] = None
    timestamp_source: str = "파일 수정일"
    file_size_bytes: int = 0
    gopro_clip_key: Optional[tuple] = None
    segments: List[Segment] = field(default_factory=list)
    analyzed: bool = False
    feature_cache: Optional[dict] = field(default=None, repr=False)

    def ensure_segments_initialized(self) -> None:
        """아직 자동 분석 전이라도 수동 편집이 가능하도록 기본 NORMAL 구간을 보장합니다."""
        if not self.segments and self.duration_sec > 0:
            self.segments = [Segment(0.0, self.duration_sec, SegmentType.NORMAL)]

    @property
    def has_manual_cuts(self) -> bool:
        return any(s.segment_type == SegmentType.MANUAL_CUT for s in self.segments)

    @property
    def normal_duration(self) -> float:
        if not self.segments:
            return self.duration_sec
        return sum(s.duration for s in self.segments if s.segment_type == SegmentType.NORMAL)

    @property
    def stop_duration(self) -> float:
        return sum(s.duration for s in self.segments if s.segment_type == SegmentType.STOP)

    @property
    def straight_duration(self) -> float:
        return sum(s.duration for s in self.segments if s.segment_type == SegmentType.STRAIGHT)

    @property
    def manual_cut_duration(self) -> float:
        return sum(s.segment_type == SegmentType.MANUAL_CUT and s.duration or 0.0 for s in self.segments)

    def estimated_output_duration(
        self,
        skip_mode: SkipMode,
        speed_multiplier: float,
    ) -> float:
        if not self.segments:
            return self.duration_sec
        return sum(
            s.output_duration(skip_mode, speed_multiplier)
            for s in self.segments
        )


def apply_global_timeline_override(
    videos: List[VideoFileInfo],
    global_start_sec: float,
    global_end_sec: float,
    new_type: SegmentType,
) -> int:
    """
    전체 연결 타임라인 기준 [global_start_sec, global_end_sec] 구간을
    각 개별 영상의 로컬 시간으로 변환하여 new_type(예: MANUAL_CUT, NORMAL 등)을 적용합니다.
    """
    g_s = min(global_start_sec, global_end_sec)
    g_e = max(global_start_sec, global_end_sec)
    if g_e - g_s < 0.05:
        return 0

    cursor = 0.0
    modified_count = 0
    for v in videos:
        clip_start = cursor
        clip_end = cursor + v.duration_sec
        cursor = clip_end

        overlap_s = max(clip_start, g_s)
        overlap_e = min(clip_end, g_e)
        if overlap_e - overlap_s >= 0.05:
            v.ensure_segments_initialized()
            local_s = overlap_s - clip_start
            local_e = overlap_e - clip_start
            v.segments = apply_segment_override(
                v.segments, v.duration_sec, local_s, local_e, new_type
            )
            modified_count += 1

    return modified_count


def map_global_time_to_clip(
    videos: List[VideoFileInfo],
    global_sec: float,
) -> Optional[Tuple[int, VideoFileInfo, float, SegmentType]]:
    """
    전체 타임라인 시각(global_sec)이 몇 번째 영상의 몇 초(local_sec)에 해당하는지,
    그리고 해당 위치의 SegmentType이 무엇인지 반환합니다.
    """
    if not videos:
        return None
    total = sum(v.duration_sec for v in videos)
    t = max(0.0, min(total, global_sec))

    cursor = 0.0
    for idx, v in enumerate(videos):
        next_cursor = cursor + v.duration_sec
        if t <= next_cursor or idx == len(videos) - 1:
            local_t = max(0.0, min(v.duration_sec, t - cursor))
            seg_type = SegmentType.NORMAL
            for s in v.segments:
                if s.start_sec <= local_t <= s.end_sec + 1e-3:
                    seg_type = s.segment_type
                    break
            return idx, v, local_t, seg_type
        cursor = next_cursor
    return None


@dataclass
class AnalysisConfig:
    detect_stops: bool = True             # 신호대기 / 정차 감지 활성화
    detect_highway_straight: bool = True  # 장시간 고속도로 직진 감지 활성화
    sensitivity: float = 0.5              # 0.0 (둔감) ~ 1.0 (민감), 기본 0.5 (보통)
    min_skip_duration_sec: float = 5.0    # 최소 몇 초 이상 지속될 때만 건너뛸지 (기본 5초)
    transition_padding_sec: float = 1.0   # 변화 구간 진입/이탈 전후 여유 시간(초)
    sample_fps: float = 4.0               # 초당 분석 샘플 프레임 수 (빠른 분석용)
    roi_top_ratio: float = 0.15           # 상단 하늘/블랙박스 상단 제외 비율
    roi_bottom_ratio: float = 0.80        # 하단 대시보드/본네트 제외 비율
    use_gpu_accel: bool = True            # GPU 하드웨어 디코딩(NVDEC + scale_cuda / D3D11VA) 가속 사용


@dataclass
class ExportConfig:
    output_path: Path
    skip_mode: SkipMode = SkipMode.TIMELAPSE
    speed_multiplier: float = 8.0
    cut_keep_margin_sec: float = 0.5
    audio_mode: AudioMode = AudioMode.KEEP_NORMAL_MUTE_SKIP
    target_resolution: str = "original"   # "original", "4k", "1440p", "1080p", "720p"
    fps_mode: str = "original"            # "original", "60", "30", "24"
    target_fps: float = 0.0               # >0이면 명시적 FPS 우선 적용
    quality_preset: VideoQualityPreset = VideoQualityPreset.HIGH
    video_codec: VideoCodec = VideoCodec.H264
    overlay_speed_badge: bool = False     # 배속 구간 우측 상단에 '>> 8x' 배지 표시 여부
    use_hw_accel: bool = True             # 가능한 경우 GPU 하드웨어 인코더 사용

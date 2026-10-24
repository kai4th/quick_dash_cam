from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import imageio_ffmpeg

from .models import VideoCodec, VideoFileInfo, VideoQualityPreset

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m2ts", ".ts", ".webm"}


def get_CreationFlags() -> int:
    """Windows에서 하위 프로세스 실행 시 검은색 콘솔 창이 깜빡이지 않도록 플래그를 반환합니다."""
    if os.name == "nt":
        return subprocess.CREATE_NO_WINDOW
    return 0


@lru_cache(maxsize=1)
def get_ffmpeg_exe() -> str:
    """번들된 imageio-ffmpeg 또는 시스템 PATH의 ffmpeg 실행 파일 경로를 반환합니다."""
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    raise RuntimeError("FFmpeg 실행 파일을 찾을 수 없습니다. imageio-ffmpeg 패키지를 확인해 주세요.")


@lru_cache(maxsize=8)
def _can_use_encoder(enc_name: str) -> bool:
    """해당 비디오 인코더가 현재 PC에서 실제로 동작하는지 1프레임 테스트로 검증합니다."""
    ffmpeg = get_ffmpeg_exe()
    try:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-f", "lavfi",
            "-i", "color=c=black:s=256x256:d=0.1:r=30",
            "-frames:v", "1",
            "-c:v", enc_name,
            "-pix_fmt", "yuv420p",
            "-f", "null",
            "-",
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            creationflags=get_CreationFlags(),
        )
        return res.returncode == 0
    except Exception:
        return False


def get_software_encoder_args(
    codec: VideoCodec,
    quality: VideoQualityPreset,
) -> Tuple[str, list[str]]:
    """CPU 소프트웨어 인코더(libx264 / libx265)와 화질 프리셋 인자를 반환합니다."""
    if codec == VideoCodec.HEVC:
        table = {
            VideoQualityPreset.LOSSLESS_LIKE: ["-preset", "medium", "-crf", "16", "-x265-params", "log-level=error"],
            VideoQualityPreset.HIGH: ["-preset", "fast", "-crf", "20", "-x265-params", "log-level=error"],
            VideoQualityPreset.BALANCED: ["-preset", "veryfast", "-crf", "24", "-x265-params", "log-level=error"],
            VideoQualityPreset.COMPACT: ["-preset", "ultrafast", "-crf", "28", "-x265-params", "log-level=error"],
        }
        return "libx265", table.get(quality, table[VideoQualityPreset.HIGH])
    else:
        table = {
            VideoQualityPreset.LOSSLESS_LIKE: ["-preset", "medium", "-crf", "14"],
            VideoQualityPreset.HIGH: ["-preset", "fast", "-crf", "18"],
            VideoQualityPreset.BALANCED: ["-preset", "veryfast", "-crf", "22"],
            VideoQualityPreset.COMPACT: ["-preset", "ultrafast", "-crf", "27"],
        }
        return "libx264", table.get(quality, table[VideoQualityPreset.HIGH])


def resolve_video_encoder(
    codec: VideoCodec = VideoCodec.H264,
    quality: VideoQualityPreset = VideoQualityPreset.HIGH,
    prefer_hw: bool = True,
) -> Tuple[str, list[str]]:
    """
    선택한 코덱(H.264 / HEVC) 및 화질 등급(원본 유지급 ~ 저용량)에 맞는 최적의 인코더와 옵션을 반환합니다.
    """
    cq_map = {
        VideoQualityPreset.LOSSLESS_LIKE: 15,
        VideoQualityPreset.HIGH: 19,
        VideoQualityPreset.BALANCED: 23,
        VideoQualityPreset.COMPACT: 28,
    }
    cq = cq_map.get(quality, 19)
    nv_preset = "p5" if quality == VideoQualityPreset.LOSSLESS_LIKE else (
        "p2" if quality == VideoQualityPreset.COMPACT else "p4"
    )
    amf_qual = "quality" if quality == VideoQualityPreset.LOSSLESS_LIKE else (
        "speed" if quality == VideoQualityPreset.COMPACT else "balanced"
    )

    if prefer_hw:
        prefix = "hevc" if codec == VideoCodec.HEVC else "h264"
        candidates = [
            (f"{prefix}_nvenc", ["-preset", nv_preset, "-cq", str(cq), "-b:v", "0"]),
            (f"{prefix}_videotoolbox", ["-q:v", str(int(max(10, 100 - cq * 2.2)))]),
            (f"{prefix}_qsv", ["-global_quality", str(cq)]),
            (f"{prefix}_amf", ["-quality", amf_qual, "-rc", "cqp", "-qp_i", str(cq), "-qp_p", str(cq)]),
            (f"{prefix}_vaapi", ["-qp", str(cq)]),
        ]
        for enc_name, enc_args in candidates:
            if _can_use_encoder(enc_name):
                return enc_name, enc_args

    return get_software_encoder_args(codec, quality)


@lru_cache(maxsize=1)
def detect_best_h264_encoder(prefer_hw: bool = True) -> Tuple[str, list[str]]:
    """현재 PC에서 사용 가능한 가장 빠른 H.264 인코더와 기본 옵션을 감지합니다."""
    return resolve_video_encoder(VideoCodec.H264, VideoQualityPreset.HIGH, prefer_hw=prefer_hw)


@lru_cache(maxsize=1)
def detect_best_gpu_decoder() -> str:
    """
    영상 구간 분석 시 사용할 수 있는 최적의 GPU 하드웨어 디코더 방식을 감지합니다.
    반환값:
      - "cuda_scale": NVIDIA NVDEC 디코딩 + GPU VRAM 내부 scale_cuda 리사이징 (CPU 점유율 최소화)
      - "d3d11va": Windows Direct3D 11 하드웨어 디코딩 가속
      - "cpu": 소프트웨어 디코딩
    """
    ffmpeg = get_ffmpeg_exe()
    try:
        res = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            creationflags=get_CreationFlags(),
        )
        hw_out = (res.stdout or "") + (res.stderr or "")
        if "cuda" in hw_out:
            # CUDA 컨텍스트 및 scale_cuda 필터가 실제로 동작하는지 1프레임 테스트
            test_cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-f", "lavfi",
                "-i", "color=c=black:s=256x256:d=0.1:r=30,format=nv12",
                "-vf", "hwupload_cuda,scale_cuda=160:90:format=nv12,hwdownload,format=nv12",
                "-frames:v", "1",
                "-f", "null",
                "-",
            ]
            t_res = subprocess.run(
                test_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                creationflags=get_CreationFlags(),
            )
            if t_res.returncode == 0:
                return "cuda_scale"
        if "d3d11va" in hw_out:
            return "d3d11va"
        if "videotoolbox" in hw_out:
            return "videotoolbox"
        if "vaapi" in hw_out:
            return "vaapi"
    except Exception:
        pass
    return "cpu"


def _parse_creation_time(stderr_text: str) -> Optional[datetime]:
    """FFmpeg 메타데이터 출력에서 creation_time을 추출합니다."""
    match = re.search(
        r"creation_time\s*:\s*(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)",
        stderr_text,
    )
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        if raw.endswith("Z"):
            dt = datetime.fromisoformat(raw[:-1] + "+00:00")
            return dt.astimezone().replace(tzinfo=None)
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _parse_filename_datetime(filename: str) -> Optional[datetime]:
    """
    파일명에 날짜/시간이 포함된 경우 추출합니다.
    예: VID_20261024_093015.mp4, DJI_20261024093015_0001_D.MP4, 2026-10-24_09-30-15.mp4
    """
    patterns = [
        r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[T_\-\s](\d{2})[-_]?(\d{2})[-_]?(\d{2})",
        r"(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            try:
                y, mo, d, h, mi, s = map(int, m.groups())
                if 1 <= mo <= 12 and 1 <= d <= 31 and 0 <= h <= 23 and 0 <= mi <= 59 and 0 <= s <= 59:
                    return datetime(y, mo, d, h, mi, s)
            except ValueError:
                continue
    return None


def _parse_gopro_chapter_key(filename: str) -> Optional[tuple]:
    """
    GoPro 파일명 규칙(예: GX010042.MP4, GH020042.MP4, GOPR0042.MP4, GP010042.MP4)을 인식합니다.
    GoPro는 앞 2자리 숫자가 챕터(01, 02...), 뒤 4자리 숫자가 클립 번호(0042)이므로
    (clip_num, chapter_num) 순으로 정렬해야 시간순이 맞습니다.
    """
    stem = Path(filename).stem.upper()
    # GX010042, GH010042, GP010042
    m = re.match(r"^G[XHP](\d{2})(\d{4})$", stem)
    if m:
        chapter_num = int(m.group(1))
        clip_num = int(m.group(2))
        return (clip_num, chapter_num)
    # GOPR0042 (구형 GoPro 첫 번째 챕터)
    m2 = re.match(r"^GOPR(\d{4})$", stem)
    if m2:
        clip_num = int(m2.group(1))
        return (clip_num, 0)
    return None


def probe_video_file(path: Path | str) -> VideoFileInfo:
    """영상 파일의 길이, 해상도, FPS, 오디오 유무, 촬영 시각을 분석하여 반환합니다."""
    p = Path(path).resolve()
    stat = p.stat()
    file_size = stat.st_size
    mtime_dt = datetime.fromtimestamp(stat.st_mtime)

    width, height = 1920, 1080
    fps = 30.0
    duration = 0.0
    has_audio = False
    recorded_at: Optional[datetime] = None
    timestamp_source = "파일 수정일"

    # 1. OpenCV를 통해 빠르게 기본 비디오 스트림 제원 확인
    cap = cv2.VideoCapture(str(p))
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        f = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if w > 0 and h > 0:
            width, height = w, h
        if 1.0 <= f <= 240.0:
            fps = f
        if frame_count > 0 and fps > 0:
            duration = frame_count / fps
        cap.release()

    # 2. FFmpeg -i 출력 파싱 (오디오 스트림 존재 여부, 정확한 duration, creation_time 메타데이터)
    try:
        ffmpeg = get_ffmpeg_exe()
        res = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(p)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            creationflags=get_CreationFlags(),
        )
        stderr_text = res.stderr or ""

        # Audio stream check
        if re.search(r"Stream\s+#\d+:\d+.*:\s+Audio:", stderr_text):
            has_audio = True

        # Duration check
        dur_match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)", stderr_text)
        if dur_match:
            hh = int(dur_match.group(1))
            mm = int(dur_match.group(2))
            ss = float(dur_match.group(3))
            ff_dur = hh * 3600 + mm * 60 + ss
            if ff_dur > 0:
                duration = ff_dur

        # creation_time check
        meta_dt = _parse_creation_time(stderr_text)
        if meta_dt is not None and meta_dt.year >= 2005:
            recorded_at = meta_dt
            timestamp_source = "메타데이터 촬영일"
    except Exception:
        pass

    # 3. 메타데이터에 촬영 시각이 없으면 파일명 패턴 -> 파일 수정일 순으로 폴백
    if recorded_at is None:
        fn_dt = _parse_filename_datetime(p.name)
        if fn_dt is not None:
            recorded_at = fn_dt
            timestamp_source = "파일명 시각"
        else:
            recorded_at = mtime_dt
            timestamp_source = "파일 수정일"

    gopro_key = _parse_gopro_chapter_key(p.name)

    return VideoFileInfo(
        path=p,
        filename=p.name,
        duration_sec=max(0.0, duration),
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
        recorded_at=recorded_at,
        timestamp_source=timestamp_source,
        file_size_bytes=file_size,
        gopro_clip_key=gopro_key,
    )

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .ffmpeg_utils import detect_best_gpu_decoder, get_CreationFlags, get_ffmpeg_exe
from .models import AnalysisConfig, Segment, SegmentType, VideoFileInfo

ANALYSIS_W = 160
ANALYSIS_H = 90
CACHE_VERSION = "v4_sobel_roadside"


def _get_disk_cache_path(video_path: Path, sample_fps: float) -> Optional[Path]:
    """동일한 대용량 영상(예: 2시간 주행 영상) 재분석 시 0.05초 만에 불러오기 위한 캐시 경로를 반환합니다."""
    try:
        st = video_path.stat()
        key_str = f"{CACHE_VERSION}|{video_path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{sample_fps:.2f}"
        digest = hashlib.md5(key_str.encode("utf-8")).hexdigest()
        cache_dir = Path(tempfile.gettempdir()) / "quick_dash_cam_fcache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"feat_{digest}.npz"
    except Exception:
        return None


def _build_decode_commands(
    ffmpeg: str,
    video_path: Path,
    sample_fps: float,
    use_gpu_accel: bool = True,
) -> List[List[str]]:
    """
    GPU 하드웨어 디코딩(CUDA NVDEC + VRAM scale_cuda, D3D11VA, VideoToolbox, VAAPI)을 우선 시도하고,
    지원되지 않는 코덱이거나 실패할 경우 CPU 디코딩으로 자동 폴백하기 위한 명령어 후보 리스트를 반환합니다.
    """
    candidates: List[List[str]] = []
    if use_gpu_accel:
        gpu_mode = detect_best_gpu_decoder()
        if gpu_mode == "cuda_scale":
            candidates.append([
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-skip_frame", "noref",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-i", str(video_path),
                "-vf", f"fps={sample_fps:.2f},scale_cuda={ANALYSIS_W}:{ANALYSIS_H}:format=nv12,hwdownload,format=nv12",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-",
            ])
        if gpu_mode in ("cuda_scale", "d3d11va"):
            candidates.append([
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-skip_frame", "noref",
                "-hwaccel", "d3d11va",
                "-i", str(video_path),
                "-vf", f"fps={sample_fps:.2f},scale={ANALYSIS_W}:{ANALYSIS_H}:flags=fast_bilinear",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-",
            ])
        if gpu_mode == "videotoolbox":
            candidates.append([
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-skip_frame", "noref",
                "-hwaccel", "videotoolbox",
                "-i", str(video_path),
                "-vf", f"fps={sample_fps:.2f},scale={ANALYSIS_W}:{ANALYSIS_H}:flags=fast_bilinear",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-",
            ])
        if gpu_mode == "vaapi":
            candidates.append([
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-skip_frame", "noref",
                "-hwaccel", "vaapi",
                "-i", str(video_path),
                "-vf", f"fps={sample_fps:.2f},scale={ANALYSIS_W}:{ANALYSIS_H}:flags=fast_bilinear",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-",
            ])

    candidates.append([
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-skip_frame", "noref",
        "-i", str(video_path),
        "-vf", f"fps={sample_fps:.2f},scale={ANALYSIS_W}:{ANALYSIS_H}:flags=fast_bilinear",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ])
    return candidates


def _iter_downsampled_frames(
    video_path: Path,
    sample_fps: float,
    use_gpu_accel: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Iterator[Tuple[float, np.ndarray]]:
    """
    GPU(CUDA NVDEC + scale_cuda) 또는 FFmpeg 파이프를 사용하여 영상을 지정된 sample_fps 및
    160x90 해상도로 고속 디코딩하여 (timestamp_sec, bgr_frame) 튜플을 순차적으로 반환합니다.
    """
    ffmpeg = get_ffmpeg_exe()
    frame_bytes = ANALYSIS_W * ANALYSIS_H * 3
    cmd_candidates = _build_decode_commands(ffmpeg, video_path, sample_fps, use_gpu_accel)

    for cmd in cmd_candidates:
        used_ffmpeg = False
        proc: Optional[subprocess.Popen] = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_bytes * 16,
                creationflags=get_CreationFlags(),
            )
            idx = 0
            while True:
                if cancel_check and cancel_check():
                    proc.terminate()
                    return
                assert proc.stdout is not None
                raw = proc.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                used_ffmpeg = True
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((ANALYSIS_H, ANALYSIS_W, 3))
                t_sec = idx / sample_fps
                yield t_sec, frame
                idx += 1
        except Exception:
            pass
        finally:
            if proc is not None:
                try:
                    if proc.stdout:
                        proc.stdout.close()
                    if proc.stderr:
                        proc.stderr.close()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        if used_ffmpeg:
            return

    # FFmpeg 파이프 실패 시 OpenCV 폴백
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if native_fps <= 0 or native_fps > 240:
            native_fps = 30.0
        step = max(1, int(round(native_fps / sample_fps)))
        frame_idx = 0
        while True:
            if cancel_check and cancel_check():
                return
            if frame_idx % step == 0:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                small = cv2.resize(frame, (ANALYSIS_W, ANALYSIS_H), interpolation=cv2.INTER_AREA)
                yield frame_idx / native_fps, small
            else:
                if not cap.grab():
                    break
            frame_idx += 1
    finally:
        cap.release()


def _compute_ego_motion_score(
    curr_patch: np.ndarray,
    prev_patch: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 8,
) -> float:
    """
    내 차(Ego-Vehicle)의 실제 이동 여부를 판별하는 격자 기반 움직임 점수를 계산합니다.
    - 지평선 아래 도로/교차로 영역(y: 44% ~ 80%)을 4 x 8 = 32개 블록으로 분할합니다.
    - 신호대기 중 전방 교차로나 옆 차선에 다른 차량들이 지나가더라도(화면의 35~45% 영역 차지),
      나머지 정지 배경(차선, 정지선, 연석, 정차 중인 타 차선 등 55% 이상)의 블록 변화량은 0에 가깝습니다.
    - 반면 내 차가 주행 중일 때는 도로 위 텍스처(윤곽선) 블록들의 대다수가 동시에 이동합니다.
    """
    gh = (curr_patch.shape[0] // grid_rows) * grid_rows
    gw = (curr_patch.shape[1] // grid_cols) * grid_cols
    c_crop = curr_patch[:gh, :gw].astype(np.float32)
    p_crop = prev_patch[:gh, :gw].astype(np.float32)

    bh = gh // grid_rows
    bw = gw // grid_cols

    # 프레임 간 밝기 편차(자동 노출 변화) 제거를 위해 중앙값 편차 보정
    diff_raw = c_crop - p_crop
    illum_shift = float(np.median(diff_raw))
    abs_diff = np.abs(diff_raw - illum_shift)

    # 각 격자 블록의 시간적 변화량 (4 x 8 = 32개 블록)
    block_diffs = np.mean(
        abs_diff.reshape(grid_rows, bh, grid_cols, bw), axis=(1, 3)
    ).ravel()

    # 각 격자 블록의 공간 텍스처(윤곽선) 강도 계산
    gx = np.abs(np.diff(c_crop, axis=1, append=c_crop[:, -1:]))
    gy = np.abs(np.diff(c_crop, axis=0, append=c_crop[-1:, :]))
    block_grads = np.mean(
        (gx + gy).reshape(grid_rows, bh, grid_cols, bw), axis=(1, 3)
    ).ravel()

    # 텍스처가 존재하는 블록 선별 (최소 4개 블록 보장: 야간/매끄러운 고속도로에서 차선 블록만 있는 경우 대응)
    grad_cut = max(2.0, float(np.percentile(block_grads, 45)))
    tex_mask = block_grads >= grad_cut
    if int(np.sum(tex_mask)) < 4:
        top_idx = np.argsort(block_grads)[-4:]
        tex_diffs = block_diffs[top_idx]
    else:
        tex_diffs = block_diffs[tex_mask]

    # 신호대기 중 다른 차량이 텍스처 블록의 최대 45%를 가리며 지나가더라도
    # 정지해 있는 나머지 55% 배경 블록을 대표하는 40백분위수(p40)와 전체 블록의 35백분위수(p35)를 결합
    p40_tex = float(np.percentile(tex_diffs, 40))
    p35_all = float(np.percentile(block_diffs, 35))
    return 0.80 * p40_tex + 0.20 * p35_all


def _sliding_median(arr: np.ndarray, window: int) -> np.ndarray:
    """
    신호대기 중 대형 버스/트럭이 1~2초간 화면 앞을 가로질러 지나갈 때 발생하는
    일시적 스파이크 노이즈를 완벽히 제거하기 위한 슬라이딩 중앙값(Median) 필터.
    """
    n = len(arr)
    if n == 0 or window <= 1:
        return arr.copy()
    half = window // 2
    out = np.empty_like(arr)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        out[i] = float(np.median(arr[s:e]))
    return out


def classify_features_to_segments(
    timestamps: np.ndarray,
    motion_arr: np.ndarray,
    signed_dx_arr: np.ndarray,
    scene_arr: np.ndarray,
    duration: float,
    sample_fps: float,
    config: AnalysisConfig,
) -> List[Segment]:
    """
    추출된 프레임별 특징 배열(motion, signed_dx, scene)로부터 현재 설정(민감도, 최소시간, 여유시간)에 맞춰
    즉시(0.01초 내) 구간(NORMAL / STOP / STRAIGHT)을 분류합니다.
    """
    n_frames = len(timestamps)
    if n_frames == 0 or duration <= 0.5:
        return [Segment(0.0, max(0.0, duration), SegmentType.NORMAL)]

    # 1) 정차 감지용 움직임 점수: 2.5초 슬라이딩 중앙값 필터 적용
    #    (신호대기 중 교차로에 다른 차량이 1~2초간 지나가도 중앙값은 정지 상태를 유지함)
    med_win = max(3, int(round(2.5 * sample_fps)))
    if med_win % 2 == 0:
        med_win += 1
    smooth_motion = _sliding_median(motion_arr, med_win)

    # 2) 조향(회전/차선변경) 점수: 부호가 있는 dx를 먼저 1.5초 이동평균하여
    #    단순 노면 진동(+/- 상쇄)은 제거하고 실제 코너링/회전만 추출
    steer_win = max(2, int(round(1.5 * sample_fps)))
    steer_kernel = np.ones(steer_win, dtype=np.float32) / steer_win
    rolling_signed_dx = np.convolve(signed_dx_arr, steer_kernel, mode="same")
    smooth_steer = np.abs(rolling_signed_dx)

    # 3) 전방 풍경/구도 변화 점수: 2.0초 슬라이딩 중앙값 + 이동평균
    smooth_scene = _sliding_median(scene_arr, max(3, int(round(2.0 * sample_fps))))

    # 민감도(0.0 ~ 1.0, 기본 초기값 0.5)에 따른 임계값
    sens = max(0.0, min(1.0, config.sensitivity))

    # [정차(STOP) 임계값]
    # sens=0.5(초기값)일 때 stop_thresh = 4.2
    # 교차로에서 다른 차들이 움직여도 내 차가 멈춰 있으면 smooth_motion은 보통 0.4 ~ 3.2 사이로 유지됨
    stop_thresh = 1.8 + sens * 4.8

    # [고속도로 직진 / 단조 주행(STRAIGHT) 임계값]
    # 시내/국도 주행에서는 좌우 상가·가옥·전신주·교행 차량의 윤곽선 및 색상 변화로 인해
    # smooth_scene이 39 ~ 60 이상으로 높게 나타나며, 실제 고속도로/터널 직진에서는 12 ~ 20 수준으로 낮게 유지됨.
    # - sens=0.1(10% 최소): scene_thresh = 14.0, steer_thresh = 0.45
    # - sens=0.5(50% 기본): scene_thresh = 22.0, steer_thresh = 0.95
    # - sens=0.9(90% 최대): scene_thresh = 30.0, steer_thresh = 1.45
    steer_thresh = 0.32 + sens * 1.25
    scene_thresh = 12.0 + sens * 20.0

    raw_labels: List[SegmentType] = []
    for i in range(n_frames):
        if config.detect_stops and (smooth_motion[i] <= stop_thresh):
            raw_labels.append(SegmentType.STOP)
            continue

        if (
            config.detect_highway_straight
            and (smooth_steer[i] <= steer_thresh)
            and (smooth_scene[i] <= scene_thresh)
        ):
            raw_labels.append(SegmentType.STRAIGHT)
        else:
            raw_labels.append(SegmentType.NORMAL)

    # 영상 시작 직후(처음 1.25초 이내) 노출 조정 등으로 인한 NORMAL 노이즈를 뒤따르는 구간에 흡수
    startup_steps = max(1, int(round(1.25 * sample_fps)))
    if n_frames > startup_steps and raw_labels[startup_steps] != SegmentType.NORMAL:
        for k in range(startup_steps):
            raw_labels[k] = raw_labels[startup_steps]

    # 후처리 1:
    # - STOP(신호대기)은 교차로 통과 차량 때문에 잠깐 끊기는 구간을 최대 2.5초까지 연결
    # - STRAIGHT(고속도로 직진)는 시내 주행 구간이 잘못 합쳐지지 않도록 짧은 1.0초 이내만 연결
    labels = _bridge_short_gaps_typed(
        raw_labels,
        stop_gap_steps=max(1, int(round(2.5 * sample_fps))),
        straight_gap_steps=max(1, int(round(1.5 * sample_fps))),
    )

    # 후처리 2: 연속된 프레임 라벨을 시간 구간(Segment)으로 변환
    raw_segments: List[Segment] = []
    cur_type = labels[0]
    start_t = 0.0
    for i in range(1, n_frames):
        if labels[i] != cur_type:
            end_t = float(timestamps[i])
            raw_segments.append(Segment(start_t, end_t, cur_type))
            cur_type = labels[i]
            start_t = end_t
    raw_segments.append(Segment(start_t, duration, cur_type))
    raw_segments = _merge_adjacent_segments(raw_segments)

    # 후처리 3: 최소 지속 시간 필터링 및 전환 여유 시간(transition_padding_sec) 적용
    # 사용자가 설정한 '최소 지속 시간(config.min_skip_duration_sec)' 미만의 스킵 구간은 전면 제외
    min_stop_dur = max(1.0, config.min_skip_duration_sec)
    min_straight_dur = max(min_stop_dur, 6.0)
    pad = max(0.0, config.transition_padding_sec)

    refined: List[Segment] = []
    for seg in raw_segments:
        if seg.segment_type == SegmentType.NORMAL:
            refined.append(seg)
            continue

        req_dur = min_stop_dur if seg.segment_type == SegmentType.STOP else min_straight_dur

        # 영상 맨 처음(0초 근처)부터 시작하는 신호대기/직진은 앞쪽 여유(lead_pad) 없이 0초부터 바로 스킵 적용
        lead_pad = pad if seg.start_sec > 1.0 else 0.0
        tail_pad = pad if (duration - seg.end_sec) > 1.0 else 0.0

        inner_start = seg.start_sec + lead_pad
        inner_end = seg.end_sec - tail_pad

        # 전환 여유(pad)를 둔 후의 최종 스킵 구간(inner_end - inner_start)이
        # 사용자가 설정한 '최소 지속 시간(req_dur)' 이상일 때만 스킵 구간으로 채택
        if (inner_end - inner_start) < req_dur:
            refined.append(Segment(seg.start_sec, seg.end_sec, SegmentType.NORMAL))
        else:
            if lead_pad > 0:
                refined.append(Segment(seg.start_sec, inner_start, SegmentType.NORMAL))
            refined.append(Segment(inner_start, inner_end, seg.segment_type))
            if tail_pad > 0:
                refined.append(Segment(inner_end, seg.end_sec, SegmentType.NORMAL))

    return _merge_adjacent_segments(refined)


def reclassify_from_cache(
    video_info: VideoFileInfo,
    config: AnalysisConfig,
) -> Optional[List[Segment]]:
    """
    이미 분석되어 feature_cache가 있는 영상에 대해 디코딩 없이 0.01초 만에 구간을 재계산합니다.
    """
    fc = video_info.feature_cache
    if not fc:
        return None
    return classify_features_to_segments(
        timestamps=fc["timestamps"],
        motion_arr=fc["motion"],
        signed_dx_arr=fc["signed_dx"],
        scene_arr=fc["scene"],
        duration=video_info.duration_sec,
        sample_fps=float(fc["sample_fps"]),
        config=config,
    )


def analyze_video(
    video_info: VideoFileInfo,
    config: AnalysisConfig,
    progress_callback: Optional[Callable[[float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    force_decode: bool = False,
) -> List[Segment]:
    """
    단일 비디오 파일을 분석하여 NORMAL, STOP(신호대기/정차), STRAIGHT(고속도로 직진) 구간 리스트를 생성합니다.
    메모리 또는 디스크에 추출된 특징 캐시가 있으면 즉시(0.01초) 재분류합니다.
    """
    duration = video_info.duration_sec
    if duration <= 0.5:
        return [Segment(0.0, max(0.0, duration), SegmentType.NORMAL)]

    sample_fps = max(1.0, min(10.0, config.sample_fps))

    # 1. 메모리 캐시 확인
    if not force_decode and video_info.feature_cache is not None:
        if abs(float(video_info.feature_cache.get("sample_fps", 0.0)) - sample_fps) < 0.05:
            if progress_callback:
                progress_callback(1.0)
            return reclassify_from_cache(video_info, config) or []

    # 2. 디스크 캐시 확인 (2시간짜리 영상도 재실행 시 0.05초 만에 로드)
    disk_cache = _get_disk_cache_path(video_info.path, sample_fps)
    if not force_decode and disk_cache is not None and disk_cache.exists():
        try:
            with np.load(disk_cache) as data:
                video_info.feature_cache = {
                    "timestamps": data["timestamps"],
                    "motion": data["motion"],
                    "signed_dx": data["signed_dx"],
                    "scene": data["scene"],
                    "sample_fps": sample_fps,
                }
            if progress_callback:
                progress_callback(1.0)
            return reclassify_from_cache(video_info, config) or []
        except Exception:
            pass

    # 중·근거리 도로/교차로 주행 영역 (지평선 아래 노면·차선·교차로 영역: y 44% ~ 80%)
    ego_y1 = int(ANALYSIS_H * 0.44)
    ego_y2 = int(ANALYSIS_H * 0.80)
    ego_x1 = int(ANALYSIS_W * 0.05)
    ego_x2 = int(ANALYSIS_W * 0.95)

    # 조향(수평 이동) 감지를 위한 지평선/중거리 대역
    steer_y1 = int(ANALYSIS_H * 0.20)
    steer_y2 = int(ANALYSIS_H * 0.56)
    steer_x1 = int(ANALYSIS_W * 0.15)
    steer_x2 = int(ANALYSIS_W * 0.85)
    hann_win = cv2.createHanningWindow(
        (steer_x2 - steer_x1, steer_y2 - steer_y1), cv2.CV_32F
    )

    # 도로변 및 중거리 풍경 변화 감지 대역 (하늘 상단과 본네트 하단을 제외한 좌우 전체 폭: y 25% ~ 78%, x 0% ~ 100%)
    # 시내/국도 주행 시 좌우 상가·가옥·전신주·교행 차량의 윤곽선(Sobel Edge) 및 구도 변화를 포착합니다.
    sc_y1 = int(ANALYSIS_H * 0.25)
    sc_y2 = int(ANALYSIS_H * 0.78)

    timestamps: List[float] = []
    motion_scores: List[float] = []
    signed_dx_scores: List[float] = []
    scene_scores: List[float] = []

    prev_ego_gray: Optional[np.ndarray] = None
    prev_steer_f32: Optional[np.ndarray] = None
    prev_sc_sobel: Optional[np.ndarray] = None
    coarse_history: List[np.ndarray] = []

    lookback_steps = max(1, int(round(0.5 * sample_fps)))  # 0.5초 전 프레임과 전방/도로변 구도 비교

    for t_sec, frame in _iter_downsampled_frames(
        video_info.path, sample_fps, config.use_gpu_accel, cancel_check
    ):
        if cancel_check and cancel_check():
            return []

        timestamps.append(t_sec)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        ego_gray = blurred[ego_y1:ego_y2, ego_x1:ego_x2]
        steer_patch = blurred[steer_y1:steer_y2, steer_x1:steer_x2].astype(np.float32)

        # 도로변 포함 전체 폭(16x8) 구도 특징 (노출 변화 보정)
        coarse_raw = cv2.resize(
            frame[sc_y1:sc_y2, :], (16, 8), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        coarse_norm = coarse_raw - np.mean(coarse_raw, axis=(0, 1), keepdims=True)

        # 도로변 건물·전신주·교행 차량의 윤곽선 이동량 감지를 위한 Sobel 에지 맵
        sc_sobel = cv2.Sobel(gray[sc_y1:sc_y2, :], cv2.CV_32F, 1, 1)

        if prev_ego_gray is None:
            motion_scores.append(0.0)
            signed_dx_scores.append(0.0)
            scene_scores.append(0.0)
        else:
            # 1) 격자 기반 내 차(Ego) 움직임 점수 (교차로 타 차량 이동에 강인함)
            m_score = _compute_ego_motion_score(ego_gray, prev_ego_gray) * (sample_fps / 4.0)
            motion_scores.append(m_score)

            # 2) 부호 있는 수평 조향 이동량(signed dx)
            assert prev_steer_f32 is not None
            try:
                (dx, _dy), _resp = cv2.phaseCorrelate(prev_steer_f32, steer_patch, hann_win)
                s_dx = float(dx) * (sample_fps / 4.0)
            except Exception:
                s_dx = 0.0
            signed_dx_scores.append(s_dx)

            # 3) 도로변·전방 풍경 변화 점수:
            #    하늘 영역(변화 0)에 가려지지 않도록 상위 활성 영역(80~85백분위수)의 구도 및 윤곽선 변화를 결합
            ref_idx = max(0, len(coarse_history) - lookback_steps)
            ref_coarse = coarse_history[ref_idx]
            cell_diffs = np.mean(np.abs(coarse_norm - ref_coarse), axis=2).ravel()
            p80_scene = float(np.percentile(cell_diffs, 80))

            assert prev_sc_sobel is not None
            p85_edge = float(np.percentile(np.abs(sc_sobel - prev_sc_sobel), 85))
            sc_score = (0.60 * p85_edge + 0.40 * p80_scene) * (sample_fps / 4.0)
            scene_scores.append(sc_score)

        prev_ego_gray = ego_gray
        prev_steer_f32 = steer_patch
        prev_sc_sobel = sc_sobel
        coarse_history.append(coarse_norm)
        if len(coarse_history) > lookback_steps + 2:
            coarse_history.pop(0)

        if progress_callback and duration > 0:
            progress_callback(min(0.99, t_sec / duration))

    n_frames = len(timestamps)
    if n_frames == 0:
        return [Segment(0.0, duration, SegmentType.NORMAL)]

    if n_frames > 1:
        motion_scores[0] = motion_scores[1]
        signed_dx_scores[0] = signed_dx_scores[1]
        scene_scores[0] = scene_scores[1]

    ts_arr = np.array(timestamps, dtype=np.float32)
    motion_arr = np.array(motion_scores, dtype=np.float32)
    signed_dx_arr = np.array(signed_dx_scores, dtype=np.float32)
    scene_arr = np.array(scene_scores, dtype=np.float32)

    video_info.feature_cache = {
        "timestamps": ts_arr,
        "motion": motion_arr,
        "signed_dx": signed_dx_arr,
        "scene": scene_arr,
        "sample_fps": sample_fps,
    }

    if disk_cache is not None:
        try:
            np.savez_compressed(
                disk_cache,
                timestamps=ts_arr,
                motion=motion_arr,
                signed_dx=signed_dx_arr,
                scene=scene_arr,
            )
        except Exception:
            pass

    segments = classify_features_to_segments(
        timestamps=ts_arr,
        motion_arr=motion_arr,
        signed_dx_arr=signed_dx_arr,
        scene_arr=scene_arr,
        duration=duration,
        sample_fps=sample_fps,
        config=config,
    )
    if progress_callback:
        progress_callback(1.0)
    return segments


def _bridge_short_gaps_typed(
    labels: List[SegmentType],
    stop_gap_steps: int,
    straight_gap_steps: int,
) -> List[SegmentType]:
    """
    STOP 구간 사이의 짧은 끊김(예: 교차로 차량 통과 2.5초)과
    STRAIGHT 구간 사이의 짧은 끊김(1.0초)을 각각 다른 허용 길이로 메웁니다.
    """
    if not labels:
        return []

    out = list(labels)
    n = len(out)
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if 0 < i and j < n and out[i] == SegmentType.NORMAL:
            prev_label = out[i - 1]
            next_label = out[j]
            if prev_label == next_label and prev_label != SegmentType.NORMAL:
                max_steps = (
                    stop_gap_steps
                    if prev_label == SegmentType.STOP
                    else straight_gap_steps
                )
                if run_len <= max_steps:
                    for k in range(i, j):
                        out[k] = prev_label
        i = j
    return out


def _merge_adjacent_segments(segments: List[Segment]) -> List[Segment]:
    """인접한 동일 타입의 Segment를 하나로 병합합니다."""
    if not segments:
        return []
    merged: List[Segment] = [
        Segment(segments[0].start_sec, segments[0].end_sec, segments[0].segment_type)
    ]
    for seg in segments[1:]:
        if seg.duration <= 0.01:
            continue
        last = merged[-1]
        if last.segment_type == seg.segment_type:
            last.end_sec = seg.end_sec
        else:
            merged.append(Segment(seg.start_sec, seg.end_sec, seg.segment_type))
    return merged

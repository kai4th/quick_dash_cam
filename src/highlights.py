"""
주행 영상 하이라이트 쇼츠 생성기.

1) 점수 계산: 구간 분석(analyzer) 단계에서 이미 추출·캐싱된 프레임별 특징값(조향, 풍경 변화, 움직임)과
   오디오 음량 급변(경적·큰 소리)을 결합해 "볼 만한 순간" 점수 곡선을 만듭니다.
   신호대기/정차·수동 잘라내기 구간은 제외하고, 고속도로 직진 구간은 가중치를 낮춥니다.
2) 장면 선정: 점수가 가장 높은 N초 창(window)부터 차례로 고르되, 전체 주행이 고루 담기도록
   이미 고른 장면 주변은 일정 간격만큼 건너뜁니다. 고른 장면은 시간순으로 정렬됩니다.
3) 렌더링: 장면별로 세로 9:16(흐림 배경 / 꽉 채우기) 또는 가로 16:9 레이아웃, 페이드 전환,
   제목·촬영 시각 오버레이를 입혀 인코딩한 뒤 하나로 이어 붙이고, 필요 시 배경음악을 입힙니다.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .ffmpeg_utils import (
    get_CreationFlags,
    get_ffmpeg_exe,
    get_software_encoder_args,
    resolve_video_encoder,
)
from .models import (
    HighlightClip,
    HighlightConfig,
    SegmentType,
    ShortsAudioMode,
    ShortsExportConfig,
    ShortsLayout,
    VideoCodec,
    VideoFileInfo,
)

AUDIO_SAMPLE_RATE = 8000
AUDIO_CACHE_VERSION = "audio_rms_v1"

# 하이라이트 점수 구성 요소별 가중치
COMPONENT_WEIGHTS: Dict[str, float] = {
    "steer": 0.30,     # 회전·커브·차선 변경 (수평 조향 이동량)
    "scene": 0.25,     # 도로변 풍경·구도 변화 (시내, 산길, 해안도로 등)
    "surprise": 0.20,  # 주변 흐름 대비 갑작스러운 화면 변화 (끼어들기, 돌발 상황)
    "audio": 0.15,     # 경적·큰 소리 등 주변 대비 음량 급증
    "motion": 0.10,    # 노면 움직임 (역동적인 주행)
}

REASON_LABELS: Dict[str, str] = {
    "steer": "회전·커브",
    "scene": "풍경 변화",
    "surprise": "돌발 장면",
    "audio": "소리 이벤트",
    "motion": "역동적 주행",
}


# ── 공용 신호 처리 ──

def _odd(n: float) -> int:
    k = max(1, int(round(n)))
    return k if k % 2 == 1 else k + 1


def _fast_sliding_median(arr: np.ndarray, window: int) -> np.ndarray:
    """벡터화된 슬라이딩 중앙값 (2시간 영상 수만 개 샘플도 수십 ms 내 처리)."""
    arr = np.asarray(arr, dtype=np.float32)
    n = len(arr)
    if n == 0 or window <= 1:
        return arr.copy()
    window = _odd(window)
    if window > n:
        window = n if n % 2 == 1 else max(1, n - 1)
    if window <= 1:
        return arr.copy()
    half = window // 2
    padded = np.pad(arr, (half, half), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=1).astype(np.float32)[:n]


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0 or window <= 1:
        return arr.copy()
    half = window // 2
    padded = np.pad(arr, (half, window - 1 - half), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    if len(arr) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.pad(arr, (0, n - len(arr)), mode="edge")


# ── 오디오 음량 추출 ──

def _audio_cache_path(video_path: Path, sample_fps: float) -> Optional[Path]:
    try:
        st = video_path.stat()
        key = f"{AUDIO_CACHE_VERSION}|{video_path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{sample_fps:.2f}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        cache_dir = Path(tempfile.gettempdir()) / "quick_dash_cam_fcache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"audio_{digest}.npy"
    except Exception:
        return None


def extract_audio_levels(
    video_path: Path,
    sample_fps: float,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[np.ndarray]:
    """
    오디오 트랙을 8kHz 모노로 빠르게 디코딩해 샘플 구간(1/sample_fps초)마다 음량(dB)을 계산합니다.
    결과는 디스크에 캐싱되어 두 번째부터는 즉시 불러옵니다. 오디오가 없으면 None.
    """
    cache = _audio_cache_path(video_path, sample_fps)
    if cache is not None and cache.exists():
        try:
            return np.load(cache)
        except Exception:
            pass

    hop = max(1, int(round(AUDIO_SAMPLE_RATE / sample_fps)))
    hop_bytes = hop * 2
    cmd = [
        get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-map", "0:a:0",
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-f", "s16le",
        "-",
    ]
    levels: List[np.ndarray] = []
    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=get_CreationFlags(),
        )
        assert proc.stdout is not None
        buf = bytearray()
        while True:
            if cancel_check and cancel_check():
                return None
            chunk = proc.stdout.read(hop_bytes * 512)
            if not chunk:
                break
            buf.extend(chunk)
            n_full = len(buf) // hop_bytes
            if n_full == 0:
                continue
            block = np.frombuffer(bytes(buf[: n_full * hop_bytes]), dtype=np.int16)
            block = block.reshape(n_full, hop).astype(np.float32)
            rms = np.sqrt(np.mean(block * block, axis=1))
            levels.append(20.0 * np.log10(rms / 32768.0 + 1e-5))
            del buf[: n_full * hop_bytes]
    except Exception:
        return None
    finally:
        if proc is not None:
            try:
                if proc.stdout:
                    proc.stdout.close()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    if not levels:
        return None
    result = np.concatenate(levels).astype(np.float32)
    if cache is not None:
        try:
            np.save(cache, result)
        except Exception:
            pass
    return result


def ensure_audio_levels(
    video: VideoFileInfo,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[np.ndarray]:
    """영상의 feature_cache에 오디오 음량 곡선이 없으면 추출해서 함께 보관합니다."""
    fc = video.feature_cache
    if not fc or not video.has_audio:
        return None
    if "audio_db" in fc:
        return fc["audio_db"]
    levels = extract_audio_levels(video.path, float(fc["sample_fps"]), cancel_check)
    if levels is not None:
        fc["audio_db"] = levels
    return levels


# ── 하이라이트 점수 계산 ──

def compute_highlight_components(video: VideoFileInfo) -> Optional[Dict[str, np.ndarray]]:
    """
    분석 캐시(feature_cache)로부터 하이라이트 점수 구성 요소(정규화 전)를 계산합니다.
    반환: {"ts", "steer", "scene", "surprise", "motion", ["audio"]} — 모두 같은 길이의 배열.
    """
    fc = video.feature_cache
    if not fc:
        return None
    ts = np.asarray(fc["timestamps"], dtype=np.float32)
    n = len(ts)
    if n == 0:
        return None
    fps = float(fc["sample_fps"])

    motion = _fast_sliding_median(fc["motion"], _odd(2.5 * fps))
    signed_dx = np.asarray(fc["signed_dx"], dtype=np.float32)
    steer = np.abs(_moving_average(signed_dx, max(2, int(round(1.5 * fps)))))
    steer = _moving_average(steer, max(1, int(round(1.0 * fps))))
    scene = _fast_sliding_median(fc["scene"], _odd(2.0 * fps))
    # 최근 20초 흐름 대비 갑자기 화면이 크게 변한 순간 (끼어들기, 터널 진출입, 돌발 상황 등)
    scene_ctx = _fast_sliding_median(fc["scene"], _odd(20.0 * fps))
    surprise = np.clip(scene - scene_ctx, 0.0, None)

    comps: Dict[str, np.ndarray] = {
        "ts": ts,
        "steer": steer,
        "scene": scene,
        "surprise": surprise,
        "motion": motion,
    }

    audio_db = fc.get("audio_db")
    if audio_db is not None and len(audio_db) > 0:
        a = _fit_length(np.asarray(audio_db, dtype=np.float32), n)
        a_ctx = _fast_sliding_median(a, _odd(15.0 * fps))
        # 주변 15초 평균 소음 대비 몇 dB 튀었는지 (경적, 감탄사, 급제동 소리 등)
        prominence = np.clip(a - a_ctx, 0.0, 25.0)
        # 거의 무음인 구간의 미세한 변화는 무시
        prominence[a < -60.0] = 0.0
        comps["audio"] = _moving_average(prominence, max(1, int(round(0.75 * fps))))
    return comps


def _type_weights(video: VideoFileInfo, ts: np.ndarray, straight_weight: float) -> Tuple[np.ndarray, np.ndarray]:
    """구간 유형별 점수 가중치와 수동 잘라내기(사용 금지) 마스크를 반환합니다."""
    weights = np.ones(len(ts), dtype=np.float32)
    cut = np.zeros(len(ts), dtype=bool)
    for seg in video.segments:
        idx = (ts >= seg.start_sec) & (ts < seg.end_sec)
        if seg.segment_type == SegmentType.STOP:
            weights[idx] = 0.0
        elif seg.segment_type == SegmentType.STRAIGHT:
            weights[idx] = max(0.0, straight_weight)
        elif seg.segment_type == SegmentType.MANUAL_CUT:
            weights[idx] = 0.0
            cut[idx] = True
    return weights, cut


def compute_highlight_scores(
    videos: List[VideoFileInfo],
    config: HighlightConfig,
) -> List[Optional[Dict[str, np.ndarray]]]:
    """
    전체 영상에 공통 기준(백분위 정규화)을 적용해 영상별 하이라이트 점수 곡선을 계산합니다.
    반환 리스트의 각 원소: {"ts", "score", "cut", "fps", "<component>_n"...} 또는 분석 전이면 None.
    """
    raw: List[Optional[Dict[str, np.ndarray]]] = []
    for v in videos:
        comps = compute_highlight_components(v)
        if comps is not None and not config.use_audio_peaks:
            comps.pop("audio", None)
        raw.append(comps)

    # 영상 전체에서 공통 정규화 기준(95백분위)을 구해 영상 간 점수를 비교 가능하게 맞춤
    scales: Dict[str, float] = {}
    for key in COMPONENT_WEIGHTS:
        pool: List[np.ndarray] = []
        for v, comps in zip(videos, raw):
            if comps is None or key not in comps:
                continue
            w, _cut = _type_weights(v, comps["ts"], config.straight_weight)
            vals = comps[key][w > 0]
            pool.append(vals if len(vals) else comps[key])
        if not pool:
            continue
        allv = np.concatenate(pool)
        if len(allv) == 0:
            continue
        sc = float(np.percentile(allv, 95))
        if sc <= 1e-6:
            sc = float(np.max(allv))
        if sc > 1e-6:
            scales[key] = sc

    results: List[Optional[Dict[str, np.ndarray]]] = []
    for v, comps in zip(videos, raw):
        if comps is None:
            results.append(None)
            continue
        ts = comps["ts"]
        type_w, cut = _type_weights(v, ts, config.straight_weight)
        total_w = 0.0
        score = np.zeros(len(ts), dtype=np.float32)
        out: Dict[str, np.ndarray] = {"ts": ts, "cut": cut}
        for key, wt in COMPONENT_WEIGHTS.items():
            if key not in comps:
                continue
            total_w += wt
            if key in scales:
                norm = np.clip(comps[key] / scales[key], 0.0, 1.5).astype(np.float32)
            else:
                norm = np.zeros(len(ts), dtype=np.float32)
            out[f"{key}_n"] = norm * wt * type_w
            score += norm * wt
        if total_w > 0:
            score /= total_w
            for key in COMPONENT_WEIGHTS:
                if f"{key}_n" in out:
                    out[f"{key}_n"] = out[f"{key}_n"] / total_w
        out["score"] = score * type_w
        out["blocked"] = type_w <= 0.0  # 정차·잘라내기 등 하이라이트로 쓰지 않을 샘플
        out["fps"] = np.array([float(v.feature_cache["sample_fps"])], dtype=np.float32)  # type: ignore[index]
        results.append(out)
    return results


# ── 장면 선정 ──

def select_highlights(
    videos: List[VideoFileInfo],
    config: HighlightConfig,
) -> List[HighlightClip]:
    """
    점수가 높은 장면부터 골라 목표 길이를 채우고, 시간순으로 정렬된 하이라이트 장면 목록을 반환합니다.
    분석(feature_cache)이 안 된 영상은 건너뜁니다.
    """
    clip_len = max(1.0, float(config.clip_duration_sec))
    target = max(clip_len, float(config.target_duration_sec))
    scored = compute_highlight_scores(videos, config)

    # 영상별 "창 시작 위치 → 창 평균 점수" 배열
    cands: List[Tuple[int, np.ndarray, float, int]] = []  # (video_idx, window_means, fps, win_steps)
    for vi, (v, sd) in enumerate(zip(videos, scored)):
        if sd is None:
            continue
        fps = float(sd["fps"][0])
        score = sd["score"]
        cut = sd["cut"].astype(np.float32)
        blocked = sd["blocked"].astype(np.float32)
        n = len(score)
        win = max(1, int(round(clip_len * fps)))
        if n >= win:
            csum = np.concatenate([[0.0], np.cumsum(score, dtype=np.float64)])
            means = (csum[win:] - csum[:-win]) / win
            ccut = np.concatenate([[0.0], np.cumsum(cut, dtype=np.float64)])
            has_cut = (ccut[win:] - ccut[:-win]) > 0
            cblk = np.concatenate([[0.0], np.cumsum(blocked, dtype=np.float64)])
            # 장면의 25% 이상이 정차 중인 화면이면 후보에서 제외 (멈춰 있다 출발하는 지루한 장면 방지)
            mostly_blocked = (cblk[win:] - cblk[:-win]) > 0.25 * win
            means = means.astype(np.float64)
            means[has_cut | mostly_blocked] = -np.inf
        elif v.duration_sec >= 1.0 and not cut.any():
            means = np.array([float(np.mean(score))], dtype=np.float64)
        else:
            continue
        means[~np.isfinite(means)] = -np.inf
        means[means <= 1e-4] = -np.inf
        cands.append((vi, means, fps, win))

    if not cands:
        return []

    total_src = sum(v.duration_sec for v in videos)
    want_count = max(1.0, target / clip_len)
    # 전체 주행이 고루 담기도록 장면 사이 최소 간격을 둠 (2시간 영상 → 약 2~3분 간격)
    spread_gap = float(min(180.0, max(clip_len * 0.5, total_src / (want_count * 3.0))))

    def window_bounds(vi: int, i: int, fps: float) -> Tuple[float, float]:
        dur = videos[vi].duration_sec
        s = i / fps
        e = min(dur, s + clip_len)
        return s, e

    picked: List[Tuple[int, int, float]] = []  # (video_idx, start_index, mean_score)
    total = 0.0
    for gap in (spread_gap, 0.0):
        work = [m.copy() for (_vi, m, _f, _w) in cands]
        # 이미 고른 장면 주변 억제
        for (pvi, pidx, _ps) in picked:
            for ci, (vi, _m, fps, _w) in enumerate(cands):
                if vi != pvi:
                    continue
                ps, pe = window_bounds(vi, pidx, fps)
                lo = int(np.floor((ps - clip_len - gap) * fps)) + 1
                hi = int(np.ceil((pe + gap) * fps))
                work[ci][max(0, lo): max(0, hi)] = -np.inf

        while total < target - 0.25 * clip_len:
            best_ci, best_i, best_val = -1, -1, -np.inf
            for ci, arr in enumerate(work):
                if len(arr) == 0:
                    continue
                i = int(np.argmax(arr))
                if arr[i] > best_val:
                    best_ci, best_i, best_val = ci, i, float(arr[i])
            if best_ci < 0 or not np.isfinite(best_val):
                break
            vi, _m, fps, _w = cands[best_ci]
            s, e = window_bounds(vi, best_i, fps)
            picked.append((vi, best_i, best_val))
            total += e - s
            lo = int(np.floor((s - clip_len - gap) * fps)) + 1
            hi = int(np.ceil((e + gap) * fps))
            work[best_ci][max(0, lo): max(0, hi)] = -np.inf
        if total >= target - 0.25 * clip_len:
            break

    clips: List[HighlightClip] = []
    for vi, idx, val in picked:
        fps = next(f for (cvi, _m, f, _w) in cands if cvi == vi)
        s, e = window_bounds(vi, idx, fps)
        sd = scored[vi]
        reason = ""
        if sd is not None:
            i1 = idx
            i2 = max(i1 + 1, min(len(sd["score"]), i1 + int(round(clip_len * fps))))
            best_key, best_c = "", -1.0
            for key in COMPONENT_WEIGHTS:
                arr = sd.get(f"{key}_n")
                if arr is None:
                    continue
                c = float(np.mean(arr[i1:i2]))
                if c > best_c:
                    best_key, best_c = key, c
            # 두드러진 요소가 없으면(모든 기여도가 미미) 일반 주행 장면으로 표기
            reason = REASON_LABELS.get(best_key, "") if best_c >= 0.03 else "주행 장면"
        clips.append(HighlightClip(videos[vi], s, e, score=val, reason=reason))

    order = {id(v): i for i, v in enumerate(videos)}
    clips.sort(key=lambda c: (order[id(c.video)], c.start_sec))
    return clips


# ── 쇼츠 렌더링 ──

def shorts_dimensions(layout: ShortsLayout, resolution: str) -> Tuple[int, int]:
    small = resolution == "720p"
    if layout == ShortsLayout.LANDSCAPE:
        return (1280, 720) if small else (1920, 1080)
    return (720, 1280) if small else (1080, 1920)


def _find_overlay_font(need_cjk: bool) -> Optional[str]:
    """오버레이 텍스트용 글꼴 경로 (한글 제목이면 한글 글꼴 우선)."""
    cjk = [
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    ]
    latin = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for fp in (cjk + latin) if need_cjk else (latin + cjk):
        if os.path.exists(fp):
            return fp
    return None


def _filter_path(p: str | Path) -> str:
    """FFmpeg 필터 인자 안에서 쓸 수 있도록 경로를 이스케이프합니다 (Windows 'C:' 대응)."""
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _text_units(text: str) -> float:
    """글꼴 크기 1 기준 대략적인 문자열 폭 (한글·한자는 1.0, 영문·숫자는 0.6)."""
    return sum(1.0 if ord(ch) >= 0x2E80 else 0.6 for ch in text) or 1.0


def clip_timestamp_text(clip: HighlightClip) -> Optional[str]:
    """장면의 실제 촬영 시각 문자열. 촬영 시각이 신뢰할 수 없으면(파일 수정일 기준) None."""
    v = clip.video
    if v.recorded_at is None or v.timestamp_source == "파일 수정일":
        return None
    t = v.recorded_at + timedelta(seconds=clip.start_sec)
    return t.strftime("%Y.%m.%d  %H:%M:%S")


def _drawtext(font: str, text_file: Path, fontsize: int, y: str, boxed: bool) -> str:
    base = (
        f"drawtext=fontfile='{_filter_path(font)}':textfile='{_filter_path(text_file)}':expansion=none:"
        f"fontsize={fontsize}:fontcolor=white:x=(w-tw)/2:y={y}"
    )
    if boxed:
        pad = max(6, fontsize // 3)
        return base + f":box=1:boxcolor=black@0.45:boxborderw={pad}"
    return base + ":shadowcolor=black@0.7:shadowx=3:shadowy=3"


def _build_clip_filtergraph(
    clip: HighlightClip,
    cfg: ShortsExportConfig,
    out_w: int,
    out_h: int,
    out_dur: float,
    title_file: Optional[Path],
    title_text: str,
    ts_file: Optional[Path],
    font_title: Optional[str],
    font_ts: Optional[str],
    caption_file: Optional[Path] = None,
    font_caption: Optional[str] = None,
) -> str:
    v = clip.video
    if cfg.layout == ShortsLayout.VERTICAL_BLUR:
        bw, bh = max(2, (out_w // 4) // 2 * 2), max(2, (out_h // 4) // 2 * 2)
        graph = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"boxblur=8:2,scale={out_w}:{out_h},eq=brightness=-0.10[bgb];"
            f"[fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:force_divisible_by=2[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
        )
    elif cfg.layout == ShortsLayout.VERTICAL_CROP:
        graph = f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}"
    else:
        graph = (
            f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
        )

    graph += f",fps={cfg.fps:.3f}"

    vertical = cfg.layout != ShortsLayout.LANDSCAPE
    # 흐림 배경 레이아웃에서는 원본 영상 위·아래 띠(band) 영역에 글자를 배치
    src_aspect = (v.height / v.width) if v.width > 0 else 9 / 16
    fg_h = min(out_h, out_w * src_aspect)
    band = max(0.0, (out_h - fg_h) / 2.0)
    use_band = cfg.layout == ShortsLayout.VERTICAL_BLUR and band >= out_h * 0.12

    if title_file is not None and font_title:
        base_fs = out_w * (0.075 if vertical else 0.042)
        fs = int(max(18, min(base_fs, 0.88 * out_w / _text_units(title_text))))
        if use_band:
            y = f"{max(out_h * 0.04, band * 0.5 - fs * 0.6):.0f}"
        else:
            y = f"{out_h * 0.06:.0f}"
        graph += "," + _drawtext(font_title, title_file, fs, y, boxed=not use_band)

    has_caption = caption_file is not None and bool(font_caption) and bool(clip.caption.strip())
    if has_caption:
        base_fs = out_w * (0.065 if vertical else 0.036)
        fs = int(max(16, min(base_fs, 0.88 * out_w / _text_units(clip.caption.strip()))))
        if use_band:
            y = f"{out_h - band + band * 0.14:.0f}"
        else:
            y = f"{out_h * 0.78:.0f}"
        graph += "," + _drawtext(font_caption, caption_file, fs, y, boxed=not use_band)  # type: ignore[arg-type]

    if ts_file is not None and font_ts:
        fs = int(max(14, out_w * (0.042 if vertical else 0.022)))
        if use_band:
            y = f"{out_h - band + band * (0.50 if has_caption else 0.25):.0f}"
        else:
            y = f"{out_h * 0.90:.0f}"
        graph += "," + _drawtext(font_ts, ts_file, fs, y, boxed=not use_band)

    f = max(0.0, cfg.fade_sec)
    if f > 0 and out_dur > f * 3:
        graph += f",fade=t=in:st=0:d={f:.3f},fade=t=out:st={out_dur - f:.3f}:d={f:.3f}"
    graph += ",setsar=1,format=yuv420p[v]"
    return graph


def _run_ffmpeg(cmd: List[str], cancel_check: Optional[Callable[[], bool]]) -> Tuple[bool, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=get_CreationFlags(),
    )
    err_chunks: List[str] = []
    while True:
        if cancel_check and cancel_check():
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return False, "사용자에 의해 작업이 취소되었습니다."
        try:
            _out, err = proc.communicate(timeout=0.25)
            if err:
                err_chunks.append(err.decode("utf-8", errors="replace"))
            break
        except subprocess.TimeoutExpired:
            continue
    return proc.returncode == 0, "".join(err_chunks).strip()


def _encode_highlight_clip(
    ffmpeg: str,
    clip: HighlightClip,
    cfg: ShortsExportConfig,
    out_path: Path,
    out_w: int,
    out_h: int,
    encoder_name: str,
    encoder_args: List[str],
    with_audio: bool,
    title_file: Optional[Path],
    ts_file: Optional[Path],
    font_title: Optional[str],
    font_ts: Optional[str],
    cancel_check: Optional[Callable[[], bool]],
    caption_file: Optional[Path] = None,
    font_caption: Optional[str] = None,
) -> Tuple[bool, str]:
    dur = clip.duration
    graph = _build_clip_filtergraph(
        clip, cfg, out_w, out_h, dur, title_file, cfg.title_text.strip(), ts_file, font_title, font_ts,
        caption_file, font_caption,
    )
    cmd: List[str] = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{clip.start_sec:.3f}",
        "-t", f"{dur:.3f}",
        "-i", str(clip.video.path),
    ]
    use_src_audio = with_audio and clip.video.has_audio
    if with_audio and not use_src_audio:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]

    if with_audio:
        a_in = "[0:a:0]" if use_src_audio else "[1:a]"
        a_chain = "aformat=sample_rates=48000:channel_layouts=stereo,aresample=async=1:first_pts=0"
        f = max(0.0, cfg.fade_sec)
        if f > 0 and dur > f * 3:
            a_chain += f",afade=t=in:st=0:d={f:.3f},afade=t=out:st={dur - f:.3f}:d={f:.3f}"
        graph += f";{a_in}{a_chain}[a]"

    cmd += ["-filter_complex", graph, "-map", "[v]"]
    if with_audio:
        cmd += ["-map", "[a]"]
    cmd += ["-c:v", encoder_name, *encoder_args]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += ["-t", f"{dur:.3f}", "-f", "mpegts", str(out_path)]

    ok, err = _run_ffmpeg(cmd, cancel_check)
    ok = ok and out_path.exists() and out_path.stat().st_size > 0
    return ok, err


def _mix_background_music(
    ffmpeg: str,
    src_video: Path,
    cfg: ShortsExportConfig,
    total_dur: float,
    output_path: Path,
    cancel_check: Optional[Callable[[], bool]],
) -> None:
    assert cfg.bgm_path is not None
    vol = max(0.0, min(2.0, cfg.bgm_volume))
    fade_out = min(1.5, max(0.3, total_dur / 4.0))
    bgm_chain = (
        f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={vol:.3f},"
        f"atrim=duration={total_dur:.3f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d=0.5,afade=t=out:st={max(0.0, total_dur - fade_out):.3f}:d={fade_out:.3f}"
    )
    graphs: List[str] = []
    if cfg.audio_mode == ShortsAudioMode.BGM_MIX:
        base = f"{bgm_chain}[bgm];[0:a]aformat=sample_rates=48000:channel_layouts=stereo[orig];"
        graphs.append(base + "[orig][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
        # 구버전 FFmpeg(normalize 옵션 미지원) 폴백
        graphs.append(base + "[orig][bgm]amix=inputs=2:duration=first:dropout_transition=0,volume=2.0[a]")
    else:
        graphs.append(f"{bgm_chain}[a]")

    last_err = ""
    for graph in graphs:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src_video),
            "-stream_loop", "-1",
            "-i", str(cfg.bgm_path),
            "-filter_complex", graph,
            "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{total_dur:.3f}",
            "-movflags", "+faststart",
            str(output_path),
        ]
        ok, err = _run_ffmpeg(cmd, cancel_check)
        if cancel_check and cancel_check():
            raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")
        if ok and output_path.exists():
            return
        last_err = err
    raise RuntimeError(f"배경음악을 입히는 중 오류가 발생했습니다:\n{last_err}")


def render_highlight_shorts(
    clips: List[HighlightClip],
    cfg: ShortsExportConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Path:
    """선정된 하이라이트 장면들을 쇼츠 영상(MP4) 하나로 만듭니다."""
    use_clips = [c for c in clips if c.included and c.duration >= 0.5]
    if not use_clips:
        raise ValueError("쇼츠에 넣을 하이라이트 장면이 없습니다. 하이라이트를 먼저 찾아 주세요.")

    needs_bgm = cfg.audio_mode in (ShortsAudioMode.BGM, ShortsAudioMode.BGM_MIX)
    if needs_bgm and (cfg.bgm_path is None or not Path(cfg.bgm_path).exists()):
        raise ValueError("배경음악 파일을 선택해 주세요.")

    ffmpeg = get_ffmpeg_exe()
    out_w, out_h = shorts_dimensions(cfg.layout, cfg.resolution)
    enc_name, enc_args = resolve_video_encoder(VideoCodec.H264, cfg.quality_preset, prefer_hw=cfg.use_hw_accel)
    sw_name, sw_args = get_software_encoder_args(VideoCodec.H264, cfg.quality_preset)

    # BGM 단독 모드는 장면 인코딩 시 오디오를 빼고 마지막에 배경음악만 입힘
    clip_audio = cfg.audio_mode in (ShortsAudioMode.ORIGINAL, ShortsAudioMode.BGM_MIX)

    title = cfg.title_text.strip()
    font_title = _find_overlay_font(need_cjk=any(ord(ch) >= 0x2E80 for ch in title)) if title else None
    font_ts = _find_overlay_font(need_cjk=False) if cfg.show_timestamp else None
    font_caption = (
        _find_overlay_font(need_cjk=True)
        if cfg.show_captions and any(c.caption.strip() for c in use_clips) else None
    )

    output_path = Path(cfg.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_dur = sum(c.duration for c in use_clips)
    done_dur = 0.0

    with tempfile.TemporaryDirectory(prefix="quick_dash_cam_shorts_") as tmp_str:
        tmp = Path(tmp_str)
        title_file: Optional[Path] = None
        if title and font_title:
            title_file = tmp / "title.txt"
            title_file.write_text(title, encoding="utf-8")

        seg_files: List[Path] = []
        for idx, clip in enumerate(use_clips):
            if cancel_check and cancel_check():
                raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")
            if progress_callback:
                progress_callback(
                    min(0.9, done_dur / max(0.1, total_dur) * 0.9),
                    f"쇼츠 장면 인코딩 [{idx + 1}/{len(use_clips)}]: {clip.video.filename} "
                    f"({clip.start_sec:.1f}s~, {clip.reason or '하이라이트'})",
                )

            ts_file: Optional[Path] = None
            ts_text = clip_timestamp_text(clip) if (cfg.show_timestamp and font_ts) else None
            if ts_text:
                ts_file = tmp / f"ts_{idx:04d}.txt"
                ts_file.write_text(ts_text, encoding="utf-8")

            caption_file: Optional[Path] = None
            if font_caption and clip.caption.strip():
                caption_file = tmp / f"cap_{idx:04d}.txt"
                caption_file.write_text(clip.caption.strip(), encoding="utf-8")

            seg_path = tmp / f"hl_{idx:04d}.ts"
            args = (ffmpeg, clip, cfg, seg_path, out_w, out_h)
            extra = (clip_audio, title_file, ts_file, font_title, font_ts, cancel_check, caption_file, font_caption)
            ok, err = _encode_highlight_clip(*args, enc_name, enc_args, *extra)
            if not ok and enc_name != sw_name and not (cancel_check and cancel_check()):
                # 하드웨어 인코더 실패 시 소프트웨어 인코더로 자동 폴백
                enc_name, enc_args = sw_name, sw_args
                ok, err = _encode_highlight_clip(*args, enc_name, enc_args, *extra)
            if not ok:
                if cancel_check and cancel_check():
                    raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")
                detail = f"\n[오류 상세]: {err}" if err else ""
                raise RuntimeError(
                    f"쇼츠 장면 인코딩에 실패했습니다: {clip.video.filename} "
                    f"({clip.start_sec:.1f}s ~ {clip.end_sec:.1f}s){detail}"
                )
            seg_files.append(seg_path)
            done_dur += clip.duration

        if progress_callback:
            progress_callback(0.92, "하이라이트 장면들을 하나의 쇼츠 영상으로 잇는 중...")

        concat_list = tmp / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as fh:
            for sf in seg_files:
                fh.write(f"file '{str(sf.resolve()).replace(chr(92), '/')}'\n")

        joined = tmp / "joined.mp4" if needs_bgm else output_path
        concat_cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy",
        ]
        if clip_audio:
            concat_cmd += ["-bsf:a", "aac_adtstoasc"]
        concat_cmd += ["-movflags", "+faststart", str(joined)]
        ok, err = _run_ffmpeg(concat_cmd, cancel_check)
        if not ok or not joined.exists():
            if cancel_check and cancel_check():
                raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")
            raise RuntimeError(f"쇼츠 영상 병합 중 오류가 발생했습니다:\n{err}")

        if needs_bgm:
            if progress_callback:
                progress_callback(0.96, "배경음악을 입히는 중...")
            _mix_background_music(ffmpeg, joined, cfg, total_dur, output_path, cancel_check)

    if progress_callback:
        progress_callback(1.0, f"완료! 저장 위치: {output_path.name}")
    return output_path

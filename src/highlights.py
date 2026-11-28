"""
주행 영상 하이라이트 쇼츠 생성기.

1) 컷 품질 점수: 구간 분석(analyzer)에서 캐싱된 프레임별 특징값(조향, 화면 변화, 노면 움직임)으로
   "차가 안정적으로 앞으로 달리고 컷 안의 화면이 차분한 정도"를 계산합니다.
   좌·우회전, 주차장 출입(출발·도착 부근·저속), 급격한 화면 변화는 감점/제외하고, 정차·잘라내기 구간은 쓰지 않습니다.
2) 장면 선정: 컷 안에서는 차분하되 컷끼리는 서로 달라 보이도록, 구간 분석 때 저장한 썸네일(색 분포·구도)로
   이미 고른 컷들과 가장 다른 장면을 차례로 고릅니다. 고른 장면은 시간순으로 정렬됩니다.
3) 렌더링: 수평 보정 → 세로 9:16(흐림 배경 / 꽉 채우기) 또는 가로 16:9 레이아웃 → 제목·자막·촬영 시각 오버레이 →
   컷 사이 크로스 디졸브로 이어 붙이고, 필요 시 배경음악을 입힙니다.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
    ShortsTransition,
    VideoCodec,
    VideoFileInfo,
)

AUDIO_SAMPLE_RATE = 8000
AUDIO_CACHE_VERSION = "audio_rms_v1"

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


TRIP_GAP_SEC = 300.0      # 녹화 시각 사이가 5분 넘게 비면 다른 주행(출발·도착이 따로 있음)으로 봄
TRIP_EDGE_MAX_SEC = 180.0  # 주행 시작·끝 이 시간만큼은 주차장·골목일 가능성이 높아 점수를 낮춤
TRIP_EDGE_MIN_FACTOR = 0.1


def _trip_edge_factors(
    videos: List[VideoFileInfo],
    ts_list: List[Optional[np.ndarray]],
) -> List[Optional[np.ndarray]]:
    """
    주행(trip) 단위로 시작·끝 부근 점수 배율(0.3~1.0)을 계산합니다.
    출발 직후·도착 직전은 대부분 주차장·골목이라, 가까운 차·기둥이 빠르게 지나가며 '화면 변화'가 커도 볼거리가 아닙니다.
    촬영 시각을 믿을 수 있는 영상끼리 5분 넘게 비어 있으면 서로 다른 주행으로 나눕니다.
    """
    trips: List[List[int]] = []
    prev_end = None
    for i, v in enumerate(videos):
        reliable = v.recorded_at is not None and v.timestamp_source != "파일 수정일"
        start = v.recorded_at if reliable else None
        new_trip = not trips
        if not new_trip and start is not None and prev_end is not None:
            new_trip = (start - prev_end).total_seconds() > TRIP_GAP_SEC
        if new_trip:
            trips.append([i])
        else:
            trips[-1].append(i)
        prev_end = (start + timedelta(seconds=v.duration_sec)) if start is not None else None

    factors: List[Optional[np.ndarray]] = [None] * len(videos)
    for trip in trips:
        trip_len = sum(videos[i].duration_sec for i in trip)
        edge = min(TRIP_EDGE_MAX_SEC, trip_len * 0.2)
        if edge <= 0:
            continue
        offset = 0.0
        for i in trip:
            ts = ts_list[i]
            if ts is not None:
                g = offset + ts.astype(np.float64)
                d = np.minimum(g, trip_len - g)
                factors[i] = (TRIP_EDGE_MIN_FACTOR + (1.0 - TRIP_EDGE_MIN_FACTOR) * np.clip(d / edge, 0.0, 1.0)).astype(np.float32)
            offset += videos[i].duration_sec
    return factors


# 좋은 컷 = 차가 안정적으로 앞으로 달리고 컷 안에서 화면이 차분한 장면.
# (컷 안의 큰 변화 — 좌·우회전, 주차장 출입, 터널 진출입, 급변 — 는 쇼츠에서 어지러움만 줌)
STEER_REJECT = 0.7       # 조향(회전) 정규값이 이 이상이면 해당 순간은 사용 불가
SURPRISE_PENALTY = 0.6   # 주변 흐름 대비 급격한 화면 변화 감점 정도
SCENE_EXCESS_START = 0.7  # 화면 변화량이 상위 수준을 넘는 만큼 감점 시작
SLOW_SPEED_MIN = 0.35    # 아주 느린 주행(주차장·골목)의 최소 배율
STEER_SCALE_FLOOR = 1.5   # 회전이 거의 없는 주행에서 차선 안 미세한 흔들림을 '회전'으로 오인하지 않도록 정규화 하한
MIN_WINDOW_QUALITY = 0.15  # 이보다 불안정한 컷은 '다르게 보여도' 쓰지 않음 (어지러운 컷보다 컷 수가 모자란 게 나음)


def compute_highlight_scores(
    videos: List[VideoFileInfo],
    config: HighlightConfig,
) -> List[Optional[Dict[str, np.ndarray]]]:
    """
    영상별 '안정적인 컷' 점수 곡선(0~1)을 계산합니다. 전체 영상 공통 기준(백분위)으로 정규화.
    - 회전 중(조향 큼)이면 0에 가깝게, 급격한 화면 변화·과도한 변화량은 감점
    - 정차·잘라내기 구간은 0, 출발·도착 부근(주차장 가능성)과 아주 느린 주행은 감점
    반환 원소: {"ts", "score", "cut", "blocked", "straight", "fps"} 또는 분석 전이면 None.
    """
    raw: List[Optional[Dict[str, np.ndarray]]] = []
    for v in videos:
        comps = compute_highlight_components(v)
        if comps is not None and not config.use_audio_peaks:
            comps.pop("audio", None)
        raw.append(comps)

    # 움직이는 구간 기준 공통 정규화 값 (조향·변화량은 95백분위, 속도 기준은 중앙값)
    scales: Dict[str, float] = {}
    motion_ref = 0.0
    for key in ("steer", "scene", "surprise", "motion"):
        pool: List[np.ndarray] = []
        for v, comps in zip(videos, raw):
            if comps is None:
                continue
            w, _cut = _type_weights(v, comps["ts"], 1.0)
            vals = comps[key][w > 0]
            pool.append(vals if len(vals) else comps[key])
        if not pool:
            continue
        allv = np.concatenate(pool)
        if len(allv) == 0:
            continue
        if key == "motion":
            motion_ref = float(np.median(allv))
            continue
        sc = float(np.percentile(allv, 95))
        if sc <= 1e-6:
            sc = float(np.max(allv))
        if key == "steer":
            sc = max(sc, STEER_SCALE_FLOOR)
        if sc > 1e-6:
            scales[key] = sc

    edge_factors = _trip_edge_factors(videos, [c["ts"] if c is not None else None for c in raw])

    def norm(comps: Dict[str, np.ndarray], key: str) -> np.ndarray:
        if key not in scales:
            return np.zeros(len(comps["ts"]), dtype=np.float32)
        return np.clip(comps[key] / scales[key], 0.0, 1.5).astype(np.float32)

    results: List[Optional[Dict[str, np.ndarray]]] = []
    for v, comps, edge in zip(videos, raw, edge_factors):
        if comps is None:
            results.append(None)
            continue
        ts = comps["ts"]
        type_w, cut = _type_weights(v, ts, config.straight_weight)
        blocked = type_w <= 0.0
        straight = np.zeros(len(ts), dtype=bool)
        for seg in v.segments:
            if seg.segment_type == SegmentType.STRAIGHT:
                straight |= (ts >= seg.start_sec) & (ts < seg.end_sec)

        steer_n = norm(comps, "steer")
        surprise_n = norm(comps, "surprise")
        scene_n = norm(comps, "scene")
        stable = np.clip(1.0 - steer_n / STEER_REJECT, 0.0, 1.0) ** 2           # 회전 중이면 0
        stable *= 1.0 - SURPRISE_PENALTY * np.clip(surprise_n, 0.0, 1.0)        # 급변 감점
        stable *= 1.0 - 0.5 * np.clip(scene_n - SCENE_EXCESS_START, 0.0, 1.0)  # 과도한 변화 감점
        if motion_ref > 1e-6:
            # 아주 느린 주행(주차장·골목) 감점. 노면 질감이 적은 고속도로는 속도 판단이 부정확해 직진 구간엔 적용 안 함
            speed = np.clip((comps["motion"] / motion_ref - 0.25) / 0.5, SLOW_SPEED_MIN, 1.0)
            stable *= np.where(straight, 1.0, speed)
        if edge is not None:
            stable *= edge  # 출발·도착 부근(주차장일 가능성 높음) 감점

        results.append({
            "ts": ts,
            "cut": cut,
            "score": (stable * type_w).astype(np.float32),
            "blocked": blocked,  # 정차·잘라내기 등 하이라이트로 쓰지 않을 샘플
            "straight": straight,
            "fps": np.array([float(v.feature_cache["sample_fps"])], dtype=np.float32),  # type: ignore[index]
        })
    return results


# ── 장면 선정 ──

GENERIC_REASON = "주행 장면"
CRUISE_REASON = "고속도로 순항"
REASON_QUOTA_RATIO = 0.4  # (대체 선정 방식) 같은 선정 이유의 장면은 전체의 40%까지만

Window = Tuple[int, int, float]  # (video_idx, start_index, window_mean_score)


def _window_table(
    videos: List[VideoFileInfo],
    scored: List[Optional[Dict[str, np.ndarray]]],
    clip_len: float,
) -> List[Tuple[int, np.ndarray, float, int]]:
    """영상별 '창 시작 위치 → 창 평균 점수' 배열. 사용할 수 없는 창은 -inf."""
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
            # 컷 일부에만 회전·감속이 섞여도 어지러우므로, 평균과 함께 컷 안의 최저값도 반영
            mins = np.lib.stride_tricks.sliding_window_view(score, win).min(axis=1).astype(np.float64)
            means = 0.5 * means + 0.5 * mins
            ccut = np.concatenate([[0.0], np.cumsum(cut, dtype=np.float64)])
            has_cut = (ccut[win:] - ccut[:-win]) > 0
            cblk = np.concatenate([[0.0], np.cumsum(blocked, dtype=np.float64)])
            # 정차 순간이 섞인 컷은 제외 (멈춰 있다 출발하는 것도 컷 안의 큰 변화)
            any_blocked = (cblk[win:] - cblk[:-win]) > 0
            means = means.astype(np.float64)
            means[has_cut | any_blocked] = -np.inf
        elif v.duration_sec >= 1.0 and not cut.any():
            means = np.array([float(np.mean(score))], dtype=np.float64)
        else:
            continue
        means[~np.isfinite(means)] = -np.inf
        means[means < MIN_WINDOW_QUALITY] = -np.inf
        cands.append((vi, means, fps, win))
    return cands


def _window_reason(sd: Dict[str, np.ndarray], i1: int, win: int) -> str:
    """선정 이유 표기: 대부분 직진 구간이면 '고속도로 순항', 그 외 '주행 장면'. (AI 선정 시 AI의 장면 설명으로 대체)"""
    i2 = max(i1 + 1, min(len(sd["score"]), i1 + win))
    straight = sd.get("straight")
    if straight is not None and float(np.mean(straight[i1:i2])) >= 0.5:
        return CRUISE_REASON
    return GENERIC_REASON


def _greedy_pick(
    videos: List[VideoFileInfo],
    scored: List[Optional[Dict[str, np.ndarray]]],
    cands: List[Tuple[int, np.ndarray, float, int]],
    clip_len: float,
    target_sec: float,
    spread_gap: float,
    picked: List[Window],
) -> List[Window]:
    """점수 높은 창부터 고르되, 간격·같은 이유 쏠림을 제한합니다. 조건을 단계적으로 풀어 목표 길이를 채웁니다."""
    picked = list(picked)
    want_count = max(1.0, target_sec / clip_len)
    quota = max(2, int(np.ceil(REASON_QUOTA_RATIO * want_count)))
    ci_of = {vi: ci for ci, (vi, _m, _f, _w) in enumerate(cands)}

    def bounds(vi: int, i: int, fps: float) -> Tuple[float, float]:
        s = i / fps
        return s, min(videos[vi].duration_sec, s + clip_len)

    def suppress(arr: np.ndarray, s: float, e: float, fps: float, gap: float) -> None:
        lo = int(np.floor((s - clip_len - gap) * fps)) + 1
        hi = int(np.ceil((e + gap) * fps))
        arr[max(0, lo): max(0, hi)] = -np.inf

    def reason_of(vi: int, i: int) -> str:
        sd = scored[vi]
        return _window_reason(sd, i, cands[ci_of[vi]][3]) if sd is not None else GENERIC_REASON

    total = sum(bounds(vi, i, cands[ci_of[vi]][2])[1] - bounds(vi, i, cands[ci_of[vi]][2])[0] for vi, i, _v in picked)
    for gap, use_quota in ((spread_gap, True), (0.0, True), (0.0, False)):
        if total >= target_sec - 0.25 * clip_len:
            break
        work = [m.copy() for (_vi, m, _f, _w) in cands]
        counts: Dict[str, int] = {}
        for pvi, pidx, _pv in picked:
            ci = ci_of[pvi]
            ps, pe = bounds(pvi, pidx, cands[ci][2])
            suppress(work[ci], ps, pe, cands[ci][2], gap)
            r = reason_of(pvi, pidx)
            counts[r] = counts.get(r, 0) + 1

        while total < target_sec - 0.25 * clip_len:
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
            s, e = bounds(vi, best_i, fps)
            r = reason_of(vi, best_i)
            if use_quota and r not in (GENERIC_REASON, CRUISE_REASON) and counts.get(r, 0) >= quota:
                suppress(work[best_ci], s, e, fps, 0.0)  # 이 이유는 이미 충분 → 이 장면은 건너뜀
                continue
            picked.append((vi, best_i, best_val))
            counts[r] = counts.get(r, 0) + 1
            total += e - s
            suppress(work[best_ci], s, e, fps, gap)
    return picked


def _windows_to_clips(
    videos: List[VideoFileInfo],
    scored: List[Optional[Dict[str, np.ndarray]]],
    cands: List[Tuple[int, np.ndarray, float, int]],
    picked: List[Window],
    clip_len: float,
) -> List[HighlightClip]:
    ci_of = {vi: ci for ci, (vi, _m, _f, _w) in enumerate(cands)}
    clips: List[HighlightClip] = []
    for vi, idx, val in picked:
        _cvi, _m, fps, win = cands[ci_of[vi]]
        s = idx / fps
        e = min(videos[vi].duration_sec, s + clip_len)
        sd = scored[vi]
        reason = _window_reason(sd, idx, win) if sd is not None else ""
        clips.append(HighlightClip(videos[vi], s, e, score=val, reason=reason))
    order = {id(v): i for i, v in enumerate(videos)}
    clips.sort(key=lambda c: (order[id(c.video)], c.start_sec))
    return clips


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
    cands = _window_table(videos, scored, clip_len)
    if not cands:
        return []
    total_src = sum(v.duration_sec for v in videos)
    want_count = max(1.0, target / clip_len)
    # 전체 주행이 고루 담기도록 장면 사이 최소 간격을 둠 (2시간 영상 → 약 2~3분 간격)
    spread_gap = float(min(180.0, max(clip_len * 0.5, total_src / (want_count * 3.0))))
    if all(v.feature_cache is None or "appear" in v.feature_cache for v in videos):
        # 장면 모습 썸네일이 있으면: 컷끼리 배경이 겹치지 않게 고르는 방식 사용
        diverse = select_diverse_highlights(videos, config, int(round(want_count)))
        if diverse:
            return diverse
    picked = _greedy_pick(videos, scored, cands, clip_len, target, spread_gap, [])
    return _windows_to_clips(videos, scored, cands, picked, clip_len)


# ── 장면 모습(appearance) 분석: 컷끼리 배경이 겹치지 않게 고르기 위한 썸네일 ──

APPEAR_CACHE_VERSION = "appear_v2"
APPEAR_STEP_SEC = 2.0
APPEAR_W, APPEAR_H = 32, 18
DARK_BRIGHTNESS = 45.0     # 평균 밝기가 이보다 낮으면(지하주차장·심야 골목 등) 감점
MIN_CUT_GAP_SEC = 20.0     # 고른 컷끼리 최소 시간 간격 (바로 옆 장면이 연달아 뽑히지 않게)
POOL_BLOCK_SEC = 8.0       # 후보 풀: 8초 구간마다 가장 나은 창 하나씩
DUPLICATE_DISTANCE = 0.08  # 이미 고른 컷과 모습 차이가 이보다 작으면 같은 장면으로 보고 제외


def _appear_cache_path(video_path: Path) -> Optional[Path]:
    try:
        st = video_path.stat()
        key = f"{APPEAR_CACHE_VERSION}|{video_path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{APPEAR_STEP_SEC}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        cache_dir = Path(tempfile.gettempdir()) / "quick_dash_cam_fcache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"appear_{digest}.npy"
    except Exception:
        return None


def save_appearance_cache(video_path: Path, thumbs: np.ndarray) -> None:
    """구간 분석(analyzer)이 디코딩하면서 모은 썸네일을 저장합니다 (추가 디코딩 없이 재사용)."""
    cache = _appear_cache_path(video_path)
    if cache is not None and len(thumbs):
        try:
            np.save(cache, thumbs.astype(np.uint8))
        except Exception:
            pass


def extract_appearance_thumbs(
    video_path: Path,
    cancel_check: Optional[Callable[[], bool]] = None,
    use_gpu_accel: bool = True,
) -> Optional[np.ndarray]:
    """
    APPEAR_STEP_SEC(2초)마다 32x18 썸네일. 보통은 구간 분석 때 함께 저장된 캐시를 읽고,
    캐시가 없을 때만(이번 기능 이전에 분석한 영상) 분석과 같은 GPU 디코딩 경로로 한 번 뽑습니다.
    반환: (n, 18, 32, 3) uint8 BGR — i번째 썸네일은 i*2초 지점.
    """
    import cv2
    from .analyzer import _iter_downsampled_frames  # 순환 import 방지를 위해 지연 import

    cache = _appear_cache_path(video_path)
    if cache is not None and cache.exists():
        try:
            return np.load(cache)
        except Exception:
            pass
    thumbs: List[np.ndarray] = []
    for _t, frame in _iter_downsampled_frames(video_path, 1.0 / APPEAR_STEP_SEC, use_gpu_accel, cancel_check):
        if cancel_check and cancel_check():
            return None
        thumbs.append(cv2.resize(frame, (APPEAR_W, APPEAR_H), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return None
    arr = np.stack(thumbs).astype(np.uint8)
    save_appearance_cache(video_path, arr)
    return arr


def ensure_appearance(
    video: VideoFileInfo,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[np.ndarray]:
    fc = video.feature_cache
    if not fc:
        return None
    if "appear" in fc:
        return fc["appear"]
    thumbs = extract_appearance_thumbs(video.path, cancel_check)
    if thumbs is not None:
        fc["appear"] = thumbs
    return thumbs


def _appearance_descriptor(thumb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """(색 분포 히스토그램, 화면 구도 격자, 평균 밝기). 하늘·도로·건물·나무의 색과 배치가 다르면 다른 장면."""
    import cv2  # 지연 import (분석 모듈과 동일 의존성)

    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 3, 3], [0, 180, 0, 256, 0, 256]).ravel()
    hist = np.sqrt(hist / max(1.0, float(hist.sum())))  # Hellinger 거리용
    small = cv2.resize(thumb, (8, 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    layout = small.ravel()
    brightness = float(hsv[..., 2].mean())
    return hist.astype(np.float32), layout.astype(np.float32), brightness


def _appearance_distance(a: Tuple[np.ndarray, np.ndarray, float], b: Tuple[np.ndarray, np.ndarray, float]) -> float:
    """0(같은 장면) ~ 약 1(완전히 다른 장면)."""
    hell = float(np.linalg.norm(a[0] - b[0]) / np.sqrt(2.0))
    lay = float(np.sqrt(np.mean((a[1] - b[1]) ** 2)) * 2.0)
    return 0.6 * hell + 0.4 * min(1.0, lay)


def select_diverse_highlights(
    videos: List[VideoFileInfo],
    config: HighlightConfig,
    count: int,
) -> Optional[List[HighlightClip]]:
    """
    컷끼리 서로 달라 보이도록 고릅니다 (장면 안의 움직임보다 '배경이 겹치지 않는 것'이 중요).
    - 정차·잘라내기·출발/도착 부근은 기존 점수에서 이미 제외/감점, 어두운 장면은 추가 감점
    - 첫 컷은 가장 나은 장면, 이후로는 '이미 고른 컷들과의 최소 차이 × 장면 품질'이 가장 큰 장면을 선택
    장면 모습 썸네일(ensure_appearance)이 없는 영상이 있으면 None (호출 측에서 기존 방식으로 대체).
    """
    if count <= 0:
        return []
    if any(v.feature_cache is not None and "appear" not in v.feature_cache for v in videos):
        return None
    clip_len = max(1.0, float(config.clip_duration_sec))
    scored = compute_highlight_scores(videos, config)
    cands = _window_table(videos, scored, clip_len)
    if not cands:
        return []

    offsets: List[float] = []
    cursor = 0.0
    for v in videos:
        offsets.append(cursor)
        cursor += v.duration_sec

    # 후보 풀: 8초 구간마다 가장 나은 창 하나
    pool: List[Tuple[int, int, float, float, Tuple[np.ndarray, np.ndarray, float]]] = []
    for vi, means, fps, win in cands:
        thumbs = videos[vi].feature_cache["appear"]  # type: ignore[index]
        block = max(1, int(round(POOL_BLOCK_SEC * fps)))
        for b0 in range(0, len(means), block):
            seg = means[b0: b0 + block]
            j = int(np.argmax(seg))
            if not np.isfinite(seg[j]):
                continue
            idx = b0 + j
            mid_t = idx / fps + clip_len / 2.0
            ti = int(min(len(thumbs) - 1, max(0, round(mid_t / APPEAR_STEP_SEC))))
            pool.append((vi, idx, float(seg[j]), offsets[vi] + mid_t, _appearance_descriptor(thumbs[ti])))
    if not pool:
        return []

    # 장면 품질: 점수 순위(0~1)를 쓰되 비중을 낮춰 '다른 모습'이 우선되게, 어두우면 감점
    vals = np.array([p[2] for p in pool])
    ranks = vals.argsort().argsort() / max(1, len(vals) - 1)
    quality = 0.5 + 0.5 * ranks
    for k, p in enumerate(pool):
        if p[4][2] < DARK_BRIGHTNESS:
            quality[k] *= 0.35

    chosen: List[int] = []
    min_gap = min(MIN_CUT_GAP_SEC, cursor / max(1, count) * 0.5)
    alive = np.ones(len(pool), dtype=bool)
    min_dist = np.full(len(pool), np.inf)
    while len(chosen) < count and alive.any():
        if not chosen:
            obj = np.where(alive, quality, -np.inf)
        else:
            usable = alive & (min_dist >= DUPLICATE_DISTANCE)  # 거의 같은 모습의 컷은 개수가 모자라도 넣지 않음
            obj = np.where(usable, np.minimum(min_dist, 1.0) * quality, -np.inf)
        k = int(np.argmax(obj))
        if not np.isfinite(obj[k]):
            break
        chosen.append(k)
        alive[k] = False
        for m in range(len(pool)):
            if not alive[m]:
                continue
            if abs(pool[m][3] - pool[k][3]) < max(clip_len, min_gap):
                alive[m] = False  # 너무 가까운 시간대
                continue
            min_dist[m] = min(min_dist[m], _appearance_distance(pool[m][4], pool[k][4]))

    picked: List[Window] = [(pool[k][0], pool[k][1], pool[k][2]) for k in chosen]
    clips = _windows_to_clips(videos, scored, cands, picked, clip_len)
    return clips


# ── 쇼츠 렌더링 ──

SHORTS_SIZES = {"2160p": (2160, 3840), "1440p": (1440, 2560), "1080p": (1080, 1920), "720p": (720, 1280)}


def resolve_shorts_resolution(resolution: str, clips: List[HighlightClip]) -> str:
    """'auto'면 원본 화질에 맞춤: 4K 원본 → 2160p, 1440p 원본 → 1440p, 그 외 1080p."""
    if resolution != "auto":
        return resolution if resolution in SHORTS_SIZES else "1080p"
    src_h = max((min(c.video.width, c.video.height) for c in clips), default=1080)
    if src_h >= 2000:
        return "2160p"
    if src_h >= 1400:
        return "1440p"
    return "1080p"


def resolve_shorts_fps(fps: float, clips: List[HighlightClip]) -> float:
    """0이면 원본 프레임레이트 유지 (최대 60fps, 29.97/59.94 같은 값도 그대로)."""
    if fps > 0:
        return float(fps)
    src = max((c.video.fps for c in clips if 10.0 <= c.video.fps <= 240.0), default=30.0)
    return float(min(60.0, src))


def shorts_dimensions(layout: ShortsLayout, resolution: str) -> Tuple[int, int]:
    w, h = SHORTS_SIZES.get(resolution, SHORTS_SIZES["1080p"])
    if layout == ShortsLayout.LANDSCAPE:
        return h, w
    return w, h


def _find_overlay_font(need_cjk: bool, bold: bool = True) -> Optional[str]:
    """오버레이 텍스트용 글꼴 경로 (한글이면 한글 글꼴 우선, bold=False면 보통 굵기 우선)."""
    if not bold:
        regular = [
            "C:/Windows/Fonts/malgun.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ] + ([] if need_cjk else [
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ])
        for fp in regular:
            if os.path.exists(fp):
                return fp
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


def _drawtext(
    font: str,
    text_file: Path,
    fontsize: int,
    y: str,
    boxed: bool,
    x: str = "(w-tw)/2",
    alpha: Optional[str] = None,
    soft: bool = False,
) -> str:
    color = "white@0.92" if soft else "white"
    base = (
        f"drawtext=fontfile='{_filter_path(font)}':textfile='{_filter_path(text_file)}':expansion=none:"
        f"fontsize={fontsize}:fontcolor={color}:x={x}:y={y}"
    )
    if alpha:
        base += f":alpha='{alpha}'"
    if soft:
        # 작고 은은한 자막: 얇은 외곽선 + 옅은 그림자 (밝은 하늘 위에서도 읽히는 정도)
        return base + ":borderw=1:bordercolor=black@0.35:shadowcolor=black@0.45:shadowx=1:shadowy=2"
    if boxed:
        pad = max(6, fontsize // 3)
        return base + f":box=1:boxcolor=black@0.45:boxborderw={pad}"
    return base + ":shadowcolor=black@0.7:shadowx=3:shadowy=3"


def _overlay_alpha(t1: float, t2: float, fade: float = 0.3) -> Optional[str]:
    """컷 안에서 [t1, t2] 동안만 서서히 나타났다 사라지는 투명도 식 (디졸브로 겹치는 구간에는 숨김)."""
    if t2 - t1 < 0.6:
        return None
    f = min(fade, (t2 - t1) / 3.0)
    return (
        f"if(lt(t,{t1:.3f}),0,if(lt(t,{t1 + f:.3f}),(t-{t1:.3f})/{f:.3f},"
        f"if(lt(t,{t2 - f:.3f}),1,if(lt(t,{t2:.3f}),({t2:.3f}-t)/{f:.3f},0))))"
    )


# ── 수평(지평선) 자동 보정 ──

ROLL_CACHE_VERSION = "roll_v3"
ROLL_SAMPLE_FRAMES = 18
ROLL_MAX_DEG = 8.0
ROLL_MIN_APPLY_DEG = 0.3


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return float(v[int(np.searchsorted(cw, cw[-1] / 2.0))])


def frame_roll_estimate(gray: np.ndarray) -> Tuple[Optional[float], int]:
    """
    한 프레임의 카메라 기울기(도) 추정. 건물 가장자리·전봇대(수직선), 지평선·보닛·건물 처마(수평선)처럼
    수평/수직에서 ±10° 이내인 직선들이 기준축에서 얼마나 틀어졌는지를 길이 가중 중앙값으로 구합니다.
    (+값 = 화면이 시계 방향으로 기울어짐) 반환: (각도 또는 None, 사용한 직선 수)
    """
    import cv2

    h, w = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=40,
                            minLineLength=max(20, int(w * 0.12)), maxLineGap=6)
    if lines is None:
        return None, 0
    devs: List[float] = []
    weights: List[float] = []
    cx, cy = w / 2.0, h / 2.0
    half_diag = math.hypot(cx, cy)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # OpenCV 버전에 따라 (N,1,4) 또는 (N,4)
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if my > h * 0.88:
            continue  # 화면 맨 아래(보닛·대시보드)는 광각 렌즈에서 휘어 보여 기울기 판단에서 제외
        # 광각(액션캠) 렌즈는 가장자리 직선이 휘어 보이므로 화면 중앙에 가까운 선일수록 더 신뢰
        centrality = max(0.1, 1.0 - (math.hypot(mx - cx, my - cy) / half_diag) ** 2)
        a = math.degrees(math.atan2(dy, dx))
        if a <= -90.0:
            a += 180.0
        elif a > 90.0:
            a -= 180.0
        if abs(a) <= 10.0:
            dev = a
        elif abs(a) >= 80.0:
            dev = a - 90.0 if a > 0 else a + 90.0
        else:
            continue  # 차선처럼 비스듬한 선은 원근 때문에 기울기 판단에 쓰지 않음
        devs.append(dev)
        weights.append(length * centrality)
    # 선 개수보다 '믿을 만한 직선 길이의 합'으로 판단 (고속도로처럼 화면을 가로지르는 지평선 하나도 충분히 신뢰)
    if len(devs) < 3 or sum(weights) < w * 0.5:
        return None, len(devs)
    return _weighted_median(np.array(devs), np.array(weights)), len(devs)


def _roll_cache_path(video_path: Path) -> Optional[Path]:
    try:
        st = video_path.stat()
        key = f"{ROLL_CACHE_VERSION}|{video_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        cache_dir = Path(tempfile.gettempdir()) / "quick_dash_cam_fcache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"roll_{digest}.json"
    except Exception:
        return None


def _grab_gray_frames(
    video: VideoFileInfo,
    times: List[float],
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[Optional[np.ndarray]]:
    """지정한 시각들의 프레임을 640px 폭 흑백으로 가져옵니다 (4개씩 병렬)."""
    sw = 640
    sh = max(2, int(round(sw * max(1, video.height) / max(1, video.width) / 2)) * 2)
    ffmpeg = get_ffmpeg_exe()

    def grab(t: float) -> Optional[np.ndarray]:
        if cancel_check and cancel_check():
            return None
        try:
            res = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(video.path),
                 "-frames:v", "1", "-vf", f"scale={sw}:{sh}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60, creationflags=get_CreationFlags(),
            )
            if len(res.stdout) >= sw * sh:
                return np.frombuffer(res.stdout[: sw * sh], dtype=np.uint8).reshape(sh, sw)
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(grab, times))


VIDEO_ROLL_TRUST = 0.75    # 컷 단위 판단이 안 될 때 영상 전체 추정값을 쓰려면 프레임의 75% 이상이 같은 각도여야 함
CLIP_ROLL_FRAMES = 5
CLIP_ROLL_MAX_SPREAD = 1.0  # 컷 안 프레임별 추정값이 이보다 들쭉날쭉하면 컷 단위 판단을 믿지 않음
_clip_roll_cache: Dict[Tuple[str, int, int], Optional[float]] = {}


def estimate_clip_roll(
    clip: HighlightClip,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[float]:
    """
    컷 구간 안의 프레임 5장으로 그 컷의 기울기를 추정합니다 (카메라가 주행 중 조금 돌아갔거나,
    경사로·커브에서 차체가 기운 컷도 맞추기 위함). 컷 안에서는 각도를 하나로 고정해 화면이 흔들리지 않게 합니다.
    프레임별 값이 일관되지 않거나 직선이 부족하면 None (영상 전체 추정값을 대신 사용).
    """
    key = (str(clip.video.path), int(round(clip.start_sec * 10)), int(round(clip.end_sec * 10)))
    if key in _clip_roll_cache:
        return _clip_roll_cache[key]
    dur = max(0.2, clip.duration)
    times = [clip.start_sec + dur * (0.1 + 0.8 * k / max(1, CLIP_ROLL_FRAMES - 1)) for k in range(CLIP_ROLL_FRAMES)]
    frames = _grab_gray_frames(clip.video, times, cancel_check)
    ests = [e for e, _n in (frame_roll_estimate(f) for f in frames if f is not None) if e is not None]
    result: Optional[float] = None
    if len(ests) >= 3:
        med = float(np.median(ests))
        spread = float(np.median(np.abs(np.array(ests) - med)))
        if spread <= CLIP_ROLL_MAX_SPREAD:
            result = float(np.clip(med, -ROLL_MAX_DEG, ROLL_MAX_DEG))
    if not (cancel_check and cancel_check()):
        _clip_roll_cache[key] = result
    return result


def estimate_roll_angle(
    video: VideoFileInfo,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[float, float]:
    """
    영상 하나의 카메라 기울기(도)와 신뢰도(0~1). 카메라는 차에 고정돼 있으므로 영상 전체에서 프레임 18장을 뽑아
    각 프레임 추정값의 중앙값을 씁니다 (커브·경사로에서의 순간 기울기는 무시됨).
    직선이 부족하거나(야간 등) 프레임마다 값이 들쭉날쭉하면 (0, 낮은 신뢰도)를 돌려줘 보정하지 않게 합니다.
    """
    cache = _roll_cache_path(video.path)
    if cache is not None and cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            return float(d["angle"]), float(d["confidence"])
        except Exception:
            pass
    dur = max(0.5, video.duration_sec)
    times = [dur * (0.05 + 0.9 * k / max(1, ROLL_SAMPLE_FRAMES - 1)) for k in range(ROLL_SAMPLE_FRAMES)]
    frames = _grab_gray_frames(video, times, cancel_check)
    ests = [e for e, _n in (frame_roll_estimate(f) for f in frames if f is not None) if e is not None]
    if len(ests) < 4:
        angle, conf = 0.0, 0.0
    else:
        med = float(np.median(ests))
        conf = float(np.mean([abs(e - med) <= 1.0 for e in ests]))
        angle = float(np.clip(med, -ROLL_MAX_DEG, ROLL_MAX_DEG)) if conf >= 0.4 else 0.0
    if cache is not None and not (cancel_check and cancel_check()):
        try:
            cache.write_text(json.dumps({"angle": angle, "confidence": conf, "frames": len(ests)}), encoding="utf-8")
        except Exception:
            pass
    return angle, conf


# ── 컷별 화면 맞춤(기울기 + 소실점) ──

MAX_REFRAME_ZOOM = 2.0


def vp_target_for(clips: List[HighlightClip]) -> Optional[Tuple[float, float]]:
    """
    소실점을 맞출 목표 위치 (0~1 좌표). 가로는 화면 중앙, 높이는 소실점을 찍은 컷들의 중앙값
    — 컷이 바뀌어도 도로가 같은 곳으로 뻗어 화면이 안정돼 보이게 함.
    """
    ys = [c.vp[1] for c in clips if c.included and c.vp is not None]
    if not ys:
        return None
    return 0.5, float(np.median(ys))


def reframe_geometry(
    w: float,
    h: float,
    roll_deg: float,
    vp: Optional[Tuple[float, float]],
    target: Optional[Tuple[float, float]],
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    화면 맞춤 변환: 출력 = s·R·(x − p) + t
    - R: roll_deg만큼 시계 방향 회전 (화면 좌표는 y가 아래로 증가)
    - p: 원본의 소실점(없으면 화면 중심), t: 출력에서 소실점이 올 위치(없으면 회전만)
    - s: 회전·이동 후에도 가장자리에 빈 공간이 생기지 않는 최소 확대율 (최대 MAX_REFRAME_ZOOM,
      넘으면 이동량을 줄여 맞춤)
    반환: (s, R, p, t) — 모두 픽셀 단위
    """
    th = math.radians(roll_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    Rinv = R.T
    c = np.array([w / 2.0, h / 2.0])
    if vp is not None and target is not None:
        p = np.array([vp[0] * w, vp[1] * h])
        t_full = np.array([target[0] * w, target[1] * h])
    else:
        p, t_full = c.copy(), c.copy()
    p_rot = c + R @ (p - c)  # 이동 없이 회전만 했을 때 소실점 위치
    corners = [np.array([0.0, 0.0]), np.array([w, 0.0]), np.array([0.0, h]), np.array([w, h])]

    def needed_zoom(t: np.ndarray) -> float:
        s_req = 1.0
        for q in corners:
            v = Rinv @ (q - t)  # 출력 모서리 q에 대응하는 원본 위치 = p + v/s
            for k, lim in ((0, w), (1, h)):
                if v[k] > 1e-9:
                    room = lim - p[k]
                    s_req = max(s_req, v[k] / room if room > 1e-6 else float("inf"))
                elif v[k] < -1e-9:
                    room = p[k]
                    s_req = max(s_req, -v[k] / room if room > 1e-6 else float("inf"))
        return s_req

    s = needed_zoom(t_full)
    t = t_full
    if s > MAX_REFRAME_ZOOM:
        # 목표까지 다 옮기면 너무 크게 확대해야 함 → 확대 한도 안에서 가능한 만큼만 이동
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if needed_zoom(p_rot + mid * (t_full - p_rot)) <= MAX_REFRAME_ZOOM:
                lo = mid
            else:
                hi = mid
        t = p_rot + lo * (t_full - p_rot)
        s = min(MAX_REFRAME_ZOOM, needed_zoom(t))
    return s, R, p, t


def reframe_affine(w: int, h: int, roll_deg: float, vp, target) -> np.ndarray:
    """미리보기용 2x3 아핀 행렬 (원본 → 출력). cv2.warpAffine에 그대로 사용."""
    s, R, p, t = reframe_geometry(w, h, roll_deg, vp, target)
    A = s * R
    return np.hstack([A, (t - A @ p).reshape(2, 1)])


def reframe_inverse_point(w: int, h: int, roll_deg: float, vp, target, qx: float, qy: float) -> Tuple[float, float]:
    """출력(미리보기) 화면의 점 → 원본 화면의 점 (픽셀). 미리보기에서 찍은 소실점을 원본 좌표로 바꿀 때 사용."""
    s, R, p, t = reframe_geometry(w, h, roll_deg, vp, target)
    x = p + R.T @ (np.array([qx, qy]) - t) / s
    return float(x[0]), float(x[1])


def _reframe_chain(
    v: VideoFileInfo,
    roll_deg: float,
    vp: Optional[Tuple[float, float]],
    target: Optional[Tuple[float, float]],
    out_w: int,
    out_h: int,
) -> str:
    """컷별 화면 맞춤을 FFmpeg 필터로: 회전(rotate) → 소실점이 목표에 오도록 잘라내기(crop) → 원래 크기로(scale)."""
    if v.width <= 0 or v.height <= 0:
        return ""
    use_vp = vp is not None and target is not None
    if abs(roll_deg) < 0.05 and not use_vp:
        return ""
    pw = int(min(v.width, max(640, out_w, out_h))) // 2 * 2  # 4K 원본은 출력 크기에 맞춰 먼저 줄여 연산량 절감
    ph = max(2, int(round(pw * v.height / v.width / 2)) * 2)
    s, R, p, t = reframe_geometry(pw, ph, roll_deg, vp, target)
    c = np.array([pw / 2.0, ph / 2.0])
    p_rot = c + R @ (p - c)
    cw = max(2, int(round(pw / s / 2)) * 2)
    ch = max(2, int(round(ph / s / 2)) * 2)
    tl = p_rot - t / s
    x = int(round(min(max(0.0, tl[0]), pw - cw)))
    y = int(round(min(max(0.0, tl[1]), ph - ch)))
    chain = f"scale={pw}:{ph}:flags=lanczos"
    if abs(roll_deg) >= 0.05:
        # rotate 필터는 양수=시계 방향
        chain += f",rotate={math.radians(roll_deg):.6f}:ow=iw:oh=ih:c=black"
    chain += f",crop={cw}:{ch}:{x}:{y},scale={pw}:{ph}:flags=lanczos"
    return chain


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
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    overlay_window: Optional[Tuple[float, float]] = None,
    reframe: str = "",
) -> str:
    v = clip.video
    level = reframe  # 컷별 화면 맞춤(기울기·소실점) 필터
    prefix = f"[0:v]{level}[lv];" if level else ""
    src = "[lv]" if level else "[0:v]"
    if cfg.layout == ShortsLayout.VERTICAL_BLUR:
        bw, bh = max(2, (out_w // 4) // 2 * 2), max(2, (out_h // 4) // 2 * 2)
        graph = (
            f"{prefix}{src}split=2[bg][fg];"
            f"[bg]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"boxblur=8:2,scale={out_w}:{out_h},eq=brightness=-0.10[bgb];"
            f"[fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
        )
    elif cfg.layout == ShortsLayout.VERTICAL_CROP:
        graph = f"{prefix}{src}scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={out_w}:{out_h}"
    else:
        graph = (
            f"{prefix}{src}scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,"
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

    # 장면 자막·촬영 시각은 컷 중간에만 서서히 나타났다 사라지게 (디졸브로 겹치는 동안 앞뒤 글자가 섞이지 않게)
    alpha = _overlay_alpha(*overlay_window) if overlay_window else None

    has_caption = caption_file is not None and bool(font_caption) and bool(clip.caption.strip())
    if has_caption:
        # 작고 깔끔한 자막: 영상 왼쪽 아래 구석에 보통 굵기 글씨로 은은하게
        fs = int(max(14, out_w * (0.036 if vertical else 0.020)))
        margin_x = int(out_w * 0.05)
        video_bottom = (out_h + fg_h) / 2.0 if cfg.layout == ShortsLayout.VERTICAL_BLUR else out_h * (0.86 if vertical else 0.88)
        y = f"{video_bottom - fs - out_h * (0.018 if vertical else 0.03):.0f}"
        graph += "," + _drawtext(font_caption, caption_file, fs, y, boxed=False,  # type: ignore[arg-type]
                                 x=str(margin_x), alpha=alpha, soft=True)

    if ts_file is not None and font_ts:
        fs = int(max(14, out_w * (0.042 if vertical else 0.022)))
        if use_band:
            y = f"{out_h - band + band * 0.25:.0f}"
        else:
            y = f"{out_h * 0.90:.0f}"
        graph += "," + _drawtext(font_ts, ts_file, fs, y, boxed=not use_band, alpha=alpha)

    # 컷 사이 전환은 이어 붙이는 단계(xfade)에서 처리하고, 여기서는 쇼츠 맨 처음/맨 끝 페이드만 적용
    if fade_in > 0 and out_dur > fade_in * 2:
        graph += f",fade=t=in:st=0:d={fade_in:.3f}"
    if fade_out > 0 and out_dur > fade_out * 2:
        graph += f",fade=t=out:st={out_dur - fade_out:.3f}:d={fade_out:.3f}"
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
    dur: Optional[float] = None,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    overlay_window: Optional[Tuple[float, float]] = None,
    reframe: str = "",
) -> Tuple[bool, str]:
    dur = clip.duration if dur is None else dur
    graph = _build_clip_filtergraph(
        clip, cfg, out_w, out_h, dur, title_file, cfg.title_text.strip(), ts_file, font_title, font_ts,
        caption_file, font_caption, fade_in, fade_out, overlay_window, reframe,
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
        if fade_in > 0 and dur > fade_in * 2:
            a_chain += f",afade=t=in:st=0:d={fade_in:.3f}"
        if fade_out > 0 and dur > fade_out * 2:
            a_chain += f",afade=t=out:st={dur - fade_out:.3f}:d={fade_out:.3f}"
        # 이어 붙이기·크로스페이드에서 길이가 정확히 맞도록 무음으로 채운 뒤 자름
        a_chain += f",apad,atrim=duration={dur:.3f}"
        graph += f";{a_in}{a_chain}[a]"

    cmd += ["-filter_complex", graph, "-map", "[v]"]
    if with_audio:
        cmd += ["-map", "[a]"]
    cmd += ["-c:v", encoder_name, *encoder_args, "-r", f"{cfg.fps:.3f}"]  # 중간 파일에도 프레임레이트 명시
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


def _join_with_transitions(
    ffmpeg: str,
    seg_files: List[Path],
    seg_durs: List[float],
    trans_sec: float,
    transition: ShortsTransition,
    with_audio: bool,
    fps: float,
    encoder_name: str,
    encoder_args: List[str],
    out_path: Path,
    cancel_check: Optional[Callable[[], bool]],
) -> Tuple[bool, str]:
    """
    컷들을 xfade(영상) + acrossfade(소리)로 겹쳐 잇습니다.
    i번째 전환 시작 시점 = (지금까지 이어 붙인 길이) - 전환 길이.
    """
    xname = "fadeblack" if transition == ShortsTransition.FADE_BLACK else "fade"
    cmd: List[str] = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for sf in seg_files:
        cmd += ["-i", str(sf)]
    parts: List[str] = []
    for i in range(len(seg_files)):
        # xfade는 입력의 프레임레이트가 확정돼 있어야 함. GPU 인코더 출력이나 일부 FFmpeg 버전(settb 사용 시)에서
        # 프레임레이트가 '알 수 없음(1/0)'이 되는 경우가 있어 fps 필터로 명시적으로 고정
        parts.append(f"[{i}:v]setpts=PTS-STARTPTS,fps={fps:.3f}[v{i}]")
        if with_audio:
            parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
    prev_v, prev_a = "v0", "a0"
    joined_len = seg_durs[0]
    for i in range(1, len(seg_files)):
        offset = max(0.0, joined_len - trans_sec)
        parts.append(
            f"[{prev_v}][v{i}]xfade=transition={xname}:duration={trans_sec:.3f}:offset={offset:.3f}[x{i}]"
        )
        prev_v = f"x{i}"
        if with_audio:
            parts.append(f"[{prev_a}][a{i}]acrossfade=d={trans_sec:.3f}:c1=tri:c2=tri[y{i}]")
            prev_a = f"y{i}"
        joined_len = offset + seg_durs[i]
    parts.append(f"[{prev_v}]format=yuv420p[vout]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]"]
    if with_audio:
        cmd += ["-map", f"[{prev_a}]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", encoder_name, *encoder_args, "-movflags", "+faststart", str(out_path)]
    ok, err = _run_ffmpeg(cmd, cancel_check)
    return ok and out_path.exists() and out_path.stat().st_size > 0, err


def render_highlight_shorts(
    clips: List[HighlightClip],
    cfg: ShortsExportConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    warnings: Optional[List[str]] = None,
    notes: Optional[List[str]] = None,
) -> Path:
    """
    선정된 하이라이트 장면들을 쇼츠 영상(MP4) 하나로 만듭니다.
    warnings 리스트를 넘기면, 완성은 됐지만 사용자에게 알릴 내용(예: 전환 효과 실패로 바로 컷 적용)을 담아 줍니다.
    """
    use_clips = [c for c in clips if c.included and c.duration >= 0.5]
    if not use_clips:
        raise ValueError("쇼츠에 넣을 하이라이트 장면이 없습니다. 하이라이트를 먼저 찾아 주세요.")

    needs_bgm = cfg.audio_mode in (ShortsAudioMode.BGM, ShortsAudioMode.BGM_MIX)
    if needs_bgm and (cfg.bgm_path is None or not Path(cfg.bgm_path).exists()):
        raise ValueError("배경음악 파일을 선택해 주세요.")

    ffmpeg = get_ffmpeg_exe()
    # 해상도·프레임레이트 확정 ('auto'/0 → 원본에 맞춤). 이후 단계는 확정된 값을 씀
    cfg = dataclasses.replace(
        cfg,
        resolution=resolve_shorts_resolution(cfg.resolution, use_clips),
        fps=resolve_shorts_fps(cfg.fps, use_clips),
    )
    out_w, out_h = shorts_dimensions(cfg.layout, cfg.resolution)
    enc_name, enc_args = resolve_video_encoder(VideoCodec.H264, cfg.quality_preset, prefer_hw=cfg.use_hw_accel)
    sw_name, sw_args = get_software_encoder_args(VideoCodec.H264, cfg.quality_preset)

    # BGM 단독 모드는 장면 인코딩 시 오디오를 빼고 마지막에 배경음악만 입힘
    clip_audio = cfg.audio_mode in (ShortsAudioMode.ORIGINAL, ShortsAudioMode.BGM_MIX)

    title = cfg.title_text.strip()
    font_title = _find_overlay_font(need_cjk=any(ord(ch) >= 0x2E80 for ch in title)) if title else None
    font_ts = _find_overlay_font(need_cjk=False) if cfg.show_timestamp else None
    font_caption = (
        _find_overlay_font(need_cjk=True, bold=False)
        if cfg.show_captions and any(c.caption.strip() for c in use_clips) else None
    )

    output_path = Path(cfg.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(use_clips)
    use_xfade = cfg.transition != ShortsTransition.CUT and n >= 2 and cfg.transition_sec > 0
    trans = min(cfg.transition_sec, min(c.duration for c in use_clips) / 3.0) if use_xfade else 0.0
    # 겹치는 만큼 쇼츠가 짧아지지 않도록 마지막을 제외한 컷은 전환 길이만큼 뒤로 늘려서 렌더링
    seg_durs: List[float] = []
    for idx, c in enumerate(use_clips):
        extend = trans if idx < n - 1 else 0.0
        seg_durs.append(max(trans * 2.5, min(c.duration + extend, c.video.duration_sec - c.start_sec)))
    total_dur = sum(seg_durs) - trans * (n - 1)
    done_dur = 0.0
    encode_share = 0.8 if use_xfade else 0.9

    # 컷별 화면 맞춤: 수동 기울기(사용자가 조정한 값)가 우선, 없으면 (켜져 있을 때) 자동 수평 보정,
    # 소실점을 찍은 컷은 소실점이 모든 컷 공통 목표 위치에 오도록 이동·확대
    roll_by_clip: List[float] = [0.0] * n  # 회전 보정 각도 (+=시계 방향)
    vp_target = vp_target_for(use_clips)
    video_angle: Dict[int, Tuple[float, float]] = {}
    for idx, c in enumerate(use_clips):
        mm, ss = divmod(int(c.start_sec), 60)
        where = f"컷 {idx + 1} ({c.video.filename} {mm}:{ss:02d})"
        vp_txt = " · 소실점 맞춤" if (c.vp is not None and vp_target is not None) else ""
        if c.roll_adjust is not None:
            roll_by_clip[idx] = c.roll_adjust
            if notes is not None and (abs(c.roll_adjust) >= 0.05 or vp_txt):
                notes.append(f"화면 맞춤: {where} 기울기 {round(-c.roll_adjust, 1) + 0.0:+.1f}° (수동){vp_txt}")  # 화면 표시와 같은 부호(슬라이더 방향 기준)
            continue
        if cfg.auto_level:
            if cancel_check and cancel_check():
                raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")
            if progress_callback:
                progress_callback(0.0, f"수평(지평선) 기울기 확인 중 [{idx + 1}/{n}]: {c.video.filename} {c.start_sec:.0f}초~")
            cut_angle = estimate_clip_roll(c, cancel_check)
            if cut_angle is not None:
                angle, how = cut_angle, "자동·컷 기준"
            else:
                if id(c.video) not in video_angle:
                    video_angle[id(c.video)] = estimate_roll_angle(c.video, cancel_check)
                v_angle, v_conf = video_angle[id(c.video)]
                # 영상 전체 프레임이 거의 같은 각도로 일치할 때만 대신 사용 (카메라가 중간에 돌아간 영상이면 잘못 돌릴 수 있음)
                angle, how = (v_angle, "자동·영상 전체 기준") if v_conf >= VIDEO_ROLL_TRUST else (0.0, "판단 어려움")
            if abs(angle) >= ROLL_MIN_APPLY_DEG:
                roll_by_clip[idx] = -angle  # 감지된 기울기의 반대 방향으로 회전
            if notes is not None:
                if how == "판단 어려움":
                    notes.append(f"수평: {where} 직선이 부족해 그대로 둠{vp_txt}")
                elif abs(angle) >= ROLL_MIN_APPLY_DEG:
                    notes.append(f"수평: {where} {angle:+.1f}° 보정 ({how}){vp_txt}")
                elif vp_txt:
                    notes.append(f"화면 맞춤: {where}{vp_txt}")
        elif notes is not None and vp_txt:
            notes.append(f"화면 맞춤: {where}{vp_txt}")
    if cfg.auto_level and notes is not None and not any(n_.startswith(("수평:", "화면 맞춤:")) for n_ in notes):
        notes.append("수평: 모든 컷이 이미 수평이라 보정하지 않음")

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
                    min(encode_share, done_dur / max(0.1, sum(seg_durs)) * encode_share),
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
            t_in = trans if idx > 0 else 0.0
            t_out = trans if idx < n - 1 else 0.0
            timing = dict(
                dur=seg_durs[idx],
                fade_in=cfg.fade_sec if idx == 0 else 0.0,
                fade_out=cfg.fade_sec if idx == n - 1 else 0.0,
                overlay_window=(t_in + 0.15, seg_durs[idx] - t_out - 0.1),
                reframe=_reframe_chain(clip.video, roll_by_clip[idx], clip.vp, vp_target, out_w, out_h),
            )
            ok, err = _encode_highlight_clip(*args, enc_name, enc_args, *extra, **timing)
            if not ok and enc_name != sw_name and not (cancel_check and cancel_check()):
                # 하드웨어 인코더 실패 시 소프트웨어 인코더로 자동 폴백
                enc_name, enc_args = sw_name, sw_args
                ok, err = _encode_highlight_clip(*args, enc_name, enc_args, *extra, **timing)
            if not ok:
                if cancel_check and cancel_check():
                    raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")
                detail = f"\n[오류 상세]: {err}" if err else ""
                raise RuntimeError(
                    f"쇼츠 장면 인코딩에 실패했습니다: {clip.video.filename} "
                    f"({clip.start_sec:.1f}s ~ {clip.end_sec:.1f}s){detail}"
                )
            seg_files.append(seg_path)
            done_dur += seg_durs[idx]

        joined = tmp / "joined.mp4" if needs_bgm else output_path
        if use_xfade:
            if progress_callback:
                name = "크로스 디졸브" if cfg.transition == ShortsTransition.DISSOLVE else "페이드"
                progress_callback(encode_share, f"장면 사이를 {name}로 자연스럽게 잇는 중...")
            jargs = (ffmpeg, seg_files, seg_durs, trans, cfg.transition, clip_audio, cfg.fps)
            ok, err = _join_with_transitions(*jargs, enc_name, enc_args, joined, cancel_check)
            if not ok and enc_name != sw_name and not (cancel_check and cancel_check()):
                ok, err = _join_with_transitions(*jargs, sw_name, sw_args, joined, cancel_check)
            if not ok and not (cancel_check and cancel_check()):
                # 전환 효과 단계가 실패해도 쇼츠는 완성되도록 바로 컷으로 이어 붙임
                first_line = next((ln for ln in err.splitlines() if ln.strip()), "알 수 없는 오류")
                if warnings is not None:
                    warnings.append(f"장면 전환 효과를 적용하지 못해 '바로 컷'으로 이었습니다.\n(원인: {first_line[:200]})")
                use_xfade = False
                total_dur = sum(seg_durs)
        if not use_xfade:
            if progress_callback:
                progress_callback(0.92, "하이라이트 장면들을 하나의 쇼츠 영상으로 잇는 중...")
            concat_list = tmp / "concat_list.txt"
            with open(concat_list, "w", encoding="utf-8") as fh:
                for sf in seg_files:
                    fh.write(f"file '{str(sf.resolve()).replace(chr(92), '/')}'\n")
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

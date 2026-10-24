from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analyzer import analyze_video, reclassify_from_cache
from src.ffmpeg_utils import get_CreationFlags, get_ffmpeg_exe, probe_video_file
from src.models import (
    AnalysisConfig,
    AudioMode,
    ExportConfig,
    SegmentType,
    SkipMode,
    SortMode,
    VideoCodec,
    VideoFileInfo,
    VideoQualityPreset,
    apply_global_timeline_override,
    map_global_time_to_clip,
    merge_preserving_manual_cuts,
)
from src.processor import render_merged_video
from src.sorter import sort_video_files


def _create_synthetic_driving_video(out_path: Path, duration_sec: int = 32, fps: int = 20) -> None:
    """
    테스트용 주행 영상(오디오 포함)을 생성합니다.
    - 0~15초: 영상 시작부터 신호대기 정차 중이나 전방 교차로에 다른 차량들이 계속 지나가는 상황 -> STOP
    - 15~22초: 교차로 회전 및 시내 일반 주행 (전체 배경 수평 패닝 + 전방 건물 급변) -> NORMAL
    - 22~32초: 고속도로 직진 (전방 소실점과 수평 구도는 고정된 채 노면 차선 및 갓길 가로수가 일정하게 흐름) -> STRAIGHT
    """
    w, h = 640, 360
    raw_video = out_path.with_suffix(".raw_v.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_video), fourcc, float(fps), (w, h))

    rng = np.random.default_rng(42)
    total_frames = duration_sec * fps

    for f_idx in range(total_frames):
        t = f_idx / fps
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # 기본 배경: 상단 하늘, 하단 도로, 최하단 대시보드
        img[: int(h * 0.42), :] = (210, 160, 110)   # 하늘
        img[int(h * 0.42) : int(h * 0.82), :] = (70, 70, 70)  # 아스팔트 도로
        img[int(h * 0.82) :, :] = (25, 25, 25)      # 대시보드

        if t < 15.0:
            # 0~15초: 신호대기 정차 (내 차 주변 건물·차선·신호등은 고정되어 있고, 교차로에 다른 차량들이 계속 지나감)
            for b in range(5):
                bx = 40 + b * 120
                cv2.rectangle(img, (bx, int(h * 0.22)), (bx + 70, int(h * 0.42)), (95, 105, 125), -1)
                cv2.rectangle(img, (bx + 10, int(h * 0.25)), (bx + 60, int(h * 0.40)), (130, 140, 155), 2)
            # 정지선 및 차선 (고정된 도로 텍스처)
            cv2.line(img, (80, int(h * 0.68)), (560, int(h * 0.68)), (220, 220, 220), 4)
            cv2.line(img, (120, int(h * 0.80)), (250, int(h * 0.50)), (200, 200, 200), 3)
            cv2.line(img, (520, int(h * 0.80)), (390, int(h * 0.50)), (200, 200, 200), 3)
            cv2.circle(img, (320, int(h * 0.25)), 10, (0, 0, 255), -1)  # 빨간 신호등

            # 전방 교차로를 가로질러 지나가는 다른 움직이는 차량들 시뮬레이션!
            car1_x = int((t * 180) % (w + 120)) - 60
            cv2.rectangle(
                img,
                (car1_x, int(h * 0.46)),
                (car1_x + 85, int(h * 0.58)),
                (30, 180, 240),
                -1,
            )
            car2_x = w - int((t * 140 + 200) % (w + 120))
            cv2.rectangle(
                img,
                (car2_x, int(h * 0.50)),
                (car2_x + 75, int(h * 0.61)),
                (220, 60, 60),
                -1,
            )
            noise = rng.integers(-1, 2, size=img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        elif 15.0 <= t < 22.0:
            # 15~22초: 교차로 회전 및 시내 일반 주행 (수평 패닝 + 전방 풍경 급변)
            shift_x = int((t * 110) % w)
            for b in range(6):
                bx = (b * 110 + shift_x) % (w - 60)
                color = (int((b * 53 + f_idx * 7) % 255), int((b * 97 + f_idx * 3) % 255), 180)
                cv2.rectangle(img, (bx, int(h * 0.20)), (bx + 55, int(h * 0.55)), color, -1)
            # 도로 전체 텍스처 회전 이동
            for r_i in range(6):
                rx = (r_i * 100 + shift_x * 2) % (w - 40)
                cv2.rectangle(img, (rx, int(h * 0.55)), (rx + 45, int(h * 0.78)), (150, 150, 190), -1)

        else:
            # 22~32초: 고속도로 직진 (지평선 원경 산봉우리/소실점은 고정, 하단 도로 차선만 일정 속도로 이동)
            cv2.rectangle(img, (0, int(h * 0.36)), (w, int(h * 0.42)), (80, 120, 80), -1)
            for m_x in (160, 320, 480):
                pts = np.array([[m_x - 60, int(h * 0.38)], [m_x, int(h * 0.24)], [m_x + 60, int(h * 0.38)]], np.int32)
                cv2.fillPoly(img, [pts], (70, 105, 75))
            phase = int((t * 45) % 40)
            for d in range(4):
                y_pos = int(h * 0.48) + ((d * 40 + phase) % int(h * 0.30))
                offset = int((y_pos - h * 0.40) * 0.85)
                cv2.rectangle(
                    img,
                    (max(10, w // 2 - offset - 15), y_pos),
                    (max(25, w // 2 - offset + 5), min(h - 1, y_pos + 10)),
                    (180, 180, 180),
                    -1,
                )
                cv2.rectangle(
                    img,
                    (min(w - 25, w // 2 + offset - 5), y_pos),
                    (min(w - 10, w // 2 + offset + 15), min(h - 1, y_pos + 10)),
                    (180, 180, 180),
                    -1,
                )

        writer.write(img)

    writer.release()

    # FFmpeg으로 440Hz 오디오 트랙을 합성하여 실제 액션캠 MP4 형태로 생성
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(raw_video),
        "-f", "lavfi",
        "-t", str(duration_sec),
        "-i", "sine=frequency=440:sample_rate=48000",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ac", "2",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, creationflags=get_CreationFlags())
    raw_video.unlink(missing_ok=True)


def test_sorter() -> None:
    # GoPro 챕터 분할 파일(GX010042, GX020042) 및 일반 시각 정렬 검증
    v1 = VideoFileInfo(
        path=Path("GX020042.MP4"),
        filename="GX020042.MP4",
        recorded_at=datetime(2026, 10, 24, 9, 0, 5),
        gopro_clip_key=(42, 2),
    )
    v2 = VideoFileInfo(
        path=Path("GX010042.MP4"),
        filename="GX010042.MP4",
        recorded_at=datetime(2026, 10, 24, 9, 0, 10),  # 복사 과정에서 수정일이 약간 늦어진 경우 시뮬레이션
        gopro_clip_key=(42, 1),
    )
    v3 = VideoFileInfo(
        path=Path("GX010043.MP4"),
        filename="GX010043.MP4",
        recorded_at=datetime(2026, 10, 24, 9, 30, 0),
        gopro_clip_key=(43, 1),
    )
    sorted_list = sort_video_files([v3, v1, v2], SortMode.CHRONOLOGICAL)
    names = [v.filename for v in sorted_list]
    assert names == ["GX010042.MP4", "GX020042.MP4", "GX010043.MP4"], f"Unexpected sort order: {names}"
    print("[PASS] GoPro chapter & chronological sorting verified:", names)


def test_analysis_and_export() -> None:
    with tempfile.TemporaryDirectory(prefix="qdc_test_") as tmp:
        tmp_dir = Path(tmp)
        clip1 = tmp_dir / "VID_20261024_090000.mp4"
        clip2 = tmp_dir / "VID_20261024_090500.mp4"

        _create_synthetic_driving_video(clip1, duration_sec=32, fps=20)
        # 두 번째 클립으로 복사
        clip2.write_bytes(clip1.read_bytes())

        info1 = probe_video_file(clip1)
        info2 = probe_video_file(clip2)
        assert info1.has_audio, "Expected audio stream in synthetic video"
        assert abs(info1.duration_sec - 32.0) < 1.0, f"Unexpected duration: {info1.duration_sec}"

        cfg = AnalysisConfig(
            detect_stops=True,
            detect_highway_straight=True,
            sensitivity=0.5,
            min_skip_duration_sec=5.0,
            transition_padding_sec=0.5,
        )
        segs1 = analyze_video(info1, cfg)
        info1.segments = segs1
        info1.analyzed = True

        info2.segments = analyze_video(info2, cfg)
        info2.analyzed = True

        types_found = {s.segment_type for s in segs1}
        print("Detected segments:", [(round(s.start_sec, 1), round(s.end_sec, 1), s.segment_type.value) for s in segs1])
        assert segs1[0].segment_type == SegmentType.STOP and segs1[0].start_sec == 0.0, (
            f"Expected 0~15s traffic light wait with moving cross-traffic to start at 0.0s as STOP, got {segs1[0]}"
        )
        assert SegmentType.NORMAL in types_found, "Expected NORMAL segment"
        assert SegmentType.STOP in types_found, "Expected STOP segment to be detected"
        assert SegmentType.STRAIGHT in types_found, "Expected STRAIGHT segment to be detected"
        assert reclassify_from_cache(info1, cfg) is not None, "Expected feature cache for instant sensitivity tuning"
        print("[PASS] Stop (with cross-traffic from 0:00) & Highway Straight detection verified!")

        # 1) 배속 재생(TIMELAPSE 8x) 병합 검증
        out_timelapse = tmp_dir / "merged_timelapse.mp4"
        exp_tl = ExportConfig(
            output_path=out_timelapse,
            skip_mode=SkipMode.TIMELAPSE,
            speed_multiplier=8.0,
            audio_mode=AudioMode.KEEP_NORMAL_MUTE_SKIP,
        )
        render_merged_video([info1, info2], exp_tl)
        tl_probe = probe_video_file(out_timelapse)
        print(f"Original 2 clips = 64.0s -> Timelapse output = {tl_probe.duration_sec:.2f}s (has_audio={tl_probe.has_audio})")
        assert 15.0 <= tl_probe.duration_sec <= 36.0, f"Unexpected timelapse duration: {tl_probe.duration_sec}"
        assert tl_probe.has_audio, "Expected audio track in merged timelapse video"
        print("[PASS] Timelapse merge & audio verified!")

        # 2) 완전 잘라내기(JUMPCUT) 병합 검증
        out_jumpcut = tmp_dir / "merged_jumpcut.mp4"
        exp_jc = ExportConfig(
            output_path=out_jumpcut,
            skip_mode=SkipMode.JUMPCUT,
            audio_mode=AudioMode.KEEP_NORMAL_MUTE_SKIP,
        )
        render_merged_video([info1, info2], exp_jc)
        jc_probe = probe_video_file(out_jumpcut)
        print(f"Original 2 clips = 64.0s -> Jumpcut output = {jc_probe.duration_sec:.2f}s (has_audio={jc_probe.has_audio})")
        assert jc_probe.duration_sec < tl_probe.duration_sec, "Jumpcut should be shorter than timelapse"
        assert jc_probe.has_audio, "Expected audio track in merged jumpcut video"
        print("[PASS] Jumpcut merge & audio verified!")

        # 3) 전체 타임라인 탐색(map_global_time_to_clip) 및 특정 구간 수동 잘라내기(MANUAL_CUT) 검증
        # 클립 1의 일반 주행 구간 포함(15초~35초, 클립 경계 32초 통과)을 수동 잘라내기
        modified_clips = apply_global_timeline_override(
            [info1, info2], 15.0, 35.0, SegmentType.MANUAL_CUT
        )
        assert modified_clips == 2, f"Expected 2 clips modified across boundary, got {modified_clips}"
        mapped = map_global_time_to_clip([info1, info2], 33.5)
        assert mapped is not None
        c_idx, _v, local_t, seg_t = mapped
        assert c_idx == 1 and abs(local_t - 1.5) < 0.2 and seg_t == SegmentType.MANUAL_CUT

        out_manual = tmp_dir / "merged_with_manual_cut.mp4"
        exp_mc = ExportConfig(
            output_path=out_manual,
            skip_mode=SkipMode.TIMELAPSE,
            speed_multiplier=8.0,
            audio_mode=AudioMode.KEEP_AND_SPEED_UP,
            quality_preset=VideoQualityPreset.LOSSLESS_LIKE,
            video_codec=VideoCodec.HEVC,
            fps_mode="30",
            overlay_speed_badge=True,
        )
        render_merged_video([info1, info2], exp_mc)
        mc_probe = probe_video_file(out_manual)
        print(
            f"HEVC Near-Lossless + Badge + SpeedUpAudio with 10s manual cut output = {mc_probe.duration_sec:.2f}s (fps={mc_probe.fps})"
        )
        assert mc_probe.duration_sec < tl_probe.duration_sec - 4.0, (
            f"Expected manual cut to shorten output further: {mc_probe.duration_sec} vs {tl_probe.duration_sec}"
        )
        assert mc_probe.has_audio, "Expected audio track in HEVC merged video"
        print("[PASS] Global timeline scrubbing, manual cut, HEVC & near-lossless export verified!")


if __name__ == "__main__":
    test_sorter()
    test_analysis_and_export()
    print("ALL TESTS PASSED SUCCESSFULLY!")

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
    HighlightClip,
    AudioMode,
    ExportConfig,
    HighlightConfig,
    SegmentType,
    ShortsAudioMode,
    ShortsExportConfig,
    ShortsLayout,
    ShortsTransition,
    SkipMode,
    SortMode,
    VideoCodec,
    VideoFileInfo,
    VideoQualityPreset,
    apply_global_timeline_override,
    map_global_time_to_clip,
    merge_preserving_manual_cuts,
)
from src.highlights import render_highlight_shorts, select_highlights
from src.processor import render_merged_video
from src.sorter import sort_video_files


def _create_synthetic_driving_video(out_path: Path, duration_sec: int = 32, fps: int = 20, palette: int = 0) -> None:
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
        img[: int(h * 0.42), :] = (210, 160, 110) if palette == 0 else (90, 120, 230)   # 하늘 (palette 1: 노을)
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
            cv2.rectangle(img, (0, int(h * 0.36)), (w, int(h * 0.42)), (80, 120, 80) if palette == 0 else (60, 70, 120), -1)
            for m_x in (160, 320, 480):
                pts = np.array([[m_x - 60, int(h * 0.38)], [m_x, int(h * 0.24)], [m_x + 60, int(h * 0.38)]], np.int32)
                cv2.fillPoly(img, [pts], (70, 105, 75) if palette == 0 else (50, 50, 90))
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


def test_highlight_shorts() -> None:
    with tempfile.TemporaryDirectory(prefix="qdc_shorts_") as tmp:
        tmp_dir = Path(tmp)
        clip1 = tmp_dir / "VID_20261024_090000.mp4"
        clip2 = tmp_dir / "VID_20261024_091000.mp4"
        _create_synthetic_driving_video(clip1, duration_sec=32, fps=20)
        _create_synthetic_driving_video(clip2, duration_sec=32, fps=20, palette=1)  # 노을 색감의 다른 주행

        info1 = probe_video_file(clip1)
        info2 = probe_video_file(clip2)
        cfg = AnalysisConfig(sensitivity=0.5, min_skip_duration_sec=5.0, transition_padding_sec=0.5)
        for info in (info1, info2):
            info.segments = analyze_video(info, cfg)
            info.analyzed = True

        # 1) 하이라이트 선정: 정차(STOP) 구간은 절대 포함되지 않고, 회전·시내 주행 구간이 우선 선택되어야 함
        hcfg = HighlightConfig(target_duration_sec=8.0, clip_duration_sec=2.0)
        clips = select_highlights([info1, info2], hcfg)
        print("Highlights:", [(c.video.filename, round(c.start_sec, 1), round(c.end_sec, 1), c.reason) for c in clips])
        assert clips, "Expected at least one highlight clip"
        total = sum(c.duration for c in clips)
        assert 4.0 <= total <= 9.0, f"Unexpected highlight total duration: {total}"
        # 좋은 컷 = 컷 안에서 차분한 장면: 회전·패닝이 계속되는 시내 구간(15~22초)은 쓰지 않고 순항 구간을 고름
        for c in clips:
            turning = max(0.0, min(c.end_sec, 22.0) - max(c.start_sec, 15.5))
            assert turning <= 0.3, f"Turning/panning cut selected: {c.start_sec}-{c.end_sec}"
        assert {id(c.video) for c in clips} == {id(info1), id(info2)}, "Distinct-looking cruise cuts from both drives"
        for c in clips:
            for seg in c.video.segments:
                if seg.segment_type == SegmentType.STOP:
                    ov = min(c.end_sec, seg.end_sec) - max(c.start_sec, seg.start_sec)
                    assert ov <= 0.3, f"Highlight overlaps traffic-light stop: {c} vs {seg}"
        ordered = [(0 if c.video is info1 else 1, c.start_sec) for c in clips]
        assert ordered == sorted(ordered), "Highlights must be in chronological order"
        # 같은 주행 안의 거의 같은 고속도로 장면은 한 번만 (컷끼리 배경이 겹치지 않게)
        for info in (info1, info2):
            same = [c for c in clips if c.video is info and c.start_sec >= 22.0]
            assert len(same) <= 1, f"Duplicate-looking cruise cuts from one drive: {same}"
        assert all("appear" in i.feature_cache for i in (info1, info2)), "Analysis should store appearance thumbnails"
        print("[PASS] Highlight selection (calm cruise cuts, no turns/stops, no duplicate-looking cuts) verified!")

        # 2) 수동 잘라내기 구간은 하이라이트에서 제외되어야 함
        apply_global_timeline_override([info1, info2], 14.0, 24.0, SegmentType.MANUAL_CUT)
        clips_cut = select_highlights([info1, info2], hcfg)
        for c in clips_cut:
            if c.video is info1:
                ov = min(c.end_sec, 24.0) - max(c.start_sec, 14.0)
                assert ov <= 0.3, f"Highlight overlaps manual cut: {c}"
        print("[PASS] Manual cuts excluded from highlights:", [(round(c.start_sec, 1), c.reason) for c in clips_cut])

        # 3) 세로 9:16 흐림 배경 + 한글 제목 + 촬영 시각 + 원본 소리
        out_v = tmp_dir / "shorts_vertical.mp4"
        render_highlight_shorts(
            clips,
            ShortsExportConfig(
                output_path=out_v,
                layout=ShortsLayout.VERTICAL_BLUR,
                title_text="2026.10.24 주행 하이라이트",
                show_timestamp=True,
                audio_mode=ShortsAudioMode.ORIGINAL,
            ),
        )
        pv = probe_video_file(out_v)
        print(f"Vertical shorts: {pv.width}x{pv.height}, {pv.duration_sec:.2f}s, audio={pv.has_audio}")
        assert (pv.width, pv.height) == (1080, 1920), f"Unexpected shorts size: {pv.width}x{pv.height}"
        assert abs(pv.duration_sec - total) < 1.0, f"Unexpected shorts duration: {pv.duration_sec} vs {total}"
        assert pv.has_audio, "Expected audio in vertical shorts"
        print("[PASS] Vertical 9:16 shorts (blur background, title, timestamp) verified!")

        # 4) 장면 하나 제외 + 720p 꽉 채우기 + 원본 소리와 배경음악 믹스
        bgm = tmp_dir / "bgm.m4a"
        subprocess.run(
            [get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-t", "3",
             "-i", "sine=frequency=660:sample_rate=48000", "-c:a", "aac", str(bgm)],
            check=True, creationflags=get_CreationFlags(),
        )
        clips[0].included = False
        kept = sum(c.duration for c in clips if c.included)
        out_b = tmp_dir / "shorts_bgm.mp4"
        render_highlight_shorts(
            clips,
            ShortsExportConfig(
                output_path=out_b,
                layout=ShortsLayout.VERTICAL_CROP,
                resolution="720p",
                audio_mode=ShortsAudioMode.BGM_MIX,
                bgm_path=bgm,
            ),
        )
        pb = probe_video_file(out_b)
        print(f"BGM shorts: {pb.width}x{pb.height}, {pb.duration_sec:.2f}s (expected {kept:.2f}s), audio={pb.has_audio}")
        assert (pb.width, pb.height) == (720, 1280)
        assert abs(pb.duration_sec - kept) < 1.0, "Excluded clip should shorten shorts; BGM loops to full length"
        assert pb.has_audio
        print("[PASS] Excluded clip, 720p crop layout & looping BGM mix verified!")


def _frame_at(path: Path, t: float) -> np.ndarray:
    raw = subprocess.run(
        [get_ffmpeg_exe(), "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", "scale=72:128", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, check=True, creationflags=get_CreationFlags(),
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32)


def test_shorts_transitions() -> None:
    with tempfile.TemporaryDirectory(prefix="qdc_tr_") as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "VID_20261024_090000.mp4"
        _create_synthetic_driving_video(src, duration_sec=32, fps=20)
        v = probe_video_file(src)
        clips = [HighlightClip(v, 16.0, 19.0), HighlightClip(v, 24.0, 27.0), HighlightClip(v, 28.0, 31.0)]
        durs = {}
        for tr in ShortsTransition:
            out = tmp_dir / f"tr_{tr.value}.mp4"
            render_highlight_shorts(clips, ShortsExportConfig(
                output_path=out, resolution="720p", transition=tr, transition_sec=0.5, show_timestamp=False,
            ))
            p = probe_video_file(out)
            assert p.has_audio, f"{tr.value}: audio missing"
            durs[tr.value] = p.duration_sec
        print("Transition durations:", {k: round(d, 2) for k, d in durs.items()})
        for k, d in durs.items():
            assert abs(d - 9.0) < 0.6, f"{k}: crossfade must not shorten the shorts much ({d:.2f}s)"
        # 디졸브 전환 한가운데(3.25초)는 앞 컷 끝(2.9초)·뒤 컷 시작(3.6초)과 다른 '섞인' 화면이어야 함
        d = tmp_dir / "tr_dissolve.mp4"
        a, mid, b = _frame_at(d, 2.9), _frame_at(d, 3.25), _frame_at(d, 3.6)
        cut_mid = _frame_at(tmp_dir / "tr_cut.mp4", 3.25)
        diffs = [float(np.abs(mid - x).mean()) for x in (a, b, cut_mid)]
        print("Dissolve midpoint diff vs (prev cut, next cut, hard-cut frame):", [round(x, 1) for x in diffs])
        assert all(x > 3 for x in diffs), "dissolve midpoint should be a blend, not either cut"
        print("[PASS] Cross-dissolve / fade-through-black / cut transitions verified!")

        # 전환 효과 단계가 실패하면(예: FFmpeg 버전·GPU 인코더 차이) 바로 컷으로 완성하고 경고를 남겨야 함
        import src.highlights as hl
        real_join = hl._join_with_transitions
        hl._join_with_transitions = lambda *a, **k: (
            False, "[Parsed_xfade_30 @ 0x0] The inputs needs to be a constant frame rate; current rate of 1/0 is invalid")
        try:
            warns: list = []
            out = tmp_dir / "tr_fallback.mp4"
            render_highlight_shorts(clips, ShortsExportConfig(output_path=out, resolution="720p"), warnings=warns)
            p = probe_video_file(out)
            assert out.exists() and p.duration_sec > 8.0, p.duration_sec
            assert warns and "바로 컷" in warns[0] and "constant frame rate" in warns[0], warns
        finally:
            hl._join_with_transitions = real_join
        print("[PASS] Transition failure falls back to hard cuts with a warning")


def test_horizon_leveling_and_quality() -> None:
    from src.highlights import (
        estimate_roll_angle, frame_roll_estimate, resolve_shorts_fps, resolve_shorts_resolution,
    )

    # 해상도·프레임레이트 자동 결정: 4K60 → 2160p/60, 1080p30 → 1080p/30, 120fps → 최대 60
    def fake(w, h, fps):
        return HighlightClip(VideoFileInfo(path=Path("x.mp4"), filename="x.mp4", width=w, height=h, fps=fps,
                                           duration_sec=10.0), 0.0, 3.0)
    assert resolve_shorts_resolution("auto", [fake(3840, 2160, 60)]) == "2160p"
    assert resolve_shorts_resolution("auto", [fake(2560, 1440, 30)]) == "1440p"
    assert resolve_shorts_resolution("auto", [fake(1920, 1080, 30)]) == "1080p"
    assert resolve_shorts_resolution("720p", [fake(3840, 2160, 60)]) == "720p"
    assert resolve_shorts_fps(0, [fake(3840, 2160, 59.94)]) == 59.94
    assert resolve_shorts_fps(0, [fake(1920, 1080, 120)]) == 60.0
    assert resolve_shorts_fps(30, [fake(3840, 2160, 60)]) == 30.0
    print("[PASS] Auto shorts resolution (4K source -> 2160x3840) and source frame rate (max 60)")

    with tempfile.TemporaryDirectory(prefix="qdc_level_") as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "VID_20261024_090000.mp4"
        _create_synthetic_driving_video(src, duration_sec=32, fps=20)
        results = {}
        for deg in (3, -2):
            rot = tmp_dir / f"rot_{deg}.mp4"
            subprocess.run(
                [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf",
                 f"rotate={deg}*PI/180:ow=iw:oh=ih:c=black,scale=iw*1.12:-2,crop=640:360",
                 "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy", str(rot)],
                check=True, creationflags=get_CreationFlags(),
            )
            v = probe_video_file(rot)
            angle, conf = estimate_roll_angle(v)
            assert abs(angle - deg) < 0.6 and conf >= 0.5, f"rotated {deg} deg -> estimated {angle:.2f} (conf {conf:.2f})"
            out = tmp_dir / f"leveled_{deg}.mp4"
            notes: list = []
            render_highlight_shorts([HighlightClip(v, 16.0, 19.0)], ShortsExportConfig(
                output_path=out, layout=ShortsLayout.LANDSCAPE, resolution="720p", show_timestamp=False,
            ), notes=notes)
            raw = subprocess.run(
                [get_ffmpeg_exe(), "-loglevel", "error", "-ss", "1.5", "-i", str(out), "-frames:v", "1",
                 "-vf", "scale=640:360", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                stdout=subprocess.PIPE, check=True, creationflags=get_CreationFlags(),
            ).stdout
            after, _n = frame_roll_estimate(np.frombuffer(raw, dtype=np.uint8).reshape(360, 640))
            assert after is not None and abs(after) < 0.6, f"leveled output still tilted: {after}"
            assert notes and "보정" in notes[0], notes
            results[deg] = (round(angle, 2), round(after, 2))
        straight, _c = estimate_roll_angle(probe_video_file(src))
        assert abs(straight) < 0.3, f"level footage must not be rotated: {straight}"
        print("[PASS] Horizon leveling (tilt -> estimated, after):", results)

        # 주행 중 카메라가 돌아간 영상(앞 16초만 3° 기울어짐): 기울어진 컷만 보정하고 수평인 컷은 건드리지 않아야 함
        nudge = tmp_dir / "nudge.mp4"
        subprocess.run(
            [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf",
             "rotate='if(lt(t,16),3,0)*PI/180':ow=iw:oh=ih:c=black,scale=iw*1.12:-2,crop=640:360",
             "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy", str(nudge)],
            check=True, creationflags=get_CreationFlags(),
        )
        nv = probe_video_file(nudge)
        notes = []
        render_highlight_shorts([HighlightClip(nv, 2.0, 5.0), HighlightClip(nv, 24.0, 27.0)], ShortsExportConfig(
            output_path=tmp_dir / "nudge_out.mp4", layout=ShortsLayout.LANDSCAPE, resolution="720p", show_timestamp=False,
        ), notes=notes)
        print("Per-cut leveling notes:", notes)
        assert any("컷 1" in n_ and "+3.0°" in n_ for n_ in notes), notes
        assert not any("컷 2" in n_ and "보정" in n_ for n_ in notes), f"level cut must not be rotated: {notes}"
        print("[PASS] Per-cut leveling: only the tilted cut is corrected when the camera moved mid-drive")


if __name__ == "__main__":
    test_sorter()
    test_analysis_and_export()
    test_highlight_shorts()
    test_shorts_transitions()
    test_horizon_leveling_and_quality()
    print("ALL TESTS PASSED SUCCESSFULLY!")

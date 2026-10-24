from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .ffmpeg_utils import (
    get_CreationFlags,
    get_ffmpeg_exe,
    get_software_encoder_args,
    resolve_video_encoder,
)
from .models import (
    AudioMode,
    ExportConfig,
    Segment,
    SegmentType,
    SkipMode,
    VideoCodec,
    VideoFileInfo,
    VideoQualityPreset,
)


@dataclass
class RenderJob:
    video: VideoFileInfo
    segment: Segment
    speed: float          # 1.0 = 정상 속도, >1.0 = 배속 재생
    mute_audio: bool      # 이 구간에서 오디오를 음소거할지 여부
    speed_up_audio: bool  # 배속 구간에서 원본 오디오도 배속할지 여부


def _resolve_target_dimensions(
    videos: List[VideoFileInfo],
    resolution_setting: str,
    fps_mode: str = "original",
) -> Tuple[int, int, float]:
    """출력 영상의 해상도(짝수 보장)와 FPS를 결정합니다."""
    first = videos[0]
    base_w, base_h = first.width, first.height
    base_fps = first.fps if 10.0 <= first.fps <= 120.0 else 30.0

    if fps_mode == "60":
        target_fps = 60.0
    elif fps_mode == "30":
        target_fps = 30.0
    elif fps_mode == "24":
        target_fps = 24.0
    else:
        target_fps = min(60.0, base_fps)

    res_map = {
        "4k": (3840, 2160),
        "1440p": (2560, 1440),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
    }
    if resolution_setting in res_map:
        w, h = res_map[resolution_setting]
    else:
        w, h = base_w, base_h

    w = max(320, (w // 2) * 2)
    h = max(240, (h // 2) * 2)
    return w, h, target_fps


def _build_atempo_filter(speed: float) -> str:
    """FFmpeg atempo 필터 체인을 생성합니다 (배속 오디오용)."""
    parts: List[str] = []
    rem = max(1.0, float(speed))
    while rem > 2.0:
        parts.append("atempo=2.0")
        rem /= 2.0
    if rem > 1.001:
        parts.append(f"atempo={rem:.4f}")
    parts.append("aresample=async=1:first_pts=0")
    return ",".join(parts)


def _build_speed_badge_vf(speed: float, target_w: int, target_h: int) -> Optional[str]:
    """배속 재생 구간 우측 상단에 '>> 8x' 형태의 배지를 그리는 FFmpeg 비디오 필터를 반환합니다."""
    font_candidates = [
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    chosen_font: Optional[str] = None
    for fc in font_candidates:
        if os.path.exists(fc):
            chosen_font = fc.replace(":", "\\:")
            break
    if not chosen_font:
        return None

    fontsize = max(18, int(round(target_h * 0.036)))
    box_pad = max(8, int(round(target_h * 0.012)))
    x_pos = f"w-tw-{int(round(target_w * 0.03))}"
    y_pos = f"{int(round(target_h * 0.035))}"
    label = f">> {speed:g}x"

    return (
        f"drawtext=fontfile='{chosen_font}':text='{label}':"
        f"fontsize={fontsize}:fontcolor=white:x={x_pos}:y={y_pos}:"
        f"box=1:boxcolor=black@0.55:boxborderw={box_pad}"
    )


def build_render_jobs(
    videos: List[VideoFileInfo],
    export_config: ExportConfig,
) -> List[RenderJob]:
    """정렬된 비디오 및 분석된 구간으로부터 렌더링할 작업 목록을 생성합니다."""
    jobs: List[RenderJob] = []
    mute_all = export_config.audio_mode == AudioMode.MUTE_ALL
    speed_audio = export_config.audio_mode == AudioMode.KEEP_AND_SPEED_UP
    speed_mult = max(1.5, float(export_config.speed_multiplier))

    for v in videos:
        segs = v.segments if v.segments else [
            Segment(0.0, v.duration_sec, SegmentType.NORMAL)
        ]
        for seg in segs:
            if seg.duration < 0.15:
                continue

            if seg.segment_type == SegmentType.MANUAL_CUT:
                # 사용자가 직접 지정한 잘라내기 구간은 모드와 무관하게 항상 완전히 제거
                continue

            if seg.segment_type == SegmentType.NORMAL:
                jobs.append(
                    RenderJob(
                        video=v,
                        segment=seg,
                        speed=1.0,
                        mute_audio=mute_all,
                        speed_up_audio=False,
                    )
                )
            else:
                if export_config.skip_mode == SkipMode.JUMPCUT:
                    continue
                else:
                    out_dur = seg.duration / speed_mult
                    if out_dur < 0.10:
                        continue
                    jobs.append(
                        RenderJob(
                            video=v,
                            segment=seg,
                            speed=speed_mult,
                            mute_audio=mute_all or (not speed_audio),
                            speed_up_audio=(not mute_all) and speed_audio,
                        )
                    )
    return jobs


def _encode_single_segment(
    ffmpeg: str,
    job: RenderJob,
    out_seg_path: Path,
    target_w: int,
    target_h: int,
    target_fps: float,
    include_audio_track: bool,
    audio_bitrate: str,
    codec: VideoCodec,
    encoder_name: str,
    encoder_args: List[str],
    overlay_speed_badge: bool,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    """단일 구간을 세그먼트 파일(.ts 또는 .mp4)로 인코딩합니다. (성공 여부, 오류 메시지)를 반환합니다."""
    src_dur = job.segment.duration
    out_dur = src_dur / job.speed

    vf_parts: List[str] = []
    if job.speed > 1.01:
        vf_parts.append(f"setpts=PTS/{job.speed:.4f}")
    vf_parts.append(f"fps={target_fps:.2f}")

    if job.video.width != target_w or job.video.height != target_h:
        vf_parts.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
        )

    if overlay_speed_badge and job.speed > 1.01:
        badge_vf = _build_speed_badge_vf(job.speed, target_w, target_h)
        if badge_vf:
            vf_parts.append(badge_vf)

    vf_parts.append("format=yuv420p")
    vf_str = ",".join(vf_parts)

    cmd: List[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{job.segment.start_sec:.3f}",
        "-i", str(job.video.path),
        "-t", f"{src_dur:.3f}",
    ]

    use_Original_audio = (
        include_audio_track
        and (not job.mute_audio)
        and job.video.has_audio
    )

    if include_audio_track and not use_Original_audio:
        cmd.extend([
            "-f", "lavfi",
            "-t", f"{out_dur + 1.0:.3f}",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        ])

    cmd.extend(["-vf", vf_str])
    cmd.extend(["-c:v", encoder_name, *encoder_args])

    if not include_audio_track:
        cmd.append("-an")
    elif use_Original_audio:
        af_str = (
            _build_atempo_filter(job.speed)
            if (job.speed > 1.01 and job.speed_up_audio)
            else "aresample=async=1:first_pts=0"
        )
        cmd.extend([
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-af", af_str,
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-ar", "48000",
            "-ac", "2",
        ])
    else:
        cmd.extend([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-ar", "48000",
            "-ac", "2",
            "-shortest",
        ])

    if codec == VideoCodec.HEVC:
        cmd.extend([
            "-tag:v", "hvc1",
            "-video_track_timescale", "90000",
            "-f", "mp4",
            str(out_seg_path),
        ])
    else:
        cmd.extend([
            "-f", "mpegts",
            str(out_seg_path),
        ])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=get_CreationFlags(),
    )
    stderr_chunks: List[str] = []
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
                stderr_chunks.append(err.decode("utf-8", errors="replace"))
            break
        except subprocess.TimeoutExpired:
            continue

    ok = proc.returncode == 0 and out_seg_path.exists() and out_seg_path.stat().st_size > 0
    err_text = "".join(stderr_chunks).strip()
    return ok, err_text


def render_merged_video(
    videos: List[VideoFileInfo],
    export_config: ExportConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Path:
    """
    시간순으로 정렬되고 구간 분석/편집이 완료된 비디오 리스트를 하나로 병합하여 최종 MP4 파일을 생성합니다.
    """
    if not videos:
        raise ValueError("합칠 영상 파일이 없습니다.")

    jobs = build_render_jobs(videos, export_config)
    if not jobs:
        raise ValueError("조건에 맞는 출력 구간이 없습니다. 잘라내기 구간이나 설정을 확인해 주세요.")

    ffmpeg = get_ffmpeg_exe()
    target_w, target_h, target_fps = _resolve_target_dimensions(
        videos, export_config.target_resolution, export_config.fps_mode
    )
    if export_config.target_fps > 0:
        target_fps = export_config.target_fps

    any_has_audio = any(v.has_audio for v in videos)
    include_audio_track = (export_config.audio_mode != AudioMode.MUTE_ALL) and any_has_audio

    audio_bitrate_map = {
        VideoQualityPreset.LOSSLESS_LIKE: "256k",
        VideoQualityPreset.HIGH: "192k",
        VideoQualityPreset.BALANCED: "128k",
        VideoQualityPreset.COMPACT: "128k",
    }
    audio_bitrate = audio_bitrate_map.get(export_config.quality_preset, "192k")

    enc_name, enc_args = resolve_video_encoder(
        codec=export_config.video_codec,
        quality=export_config.quality_preset,
        prefer_hw=export_config.use_hw_accel,
    )

    output_path = Path(export_config.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_out_dur = sum(j.segment.duration / j.speed for j in jobs)
    completed_out_dur = 0.0
    seg_ext = ".mp4" if export_config.video_codec == VideoCodec.HEVC else ".ts"

    with tempfile.TemporaryDirectory(prefix="quick_dash_cam_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        seg_files: List[Path] = []

        for idx, job in enumerate(jobs):
            if cancel_check and cancel_check():
                raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")

            seg_type_label = {
                SegmentType.NORMAL: "일반 주행 (1x)",
                SegmentType.STOP: f"신호대기/정차 ({job.speed:g}x 배속)",
                SegmentType.STRAIGHT: f"고속도로 직진 ({job.speed:g}x 배속)",
            }.get(job.segment.segment_type, "구간")

            if progress_callback:
                frac = (completed_out_dur / max(0.1, total_out_dur)) * 0.92
                progress_callback(
                    min(0.92, frac),
                    f"인코딩 중 [{idx + 1}/{len(jobs)}]: {job.video.filename} ({seg_type_label})",
                )

            seg_path = tmp_dir / f"seg_{idx:05d}{seg_ext}"
            ok, err_msg = _encode_single_segment(
                ffmpeg=ffmpeg,
                job=job,
                out_seg_path=seg_path,
                target_w=target_w,
                target_h=target_h,
                target_fps=target_fps,
                include_audio_track=include_audio_track,
                audio_bitrate=audio_bitrate,
                codec=export_config.video_codec,
                encoder_name=enc_name,
                encoder_args=enc_args,
                overlay_speed_badge=export_config.overlay_speed_badge,
                cancel_check=cancel_check,
            )

            # 하드웨어 인코더 실패 시 소프트웨어 인코더로 자동 폴백
            sw_name, sw_args = get_software_encoder_args(
                export_config.video_codec, export_config.quality_preset
            )
            if not ok and enc_name != sw_name:
                if cancel_check and cancel_check():
                    raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")
                enc_name, enc_args = sw_name, sw_args
                ok, err_msg = _encode_single_segment(
                    ffmpeg=ffmpeg,
                    job=job,
                    out_seg_path=seg_path,
                    target_w=target_w,
                    target_h=target_h,
                    target_fps=target_fps,
                    include_audio_track=include_audio_track,
                    audio_bitrate=audio_bitrate,
                    codec=export_config.video_codec,
                    encoder_name=enc_name,
                    encoder_args=enc_args,
                    overlay_speed_badge=export_config.overlay_speed_badge,
                    cancel_check=cancel_check,
                )

            if not ok:
                if cancel_check and cancel_check():
                    raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")
                detail = f"\n[오류 상세]: {err_msg}" if err_msg else ""
                raise RuntimeError(
                    f"구간 인코딩에 실패했습니다: {job.video.filename} "
                    f"({job.segment.start_sec:.1f}s ~ {job.segment.end_sec:.1f}s){detail}"
                )

            seg_files.append(seg_path)
            completed_out_dur += job.segment.duration / job.speed

        if cancel_check and cancel_check():
            raise RuntimeError("사용자에 의해 작업이 취소되었습니다.")

        if progress_callback:
            progress_callback(0.94, "전체 구간을 하나의 MP4 파일로 병합하는 중...")

        concat_list_path = tmp_dir / "concat_list.txt"
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for sf in seg_files:
                safe_path = str(sf.resolve()).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        concat_cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c", "copy",
        ]
        if export_config.video_codec == VideoCodec.HEVC:
            concat_cmd.extend(["-tag:v", "hvc1"])
        elif include_audio_track:
            concat_cmd.extend(["-bsf:a", "aac_adtstoasc"])

        concat_cmd.extend(["-movflags", "+faststart", str(output_path)])

        res = subprocess.run(
            concat_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=get_CreationFlags(),
        )
        if res.returncode != 0 or not output_path.exists():
            raise RuntimeError(f"최종 영상 병합 중 오류가 발생했습니다:\n{res.stderr}")

    if progress_callback:
        progress_callback(1.0, f"완료! 저장 위치: {output_path.name}")

    return output_path

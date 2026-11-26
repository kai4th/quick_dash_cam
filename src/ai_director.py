"""
Claude가 하이라이트 장면을 골라 주는 "AI 편집자".

흐름
1) 알고리즘으로 목표 장면 수의 약 5배 후보를 서로 다른 모습 위주로 넉넉히 뽑습니다.
2) 후보마다 시작·중간·끝 3프레임을 가로로 이어 붙이고, 후보 6개씩 한 장의 "콘택트 시트" 이미지로 묶습니다.
   (이미지 장수를 줄여 Claude Code가 파일을 읽는 횟수와 API 토큰 비용을 모두 줄임)
3) Claude에게 후보별 점수·장면 설명·자막과, 최종으로 쓸 장면 번호를 JSON으로 받습니다.
   - Claude Code CLI(`claude -p`, 구독 계정) 또는 Claude API(anthropic SDK, API 키) 중 선택
4) 응답을 검증해 HighlightClip 목록에 반영합니다. 선택되지 않은 후보도 목록에 남겨(제외 상태)
   사용자가 직접 바꿔 넣을 수 있게 합니다.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .ffmpeg_utils import get_CreationFlags, get_ffmpeg_exe
from .highlights import clip_timestamp_text, ensure_appearance, select_diverse_highlights, select_highlights
from .models import AiDirectorMode, HighlightClip, HighlightConfig, VideoFileInfo

API_MODEL = "claude-opus-5-5"
CLI_MODEL = "opus"

FRAMES_PER_CANDIDATE = 3
FRAME_W, FRAME_H = 384, 216
LABEL_W = 64
CANDIDATES_PER_SHEET = 6

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]

RESULT_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                    "caption": {"type": "string"},
                },
                "required": ["id", "score", "reason", "caption"],
                "additionalProperties": False,
            },
        },
        "selected_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["candidates", "selected_ids"],
    "additionalProperties": False,
}


class AiDirectorError(RuntimeError):
    """AI 장면 선정 실패 (설치·로그인·네트워크·응답 형식 문제 등). 메시지는 사용자에게 그대로 보여줌."""


@dataclass
class ContactSheet:
    path: Path
    first_id: int
    last_id: int


# ── 1. 후보 장면 → 콘택트 시트 이미지 ──

def candidate_count_for(hcfg: HighlightConfig) -> Tuple[int, int]:
    """(최종 장면 수, 후보 수). 후보는 최종 장면 수의 약 5배 (최소 20, 최대 80)."""
    n_final = max(1, int(round(hcfg.target_duration_sec / max(1.0, hcfg.clip_duration_sec))))
    n_cand = int(min(80, max(20, n_final * 5)))
    return n_final, n_cand


def select_candidates(videos: List[VideoFileInfo], hcfg: HighlightConfig) -> List[HighlightClip]:
    """AI에게 보여줄 후보 장면을 넉넉히 고릅니다 (서로 다른 모습 위주, 시간순)."""
    _n_final, n_cand = candidate_count_for(hcfg)
    # 점수 순으로만 뽑으면 커브·주차장 같은 '화면 변화 큰' 장면에 쏠리므로, 서로 다른 모습의 장면을 넉넉히 뽑아 Claude가 판단하게 함
    diverse = select_diverse_highlights(videos, hcfg, n_cand)
    if diverse:
        return diverse
    cand_cfg = HighlightConfig(
        target_duration_sec=n_cand * hcfg.clip_duration_sec,
        clip_duration_sec=hcfg.clip_duration_sec,
        use_audio_peaks=hcfg.use_audio_peaks,
        straight_weight=hcfg.straight_weight,
    )
    return select_highlights(videos, cand_cfg)


def _grab_frames(clip: HighlightClip, n: int, cancel_check: Optional[CancelFn]) -> List[np.ndarray]:
    """장면 구간에서 고르게 n프레임을 FRAME_W x FRAME_H로 디코딩합니다 (FFmpeg 1회 실행)."""
    dur = max(0.2, clip.duration)
    frame_bytes = FRAME_W * FRAME_H * 3
    cmd = [
        get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-ss", f"{clip.start_sec + dur * 0.08:.3f}",
        "-t", f"{dur * 0.9:.3f}",
        "-i", str(clip.video.path),
        "-vf", (
            f"fps={n / (dur * 0.9):.4f},"
            f"scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease,"
            f"pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2"
        ),
        "-frames:v", str(n),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frames: List[np.ndarray] = []
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60,
            creationflags=get_CreationFlags(),
        )
        raw = res.stdout
        for i in range(len(raw) // frame_bytes):
            frames.append(np.frombuffer(raw[i * frame_bytes:(i + 1) * frame_bytes], dtype=np.uint8)
                          .reshape(FRAME_H, FRAME_W, 3))
    except Exception:
        pass
    while len(frames) < n:
        frames.append(frames[-1] if frames else np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
    return frames[:n]


def build_contact_sheets(
    candidates: Sequence[HighlightClip],
    out_dir: Path,
    progress: Optional[ProgressFn] = None,
    cancel_check: Optional[CancelFn] = None,
) -> List[ContactSheet]:
    """
    후보 6개씩 한 장(가로 3프레임 x 세로 6줄)의 JPEG 이미지로 묶습니다.
    각 줄 왼쪽에 후보 번호(#1, #2 ...)를 크게 적어 Claude가 번호와 장면을 정확히 대응시키게 합니다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets: List[ContactSheet] = []
    rows: List[np.ndarray] = []
    first_id = 1
    for i, clip in enumerate(candidates):
        if cancel_check and cancel_check():
            raise AiDirectorError("사용자에 의해 작업이 취소되었습니다.")
        if progress:
            progress(i / max(1, len(candidates)), f"[AI 편집] 후보 장면 사진 준비 {i + 1}/{len(candidates)}")
        cid = i + 1
        frames = _grab_frames(clip, FRAMES_PER_CANDIDATE, cancel_check)
        label = np.full((FRAME_H, LABEL_W, 3), 24, dtype=np.uint8)
        txt = f"#{cid}"
        scale = 0.9 if cid < 10 else 0.75
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(label, txt, ((LABEL_W - tw) // 2, (FRAME_H + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        row = np.hstack([label] + frames)
        cv2.line(row, (0, FRAME_H - 1), (row.shape[1], FRAME_H - 1), (255, 255, 255), 2)
        rows.append(row)
        if len(rows) == CANDIDATES_PER_SHEET or i == len(candidates) - 1:
            sheet_path = out_dir / f"candidates_{len(sheets) + 1:02d}.jpg"
            cv2.imwrite(str(sheet_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
            sheets.append(ContactSheet(sheet_path, first_id, cid))
            rows = []
            first_id = cid + 1
    return sheets


# ── 2. 요청 내용(프롬프트) ──

def _candidate_line(cid: int, clip: HighlightClip, videos: List[VideoFileInfo]) -> str:
    clip_no = next((i + 1 for i, v in enumerate(videos) if v is clip.video), 0)
    offset = sum(v.duration_sec for v in videos[: max(0, clip_no - 1)]) + clip.start_sec
    mm, ss = divmod(int(offset), 60)
    hh, mm = divmod(mm, 60)
    parts = [f"#{cid}", f"클립 {clip_no}", f"전체 타임라인 {hh:d}:{mm:02d}:{ss:02d}"]
    ts = clip_timestamp_text(clip)
    if ts:
        parts.append(f"촬영 시각 {ts.split()[-1]}")
    if clip.reason:
        parts.append(f"알고리즘 감지: {clip.reason}")
    return " · ".join(parts)


def build_prompt(
    candidates: Sequence[HighlightClip],
    videos: List[VideoFileInfo],
    sheets: Sequence[ContactSheet],
    n_final: int,
    target_sec: float,
    sheet_refs: Optional[Sequence[str]] = None,
) -> str:
    """Claude에게 보낼 지시문. sheet_refs가 있으면(Claude Code용) 이미지 파일 이름을 함께 안내합니다."""
    n_final = min(n_final, len(candidates))
    sheet_desc = []
    for k, sh in enumerate(sheets):
        ref = f" — 파일 `{sheet_refs[k]}`" if sheet_refs else ""
        sheet_desc.append(f"- 이미지 {k + 1}{ref}: 후보 #{sh.first_id} ~ #{sh.last_id}")
    cand_lines = "\n".join(_candidate_line(i + 1, c, videos) for i, c in enumerate(candidates))
    return f"""너는 자동차 블랙박스·액션캠 주행 영상으로 세로형 쇼츠를 만드는 영상 편집자야.
긴 주행 영상에서 알고리즘이 고른 후보 장면 {len(candidates)}개(각 약 {candidates[0].duration:.0f}초)가 있어.
이 중 가장 볼 만한 {n_final}개를 골라 약 {target_sec:.0f}초짜리 하이라이트 쇼츠를 만들려고 해.

## 후보 이미지 보는 법
{chr(10).join(sheet_desc)}
- 이미지 한 장에 후보가 위에서 아래로 한 줄씩 있고, 줄 왼쪽에 후보 번호(#번호)가 적혀 있어.
- 한 줄의 사진 3장은 그 장면의 시작 → 중간 → 끝 순서야.

## 후보 정보 (시간순)
{cand_lines}

## 가장 중요한 기준
1. 컷 안에서는 화면이 차분할 것
   - 한 장면은 약 {candidates[0].duration:.0f}초짜리 컷이야. 컷 안에서 화면이 크게 바뀌면 쇼츠가 어지러워져.
   - 한 줄의 사진 3장(시작·중간·끝)이 서로 많이 다르면(좌·우회전 중, 방향이 크게 바뀜, 터널 진출입) 고르지 마.
   - 차가 안정적으로 앞으로 달리며 시야가 트인 장면이 좋아 (직선 도로·고속도로 순항·완만한 커브의 풍경).
2. 컷끼리는 배경이 겹치지 않을 것
   - 고른 컷들끼리는 장소·풍경·분위기가 서로 달라야 해 (예: 시내 → 강변 → 산길 → 고속도로 → 노을).
   - 같은 도로·비슷한 풍경이 반복되면 그중 가장 나은 하나만 골라.

## 좋은 장면
- 경치가 좋은 장면: 바다·강·산·노을·야경·벚꽃·단풍·눈길, 시원하게 뻗은 도로, 다리 위
- 눈에 띄는 랜드마크·도시 풍경을 안정적으로 지나가는 장면

## 피할 장면
- 좌회전·우회전·유턴·차선 변경처럼 방향이 바뀌는 장면
- 주차장·지하주차장·골목·주차장 출입구, 아주 천천히 움직이는 장면
- 정체로 거의 멈춰 있는 장면
- 너무 어둡거나 흐리거나 역광·유리 반사가 심해 알아보기 힘든 장면
- 선택한 장면들이 한 시간대에 몰리지 않고 주행 전체에 고루 퍼지면 더 좋아

## 응답 (JSON만 출력)
- candidates: 후보 {len(candidates)}개 모두에 대해
  - id: 후보 번호
  - score: 0~10 정수 (쇼츠에 넣을 가치)
  - reason: 장면을 설명하는 한국어 15자 이내 (예: "노을 진 해안도로", "시원한 고속도로")
  - caption: 화면 구석에 작게 띄울 한국어 자막. 담백한 짧은 명사구 2~8자, 이모지·느낌표·특수문자 금지
    - 화면에 보이는 걸 그대로 읽지 말 것 (나쁜 예: "바다가 보인다", "차가 달린다", "도로 주행 중")
    - 시간대·분위기·장소의 성격을 짧게 (좋은 예: "해 질 무렵", "한산한 국도", "터널을 지나", "비 갠 오후", "퇴근길")
    - 과장·감탄·유행어 금지. 붙일 만한 말이 없으면 빈 문자열("")로 둬 (억지 자막보다 없는 게 나음)
    - 고른 장면들끼리 같은 표현을 반복하지 말 것
- selected_ids: 쇼츠에 넣을 후보 번호 정확히 {n_final}개
"""


# ── 3. 응답 → 하이라이트 목록 ──

def _extract_json_object(text: str) -> Optional[Dict]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _clean_text(text: object, limit: int) -> str:
    s = str(text or "").replace("\n", " ").strip()
    # 오버레이 글꼴에 없는 이모지 등은 제거 (한글·영문·숫자·기본 문장부호만 유지)
    s = "".join(ch for ch in s if ord(ch) < 0x2000 or 0x3000 <= ord(ch) <= 0xD7A3 or 0xFF00 <= ord(ch) <= 0xFFEF)
    return s[:limit].strip()


def apply_ai_result(
    data: Dict,
    candidates: Sequence[HighlightClip],
    n_final: int,
) -> List[HighlightClip]:
    """
    Claude 응답을 검증해 후보 목록에 반영합니다.
    - 모르는 번호·중복은 무시, 선택이 모자라면 AI 점수 → 알고리즘 점수 순으로 채우고, 넘치면 AI 점수 높은 순으로 자름
    - 반환: 후보 전체 목록 — 선택된 것만 included=True
    """
    n = len(candidates)
    n_final = min(max(1, n_final), n)
    info: Dict[int, Dict] = {}
    for item in data.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if 1 <= cid <= n and cid not in info:
            info[cid] = item

    def ai_score(cid: int) -> float:
        try:
            return float(max(0.0, min(10.0, float(info[cid].get("score", 0)))))
        except (KeyError, TypeError, ValueError):
            return -1.0

    selected: List[int] = []
    for raw in data.get("selected_ids") or []:
        try:
            cid = int(raw)
        except (TypeError, ValueError):
            continue
        if 1 <= cid <= n and cid not in selected:
            selected.append(cid)
    if len(selected) > n_final:
        selected = sorted(selected, key=lambda c: -ai_score(c))[:n_final]
    if len(selected) < n_final:
        # Claude가 덜 골랐으면 AI 점수 5점 이상(또는 평가가 빠진) 후보로만 채움 — 낮게 평가한 장면은 억지로 넣지 않음
        rest = [c for c in range(1, n + 1) if c not in selected and (c not in info or ai_score(c) >= 5.0)]
        rest.sort(key=lambda c: (-ai_score(c), -candidates[c - 1].score))
        selected += rest[: n_final - len(selected)]

    chosen = set(selected)
    result: List[HighlightClip] = []
    for i, c in enumerate(candidates):
        cid = i + 1
        item = info.get(cid, {})
        result.append(HighlightClip(
            video=c.video,
            start_sec=c.start_sec,
            end_sec=c.end_sec,
            score=c.score,
            reason=_clean_text(item.get("reason"), 20) or c.reason,
            included=cid in chosen,
            ai_score=ai_score(cid) if cid in info else None,
            caption=_clean_text(item.get("caption"), 12).rstrip("!~.,"),
        ))
    return result


# ── 4-A. Claude Code CLI (구독 계정) ──

def find_claude_cli() -> Optional[str]:
    exe = shutil.which("claude")
    if exe:
        return exe
    home = Path.home()
    for cand in (
        home / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude"),
        home / ".claude" / "local" / ("claude.exe" if os.name == "nt" else "claude"),
        Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd",
    ):
        if str(cand) and cand.exists():
            return str(cand)
    return None


def _run_process(cmd: List[str], stdin_text: str, cwd: Path, timeout: float,
                 cancel_check: Optional[CancelFn], env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=get_CreationFlags(), env=env,
    )
    t0 = time.time()
    payload: Optional[bytes] = stdin_text.encode("utf-8")
    while True:
        if cancel_check and cancel_check():
            proc.kill()
            raise AiDirectorError("사용자에 의해 작업이 취소되었습니다.")
        if time.time() - t0 > timeout:
            proc.kill()
            raise AiDirectorError(f"Claude 응답이 {int(timeout)}초 안에 오지 않았습니다. 잠시 후 다시 시도해 주세요.")
        try:
            out, err = proc.communicate(input=payload, timeout=0.5)
            return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            payload = None  # 입력은 첫 호출에서 이미 전달됨
            continue


CLI_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def _cli_error_text(stdout: str, stderr: str) -> str:
    """실패한 실행의 출력에서 사람이 읽을 오류 문구를 꺼냅니다 (JSON 출력이면 result 필드)."""
    for text in (stdout, stderr):
        obj = _extract_json_object(text) if text.strip() else None
        if isinstance(obj, dict) and obj.get("result"):
            return str(obj["result"]).strip()
    return (stderr.strip() or stdout.strip() or "(오류 메시지 없음)")


def _looks_like_auth_error(low: str) -> bool:
    return any(k in low for k in ("/login", "not logged in", "invalid api key", "authentication", "oauth", "unauthorized", "401"))


def _explain_cli_failure(exe: str, err: str) -> str:
    low = err.lower()
    msg = f"Claude Code 실행에 실패했습니다.\n\n[사용한 claude 경로]\n{exe}\n\n[Claude Code가 보낸 메시지]\n{err[:1200]}"
    if _looks_like_auth_error(low):
        # PATH에 등록되지 않은 경우가 있어(예: ~/.local/bin) 실제로 찾은 전체 경로로 안내
        on_path = shutil.which("claude") is not None
        run = "claude" if on_path else (f'& "{exe}"' if os.name == "nt" else f'"{exe}"')
        msg += (
            "\n\n[해결 방법]\n"
            f"1) PowerShell(또는 터미널)을 열고 다음을 실행합니다:\n   {run}\n"
            "2) Claude 구독 계정으로 로그인한 뒤 `/exit`로 종료합니다 (이미 입력창이 뜨면 `/login` 입력).\n"
            f"3) {run} -p \"안녕\" 을 실행해 답이 나오면 앱에서 다시 시도하세요.\n"
            "※ Claude 데스크톱 앱·웹 로그인과 터미널용 claude 로그인은 별개입니다."
        )
        if not on_path:
            msg += "\n※ claude가 PATH에 등록되어 있지 않아 터미널에서 `claude`만 입력하면 실행되지 않습니다."
    elif "credit" in low or "usage limit" in low or "rate limit" in low:
        msg += "\n\n구독 사용량 한도에 걸렸을 수 있습니다. 잠시 후 다시 시도하거나 [Claude API] 방식을 사용해 주세요."
    return msg


def parse_cli_output(stdout: str, exe: str = "claude") -> Dict:
    """`claude -p --output-format json` 출력에서 결과 JSON을 꺼냅니다."""
    try:
        outer = json.loads(stdout.strip())
    except json.JSONDecodeError:
        outer = _extract_json_object(stdout)
    if not isinstance(outer, dict):
        raise AiDirectorError(f"Claude Code 출력 형식을 이해하지 못했습니다:\n{stdout[:500]}")
    if outer.get("is_error") or outer.get("subtype", "success") != "success":
        raise AiDirectorError(_explain_cli_failure(exe, str(outer.get("result", "")) or "(오류 메시지 없음)"))
    structured = outer.get("structured_output")
    if isinstance(structured, dict):
        return structured
    if "selected_ids" in outer and "candidates" in outer:
        return outer
    inner = _extract_json_object(str(outer.get("result", "")))
    if inner is None:
        raise AiDirectorError(f"Claude 응답에서 JSON 결과를 찾지 못했습니다:\n{str(outer.get('result', ''))[:500]}")
    return inner


def run_claude_code(
    prompt: str,
    workdir: Path,
    cancel_check: Optional[CancelFn] = None,
    timeout: float = 900.0,
) -> Dict:
    """설치된 Claude Code를 헤드리스(-p)로 실행해 workdir 안의 후보 이미지를 평가받습니다."""
    exe = find_claude_cli()
    if not exe:
        raise AiDirectorError(
            "Claude Code(claude 명령)를 찾지 못했습니다.\n"
            "Claude Code를 설치하고 터미널에서 `claude`를 한 번 실행해 로그인한 뒤 다시 시도해 주세요."
        )
    full = prompt + (
        "\n\n위 이미지 파일들은 현재 폴더에 있어. Read 도구로 모든 이미지를 확인한 뒤, "
        "설명 없이 JSON 객체 하나만 출력해."
    )
    # "구독 계정" 모드: API 키 환경 변수가 있으면 Claude Code가 구독 로그인 대신 그 키를 쓰므로 빼고 실행
    env = {k: v for k, v in os.environ.items() if k not in CLI_STRIPPED_ENV}

    # .cmd/.bat(npm 설치본)은 cmd.exe를 거치며 따옴표가 깨질 수 있어 JSON 스키마 인자는 생략 (프롬프트로 형식 지정)
    is_batch = exe.lower().endswith((".cmd", ".bat"))
    newer = ["--tools", "Read", "--allowedTools", "Read", "--no-session-persistence"]
    variants: List[List[str]] = []
    if not is_batch:
        variants.append(newer + ["--json-schema", json.dumps(RESULT_SCHEMA, ensure_ascii=False)])
    variants.append(newer)
    variants.append(["--allowedTools", "Read"])  # 구버전 Claude Code 호환

    use_model = True
    last_err = ""
    i = 0
    while i < len(variants):
        cmd = [exe, "-p", "--output-format", "json"] + (["--model", CLI_MODEL] if use_model else []) + variants[i]
        code, out, err = _run_process(cmd, full, workdir, timeout, cancel_check, env)
        if code == 0 and out.strip():
            return parse_cli_output(out, exe)
        last_err = _cli_error_text(out, err)
        low = last_err.lower()
        if "unknown option" in low or "unexpected argument" in low:
            i += 1  # 이 버전에 없는 옵션 → 더 단순한 명령으로 재시도
            continue
        if use_model and "model" in low and not _looks_like_auth_error(low):
            use_model = False  # 요금제에서 Opus를 못 쓰는 경우 등 → 계정 기본 모델로 재시도
            continue
        break
    raise AiDirectorError(_explain_cli_failure(exe, last_err))


# ── 4-B. Claude API (API 키) ──

def run_claude_api(
    prompt: str,
    sheets: Sequence[ContactSheet],
    api_key: Optional[str] = None,
    cancel_check: Optional[CancelFn] = None,
) -> Dict:
    try:
        import anthropic
    except ImportError as exc:
        raise AiDirectorError(
            "Claude API를 쓰려면 anthropic 패키지가 필요합니다.\n`pip install -r requirements.txt`를 실행해 주세요."
        ) from exc

    content: List[Dict] = []
    for k, sh in enumerate(sheets):
        content.append({"type": "text", "text": f"이미지 {k + 1}: 후보 #{sh.first_id} ~ #{sh.last_id}"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(sh.path.read_bytes()).decode("ascii"),
            },
        })
    content.append({"type": "text", "text": prompt})

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=API_MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # 안전 정책으로 거절되면 서버가 대체 모델로 자동 재시도
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": RESULT_SCHEMA},
            },
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError as exc:
        raise AiDirectorError("Claude API 키가 올바르지 않습니다. 키를 확인해 주세요.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise AiDirectorError(f"이 API 키로는 요청할 권한이 없습니다: {exc.message}") from exc
    except anthropic.RateLimitError as exc:
        raise AiDirectorError("Claude API 사용량 한도에 걸렸습니다. 잠시 후 다시 시도해 주세요.") from exc
    except anthropic.BadRequestError as exc:
        raise AiDirectorError(f"Claude API 요청이 거부되었습니다: {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        raise AiDirectorError(f"Claude API 서버 오류({exc.status_code})입니다. 잠시 후 다시 시도해 주세요.") from exc
    except anthropic.APIConnectionError as exc:
        raise AiDirectorError("Claude API에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요.") from exc

    if cancel_check and cancel_check():
        raise AiDirectorError("사용자에 의해 작업이 취소되었습니다.")
    if response.stop_reason == "refusal":
        raise AiDirectorError("Claude가 이 요청을 처리하지 않았습니다(안전 정책). 알고리즘 선정 결과를 사용해 주세요.")
    if response.stop_reason == "max_tokens":
        raise AiDirectorError("Claude 응답이 너무 길어 중간에 끊겼습니다. 쇼츠 길이를 줄여 다시 시도해 주세요.")
    text = next((b.text for b in response.content if b.type == "text"), "")
    data = _extract_json_object(text)
    if data is None:
        raise AiDirectorError(f"Claude API 응답을 이해하지 못했습니다:\n{text[:500]}")
    return data


# ── 5. 전체 흐름 ──

def ai_select_highlights(
    videos: List[VideoFileInfo],
    hcfg: HighlightConfig,
    mode: AiDirectorMode,
    api_key: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    cancel_check: Optional[CancelFn] = None,
    runner: Optional[Callable[[str, Path, List[ContactSheet]], Dict]] = None,
) -> List[HighlightClip]:
    """
    후보 선정 → 콘택트 시트 → Claude 평가 → 결과 반영.
    반환: 후보 전체 HighlightClip 목록 — Claude가 고른 장면만 included=True
    runner는 테스트용(실제 Claude 대신 호출)입니다.
    """
    n_final, _n_cand = candidate_count_for(hcfg)
    for v in videos:  # 서로 다른 모습의 후보를 고르기 위한 썸네일 (캐시되어 있으면 즉시)
        if cancel_check and cancel_check():
            raise AiDirectorError("사용자에 의해 작업이 취소되었습니다.")
        ensure_appearance(v, cancel_check)
    candidates = select_candidates(videos, hcfg)
    if not candidates:
        return []

    def sub(lo: float, hi: float) -> Optional[ProgressFn]:
        if progress is None:
            return None
        return lambda f, m: progress(lo + (hi - lo) * f, m)

    with tempfile.TemporaryDirectory(prefix="quick_dash_cam_ai_") as tmp_str:
        tmp = Path(tmp_str)
        sheets = build_contact_sheets(candidates, tmp, sub(0.0, 0.35), cancel_check)
        refs = [sh.path.name for sh in sheets]
        if progress:
            progress(0.4, f"[AI 편집] Claude가 후보 {len(candidates)}개 중 {min(n_final, len(candidates))}개를 고르는 중... (보통 30초~2분)")
        if runner is not None:
            data = runner(build_prompt(candidates, videos, sheets, n_final, hcfg.target_duration_sec, refs), tmp, sheets)
        elif mode == AiDirectorMode.CLAUDE_CODE:
            prompt = build_prompt(candidates, videos, sheets, n_final, hcfg.target_duration_sec, refs)
            data = run_claude_code(prompt, tmp, cancel_check)
        elif mode == AiDirectorMode.CLAUDE_API:
            prompt = build_prompt(candidates, videos, sheets, n_final, hcfg.target_duration_sec)
            data = run_claude_api(prompt, sheets, api_key, cancel_check)
        else:
            raise AiDirectorError("AI 장면 선정 방식이 선택되지 않았습니다.")
    if progress:
        progress(1.0, "[AI 편집] Claude 선정 결과 반영 중...")
    return apply_ai_result(data, candidates, n_final)

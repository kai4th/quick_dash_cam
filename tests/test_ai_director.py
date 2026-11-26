"""
AI 편집자(ai_director) 테스트 — 실제 Claude를 호출하지 않습니다.
- Claude Code: PATH에 가짜 `claude` 실행 파일을 두고 옵션·입력·출력 처리 검증
- Claude API: 로컬 가짜 HTTP 서버로 실제 anthropic SDK가 보내는 요청 형식 검증
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai_director import (  # noqa: E402
    RESULT_SCHEMA,
    AiDirectorError,
    ai_select_highlights,
    candidate_count_for,
    apply_ai_result,
    parse_cli_output,
)
from src.analyzer import analyze_video  # noqa: E402
from src.ffmpeg_utils import probe_video_file  # noqa: E402
from src.highlights import render_highlight_shorts  # noqa: E402
from src.models import (  # noqa: E402
    AiDirectorMode,
    AnalysisConfig,
    HighlightClip,
    HighlightConfig,
    ShortsExportConfig,
    VideoFileInfo,
)
from tests.test_pipeline import _create_synthetic_driving_video  # noqa: E402


def _fake_candidates(n: int) -> list:
    v = VideoFileInfo(path=Path("x.mp4"), filename="x.mp4", duration_sec=600.0)
    return [HighlightClip(v, i * 10.0, i * 10.0 + 3.0, score=1.0 / (i + 1), reason="회전·커브") for i in range(n)]


def test_apply_ai_result() -> None:
    cands = _fake_candidates(8)
    data = {
        "title": "노을 드라이브 🌅",
        "candidates": [
            {"id": 3, "score": 9, "reason": "노을 진 해안도로", "caption": "바다가 보인다 😍"},
            {"id": 5, "score": 12, "reason": "터널 진출", "caption": "터널 탈출"},
            {"id": 99, "score": 10, "reason": "없는 후보", "caption": ""},
            {"id": "2", "score": 7, "reason": "고갯길", "caption": "구불구불"},
        ],
        "selected_ids": [5, 3, 3, 99, "x"],
    }
    clips, title = apply_ai_result(data, cands, n_final=3)
    assert title == "노을 드라이브", title
    assert len(clips) == 8
    chosen = [i + 1 for i, c in enumerate(clips) if c.included]
    # 5, 3 선택 + 부족한 1개는 AI 점수 높은 2번으로 채움, 모르는 번호·중복은 무시
    assert chosen == [2, 3, 5], chosen
    assert clips[4].ai_score == 10.0, "score must be clamped to 0..10"
    assert clips[2].caption == "바다가 보인다", clips[2].caption
    assert clips[0].ai_score is None and clips[0].reason == "회전·커브", "unrated candidates keep algorithm reason"
    print("[PASS] AI result validation (unknown ids, duplicates, clamping, fill, emoji strip)")


def test_candidate_counts() -> None:
    cases = {(30.0, 3.0): (10, 50), (15.0, 3.0): (5, 25), (90.0, 3.0): (30, 80), (6.0, 3.0): (2, 20)}
    for (target, clip), want in cases.items():
        got = candidate_count_for(HighlightConfig(target_duration_sec=target, clip_duration_sec=clip))
        assert got == want, f"{target}s/{clip}s -> {got}, expected {want}"
    print("[PASS] Candidate counts: ~5x final cuts (min 20, max 80)")


def test_parse_cli_output() -> None:
    payload = {"title": "t", "candidates": [], "selected_ids": [1]}
    out1 = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                       "result": "", "structured_output": payload})
    assert parse_cli_output(out1) == payload
    out2 = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                       "result": "설명입니다\n```json\n" + json.dumps(payload) + "\n```"})
    assert parse_cli_output(out2) == payload
    try:
        parse_cli_output(json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True, "result": "x"}))
        raise AssertionError("expected error")
    except AiDirectorError:
        pass
    print("[PASS] Claude Code JSON output parsing (structured_output / fenced result / error)")


FAKE_CLAUDE = r'''#!{python}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ["FAKE_CLAUDE_LOG"])
calls = json.loads(log.read_text()) if log.exists() else []
prompt = sys.stdin.read()
calls.append({{"args": args, "cwd": os.getcwd(), "files": sorted(os.listdir(".")), "prompt": prompt,
              "has_api_key": "ANTHROPIC_API_KEY" in os.environ}})
log.write_text(json.dumps(calls, ensure_ascii=False))
if os.environ.get("FAKE_CLAUDE_AUTH_FAIL"):
    print(json.dumps({{"type": "result", "subtype": "success", "is_error": True,
                      "result": "Invalid API key · Please run /login"}})); sys.exit(1)
if os.environ.get("FAKE_CLAUDE_OLD") and "--json-schema" in args:
    sys.stderr.write("error: unknown option '--json-schema'\n"); sys.exit(1)
import re
ids = sorted({{int(x) for x in re.findall(r"^#(\d+) ·", prompt, re.M)}})
result = {{"title": "아침 출근길", "selected_ids": ids[:2],
          "candidates": [{{"id": i, "score": 10 - i % 10, "reason": f"장면{{i}}", "caption": f"자막{{i}}"}} for i in ids]}}
out = {{"type": "result", "subtype": "success", "is_error": False, "result": json.dumps(result, ensure_ascii=False)}}
if "--json-schema" in args:
    out["structured_output"] = result
print(json.dumps(out, ensure_ascii=False))
'''


def _setup_videos(tmp: Path) -> list:
    src = tmp / "VID_20261024_090000.mp4"
    _create_synthetic_driving_video(src, duration_sec=32, fps=20)
    v = probe_video_file(src)
    v.segments = analyze_video(v, AnalysisConfig())
    v.analyzed = True
    return [v]


def test_claude_code_backend(videos: list, tmp: Path) -> None:
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "claude"
    fake.write_text(FAKE_CLAUDE.format(python=sys.executable), encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    log = tmp / "claude_calls.json"
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
    os.environ["FAKE_CLAUDE_LOG"] = str(log)
    saved_key = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-stale-key"  # 구독 모드에서는 Claude Code에 전달되면 안 됨
    try:
        hcfg = HighlightConfig(target_duration_sec=4.0, clip_duration_sec=2.0)
        clips, title = ai_select_highlights(videos, hcfg, AiDirectorMode.CLAUDE_CODE)
        calls = json.loads(log.read_text())
        assert len(calls) == 1
        args = calls[0]["args"]
        for flag in ("-p", "--output-format", "--json-schema", "--tools", "--allowedTools"):
            assert flag in args, f"missing {flag}: {args}"
        assert json.loads(args[args.index("--json-schema") + 1]) == RESULT_SCHEMA
        assert any(f.endswith(".jpg") for f in calls[0]["files"]), "contact sheets must be in the CLI working dir"
        assert "candidates_01.jpg" in calls[0]["prompt"], "prompt must reference sheet file names"
        assert not calls[0]["has_api_key"], "ANTHROPIC_API_KEY must be stripped in subscription mode"
        used = [c for c in clips if c.included]
        assert title == "아침 출근길" and len(used) == 2, (title, len(used))
        assert all(c.caption.startswith("자막") for c in clips)
        print(f"[PASS] Claude Code backend: {len(clips)} candidates, {len(used)} picked, title='{title}'")

        # 구버전 CLI(--json-schema 미지원) → 더 단순한 옵션으로 자동 재시도
        log.unlink()
        os.environ["FAKE_CLAUDE_OLD"] = "1"
        clips2, _ = ai_select_highlights(videos, hcfg, AiDirectorMode.CLAUDE_CODE)
        calls = json.loads(log.read_text())
        assert len(calls) == 2 and "--json-schema" not in calls[1]["args"]
        assert sum(c.included for c in clips2) == 2
        print("[PASS] Claude Code backend retries without unsupported options (older CLI)")

        # 로그인 오류: 실제 메시지·사용한 경로·해결 방법을 보여줘야 함
        os.environ.pop("FAKE_CLAUDE_OLD", None)
        os.environ["FAKE_CLAUDE_AUTH_FAIL"] = "1"
        try:
            ai_select_highlights(videos, hcfg, AiDirectorMode.CLAUDE_CODE)
            raise AssertionError("expected AiDirectorError")
        except AiDirectorError as exc:
            msg = str(exc)
            assert "Invalid API key" in msg and str(fake) in msg and "/login" in msg and "해결 방법" in msg, msg
        print("[PASS] Claude Code login error shows real message, claude path and fix steps")
    finally:
        os.environ["PATH"] = old_path
        os.environ.pop("FAKE_CLAUDE_OLD", None)
        os.environ.pop("FAKE_CLAUDE_AUTH_FAIL", None)
        if saved_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = saved_key


def test_claude_api_backend(videos: list, tmp: Path) -> None:
    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured["path"] = self.path
            captured["headers"] = dict(self.headers)
            captured["body"] = body
            n = sum(1 for blk in body["messages"][0]["content"] if blk["type"] == "image")
            result = {"title": "해안 드라이브", "selected_ids": [1, 2],
                      "candidates": [{"id": 1, "score": 9, "reason": "바다 풍경", "caption": "바다다"},
                                     {"id": 2, "score": 8, "reason": "커브길", "caption": "커브"}]}
            msg = {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                   "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                   "stop_reason": "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": 100 * n, "output_tokens": 50}}
            data = json.dumps(msg).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    saved = {k: os.environ.get(k) for k in ("ANTHROPIC_BASE_URL", "NO_PROXY", "no_proxy")}
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{server.server_port}"
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    try:
        hcfg = HighlightConfig(target_duration_sec=4.0, clip_duration_sec=2.0)
        clips, title = ai_select_highlights(videos, hcfg, AiDirectorMode.CLAUDE_API, api_key="sk-test")
        body = captured["body"]
        assert captured["path"].startswith("/v1/messages"), captured["path"]
        assert body["model"] == "claude-opus-5-5"
        assert body["fallbacks"] == "default"
        assert body["output_config"]["format"]["schema"] == RESULT_SCHEMA
        assert "server-side-fallback-2026-07-01" in captured["headers"].get("anthropic-beta", "")
        imgs = [b for b in body["messages"][0]["content"] if b["type"] == "image"]
        assert imgs and imgs[0]["source"]["media_type"] == "image/jpeg"
        assert title == "해안 드라이브" and sum(c.included for c in clips) == 2
        print(f"[PASS] Claude API backend request shape ({len(imgs)} contact-sheet images, fallbacks, JSON schema)")

        # AI 자막을 넣은 세로 쇼츠 렌더링
        out = tmp / "ai_shorts.mp4"
        render_highlight_shorts(clips, ShortsExportConfig(output_path=out, title_text=title,
                                                           show_captions=True, resolution="720p"))
        p = probe_video_file(out)
        assert (p.width, p.height) == (720, 1280) and 3.0 <= p.duration_sec <= 5.0, (p.width, p.height, p.duration_sec)
        print("[PASS] Shorts render with AI captions")
    finally:
        server.shutdown()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    test_apply_ai_result()
    test_candidate_counts()
    test_parse_cli_output()
    with tempfile.TemporaryDirectory(prefix="qdc_ai_") as t:
        tmp_dir = Path(t)
        vids = _setup_videos(tmp_dir)
        if os.name != "nt":
            test_claude_code_backend(vids, tmp_dir)
        test_claude_api_backend(vids, tmp_dir)
    print("ALL AI DIRECTOR TESTS PASSED!")

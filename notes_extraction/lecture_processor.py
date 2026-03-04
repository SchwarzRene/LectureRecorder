#!/usr/bin/env python3
"""
TU Wien Lecture Processor  v2
==============================
Generates structured, exam-ready Markdown notes from a recorded lecture video.

Pipeline:
  1. Audio  → Whisper large-v3  → timestamped transcript
  2. Smart frame extraction (ignores professor movement via 3x3 grid analysis)
  3. Classify each keyframe: BOARD | PDF_SLIDE | OTHER
  4. Qwen2.5-VL reads board math / PDF content per frame
  5. Align frames to transcript timestamps
  6. LLM synthesises everything into:
       - Executive summary (what happened in the lecture)
       - Topic sections with headers
       - Definitions and theorems highlighted in blockquotes
       - Key formulas collected at the top
       - Exam tips per topic
       - Full chronological notes with embedded screenshots
       - Glossary of new terms

Usage:
    python lecture_processor.py lecture.mp4 --api-key YOUR_TOGETHER_KEY
    python lecture_processor.py lecture.mp4 --qwen-api local
    python lecture_processor.py lecture.mp4 --summarize-api claude --summarize-key YOUR_ANTHROPIC_KEY
    python lecture_processor.py lecture.mp4 --skip-vision   # transcript + frames only

Requirements:
    pip install faster-whisper opencv-python numpy Pillow requests "scenedetect[opencv]" tqdm

Vision API options  (--qwen-api):
  together    -> together.ai  (recommended, ~$0.001/frame, free $1 credit on signup)
  hyperbolic  -> hyperbolic.ai
  local       -> local ollama (fully offline, needs GPU)

Summarisation API  (--summarize-api):
  together    -> Llama-3.3-70B on together.ai  (same key as vision, default)
  claude      -> Claude claude-sonnet-4-20250514 via Anthropic  (--summarize-key YOUR_ANTHROPIC_KEY)
  local       -> local ollama
  none        -> skip summarisation (raw notes only)
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests
from tqdm import tqdm

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("[warn] scenedetect not installed – falling back to basic frame sampler")

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[warn] faster-whisper not installed – transcription will be skipped")


# ══════════════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════════════

FrameType = Literal["board", "pdf_slide", "other"]

@dataclass
class Keyframe:
    index: int
    timestamp: float
    path: Path
    frame_type: FrameType = "other"
    vision_text: str = ""
    latex_math: list = field(default_factory=list)
    importance: int = 1   # 0=low, 1=medium, 2=high, 3=critical

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Transcription
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio(video_path: Path, model_size: str, language) -> list:
    if not WHISPER_AVAILABLE:
        print("[skip] faster-whisper not available")
        return []
    print(f"\n[1/5] Transcribing audio with Whisper {model_size}...")
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments_raw, info = model.transcribe(
        str(video_path), language=language,
        beam_size=5, vad_filter=True, word_timestamps=False,
    )
    segments = []
    for seg in tqdm(segments_raw, desc="  Whisper"):
        segments.append(TranscriptSegment(seg.start, seg.end, seg.text.strip()))
    print(f"  Language: {info.language} ({info.language_probability:.0%})  |  Segments: {len(segments)}")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Keyframe extraction
# ══════════════════════════════════════════════════════════════════════════════

def _lap_var(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _is_prof_movement(prev, curr: np.ndarray, threshold: float = 0.15) -> bool:
    """
    Returns True if the change between frames is localised to <=2 of 9 grid cells,
    indicating the professor moved rather than new content appearing on board/screen.
    """
    if prev is None:
        return False
    h, w = curr.shape[:2]
    ch, cw = h // 3, w // 3
    active = 0
    for r in range(3):
        for c in range(3):
            p = cv2.cvtColor(prev[r*ch:(r+1)*ch, c*cw:(c+1)*cw], cv2.COLOR_BGR2GRAY).astype(float)
            q = cv2.cvtColor(curr[r*ch:(r+1)*ch, c*cw:(c+1)*cw], cv2.COLOR_BGR2GRAY).astype(float)
            if np.abs(q - p).mean() / 255.0 > threshold:
                active += 1
    return active <= 2

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a.flatten(), b.flatten()) / d) if d else 1.0

def _thumb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (64, 64)).astype(float) if img is not None else None

def deduplicate(frames: list, sim: float = 0.92) -> list:
    if not frames:
        return frames
    kept = [frames[0]]
    prev = _thumb(frames[0].path)
    for kf in frames[1:]:
        t = _thumb(kf.path)
        if t is not None and prev is not None and _ncc(prev, t) > sim:
            continue
        kept.append(kf)
        prev = t
    return kept

def extract_keyframes(video_path: Path, output_dir: Path) -> list:
    print("\n[2/5] Extracting keyframes...")
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    frames = _extract_scenedetect(video_path, output_dir, fps) if SCENEDETECT_AVAILABLE \
             else _extract_fallback(video_path, output_dir, fps)
    frames = deduplicate(frames)
    print(f"  Kept {len(frames)} keyframes after dedup")
    return frames

def _extract_scenedetect(video_path: Path, output_dir: Path, fps: float) -> list:
    print("  Using PySceneDetect...")
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27.0, min_scene_len=30))
    sm.detect_scenes(video, show_progress=True)
    scene_list = sm.get_scene_list()
    cap = cv2.VideoCapture(str(video_path))
    frames, prev = [], None
    for i, (start_time, _) in enumerate(tqdm(scene_list, desc="  Frames")):
        fn = int(start_time.get_frames())
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret: continue
        if _is_prof_movement(prev, frame) or _lap_var(frame) < 50:
            prev = frame; continue
        p = output_dir / f"frame_{i:04d}_{fn}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames.append(Keyframe(fn, fn / fps, p))
        prev = frame
    cap.release()
    return frames

def _extract_fallback(video_path: Path, output_dir: Path, fps: float, interval: float = 8.0) -> list:
    print(f"  Sampling every {interval:.0f}s...")
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * interval)
    frames, prev, fn = [], None, 0
    with tqdm(total=total // max(step,1), desc="  Frames") as pb:
        while fn < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret: break
            if not _is_prof_movement(prev, frame) and _lap_var(frame) > 50:
                p = output_dir / f"frame_{len(frames):04d}_{fn}.jpg"
                cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                frames.append(Keyframe(fn, fn / fps, p))
                prev = frame
            fn += step; pb.update(1)
    cap.release()
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Classification: BOARD vs PDF_SLIDE vs OTHER
# ══════════════════════════════════════════════════════════════════════════════

def classify_frames(frames: list) -> list:
    print("\n[3/5] Classifying frames...")
    for kf in tqdm(frames, desc="  Classifying"):
        img = cv2.imread(str(kf.path))
        if img is None:
            kf.frame_type = "other"; continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bright_frac = (gray > 200).mean()
        dark_frac   = (gray < 60).mean()
        mean_bright = gray.mean()
        lap         = _lap_var(img)
        hue_bins    = (np.bincount(hsv[:,:,0].flatten(), minlength=180) > 50).sum()

        if bright_frac > 0.45 and lap > 80:
            kf.frame_type = "pdf_slide"
        elif dark_frac > 0.30 and mean_bright < 100:
            kf.frame_type = "board"
        elif mean_bright > 160 and hue_bins < 40:
            kf.frame_type = "board"
        elif bright_frac > 0.55:
            kf.frame_type = "other"
        else:
            kf.frame_type = "board" if lap > 60 else "other"

    counts = {t: sum(1 for kf in frames if kf.frame_type == t) for t in ("board","pdf_slide","other")}
    print(f"  board={counts['board']}  pdf_slide={counts['pdf_slide']}  other={counts['other']}")
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Vision analysis (Qwen2.5-VL)
# ══════════════════════════════════════════════════════════════════════════════

BOARD_PROMPT = """You are analysing a blackboard/whiteboard frame from a university mathematics lecture at TU Wien.

Extract ALL content written on the board:
1. Every mathematical formula and equation -> write in LaTeX ($...$ inline, $$...$$ for display)
2. All text, labels, annotations -> transcribe exactly
3. Any diagrams -> describe briefly

Also rate the importance of this frame:
- CRITICAL: contains a definition, theorem, proof, or key formula likely to be examined
- HIGH: important derivation steps or key examples
- MEDIUM: supporting calculations or examples
- LOW: minor notes or intermediate steps

Output EXACTLY in this format (keep the section headers):
IMPORTANCE: <CRITICAL|HIGH|MEDIUM|LOW>

MATH:
<all formulas in LaTeX, one per line>

TEXT:
<all non-math written text>

DIAGRAM:
<brief description, or "none">
"""

PDF_PROMPT = """You are analysing a PDF/slide frame from a university mathematics lecture at TU Wien.

Extract:
1. Slide title
2. All text content
3. All mathematical formulas -> write in LaTeX
4. Any figures or graphs -> describe

Rate the importance:
- CRITICAL: theorem, definition, key formula, or exam-relevant result
- HIGH: important concept or worked example
- MEDIUM: supporting material
- LOW: administrative or background info

Output EXACTLY in this format:
IMPORTANCE: <CRITICAL|HIGH|MEDIUM|LOW>

TITLE: <title or "untitled">

CONTENT:
<all text and math from the slide>

FIGURES:
<description, or "none">
"""

def _call_vision(qwen_api: str, api_key, local_url: str, prompt: str, image_path: Path) -> str:
    img_b64 = encode_image(image_path)
    msg = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": prompt},
    ]
    if qwen_api == "together":
        r = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "Qwen/Qwen2.5-VL-72B-Instruct", "max_tokens": 1500,
                  "messages": [{"role": "user", "content": msg}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    elif qwen_api == "hyperbolic":
        r = requests.post(
            "https://api.hyperbolic.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "Qwen/Qwen2.5-VL-72B-Instruct", "max_tokens": 1500,
                  "messages": [{"role": "user", "content": msg}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    elif qwen_api == "local":
        r = requests.post(
            f"{local_url}/api/chat",
            json={"model": "qwen2.5vl:7b", "stream": False,
                  "messages": [{"role": "user", "content": msg}],
                  "options": {"num_predict": 1500}},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    raise ValueError(f"Unknown vision API: {qwen_api}")

def _parse_importance(text: str) -> int:
    m = re.search(r'IMPORTANCE:\s*(CRITICAL|HIGH|MEDIUM|LOW)', text, re.IGNORECASE)
    if not m: return 1
    return {"critical": 3, "high": 2, "medium": 1, "low": 0}[m.group(1).lower()]

def analyse_frames_with_vision(frames: list, qwen_api: str, api_key, local_url: str, delay: float = 0.5) -> list:
    print("\n[4/5] Running vision analysis...")
    todo = [kf for kf in frames if kf.frame_type in ("board", "pdf_slide")]
    print(f"  Analysing {len(todo)} frames")
    for kf in tqdm(todo, desc="  Vision API"):
        prompt = BOARD_PROMPT if kf.frame_type == "board" else PDF_PROMPT
        try:
            result = _call_vision(qwen_api, api_key, local_url, prompt, kf.path)
            kf.vision_text = result
            kf.importance = _parse_importance(result)
            kf.latex_math = re.findall(r'\$\$.*?\$\$|\$[^$]+\$', result, re.DOTALL)
        except Exception as e:
            print(f"\n  [warn] Vision failed for {kf.path.name}: {e}")
            kf.vision_text = f"[Vision failed: {e}]"
        time.sleep(delay)
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Step 5a – AI Summarisation
# ══════════════════════════════════════════════════════════════════════════════

SUMMARY_SYSTEM = """You are an expert academic note-taker and tutor for a mathematics student at TU Wien (Vienna University of Technology).
Analyse lecture content and produce clear, structured, exam-focused notes.
Write in the same language as the lecture (German or English – detect automatically).
Be precise with mathematical notation – always use LaTeX."""

SUMMARY_PROMPT = """Below is the full content of a university mathematics lecture extracted from a video recording.
It includes a timestamped transcript and descriptions of board/slide content.

Produce a JSON response with this exact structure:

{{
  "lecture_title": "inferred title",
  "executive_summary": "3-5 sentence overview: what was taught, what are the main results",
  "topics": [
    {{
      "title": "topic name",
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "summary": "2-3 sentence summary",
      "key_points": ["point 1", "point 2"],
      "definitions": ["TermName: explanation"],
      "theorems": ["TheoremName: statement, use LaTeX for math"],
      "exam_tips": ["what students should memorise or watch out for"],
      "important_frame_indices": [list of integer frame indices most relevant to this topic]
    }}
  ],
  "key_formulas": [
    {{"name": "formula name", "latex": "$$...$$", "description": "what it means / when to use it"}}
  ],
  "glossary": [
    {{"term": "term", "definition": "plain-language definition"}}
  ],
  "what_to_study": ["3-5 most important things to focus on for an exam on this material"]
}}

LECTURE CONTENT:
{content}

Respond with ONLY valid JSON. No markdown fences, no extra text before or after.
"""

def build_context(frames: list, segments: list) -> str:
    lines = []
    chunk, chunk_start = [], 0.0
    for seg in segments:
        chunk.append(seg.text)
        if seg.end - chunk_start > 60:
            lines.append(f"[{fmt_time(chunk_start)}-{fmt_time(seg.end)}] " + " ".join(chunk))
            chunk, chunk_start = [], seg.end
    if chunk:
        lines.append(f"[{fmt_time(chunk_start)}] " + " ".join(chunk))

    lines.append("\n--- BOARD / SLIDE CONTENT ---\n")
    for i, kf in enumerate(frames):
        if kf.frame_type == "other" or not kf.vision_text:
            continue
        imp = {3:"CRITICAL", 2:"HIGH", 1:"MEDIUM", 0:"LOW"}.get(kf.importance, "MEDIUM")
        lines.append(f"\n[Frame {i} @ {fmt_time(kf.timestamp)}] [{kf.frame_type.upper()}] [{imp}]")
        lines.append(kf.vision_text[:800])
    return "\n".join(lines)

def summarize_lecture(frames: list, segments: list, summarize_api: str, api_key, summarize_key, local_url: str):
    if summarize_api == "none":
        return None
    print("\n[5a/5] Summarising lecture with AI...")
    context = build_context(frames, segments)
    prompt = SUMMARY_PROMPT.format(content=context[:28000])

    try:
        if summarize_api == "together":
            r = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "max_tokens": 4000,
                      "temperature": 0.2,
                      "messages": [{"role":"system","content":SUMMARY_SYSTEM},
                                   {"role":"user","content":prompt}]},
                timeout=120,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]

        elif summarize_api == "claude":
            key = summarize_key or api_key
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 4000,
                      "system": SUMMARY_SYSTEM,
                      "messages": [{"role":"user","content":prompt}]},
                timeout=120,
            )
            r.raise_for_status()
            raw = r.json()["content"][0]["text"]

        elif summarize_api == "local":
            r = requests.post(
                f"{local_url}/api/chat",
                json={"model": "llama3.3:70b", "stream": False,
                      "messages": [{"role":"system","content":SUMMARY_SYSTEM},
                                   {"role":"user","content":prompt}],
                      "options": {"num_predict": 4000, "temperature": 0.2}},
                timeout=300,
            )
            r.raise_for_status()
            raw = r.json()["message"]["content"]
        else:
            return None

        clean = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        clean = re.sub(r'\s*```$', '', clean)
        result = json.loads(clean)
        print(f"  Topics found: {len(result.get('topics', []))}")
        return result

    except Exception as e:
        print(f"  [warn] Summarisation failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Step 5b – Markdown generation
# ══════════════════════════════════════════════════════════════════════════════

def _parse_vision_fields(kf: Keyframe) -> dict:
    text = kf.vision_text
    out = {"math":"", "content":"", "diagram":"", "title":""}
    m = re.search(r'MATH:\s*(.*?)(?=TEXT:|CONTENT:|DIAGRAM:|TITLE:|FIGURES:|IMPORTANCE:|$)', text, re.DOTALL)
    if m: out["math"] = m.group(1).strip()
    m = re.search(r'(?:TEXT:|CONTENT:)\s*(.*?)(?=MATH:|DIAGRAM:|FIGURES:|IMPORTANCE:|$)', text, re.DOTALL)
    if m: out["content"] = m.group(1).strip()
    m = re.search(r'(?:DIAGRAM:|FIGURES:)\s*(.*?)(?=MATH:|TEXT:|CONTENT:|IMPORTANCE:|$)', text, re.DOTALL)
    if m: out["diagram"] = m.group(1).strip()
    m = re.search(r'TITLE:\s*(.+)', text)
    if m: out["title"] = m.group(1).strip()
    return out

def _frame_block(L: list, kf: Keyframe, img_dir: str, heading: str = "###"):
    imp_badge = {3: "  ⚠️ **CRITICAL**", 2: "  🔶 **Important**", 1: "  🔹 Notable", 0: ""}.get(kf.importance, "")
    type_label = "📋 PDF Slide" if kf.frame_type == "pdf_slide" else "📝 Board"
    anchor = fmt_time(kf.timestamp).replace(":", "")

    L.append(f"{heading} {type_label} @ {fmt_time(kf.timestamp)}{imp_badge} {{#{anchor}}}")
    L.append("")
    L.append(f"![{type_label} @ {fmt_time(kf.timestamp)}]({img_dir}/{kf.path.name})")
    L.append("")

    if kf.vision_text and "[Vision failed" not in kf.vision_text:
        v = _parse_vision_fields(kf)

        if v["title"] and kf.frame_type == "pdf_slide" and v["title"].lower() != "untitled":
            L.append(f"**{v['title']}**"); L.append("")

        if v["math"]:
            # Critical math gets blockquote treatment so it stands out visually
            if kf.importance == 3:
                L.append("> **Mathematical content:**"); L.append(">")
                for line in v["math"].splitlines():
                    L.append(f"> {line}")
            else:
                L.append("**Mathematical content:**"); L.append("")
                L.append(v["math"])
            L.append("")

        if v["content"]:
            L.append("**Written content:**"); L.append("")
            L.append(v["content"]); L.append("")

        if v["diagram"] and v["diagram"].lower() not in ("none", ""):
            L.append(f"**Diagram:** {v['diagram']}"); L.append("")
    else:
        L.append(f"*{kf.vision_text or 'No vision analysis available'}*"); L.append("")

    L += ["---", ""]


def generate_notes(frames: list, segments: list, summary, output_dir: Path, video_name: str) -> Path:
    print("\n[5b/5] Generating Markdown notes...")
    notes_path = output_dir / f"{video_name}_notes.md"
    img_dir = "frames"
    L = []

    has_frames   = any(kf.frame_type != "other" for kf in frames)
    has_summary  = summary is not None
    has_topics   = has_summary and bool(summary.get("topics"))
    has_formulas = has_summary and bool(summary.get("key_formulas"))
    has_glossary = has_summary and bool(summary.get("glossary"))

    # ── Title & metadata ───────────────────────────────────────────────────────
    title = summary.get("lecture_title", video_name) if has_summary else video_name
    duration = fmt_time(segments[-1].end) if segments else "unknown"
    n_frames  = sum(1 for kf in frames if kf.frame_type != "other")
    n_crit    = sum(1 for kf in frames if kf.importance == 3)

    L += [
        f"# {title}",
        "",
        "| | |",
        "|---|---|",
        f"| **Video** | `{video_name}` |",
        f"| **Duration** | {duration} |",
        f"| **Transcript segments** | {len(segments)} |",
        f"| **Frames captured** | {n_frames} ({n_crit} critical) |",
        "",
    ]

    # ── Table of contents ──────────────────────────────────────────────────────
    toc = []
    if has_summary:   toc.append("- [Lecture Summary](#lecture-summary)")
    if has_topics:    toc.append("- [Detailed Notes by Topic](#detailed-notes-by-topic)")
    toc.append("- [Chronological Notes](#chronological-notes)")
    if has_glossary:  toc.append("- [Glossary](#glossary)")
    if segments:      toc.append("- [Full Transcript](#full-transcript)")

    L += ["## Contents", ""] + toc + ["", "---", ""]

    # ── Executive Summary ──────────────────────────────────────────────────────
    if has_summary:
        L += ["## Lecture Summary", "", summary.get("executive_summary", ""), ""]

        study = summary.get("what_to_study", [])
        if study:
            L += ["### 🎯 What to Focus On", ""]
            for item in study:
                L.append(f"- **{item}**")
            L.append("")

        if has_formulas:
            L += ["### 🔑 Key Formulas", ""]
            for f in summary["key_formulas"]:
                name  = f.get("name", "")
                latex = f.get("latex", "")
                desc  = f.get("description", "")
                L += [f"**{name}**", "", latex, ""]
                if desc:
                    L += [f"> {desc}", ""]

        L += ["---", ""]

    # ── Per-topic sections ─────────────────────────────────────────────────────
    if has_topics:
        L += ["## Detailed Notes by Topic", ""]
        used_frames = set()

        for topic in summary["topics"]:
            start = topic.get("start_time", "")
            end   = topic.get("end_time", "")
            L += [
                f"### {topic['title']}",
                f"*{start} – {end}*",
                "",
                topic.get("summary", ""),
                "",
            ]

            defs = topic.get("definitions", [])
            if defs:
                L += ["> **📝 Definitions**", ">"]
                for d in defs:
                    L += [f"> **{d}**", ">"]
                L.append("")

            thms = topic.get("theorems", [])
            if thms:
                L += ["> **⚠️ Theorems & Results**", ">"]
                for t in thms:
                    L += [f"> {t}", ">"]
                L.append("")

            kps = topic.get("key_points", [])
            if kps:
                L.append("**Key points:**")
                for kp in kps:
                    L.append(f"- {kp}")
                L.append("")

            tips = topic.get("exam_tips", [])
            if tips:
                for tip in tips:
                    L.append(f"> 💡 **Exam tip:** {tip}")
                L.append("")

            for fi in topic.get("important_frame_indices", []):
                if fi < len(frames) and fi not in used_frames and frames[fi].frame_type != "other":
                    used_frames.add(fi)
                    _frame_block(L, frames[fi], img_dir, heading="####")

        L += ["---", ""]

    # ── Chronological notes ────────────────────────────────────────────────────
    L += ["## Chronological Notes", ""]

    if not has_frames and not segments:
        L += ["> *No content found in this video.*", ""]

    elif not has_frames:
        # Transcript-only: group into readable paragraphs by time (every ~2 min)
        L += [
            "> *No board or slide content was detected in this video.*  ",
            "> *Showing transcript only.*",
            "",
        ]
        para, para_start = [], segments[0].start if segments else 0.0
        for seg in segments:
            para.append(seg.text)
            # Flush every ~2 minutes or at natural sentence boundaries
            if seg.end - para_start >= 120 or seg.text.rstrip().endswith((".", "!", "?")):
                if len(para) >= 3:  # only flush once we have enough content
                    L.append(f"**[{fmt_time(para_start)}]**")
                    L.append("")
                    L.append(" ".join(para))
                    L.append("")
                    para, para_start = [], seg.end
        if para:
            L.append(f"**[{fmt_time(para_start)}]**")
            L.append("")
            L.append(" ".join(para))
            L.append("")

    else:
        # Interleave frames + transcript
        events = sorted(
            [(kf.timestamp, "frame", i, kf) for i, kf in enumerate(frames) if kf.frame_type != "other"] +
            [(seg.start, "seg", None, seg) for seg in segments],
            key=lambda x: x[0],
        )
        pending, seen = [], set()

        for ts, kind, idx, obj in events:
            if kind == "seg":
                pending.append(obj.text)
            else:
                kf = obj
                if idx in seen: continue
                seen.add(idx)
                if pending:
                    block = " ".join(pending).strip()
                    if block:
                        L.append(f"> *{block}*"); L.append("")
                    pending = []
                _frame_block(L, kf, img_dir, heading="###")

        if pending:
            L.append(f"> *{' '.join(pending).strip()}*"); L.append("")

    # ── Glossary ───────────────────────────────────────────────────────────────
    if has_glossary:
        L += ["---", "", "## Glossary", ""]
        for entry in summary["glossary"]:
            L += [f"**{entry.get('term','')}** — {entry.get('definition','')}", ""]

    # ── Full transcript ────────────────────────────────────────────────────────
    if segments:
        L += ["---", "", "## Full Transcript", ""]
        # Group consecutive segments into lines for readability
        current_line, line_start = [], segments[0].start
        for i, seg in enumerate(segments):
            current_line.append(seg.text.strip())
            # New paragraph every ~30s or at sentence end
            is_last = i == len(segments) - 1
            ends_sentence = seg.text.rstrip().endswith((".", "?", "!"))
            long_enough = seg.end - line_start >= 30
            if (ends_sentence and long_enough) or is_last:
                L.append(f"**[{fmt_time(line_start)}]** {' '.join(current_line)}")
                L.append("")
                current_line, line_start = [], seg.end

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    return notes_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TU Wien Lecture Processor v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to lecture video (mp4, mkv, ...)")
    parser.add_argument("--whisper-model", default="large-v3",
                        choices=["tiny","base","small","medium","large-v2","large-v3"])
    parser.add_argument("--lang", default=None, help="Force language: 'de' or 'en'")
    parser.add_argument("--qwen-api", default="together", choices=["together","hyperbolic","local"])
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"),
                        help="together.ai / hyperbolic key (or set QWEN_API_KEY env var)")
    parser.add_argument("--summarize-api", default="together",
                        choices=["together","claude","local","none"],
                        help="API for the summarisation step (default: together = Llama-3.3-70B)")
    parser.add_argument("--summarize-key", default=os.environ.get("SUMMARIZE_API_KEY"),
                        help="Separate key for summarisation (e.g. your Anthropic key if --summarize-api claude)")
    parser.add_argument("--local-url", default="http://localhost:11434")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision API")
    parser.add_argument("--skip-transcription", action="store_true", help="Skip Whisper")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"[error] File not found: {video_path}"); sys.exit(1)

    video_name = video_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else video_path.parent / f"{video_name}_notes"
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TU Wien Lecture Processor  v2")
    print(f"  Video  : {video_path.name}")
    print(f"  Output : {output_dir}")
    print("=" * 60)

    if not args.skip_vision and args.qwen_api in ("together","hyperbolic") and not args.api_key:
        print(f"[error] --api-key required for --qwen-api={args.qwen_api}")
        print("  Get a free key at https://api.together.xyz  (~$0.001/frame, $1 free credit)")
        print("  Or use --qwen-api local  or  --skip-vision")
        sys.exit(1)

    # Pipeline
    segments = []
    if not args.skip_transcription:
        segments = transcribe_audio(video_path, args.whisper_model, args.lang)
        tp = output_dir / f"{video_name}_transcript.txt"
        with open(tp, "w", encoding="utf-8") as f:
            for s in segments:
                f.write(f"[{fmt_time(s.start)}] {s.text}\n")
        print(f"  Transcript saved -> {tp.name}")

    frames = extract_keyframes(video_path, frames_dir)
    frames = classify_frames(frames)

    if not args.skip_vision:
        frames = analyse_frames_with_vision(frames, args.qwen_api, args.api_key, args.local_url)

    summary = summarize_lecture(
        frames, segments,
        summarize_api=args.summarize_api,
        api_key=args.api_key,
        summarize_key=args.summarize_key,
        local_url=args.local_url,
    )

    if summary:
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    notes_path = generate_notes(frames, segments, summary, output_dir, video_name)

    meta = [{"index": kf.index, "timestamp": fmt_time(kf.timestamp),
              "type": kf.frame_type, "importance": kf.importance,
              "file": kf.path.name} for kf in frames]
    with open(output_dir / "frames_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  Notes    -> {notes_path}")
    print(f"  Frames   -> {frames_dir}  ({len(frames)} images)")
    if summary:
        print(f"  Topics   -> {len(summary.get('topics', []))} sections")
        print(f"  Formulas -> {len(summary.get('key_formulas', []))} key formulas")
    print()
    print("  Tip: open notes in Obsidian for LaTeX + image rendering")
    print("=" * 60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
TU Wien Lecture Processor
=========================
Generates structured Markdown notes from a recorded lecture video.

Pipeline:
  1. Extract audio  → Whisper large-v3  → timestamped transcript
  2. Smart frame extraction via scene-change detection (ignores professor movement)
  3. Classify each keyframe: BOARD | PDF_SLIDE | OTHER
  4. Vision model (Qwen2.5-VL via API) reads board math / PDF content per frame
  5. Align frames to transcript timestamps
  6. LLM synthesises everything into structured Markdown notes with embedded images

Usage:
    python lecture_processor.py <video_file> [options]

    python lecture_processor.py lecture.mp4
    python lecture_processor.py lecture.mp4 --whisper-model large-v3 --lang de
    python lecture_processor.py lecture.mp4 --qwen-api together  --api-key YOUR_KEY
    python lecture_processor.py lecture.mp4 --qwen-api local     --local-url http://localhost:8000

Requirements:
    pip install faster-whisper opencv-python numpy Pillow requests scenedetect[opencv] tqdm

For Qwen2.5-VL you have three options:
  A) together.ai API  (--qwen-api together  --api-key <key>)   fast, cheap (~$0.001/frame)
  B) hyperbolic.ai    (--qwen-api hyperbolic --api-key <key>)   alternative
  C) local ollama     (--qwen-api local)                         fully offline, needs GPU
"""

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# ─── optional: scenedetect ────────────────────────────────────────────────────
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector, ThresholdDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("[warn] scenedetect not installed – falling back to basic frame diff")

# ─── optional: faster-whisper ─────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[warn] faster-whisper not installed – transcript step will be skipped")


# ══════════════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════════════

FrameType = Literal["board", "pdf_slide", "other"]

@dataclass
class Keyframe:
    index: int                        # frame number in video
    timestamp: float                  # seconds from start
    path: Path                        # saved image path
    frame_type: FrameType = "other"
    vision_text: str = ""             # raw OCR/description from Qwen
    latex_math: list[str] = field(default_factory=list)

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Audio transcription (Whisper)
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio(video_path: Path, model_size: str, language: str | None) -> list[TranscriptSegment]:
    if not WHISPER_AVAILABLE:
        print("[skip] faster-whisper not available, skipping transcription")
        return []

    print(f"\n[1/4] Transcribing audio with Whisper {model_size}…")
    model = WhisperModel(model_size, device="auto", compute_type="auto")

    segments_raw, info = model.transcribe(
        str(video_path),
        language=language,
        beam_size=5,
        vad_filter=True,               # skip silence
        word_timestamps=False,
    )

    segments = []
    for seg in tqdm(segments_raw, desc="  Whisper segments"):
        segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
        ))

    detected_lang = info.language
    print(f"  Detected language: {detected_lang} (confidence {info.language_probability:.0%})")
    print(f"  Total segments: {len(segments)}")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Smart keyframe extraction
# ══════════════════════════════════════════════════════════════════════════════

def _laplacian_variance(frame: np.ndarray) -> float:
    """Higher = sharper / more structured content on screen."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _is_professor_movement(prev: np.ndarray, curr: np.ndarray, threshold: float = 0.15) -> bool:
    """
    Detect if the change between two frames is caused by the professor moving
    rather than a meaningful content change on the board/screen.

    Strategy: divide frame into a 3×3 grid.
    If the difference is concentrated in only 1-2 cells (local movement),
    it's likely the professor. If spread across many cells, it's a content change.
    """
    if prev is None:
        return False

    h, w = curr.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    diffs = []
    for row in range(3):
        for col in range(3):
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w
            c_prev = cv2.cvtColor(prev[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
            c_curr = cv2.cvtColor(curr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
            diff = np.abs(c_curr - c_prev).mean() / 255.0
            diffs.append(diff)

    active_cells = sum(1 for d in diffs if d > threshold)
    # ≤2 active cells out of 9 → local movement (professor)
    return active_cells <= 2


def extract_keyframes_scenedetect(video_path: Path, output_dir: Path) -> list[Keyframe]:
    """Use PySceneDetect for robust scene boundary detection."""
    print("  Using PySceneDetect…")
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    # ContentDetector catches cuts + gradual changes; threshold tuned for lectures
    scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=30))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    keyframes = []
    prev_frame = None

    for i, (start_time, _) in enumerate(tqdm(scene_list, desc="  Extracting frames")):
        frame_no = int(start_time.get_frames())
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            continue

        # Skip if this looks like pure professor movement
        if _is_professor_movement(prev_frame, frame):
            prev_frame = frame
            continue

        # Skip blurry frames (motion blur while professor is at board)
        if _laplacian_variance(frame) < 50:
            prev_frame = frame
            continue

        img_path = output_dir / f"frame_{i:04d}_{frame_no}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        keyframes.append(Keyframe(
            index=frame_no,
            timestamp=frame_no / fps,
            path=img_path,
        ))
        prev_frame = frame

    cap.release()
    return keyframes


def extract_keyframes_fallback(video_path: Path, output_dir: Path, interval_sec: float = 8.0) -> list[Keyframe]:
    """Fallback: sample every N seconds, skip professor-movement frames."""
    print("  Using fallback frame sampler (every {:.0f}s)…".format(interval_sec))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * interval_sec)

    keyframes = []
    prev_kept = None
    frame_no = 0

    with tqdm(total=total_frames // step, desc="  Sampling frames") as pbar:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                break

            if not _is_professor_movement(prev_kept, frame) and _laplacian_variance(frame) > 50:
                img_path = output_dir / f"frame_{len(keyframes):04d}_{frame_no}.jpg"
                cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                keyframes.append(Keyframe(
                    index=frame_no,
                    timestamp=frame_no / fps,
                    path=img_path,
                ))
                prev_kept = frame

            frame_no += step
            pbar.update(1)
            if frame_no >= total_frames:
                break

    cap.release()
    return keyframes


def extract_keyframes(video_path: Path, output_dir: Path) -> list[Keyframe]:
    print("\n[2/4] Extracting keyframes…")
    output_dir.mkdir(parents=True, exist_ok=True)
    if SCENEDETECT_AVAILABLE:
        frames = extract_keyframes_scenedetect(video_path, output_dir)
    else:
        frames = extract_keyframes_fallback(video_path, output_dir)

    # Deduplicate very similar consecutive frames (structural similarity)
    frames = deduplicate_frames(frames)
    print(f"  Kept {len(frames)} keyframes after deduplication")
    return frames


def deduplicate_frames(frames: list[Keyframe], similarity_threshold: float = 0.92) -> list[Keyframe]:
    """Remove near-duplicate frames using normalised cross-correlation on thumbnails."""
    if not frames:
        return frames
    kept = [frames[0]]
    prev_thumb = _thumbnail(frames[0].path)

    for kf in frames[1:]:
        thumb = _thumbnail(kf.path)
        if prev_thumb is not None and thumb is not None:
            score = _ncc(prev_thumb, thumb)
            if score > similarity_threshold:
                continue  # too similar – skip
        kept.append(kf)
        prev_thumb = thumb

    return kept


def _thumbnail(path: Path, size: tuple = (64, 64)) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, size).astype(float)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation in [0, 1]."""
    a = a - a.mean(); b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(np.dot(a.flatten(), b.flatten()) / denom)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Frame classification: BOARD vs PDF_SLIDE vs OTHER
# ══════════════════════════════════════════════════════════════════════════════

def classify_frames(frames: list[Keyframe]) -> list[Keyframe]:
    """
    Lightweight local classifier – no GPU needed.

    Heuristics:
      BOARD       – dark/coloured background, bright chalk marks, low colour diversity
      PDF_SLIDE   – white/light background, high colour diversity, sharp edges typical of
                    rendered text (high Laplacian variance on light background)
      OTHER       – everything else (professor face-cam, etc.)
    """
    print("\n[3/4] Classifying frames (board / PDF slide / other)…")

    for kf in tqdm(frames, desc="  Classifying"):
        img = cv2.imread(str(kf.path))
        if img is None:
            kf.frame_type = "other"
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mean_brightness = gray.mean()
        # Value channel spread (how bright the overall scene is)
        v_channel = hsv[:, :, 2]
        brightness_std = v_channel.std()

        # Fraction of very bright pixels (>200) – PDFs tend to have large white areas
        bright_fraction = (gray > 200).mean()

        # Fraction of very dark pixels (<60) – blackboards
        dark_fraction = (gray < 60).mean()

        # Colour diversity (number of unique hues, binned)
        h_channel = hsv[:, :, 0]
        hue_hist = np.bincount(h_channel.flatten(), minlength=180)
        active_hue_bins = (hue_hist > 50).sum()

        lap_var = _laplacian_variance(img)

        # ── Decision rules ────────────────────────────────────────────────────
        if bright_fraction > 0.45 and lap_var > 80:
            # Large white area + structured content → PDF/slide
            kf.frame_type = "pdf_slide"
        elif dark_fraction > 0.30 and mean_brightness < 100:
            # Predominantly dark → blackboard
            kf.frame_type = "board"
        elif mean_brightness > 160 and active_hue_bins < 40:
            # Very bright, low colour variety → whiteboard or bright slide
            kf.frame_type = "board"
        elif bright_fraction > 0.55:
            # Mostly white but low sharpness → likely a blank/transitional frame
            kf.frame_type = "other"
        else:
            # Mixed – treat as board if decently sharp
            kf.frame_type = "board" if lap_var > 60 else "other"

    counts = {t: sum(1 for kf in frames if kf.frame_type == t) for t in ("board", "pdf_slide", "other")}
    print(f"  board={counts['board']}  pdf_slide={counts['pdf_slide']}  other={counts['other']}")
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Vision analysis per frame (Qwen2.5-VL)
# ══════════════════════════════════════════════════════════════════════════════

BOARD_PROMPT = """You are analysing a frame from a mathematics lecture at TU Wien.
The image shows a blackboard or whiteboard with handwritten content.

Please extract ALL content that is written on the board:
1. All mathematical formulas, equations, definitions – write them in LaTeX (use $...$ for inline, $$...$$ for display)
2. Any written text, labels, or annotations (transcribe exactly)
3. Describe any diagrams or figures briefly

Be thorough and accurate. Even if handwriting is messy, do your best.
Output format:
MATH:
<all formulas in LaTeX>

TEXT:
<all non-math text>

DIAGRAM:
<brief description if any diagram present, else "none">
"""

PDF_PROMPT = """You are analysing a frame from a mathematics lecture at TU Wien.
The image shows a PDF document or slide being presented on screen.

Please extract:
1. The slide/page title (if visible)
2. ALL text content visible on the slide
3. Any mathematical formulas (write in LaTeX)
4. Describe any figures, graphs, or diagrams

Output format:
TITLE: <title or "untitled">

CONTENT:
<all text and math from the slide>

FIGURES:
<description of any visual elements, else "none">
"""


def _encode_image(path: Path) -> str:
    """Base64-encode image for API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_together_ai(api_key: str, prompt: str, image_path: Path) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "max_tokens": 1500,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    resp = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_hyperbolic(api_key: str, prompt: str, image_path: Path) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "max_tokens": 1500,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    resp = requests.post("https://api.hyperbolic.xyz/v1/chat/completions", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_local_ollama(base_url: str, prompt: str, image_path: Path) -> str:
    """Call a locally running Qwen2.5-VL via ollama or vllm."""
    payload = {
        "model": "qwen2.5vl:7b",   # adjust to your local model name
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "stream": False,
        "options": {"num_predict": 1500},
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def analyse_frames_with_vision(
    frames: list[Keyframe],
    qwen_api: str,
    api_key: str | None,
    local_url: str,
    delay: float = 0.5,
) -> list[Keyframe]:
    print("\n[4/4] Running vision analysis on board & slide frames…")

    processable = [kf for kf in frames if kf.frame_type in ("board", "pdf_slide")]
    print(f"  Analysing {len(processable)} frames (skipping 'other')")

    for kf in tqdm(processable, desc="  Vision API"):
        prompt = BOARD_PROMPT if kf.frame_type == "board" else PDF_PROMPT

        try:
            if qwen_api == "together":
                result = _call_together_ai(api_key, prompt, kf.path)
            elif qwen_api == "hyperbolic":
                result = _call_hyperbolic(api_key, prompt, kf.path)
            elif qwen_api == "local":
                result = _call_local_ollama(local_url, prompt, kf.path)
            else:
                raise ValueError(f"Unknown API: {qwen_api}")

            kf.vision_text = result
            # Extract LaTeX blocks for later rendering hints
            kf.latex_math = re.findall(r'\$\$.*?\$\$|\$[^$]+\$', result, re.DOTALL)

        except Exception as e:
            print(f"\n  [warn] Vision API failed for {kf.path.name}: {e}")
            kf.vision_text = f"[Vision analysis failed: {e}]"

        time.sleep(delay)   # be polite to rate limits

    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 – Align frames to transcript & generate Markdown notes
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _segments_in_window(segments: list[TranscriptSegment], t_start: float, t_end: float) -> str:
    """Return transcript text for time window [t_start, t_end]."""
    texts = [
        seg.text for seg in segments
        if seg.start >= t_start and seg.end <= t_end
    ]
    return " ".join(texts).strip()


def generate_notes(
    frames: list[Keyframe],
    segments: list[TranscriptSegment],
    output_dir: Path,
    video_name: str,
) -> Path:
    """Combine transcript + vision output into a structured Markdown document."""

    notes_path = output_dir / f"{video_name}_notes.md"
    img_rel_dir = "frames"   # relative path in the notes file

    lines = [
        f"# Lecture Notes – {video_name}",
        "",
        "> Auto-generated by TU Wien Lecture Processor  ",
        f"> Video: `{video_name}`  ",
        "",
        "---",
        "",
        "## Contents",
        "",
    ]

    # Build section list
    content_frames = [kf for kf in frames if kf.frame_type != "other"]
    for i, kf in enumerate(content_frames):
        type_label = "📋 Slide/PDF" if kf.frame_type == "pdf_slide" else "📝 Board"
        lines.append(f"{i+1}. [{type_label} @ {_fmt_time(kf.timestamp)}](#{_fmt_time(kf.timestamp).replace(':', '')})")

    lines += ["", "---", ""]

    # Interleave transcript + frames chronologically
    all_timestamps = sorted(
        [(kf.timestamp, "frame", kf) for kf in frames if kf.frame_type != "other"] +
        [(seg.start, "seg_start", seg) for seg in segments],
        key=lambda x: x[0],
    )

    current_section_start = 0.0
    last_frame_time = 0.0
    pending_transcript: list[str] = []

    processed_frames: set[int] = set()

    for ts, kind, obj in all_timestamps:
        if kind == "seg_start":
            pending_transcript.append(obj.text)

        elif kind == "frame":
            kf: Keyframe = obj
            if kf.index in processed_frames:
                continue
            processed_frames.add(kf.index)

            # Flush accumulated transcript before this frame
            if pending_transcript:
                block = " ".join(pending_transcript).strip()
                if block:
                    lines.append(f"> 🎙️ *{block}*")
                    lines.append("")
                pending_transcript = []

            # Frame anchor + image
            anchor = _fmt_time(kf.timestamp).replace(":", "")
            type_label = "📋 PDF Slide" if kf.frame_type == "pdf_slide" else "📝 Board"
            lines.append(f"### {type_label} @ {_fmt_time(kf.timestamp)} {{#{anchor}}}")
            lines.append("")
            # Embed image (relative path)
            lines.append(f"![Frame at {_fmt_time(kf.timestamp)}]({img_rel_dir}/{kf.path.name})")
            lines.append("")

            # Vision content
            if kf.vision_text and "[Vision analysis failed" not in kf.vision_text:
                # Parse structured output
                text = kf.vision_text

                # Extract MATH section
                math_match = re.search(r'MATH:\s*(.*?)(?=TEXT:|CONTENT:|DIAGRAM:|TITLE:|FIGURES:|$)', text, re.DOTALL)
                content_match = re.search(r'(?:TEXT:|CONTENT:)\s*(.*?)(?=MATH:|DIAGRAM:|FIGURES:|$)', text, re.DOTALL)
                diagram_match = re.search(r'(?:DIAGRAM:|FIGURES:)\s*(.*?)$', text, re.DOTALL)
                title_match = re.search(r'TITLE:\s*(.+)', text)

                if title_match and kf.frame_type == "pdf_slide":
                    title = title_match.group(1).strip()
                    if title and title.lower() != "untitled":
                        lines.append(f"**{title}**")
                        lines.append("")

                if math_match:
                    math_content = math_match.group(1).strip()
                    if math_content:
                        lines.append("**Mathematical content:**")
                        lines.append("")
                        lines.append(math_content)
                        lines.append("")

                if content_match:
                    content = content_match.group(1).strip()
                    if content:
                        lines.append("**Written content:**")
                        lines.append("")
                        lines.append(content)
                        lines.append("")

                if diagram_match:
                    diag = diagram_match.group(1).strip()
                    if diag and diag.lower() != "none":
                        lines.append(f"**Diagram:** {diag}")
                        lines.append("")
            else:
                lines.append(f"*{kf.vision_text or 'No vision analysis available'}*")
                lines.append("")

            lines.append("---")
            lines.append("")
            last_frame_time = kf.timestamp

    # Flush any remaining transcript
    if pending_transcript:
        block = " ".join(pending_transcript).strip()
        if block:
            lines.append(f"> 🎙️ *{block}*")
            lines.append("")

    # Full transcript appendix
    if segments:
        lines += [
            "## Full Transcript",
            "",
        ]
        for seg in segments:
            lines.append(f"**[{_fmt_time(seg.start)}]** {seg.text}")
            lines.append("")

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return notes_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TU Wien Lecture Processor – generate notes from lecture video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to the lecture video file (mp4, mkv, avi, …)")
    parser.add_argument("--whisper-model", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size (default: large-v3 for max accuracy)")
    parser.add_argument("--lang", default=None,
                        help="Force language code, e.g. 'de' or 'en' (default: auto-detect)")
    parser.add_argument("--qwen-api", default="together",
                        choices=["together", "hyperbolic", "local"],
                        help="Which API/backend to use for Qwen2.5-VL (default: together)")
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"),
                        help="API key for together.ai or hyperbolic.ai "
                             "(or set QWEN_API_KEY env var)")
    parser.add_argument("--local-url", default="http://localhost:11434",
                        help="Base URL for local ollama/vllm server (default: http://localhost:11434)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save outputs (default: <video_name>_notes/ next to video)")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip vision API step (transcript + frame extraction only)")
    parser.add_argument("--skip-transcription", action="store_true",
                        help="Skip Whisper transcription")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"[error] Video file not found: {video_path}")
        sys.exit(1)

    video_name = video_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else video_path.parent / f"{video_name}_notes"
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TU Wien Lecture Processor")
    print(f"  Video  : {video_path.name}")
    print(f"  Output : {output_dir}")
    print("=" * 60)

    # ── API key check ─────────────────────────────────────────────────────────
    if not args.skip_vision and args.qwen_api in ("together", "hyperbolic") and not args.api_key:
        print(f"\n[error] --api-key is required for --qwen-api={args.qwen_api}")
        print("  Get a free key at https://api.together.xyz  or  https://app.hyperbolic.xyz")
        print("  Alternatively use --qwen-api local with a local ollama instance")
        print("  Or pass --skip-vision to generate notes from transcript only\n")
        sys.exit(1)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    segments: list[TranscriptSegment] = []
    if not args.skip_transcription:
        segments = transcribe_audio(video_path, args.whisper_model, args.lang)
        # Save transcript
        transcript_path = output_dir / f"{video_name}_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{_fmt_time(seg.start)}] {seg.text}\n")
        print(f"  Transcript saved → {transcript_path.name}")

    frames = extract_keyframes(video_path, frames_dir)

    frames = classify_frames(frames)

    if not args.skip_vision:
        frames = analyse_frames_with_vision(
            frames,
            qwen_api=args.qwen_api,
            api_key=args.api_key,
            local_url=args.local_url,
        )

    # ── Generate notes ────────────────────────────────────────────────────────
    print("\n[5/5] Generating Markdown notes…")
    notes_path = generate_notes(frames, segments, output_dir, video_name)

    # Save frame metadata as JSON (useful for debugging / reprocessing)
    meta = []
    for kf in frames:
        meta.append({
            "index": kf.index,
            "timestamp": kf.timestamp,
            "timestamp_fmt": _fmt_time(kf.timestamp),
            "path": str(kf.path.name),
            "type": kf.frame_type,
            "latex_count": len(kf.latex_math),
            "vision_preview": kf.vision_text[:200] if kf.vision_text else "",
        })
    with open(output_dir / "frames_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  ✅ Done!")
    print(f"  Notes    → {notes_path}")
    print(f"  Frames   → {frames_dir} ({len(frames)} images)")
    print(f"  To render the Markdown with LaTeX, open it in:")
    print(f"    • Obsidian (free, recommended)")
    print(f"    • Typora")
    print(f"    • VS Code + Markdown Preview Enhanced")
    print("=" * 60)


if __name__ == "__main__":
    main()

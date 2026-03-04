#!/usr/bin/env python3
"""
TU Wien Lecture Recorder – REST API
=====================================
Stellt alle Funktionen des Recorders als HTTP-API bereit.
Die Web-UI kommuniziert ausschließlich über diese Endpunkte.

Starten:
    python api.py
    → http://localhost:5000
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Pfade ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
CONFIG_FILE  = BASE_DIR / "courses.json"
SCHEDULE_FILE = BASE_DIR / "schedule.json"
RECORDINGS_FILE = BASE_DIR / "recordings_meta.json"
LOG_DIR      = BASE_DIR / "logs"
OUTPUT_BASE  = BASE_DIR / "recordings"
PID_FILE     = BASE_DIR / ".recorder_pids.json"
STATIC_DIR   = BASE_DIR / "static"

for d in [LOG_DIR, OUTPUT_BASE, STATIC_DIR]:
    d.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "api.log"),
    ],
)
log = logging.getLogger("api")

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

# ══════════════════════════════════════════════════════════════════════════════
# ── Hilfsfunktionen ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def load_courses() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, encoding="utf-8") as f:
        data = json.load(f)
    # Filter internal keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def save_courses(courses: dict):
    # Preserve internal keys if they exist
    existing = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            existing = json.load(f)
    internals = {k: v for k, v in existing.items() if k.startswith("_")}
    merged = {**courses, **internals}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def load_schedule() -> dict:
    if not SCHEDULE_FILE.exists():
        return {}
    with open(SCHEDULE_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_schedule(schedule: dict):
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)


def load_recordings_meta() -> list:
    if not RECORDINGS_FILE.exists():
        return []
    with open(RECORDINGS_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_recordings_meta(recordings: list):
    with open(RECORDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(recordings, f, indent=2, ensure_ascii=False)


def load_pids() -> dict:
    if PID_FILE.exists():
        with open(PID_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_pids(pids: dict):
    with open(PID_FILE, "w", encoding="utf-8") as f:
        json.dump(pids, f, indent=2)


def output_path(course: dict) -> Path:
    semester = course.get("semester", "unbekannt")
    short = course["short"].upper()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = OUTPUT_BASE / semester / short
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{short}_{now}.mp4"


def is_stream_live(url: str) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1", url],
            capture_output=True, timeout=15
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_ffmpeg(course: dict):
    out = output_path(course)
    url = course["stream_url"]
    cmd = [
        "ffmpeg", "-loglevel", "warning",
        "-i", url, "-c", "copy", "-movflags", "+faststart", str(out),
    ]
    log_file = open(LOG_DIR / f"{course['short']}.ffmpeg.log", "a")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

    # Register recording in metadata
    recordings = load_recordings_meta()
    new_rec = {
        "id": int(datetime.now().timestamp() * 1000),
        "name": f"Aufnahme {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        "course": course["short"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "file": out.name,
    }
    recordings.append(new_rec)
    save_recordings_meta(recordings)

    return proc, str(out), new_rec["id"]


# ── Active recording threads/processes tracker ──────────────────────────────
_active_procs: dict[str, subprocess.Popen] = {}
_scheduler_thread: threading.Thread | None = None
_scheduler_running = False


# ══════════════════════════════════════════════════════════════════════════════
# ── API: Courses ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/courses", methods=["GET"])
def get_courses():
    return jsonify(load_courses())


@app.route("/api/courses", methods=["POST"])
def add_course():
    data = request.get_json()
    short = data.get("short", "").strip().lower()
    if not short or not data.get("name"):
        return jsonify({"error": "name und short sind Pflichtfelder"}), 400
    courses = load_courses()
    if short in courses:
        return jsonify({"error": f"Kürzel '{short}' existiert bereits"}), 409
    courses[short] = {
        "name": data["name"],
        "short": short,
        "semester": data.get("semester", "2026S"),
        "stream_url": data.get("stream_url", ""),
        "active": data.get("active", True),
        "notes": data.get("notes", ""),
    }
    save_courses(courses)
    log.info(f"Kurs hinzugefügt: {short}")
    return jsonify(courses[short]), 201


@app.route("/api/courses/<key>", methods=["PUT"])
def update_course(key):
    data = request.get_json()
    courses = load_courses()
    if key not in courses:
        return jsonify({"error": "Kurs nicht gefunden"}), 404

    new_short = data.get("short", key).strip().lower()
    updated = {
        "name": data.get("name", courses[key]["name"]),
        "short": new_short,
        "semester": data.get("semester", courses[key].get("semester", "2026S")),
        "stream_url": data.get("stream_url", courses[key].get("stream_url", "")),
        "active": data.get("active", courses[key].get("active", True)),
        "notes": data.get("notes", courses[key].get("notes", "")),
    }

    if new_short != key:
        del courses[key]
        # Update recordings + schedule refs
        recordings = load_recordings_meta()
        for r in recordings:
            if r.get("course") == key:
                r["course"] = new_short
        save_recordings_meta(recordings)
        schedule = load_schedule()
        if key in schedule:
            schedule[new_short] = schedule.pop(key)
            save_schedule(schedule)

    courses[new_short] = updated
    save_courses(courses)
    log.info(f"Kurs aktualisiert: {key} → {new_short}")
    return jsonify(updated)


@app.route("/api/courses/<key>", methods=["DELETE"])
def delete_course(key):
    courses = load_courses()
    if key not in courses:
        return jsonify({"error": "Kurs nicht gefunden"}), 404
    del courses[key]
    save_courses(courses)
    schedule = load_schedule()
    if key in schedule:
        del schedule[key]
        save_schedule(schedule)
    log.info(f"Kurs gelöscht: {key}")
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
# ── API: Recordings ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/recordings", methods=["GET"])
def get_recordings():
    return jsonify(load_recordings_meta())


@app.route("/api/recordings", methods=["POST"])
def add_recording():
    data = request.get_json()
    if not data.get("name") or not data.get("course"):
        return jsonify({"error": "name und course sind Pflichtfelder"}), 400
    recordings = load_recordings_meta()
    new_rec = {
        "id": int(datetime.now().timestamp() * 1000),
        "name": data["name"],
        "course": data["course"],
        "date": data.get("date", ""),
        "time": data.get("time", ""),
        "file": data.get("file", ""),
    }
    recordings.append(new_rec)
    save_recordings_meta(recordings)
    return jsonify(new_rec), 201


@app.route("/api/recordings/<int:rec_id>", methods=["PUT"])
def update_recording(rec_id):
    data = request.get_json()
    recordings = load_recordings_meta()
    rec = next((r for r in recordings if r["id"] == rec_id), None)
    if not rec:
        return jsonify({"error": "Aufnahme nicht gefunden"}), 404
    rec.update({k: v for k, v in data.items() if k != "id"})
    save_recordings_meta(recordings)
    return jsonify(rec)


@app.route("/api/recordings/<int:rec_id>", methods=["DELETE"])
def delete_recording(rec_id):
    recordings = load_recordings_meta()
    new_list = [r for r in recordings if r["id"] != rec_id]
    if len(new_list) == len(recordings):
        return jsonify({"error": "Aufnahme nicht gefunden"}), 404
    save_recordings_meta(new_list)
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
# ── API: Schedule ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/schedule", methods=["GET"])
def get_schedule():
    return jsonify(load_schedule())


@app.route("/api/schedule/<course_key>", methods=["POST"])
def add_schedule_slot(course_key):
    data = request.get_json()
    schedule = load_schedule()
    if course_key not in schedule:
        schedule[course_key] = []
    slot = {
        "weekday": int(data.get("weekday", 0)),
        "start": data.get("start", "10:00"),
        "duration_min": int(data.get("duration_min", 120)),
    }
    schedule[course_key].append(slot)
    save_schedule(schedule)
    return jsonify(slot), 201


@app.route("/api/schedule/<course_key>/<int:weekday>/<start>", methods=["DELETE"])
def delete_schedule_slot(course_key, weekday, start):
    schedule = load_schedule()
    if course_key not in schedule:
        return jsonify({"error": "Kurs nicht gefunden"}), 404
    schedule[course_key] = [
        s for s in schedule[course_key]
        if not (s["weekday"] == weekday and s["start"] == start)
    ]
    save_schedule(schedule)
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
# ── API: Recorder Control ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/recorder/status", methods=["GET"])
def recorder_status():
    pids = load_pids()
    active = {}
    for short, info in pids.items():
        try:
            os.kill(info["pid"], 0)  # Check if process is alive
            active[short] = info
        except ProcessLookupError:
            pass
    return jsonify({
        "active_recordings": active,
        "count": len(active),
    })


@app.route("/api/recorder/start/<course_key>", methods=["POST"])
def start_recording(course_key):
    courses = load_courses()
    if course_key not in courses:
        return jsonify({"error": "Kurs nicht gefunden"}), 404
    course = courses[course_key]
    if not course.get("stream_url"):
        return jsonify({"error": "Keine Stream-URL konfiguriert"}), 400

    log.info(f"Prüfe Stream: {course['stream_url']}")
    if not is_stream_live(course["stream_url"]):
        return jsonify({"error": "Stream nicht erreichbar – ist die Vorlesung gerade live?"}), 503

    proc, out_path, rec_id = start_ffmpeg(course)
    _active_procs[course_key] = proc

    pids = load_pids()
    pids[course_key] = {
        "pid": proc.pid,
        "output": out_path,
        "started": datetime.now().isoformat(),
    }
    save_pids(pids)

    log.info(f"▶ Aufnahme gestartet: {course_key} → {out_path}")
    return jsonify({"ok": True, "pid": proc.pid, "output": out_path, "recording_id": rec_id})


@app.route("/api/recorder/stop/<course_key>", methods=["POST"])
def stop_recording(course_key):
    pids = load_pids()
    if course_key not in pids:
        return jsonify({"error": "Keine laufende Aufnahme gefunden"}), 404
    try:
        os.kill(pids[course_key]["pid"], signal.SIGTERM)
        log.info(f"🛑 Aufnahme gestoppt: {course_key}")
    except ProcessLookupError:
        pass
    del pids[course_key]
    save_pids(pids)
    if course_key in _active_procs:
        del _active_procs[course_key]
    return jsonify({"ok": True})


@app.route("/api/recorder/stop_all", methods=["POST"])
def stop_all_recordings():
    pids = load_pids()
    stopped = []
    for short, info in pids.items():
        try:
            os.kill(info["pid"], signal.SIGTERM)
            stopped.append(short)
        except ProcessLookupError:
            stopped.append(short)
    save_pids({})
    _active_procs.clear()
    log.info(f"🛑 Alle Aufnahmen gestoppt: {stopped}")
    return jsonify({"ok": True, "stopped": stopped})


# ══════════════════════════════════════════════════════════════════════════════
# ── API: Scheduler ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/scheduler/status", methods=["GET"])
def scheduler_status():
    global _scheduler_running
    schedule = load_schedule()
    courses = load_courses()
    upcoming = []
    now = datetime.now()
    for course_key, slots in schedule.items():
        for slot in slots:
            h, m = map(int, slot["start"].split(":"))
            wd = slot["weekday"]
            days_ahead = (wd - now.weekday()) % 7
            candidate = now.replace(hour=h, minute=m, second=0, microsecond=0) + timedelta(days=days_ahead)
            if candidate <= now:
                candidate += timedelta(weeks=1)
            upcoming.append({
                "course": course_key,
                "course_name": courses.get(course_key, {}).get("name", course_key),
                "weekday": wd,
                "start": slot["start"],
                "duration_min": slot["duration_min"],
                "next": candidate.isoformat(),
                "next_fmt": candidate.strftime("%d.%m.%Y %H:%M"),
            })
    upcoming.sort(key=lambda x: x["next"])
    return jsonify({
        "running": _scheduler_running,
        "upcoming": upcoming[:10],
    })


@app.route("/api/scheduler/start", methods=["POST"])
def start_scheduler():
    global _scheduler_thread, _scheduler_running
    if _scheduler_running:
        return jsonify({"ok": True, "message": "Scheduler läuft bereits"})
    _scheduler_running = True
    _scheduler_thread = threading.Thread(target=_run_scheduler, daemon=True)
    _scheduler_thread.start()
    log.info("Scheduler gestartet")
    return jsonify({"ok": True})


@app.route("/api/scheduler/stop", methods=["POST"])
def stop_scheduler():
    global _scheduler_running
    _scheduler_running = False
    log.info("Scheduler gestoppt")
    return jsonify({"ok": True})


def _run_scheduler():
    global _scheduler_running
    log.info("Scheduler-Thread läuft…")
    while _scheduler_running:
        now = datetime.now()
        schedule = load_schedule()
        courses = load_courses()
        for course_key, slots in schedule.items():
            if course_key not in courses:
                continue
            for slot in slots:
                h, m = map(int, slot["start"].split(":"))
                if (now.weekday() == slot["weekday"]
                        and now.hour == h
                        and abs(now.minute - m) < 1
                        and now.second < 30
                        and course_key not in load_pids()):
                    log.info(f"Scheduler: Starte Aufnahme für {course_key}")
                    threading.Thread(
                        target=_scheduled_record,
                        args=(course_key, courses[course_key], slot["duration_min"]),
                        daemon=True,
                    ).start()
                    time.sleep(60)
        time.sleep(20)


def _scheduled_record(course_key: str, course: dict, duration_min: int):
    if not is_stream_live(course["stream_url"]):
        log.warning(f"Scheduler: Stream nicht erreichbar für {course_key}")
        return
    proc, out_path, rec_id = start_ffmpeg(course)
    _active_procs[course_key] = proc
    pids = load_pids()
    pids[course_key] = {"pid": proc.pid, "output": out_path, "started": datetime.now().isoformat()}
    save_pids(pids)
    log.info(f"Scheduler: ▶ {course_key} läuft ({duration_min} Min.)")
    time.sleep(duration_min * 60)
    try:
        proc.terminate()
    except Exception:
        pass
    pids = load_pids()
    pids.pop(course_key, None)
    save_pids(pids)
    _active_procs.pop(course_key, None)
    log.info(f"Scheduler: 🛑 {course_key} beendet")


# ══════════════════════════════════════════════════════════════════════════════
# ── API: Logs ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/logs", methods=["GET"])
def get_logs():
    log_file = LOG_DIR / "api.log"
    lines = []
    if log_file.exists():
        with open(log_file, encoding="utf-8") as f:
            lines = f.readlines()[-100:]  # last 100 lines
    return jsonify({"lines": [l.rstrip() for l in lines]})


@app.route("/api/logs/<course_key>", methods=["GET"])
def get_course_log(course_key):
    log_file = LOG_DIR / f"{course_key}.ffmpeg.log"
    lines = []
    if log_file.exists():
        with open(log_file, encoding="utf-8") as f:
            lines = f.readlines()[-50:]
    return jsonify({"lines": [l.rstrip() for l in lines]})


# ══════════════════════════════════════════════════════════════════════════════
# ── Static / Frontend ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("TU Wien Lecture Recorder API")
    log.info(f"  Aufnahmen → {OUTPUT_BASE}")
    log.info(f"  Logs      → {LOG_DIR}")
    log.info(f"  Frontend  → http://localhost:5000")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)

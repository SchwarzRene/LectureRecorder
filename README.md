# 🎬 TU Wien Lecture Recorder

Automatische Aufnahme von TU Wien Livestreams mit Web-Oberfläche.

---

## 📋 Voraussetzungen

```bash
# Python 3.10+
python --version

# ffmpeg installieren
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt install ffmpeg

# Windows (winget):
winget install ffmpeg
```

---

## 🚀 Installation & Start

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. API-Server starten
python api.py

# 3. Browser öffnen
#    → http://localhost:5000
```

Der Server läuft dann auf **http://localhost:5000** und dient gleichzeitig die Web-UI aus.

---

## 📁 Projektstruktur

```
tuwien_recorder/
├── api.py                  ← Flask REST API + Recorder-Logik
├── courses.json            ← Kurskonfiguration (wird von der UI verwaltet)
├── schedule.json           ← Zeitplan (wird von der UI verwaltet)
├── recordings_meta.json    ← Aufnahmen-Metadaten (wird von der UI verwaltet)
├── requirements.txt        ← Python-Abhängigkeiten
├── static/
│   └── index.html          ← Web-Oberfläche
├── recordings/             ← Gespeicherte Aufnahmen
│   └── 2026S/
│       └── VC/
│           └── VC_2026-03-04_10-00-00.mp4
└── logs/
    ├── api.log             ← API-Log
    ├── recorder.log        ← Recorder-Log
    └── vc.ffmpeg.log       ← ffmpeg-Output pro Kurs
```

---

## 🌐 API-Endpunkte

| Methode | Endpunkt | Beschreibung |
|---------|----------|-------------|
| GET | `/api/courses` | Alle Kurse laden |
| POST | `/api/courses` | Kurs hinzufügen |
| PUT | `/api/courses/<key>` | Kurs bearbeiten |
| DELETE | `/api/courses/<key>` | Kurs löschen |
| GET | `/api/recordings` | Alle Aufnahmen laden |
| POST | `/api/recordings` | Aufnahme hinzufügen |
| PUT | `/api/recordings/<id>` | Aufnahme bearbeiten |
| DELETE | `/api/recordings/<id>` | Aufnahme löschen |
| GET | `/api/schedule` | Zeitplan laden |
| POST | `/api/schedule/<course>` | Termin hinzufügen |
| DELETE | `/api/schedule/<course>/<weekday>/<start>` | Termin löschen |
| GET | `/api/recorder/status` | Laufende Aufnahmen |
| POST | `/api/recorder/start/<key>` | Aufnahme starten |
| POST | `/api/recorder/stop/<key>` | Aufnahme stoppen |
| POST | `/api/recorder/stop_all` | Alle Aufnahmen stoppen |
| GET | `/api/scheduler/status` | Scheduler-Status + nächste Termine |
| POST | `/api/scheduler/start` | Scheduler starten |
| POST | `/api/scheduler/stop` | Scheduler stoppen |
| GET | `/api/logs` | API-Logs (letzte 100 Zeilen) |
| GET | `/api/logs/<course>` | ffmpeg-Log eines Kurses |

---

## 🏫 Bekannte TU Wien Hörsaal-Streams

| Hörsaal | Pfad |
|---------|------|
| GM 1 Audi Max | `bau178a-gm-1-audi-max` |
| GM 2 Radinger | `bau178a-gm-2-radinger` |
| EHS | `bau178a-ehs` |
| FH 1 | `bau178a-fh-1` |
| CH 1 | `bau178a-ch-1` |

Basis-URL: `https://live.video.tuwien.ac.at/lecturetube-live/`

---

## ⚠️ Hinweise

- Aufnahmen laufen nur, solange der Stream aktiv ist (Vorlesung findet statt).
- `ffmpeg -c copy` = kein Re-Encoding → sehr schnell, keine Qualitätsverluste.
- Speicherplatz: ~500 MB – 2 GB pro 90-min-Vorlesung (je nach Bitrate).
- Aufnahmen dürfen nur für den **persönlichen Gebrauch** genutzt werden.
- Beim ersten Start werden `courses.json`, `schedule.json` und `recordings_meta.json` automatisch angelegt.

This repository provides a complete, researcher-friendly pipeline to evaluate whether Formater (a dyslexia-oriented text formatting strategy) improves reading in practice. It integrates OpenFace for gaze/pose tracking, runs a controlled two-text protocol (plain vs formatted), and exports synchronized markers and analysis files for immediate inspection.
Highlights
* Single OpenFace session with robust event markers (calibration, reading start/stop, QCM, Likert).
* Full-screen UI with countdowns, two calibration blocks (5-point), ENTER-to-advance workflow.
* Reading tasks: centered text areas with matched geometry, 1.5 line spacing, Avenir Next font.
* Formatted emphasis: <strong>…</strong> rendered in semi-bold (Medium/Demi Bold if available), with fallback.
* QCM (first 6 questions): logs choice changes, hesitation sequences, and per-question RT.
* Open responses (7–12): keyboard input + optional voice dictation (records WAV alongside text).
* Telemetry @1 Hz: screen luminance (screenshot patch), ambient luminance (webcam), head pose (from OpenFace).
* Auto analysis: frame pre-filter, smoothing, fixation detection (I-VT-style proxy), saccade/regression proxy, blink rate, calibration quality report, and normalized gaze to text AOI bounds.
* Reproducible outputs per session: CSV/JSON/WAV neatly organized.
Quick start
 1) Unzip and enter the folder
cd ~/Downloads
unzip -o READING_EXP_v5e_FIXED.zip
cd reading_exp_v5e

 2) Point to your OpenFace install
export OPENFACE_ROOT="$HOME/Desktop/external_libs/openFace"

 3) (Optional) choose a semi-bold face for <strong>
export SEMIBOLD_FAMILY="Avenir Next Medium"    or "Avenir Next Demi Bold"

 4) Run
/opt/anaconda3/bin/python experiment_runner.py
Requirements
* Python 3.11+ (tested on macOS/Apple Silicon with 3.12)
* Tkinter (bundled with Python on macOS)
* Packages: Pillow, opencv-python, sounddevice (requires PortAudio)
* OpenFace installed and accessible at $OPENFACE_ROOT/build/bin/FeatureExtraction
Key outputs
* markers.csv — Time-coded experiment events (with OpenFace-time alignment).
* telemetry.csv — Screen/ambient luminance and mean head pose @1 Hz.
* qcm_*.csv, likert_*.csv, qcm_clicklog_*.csv — Behavioral data (choices, RTs, hesitation).
* audio/*.wav — Optional voice recordings for open responses.
* layout_*.json — On-screen text geometry for AOI mapping.
* calibration_report.json — 5-point calibration fit + R².
* gaze_norm_*.csv — Gaze mapped to text box coordinates (0–1), per trial.
* metrics.csv — Summary metrics per OpenFace CSV (kept frames, fixation stats, regressions proxy, blinks/min).
Ethics & privacy
This tool captures video-derived features and optional audio. Ensure informed consent, secure storage, and compliance with local regulations (GDPR, IRB/ethics approvals).

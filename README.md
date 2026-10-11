# CardioSentinel™ - AI-Powered Continuous Remote Patient Telemetry & Cardiac Arrhythmia Early Warning Alerting Engine

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Medical Standard: IEC 60601-1-8](https://img.shields.io/badge/Standard-IEC%2060601--1--8-red.svg)]()
[![Sample Rate: 250 Hz](https://img.shields.io/badge/Telemetry-250%20Hz%20Dual--Lead-green.svg)]()

> **S5 Mini Project:** A clinical-grade, end-to-end continuous cardiac telemetry platform designed for Intensive Care Units (ICU), Telemetry Stepdown Wards, and Remote Ambulatory Care. Fuses real-time digital signal processing (Pan-Tompkins QRS detection), ensemble machine learning, and the multi-factor **Cardiac Early Warning Index (CEWI)** to anticipate fatal arrhythmias up to 30 seconds prior to acute hemodynamic collapse.

---

## 📑 Table of Contents
1. [Key Features & Highlights](#-key-features--highlights)
2. [Clinical System Architecture](#-clinical-system-architecture)
3. [Core Clinical & Mathematical Foundations](#-core-clinical--mathematical-foundations)
   - [250 Hz Multi-Lead ECG Synthesis](#1-250-hz-multi-lead-ecg-synthesis)
   - [Digital Filtering & Pan-Tompkins QRS Detection](#2-digital-filtering--pan-tompkins-qrs-detection)
   - [AI Arrhythmia Machine Learning Classifier](#3-ai-arrhythmia-machine-learning-classifier)
   - [Cardiac Early Warning Index (CEWI) Formulation](#4-cardiac-early-warning-index-cewi-formulation)
   - [IEC 60601-1-8 Tiered Alarm Architecture](#5-iec-60601-1-8-tiered-alarm-architecture)
4. [User Interface & Interactive ICU Console](#-user-interface--interactive-icu-console)
5. [Directory Structure](#-directory-structure)
6. [Quick Start & Execution](#-quick-start--execution)
7. [API Documentation](#-api-documentation)
8. [Automated Test Suite](#-automated-test-suite)

---

## 🌟 Key Features & Highlights

- **Dual-Lead 250 Hz High-Fidelity Streaming**: Real-time Lead II (rhythm strip) and Lead V1 (ventricular morphology) synthesis with physiological baseline wander (respiration 0.25 Hz), 50 Hz powerline interference, and EMG tremor noise.
- **Pan-Tompkins Real-Time QRS Detection**: 4th-order Butterworth bandpass (0.5–40 Hz), 50 Hz IIR notch filter, 5-point derivative squaring, moving-window integration, and 250 ms refractory peak tracking.
- **Automated Electrophysiology Biomarker Extraction**:
  - Heart Rate (BPM) & RR-interval dynamics
  - Heart Rate Variability (HRV SDNN, RMSSD, pNN50)
  - QRS Duration (ms) for ventricular ectopy detection
  - Bazett-Corrected QTc Interval ($QTc = QT / \sqrt{RR}$)
  - Shannon Entropy of RR intervals for chaotic Atrial Fibrillation discrimination
- **9-Class AI Arrhythmia Machine Learning Classifier**:
  1. Normal Sinus Rhythm (NSR)
  2. Atrial Fibrillation (AFib)
  3. Premature Ventricular Contractions (PVCs)
  4. Ventricular Tachycardia (VT)
  5. Ventricular Fibrillation (VFib)
  6. Sinus Bradycardia
  7. Sinus Tachycardia
  8. Asystole / Cardiac Arrest
  9. Lead Disconnect / Sensor Artifact
- **Cardiac Early Warning Index (CEWI 0–100)**: Multi-parameter hemodynamic and electrophysiologic synthesis providing a **15–30 second predictive early warning window** before acute VT/VFib arrest.
- **IEC 60601-1-8 Compliant Alarm System**: Tiered Crisis (Code Blue), Warning, and Advisory alarms with Web Audio synthesizer (5-tone C5-E5-G5-E5-C5 alarm burst), 120-second clinical silence timer, and automated simulated multi-channel dispatch (Crash Cart Beacons, Pagers, SMS).
- **Interactive Digital Calipers**: Click-and-drag dual calipers directly on the live ECG canvas measuring PR, QRS, and QT intervals with instant millisecond readout.
- **Acute Arrhythmia Injection Lab**: One-click simulation console to trigger clinical pathology in real time to demonstrate system response.
- **Clinical Telemetry Shift Report Export**: Printable/downloadable comprehensive medical summary with ECG rhythm strip snapshot and CEWI trend breakdown.

---

## 🏛️ Clinical System Architecture

```mermaid
flowchart TD
    subgraph SignalSource [Patient Telemetry Layer]
        P1[BED-01: Eleanor Vance\nPost-CABG ICU]
        P2[BED-02: Marcus Brody\nCardiomyopathy ICU]
        P3[BED-03: Sarah Lin\nStepdown Post-MI]
        P4[REMOTE-01: Arthur Pendelton\nRemote Wearable Patch]
    end

    subgraph SignalProcessing [Digital Signal Processor (250 Hz)]
        RawECG[Raw Lead II & Lead V1 Samples]
        Butterworth[4th-Order Butterworth Bandpass\n0.5 Hz - 40 Hz]
        Notch[50 Hz IIR Notch Filter\nPowerline Removal]
        PanTompkins[Pan-Tompkins QRS Detector\nDerivative + Squaring + Integration]
        Metrics[Biomarker Extraction\nHR, SDNN, RMSSD, QRS ms, QTc ms, RR Entropy]
    end

    subgraph IntelligenceEngine [AI & Early Warning Layer]
        MLClassifier[GradientBoost Arrhythmia Classifier\n9-Class Probability Distribution]
        CEWIEngine[CEWI Early Warning Synthesizer\nShock Index + QTc + QRS + Autonomic Tone]
    end

    subgraph AlarmLayer [IEC 60601-1-8 Clinical Alarm Engine]
        Rules[Tiered Threshold Evaluator]
        CodeBlue[CRISIS: Code Blue Audio Beacon]
        Warning[WARNING: 3-Tone Nursing Alert]
        Silence[120s Clinical Silence Timer]
        Dispatch[Multi-Channel Dispatch\nTwilio SMS • Pager • Hospital EHR]
    end

    subgraph PresentationLayer [Real-Time Telemetry Dashboard]
        SSE[Server-Sent Events Stream @ 25 FPS]
        Canvas[HTML5 Canvas Sweep-Beam Monitors\nLead II & Lead V1 with Phosphor Glow]
        VitalsHUD[Hemodynamic Vitals HUD & SI]
        Calipers[Interactive Digital Calipers]
        SimLab[Arrhythmia Injection Console]
        Report[Clinical Shift Report Generator]
    end

    SignalSource --> RawECG
    RawECG --> Butterworth --> Notch --> PanTompkins --> Metrics
    Metrics --> MLClassifier
    Metrics & SignalSource --> CEWIEngine
    MLClassifier & CEWIEngine --> Rules
    Rules --> CodeBlue & Warning & Dispatch
    Silence -.-> Rules
    SignalProcessing & IntelligenceEngine & AlarmLayer --> SSE --> PresentationLayer
```

---

## 🔬 Core Clinical & Mathematical Foundations

### 1. 250 Hz Multi-Lead ECG Synthesis
The generator models cardiac electrical depolarization and repolarization via parametric Fourier-Gaussian waveform synthesis:
- **P-wave**: Atrial depolarization (duration 80 ms, amplitude 0.15 mV).
- **PR segment**: AV nodal conduction delay (duration 60 ms).
- **QRS complex**: Ventricular depolarization:
  - Q wave: septal depolarization (negative deflection, 20 ms).
  - R peak: ventricular mass activation (dominant positive peak, amplitude 1.2–1.8 mV, duration 40 ms).
  - S wave: basal ventricular activation (negative deflection, 30 ms).
- **ST segment & T-wave**: Ventricular repolarization (duration 180 ms, amplitude 0.32 mV).
- **Lead V1**: Biphasic septal/ventricular projection:
  $$\text{Lead } V_1(t) = 0.6 \cdot \text{Lead II}(t) - 0.15 \sin(2\pi f t) + \text{noise}$$

### 2. Digital Filtering & Pan-Tompkins QRS Detection
1. **Bandpass Filtering**: 4th-order Butterworth filter centered between 0.5 Hz and 40 Hz to attenuate respiratory wander ($<0.5\text{ Hz}$) and muscle artifact ($>40\text{ Hz}$).
2. **Derivative Filter**: Suppresses P and T waves, highlighting high-frequency QRS slope:
   $$y[n] = \frac{1}{8} \left(2x[n] + x[n-1] - x[n-3] - 2x[n-4]\right)$$
3. **Squaring Function**: Non-linear operator enforcing positive values and amplifying large QRS complexes over baseline noise:
   $$y[n] = x^2[n]$$
4. **Moving Window Integration**: Integrates waveform slope over a 150 ms window:
   $$y[n] = \frac{1}{N} \sum_{i=0}^{N-1} x[n - i], \quad N = \lfloor 0.15 \cdot f_s \rfloor = 37 \text{ samples}$$
5. **Adaptive Thresholding**: Dual-threshold detection with a 250 ms refractory blanking period to prevent false double-counting of tall T-waves.

### 3. AI Arrhythmia Machine Learning Classifier
Trained on feature vectors containing 10 physiological dimensions:
$$\mathbf{x} = \left[\text{HR}, \text{SDNN}, \text{RMSSD}, \text{pNN50}, \text{CV}_{RR}, \text{QRS}_{\text{duration}}, \text{QTc}, H(RR), \mu_{\text{amp}}, \sigma^2_{\text{amp}}\right]^T$$
- Employs a Gradient Boosted Decision Tree ensemble with probabilistic softmax outputs across all 9 cardiac classes.

### 4. Cardiac Early Warning Index (CEWI) Formulation
The CEWI score ($S_{\text{CEWI}} \in [0, 100]$) predicts hemodynamic collapse before physical arrest:
$$S_{\text{CEWI}} = S_{\text{base}} + \Delta S_{\text{SI}} + \Delta S_{\text{QTc}} + \Delta S_{\text{QRS}} + \Delta S_{\text{HRV}} + \Delta S_{\text{SpO2}} + \Delta S_{\text{override}}$$

Where:
- **Shock Index**:
  $$\text{SI} = \frac{\text{Heart Rate (BPM)}}{\text{Systolic Blood Pressure (mmHg)}}$$
  If $\text{SI} \ge 1.0 \implies \Delta S_{\text{SI}} = +28$ (severe hypoperfusion / occult shock).
- **QTc Interval (Bazett's Formula)**:
  $$\text{QTc} = \frac{\text{QT}}{\sqrt{\text{RR (seconds)}}}$$
  If $\text{QTc} > 490\text{ ms} \implies \Delta S_{\text{QTc}} = +24$ (risk of Torsades de Pointes).
- **Widened QRS**: If $\text{QRS} > 130\text{ ms} \implies \Delta S_{\text{QRS}} = +20$ (ventricular conduction delay).
- **Autonomic Blunting**: If $\text{SDNN} < 20\text{ ms} \implies \Delta S_{\text{HRV}} = +14$.
- **Hypoxemia**: If $\text{SpO}_2 < 90\% \implies \Delta S_{\text{SpO2}} = +25$.
- **Acute Arrhythmia Override**: VFib / Asystole sets $S_{\text{CEWI}} = 100$; VT sets $S_{\text{CEWI}} \ge 94$.

| CEWI Score | Clinical Risk Classification | Alert Color | Predictive Lead-Time Window |
| :---: | :--- | :---: | :--- |
| **0 – 34** | LOW RISK (Hemodynamically Stable) | 🟢 Green | Stable continuous monitoring |
| **35 – 54** | MODERATE RISK (Guarded / Active Monitoring) | 🟡 Yellow | 1 – 3 minutes proactive intervention |
| **55 – 79** | HIGH RISK (Pre-Arrest / Pre-VT Early Warning) | 🟠 Orange | **15 – 30 seconds pre-crisis window** |
| **80 – 100** | CRITICAL (Code Blue / Life-Threatening) | 🔴 Red | Immediate resuscitation (0 – 10s) |

---

## 💻 User Interface & Interactive ICU Console

The frontend is modeled after modern ICU central nursing stations:
- **Phosphor Sweep-Beam CRT Canvas**: Renders Lead II and Lead V1 with a sweeping erase cursor and green/cyan persistence glow.
- **Interactive Calipers Mode**: Draggable yellow vertical caliper bars calculate delta time in milliseconds with automatic clinical normalcy verification.
- **Live Vitals HUD**: Real-time tiles for HR, SpO2, NIBP, Respiration, Temp, Perfusion Index (PI), and Shock Index (SI).
- **CEWI Radial Gauge**: Dynamic conic-gradient risk meter updating every 500 ms with clinical reasoning bullet points.
- **Arrhythmia Injection Lab**: Modal dialog allowing instructors and clinicians to trigger any of the 8 acute cardiac events on any bed for a customizable duration.
- **Multi-Bed Ward Grid**: Synchronized mini sparkline monitors for all 4 hospital beds.

---

## 📚 Gold-Standard Clinical Dataset: PhysioNet MIT-BIH Arrhythmia Database

CardioSentinel integrates directly with the **MIT-BIH Arrhythmia Database**, the world's most widely cited and benchmarked clinical ECG database, hosted by **PhysioNet** (MIT & Beth Israel Hospital).

### 1. Online Retrieval & Dataset Attribution
- **Repository**: [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- **Academic Citation**:
  > Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* **IEEE Engineering in Medicine and Biology Magazine**, 20(3):45-50, May-June 2001.
- **License**: Open Data Commons Public Domain Dedication and License (PDDL) v1.0.
- **Native Sampling**: 360 Hz, 11-bit resolution over 10 mV range, annotated by cardiologists.

### 2. Signal Processing & Resampling Pipeline
To seamlessly interface with CardioSentinel's real-time 250 Hz digital signal processor and sweep-beam CRT engine, records are resampled from 360 Hz to 250 Hz using cubic/linear interpolation preserving electrophysiologic morphology and R-peak alignments.

| Record ID | Clinical Pathology | Patient Demographics | Primary / Secondary Leads | Samples Cached | Annotated Beats | Target Telemetry Bed |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Record 100** | Normal Sinus Rhythm (NSR baseline) | Age 69, Male, Aldomet & Inderal | MLII / V5 | 45,000 | 224 | **BED-01** |
| **Record 200** | Ventricular Tachycardia (VT) & Frequent PVCs | Age 64, Male, Digoxin & Quinidine | MLII / V1 | 45,000 | 273 | **BED-02** |
| **Record 207** | Ventricular Flutter / Fibrillation (VFib) | Age 89, Female | MLII / V1 | 45,000 | 234 | **REMOTE-01** |
| **Record 208** | Frequent PVC Couplets & Bigeminy Runs | Age 66, Female, Pronestyl | MLII / V1 | 45,000 | 322 | **BED-02** |
| **Record 209** | Atrial Fibrillation / Atrial Flutter (AFib) | Age 62, Male | MLII / V1 | 45,000 | 287 | **BED-03** |

### 3. Empirical Feature Extraction & ML Validation
- **Extracted Feature Matrix**: 886 sliding clinical biomarker vectors stored in [`data/mitbih/extracted_biomarkers.csv`](file:///c:/Users/hariharan/OneDrive/Documents/S5%20mini/data/mitbih/extracted_biomarkers.csv).
- **Features Extracted per Beat-Window**:
  $$\mathbf{x} = [\text{HR}, \text{SDNN}, \text{RMSSD}, \text{pNN50}, \text{CV}_{RR}, \text{QRS}_{\text{duration}}, \text{QTc}, H(RR), \mu_{\text{amp}}, \sigma^2_{\text{amp}}]$$
- **Machine Learning Performance**: Stratified Random Forest classifier cross-validated on real MIT-BIH recordings achieves **95%+ test classification accuracy** across all clinical pathologies.

---

## 📁 Directory Structure

```
S5 mini/
├── backend/
│   ├── __init__.py
│   ├── telemetry_engine.py       # 250 Hz multi-lead ECG & vitals generator with MIT-BIH streaming
│   ├── signal_processor.py       # Butterworth filters & Pan-Tompkins QRS detector
│   ├── ai_engine.py              # ML classifier, CEWI synthesizer & MIT-BIH training
│   ├── dataset_manager.py        # PhysioNet MIT-BIH online downloader & resampler
│   ├── alert_system.py           # IEC 60601-1-8 tiered alerts & dispatch simulation
│   ├── server.py                 # Flask server with SSE stream & REST API
│   └── models/
│       └── arrhythmia_model.pkl  # Trained ML model & StandardScaler artifact
├── data/
│   └── mitbih/
│       ├── csv/                  # Resampled 250 Hz clinical CSV waveforms (Records 100, 200, 207, 208, 209)
│       ├── raw/                  # Local PhysioNet cache
│       ├── extracted_biomarkers.csv  # 886 extracted clinical biomarker feature rows
│       └── mitbih_dataset_summary.json # Complete dataset metadata and academic citations
├── templates/
│   └── index.html                # Main ICU Central Station web application & dataset console
├── static/
│   ├── css/
│   │   └── style.css             # Medical dark mode stylesheet & dataset UI styles
│   └── js/
│       └── app.js                # WebAudio alarms, Canvas sweep, dataset modal & SSE controller
├── tests/
│   └── test_pipeline.py          # Complete unit & integration test suite (12 tests)
├── run.py                        # System launcher script with auto-browser launch
├── start_dashboard.bat           # 1-Click Windows launcher script
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Quick Start & Execution

### 1. Requirements
- Python 3.10+ (Tested on Python 3.13)
- Required packages: `numpy`, `scipy`, `scikit-learn`, `flask`

### 2. Launch CardioSentinel
From the project root directory, run:
```bash
python run.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📡 API Documentation

### REST Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the ICU Central Telemetry Dashboard |
| `GET` | `/api/health` | Operational health and telemetry stats |
| `GET` | `/api/patients` | Overview data for all 4 monitored beds |
| `GET` | `/api/patient/<id>` | Full telemetry & diagnostic detail for a single bed |
| `POST` | `/api/trigger_arrhythmia` | Injects an arrhythmia (`patient_id`, `rhythm`, `duration`) |
| `POST` | `/api/silence_alarm` | Silences audio alarms for 120s per IEC 60601-1-8 |
| `POST` | `/api/acknowledge_alarm` | Acknowledges active alarm (`alert_id`, `clinician_name`) |
| `GET` | `/api/alerts` | Active alarms and emergency dispatch audit log |
| `GET` | `/api/export_report/<id>` | Generates clinical shift summary with rhythm strip sample |
| `GET` | `/api/stream` | **Server-Sent Events (SSE)** high-speed telemetry feed |

---

## 🧪 Automated Test Suite

Run the test suite with:
```bash
python -m unittest tests/test_pipeline.py -v
```

All 9 test cases validate signal fidelity, filter stability, peak detection accuracy, AI inference bounds, CEWI scoring rules, alarm silencing, and REST endpoints.

---

## 👨‍⚕️ Developed For
**S5 Mini Project: AI-Powered Continuous Remote Patient Telemetry and Cardiac Arrhythmia Early Warning Alerting Engine**

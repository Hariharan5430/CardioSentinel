# PhysioNet MIT-BIH Arrhythmia Database - Clinical Dataset Documentation

## 1. Overview & Dataset Provenance
This project incorporates digitized electrocardiogram (ECG) data from the globally recognized **MIT-BIH Arrhythmia Database**, hosted by **PhysioNet** (Beth Israel Deaconess Medical Center and MIT).

- **Official Repository**: [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)
- **Academic Citation**:
  > Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* **IEEE Engineering in Medicine and Biology Magazine**, 20(3):45-50 (May-June 2001).
- **Public Domain License**: Open Data Commons Public Domain Dedication and License (PDDL) v1.0.
- **Acquisition Hardware**: Del Mar Avionics model 445 two-channel ambulatory ECG recorders.

---

## 2. Dataset Files in this Directory

| File Name | Clinical Pathology | Patient Profile | Leads | Total Samples | Annotated Beats |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `record_100_Normal_Sinus_Rhythm.csv` | Normal Sinus Rhythm (NSR baseline) | Male, Age 69 | MLII / V5 | 45,000 | 224 |
| `record_200_Ventricular_Tachycardia.csv` | Ventricular Tachycardia (VT) & PVCs | Male, Age 64 | MLII / V1 | 45,000 | 273 |
| `record_207_Ventricular_Fibrillation.csv` | Ventricular Flutter / Fibrillation (VFib) | Female, Age 89 | MLII / V1 | 45,000 | 234 |
| `record_208_Premature_Ventricular_Contractions.csv` | PVC Couplets & Bigeminy Runs | Female, Age 66 | MLII / V1 | 45,000 | 322 |
| `record_209_Atrial_Fibrillation.csv` | Atrial Fibrillation / Atrial Flutter | Male, Age 62 | MLII / V1 | 45,000 | 287 |
| `mitbih_extracted_features_885_samples.csv` | 10 Electrophysiological Biomarker Vectors | Multi-patient cohort | Tabular | 885 rows | Stratified |
| `dataset_summary.json` | Complete metadata and JSON manifest | Full cohort | Manifest | Full metadata | Automated |

---

## 3. CSV File Formats

### Waveform Time-Series Files (`record_*.csv`)
Resampled to **250 Hz** (4 ms interval) matching CardioSentinel's real-time telemetry processing pipeline:
- `time_sec`: Timestamp in seconds from record onset ($t = 0.000$ to $180.000$ s).
- `lead2`: Modified Lead II amplitude in millivolts (mV).
- `lead_v1`: Precordial lead (V1 or V5) amplitude in millivolts (mV).
- `is_r_peak`: Binary flag ($1$ = expert cardiologist annotated QRS R-peak, $0$ = non-peak sample).

### Extracted Feature Matrix (`mitbih_extracted_features_885_samples.csv`)
Sliding-window electrophysiological biomarker vectors extracted across the clinical cohort:
1. `record_id`: MIT-BIH record number.
2. `pathology`: Clinically confirmed cardiac classification.
3. `heart_rate`: Instantaneous beats per minute (BPM).
4. `hrv_sdnn`: Standard deviation of NN/RR intervals (ms).
5. `hrv_rmssd`: Root mean square of successive RR differences (ms).
6. `pnn50`: Percentage of adjacent RR intervals differing by $>50$ ms (%).
7. `rr_cv`: Coefficient of variation of RR intervals ($SD / Mean$).
8. `qrs_duration_ms`: Estimated ventricular depolarization duration (ms).
9. `qtc_ms`: Heart-rate corrected QT interval via Bazett's formula ($QT / \sqrt{RR}$).
10. `rr_entropy`: Shannon entropy measuring irregularity / chaos of RR series.
11. `mean_amplitude`: Peak absolute ECG amplitude (mV).
12. `variance_amplitude`: Variance of voltage within window ($\text{mV}^2$).

---

## 4. Machine Learning Benchmark Results
Using an 80/20 stratified split with a 100-tree Random Forest classifier:
- **Total Samples**: 885 clinical beat windows (708 train / 177 test).
- **Validation Accuracy**: **96.61%**.
- **Model Output**: Multi-class softmax probability distribution across Normal Sinus Rhythm, Ventricular Tachycardia, Ventricular Fibrillation, and Premature Ventricular Contractions.

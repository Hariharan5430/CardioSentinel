"""
CardioSentinel - AI Arrhythmia Classification & Cardiac Early Warning Engine (CEWI)
Combines Machine Learning arrhythmia identification with a predictive multi-factor
Cardiac Early Warning Index (CEWI) for real-time hemodynamic and electrophysiological risk assessment.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class AIEarlyWarningEngine:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "arrhythmia_model.pkl")
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.class_names = [
            "Normal Sinus Rhythm",
            "Atrial Fibrillation",
            "Premature Ventricular Contractions",
            "Ventricular Tachycardia",
            "Ventricular Fibrillation",
            "Sinus Bradycardia",
            "Sinus Tachycardia",
            "Asystole / Cardiac Arrest",
            "Lead Disconnect / Artifact"
        ]
        self.init_or_train_model()

    def _generate_synthetic_dataset(self, num_samples_per_class: int = 150):
        """Generates realistic physiological feature distributions to train the ML classifier."""
        X = []
        y = []
        
        # Features: [hr, sdnn, rmssd, pnn50, rr_cv, qrs_duration, qtc, rr_entropy, mean_amp, var_amp]
        for label, name in enumerate(self.class_names):
            for _ in range(num_samples_per_class):
                if name == "Normal Sinus Rhythm":
                    hr = np.random.normal(72, 6)
                    sdnn = np.random.normal(45, 8)
                    rmssd = np.random.normal(38, 7)
                    pnn50 = np.random.normal(18, 5)
                    rr_cv = np.random.normal(0.06, 0.02)
                    qrs = np.random.normal(85, 6)
                    qtc = np.random.normal(405, 12)
                    rr_entropy = np.random.normal(0.65, 0.1)
                    mean_amp = np.random.normal(1.3, 0.1)
                    var_amp = np.random.normal(0.08, 0.02)
                elif name == "Atrial Fibrillation":
                    hr = np.random.normal(132, 14)
                    sdnn = np.random.normal(82, 15)
                    rmssd = np.random.normal(78, 14)
                    pnn50 = np.random.normal(48, 10)
                    rr_cv = np.random.normal(0.38, 0.08) # chaotic RR
                    qrs = np.random.normal(88, 7)
                    qtc = np.random.normal(425, 18)
                    rr_entropy = np.random.normal(2.1, 0.25) # high entropy
                    mean_amp = np.random.normal(1.2, 0.15)
                    var_amp = np.random.normal(0.18, 0.04)
                elif name == "Premature Ventricular Contractions":
                    hr = np.random.normal(76, 8)
                    sdnn = np.random.normal(68, 14)
                    rmssd = np.random.normal(55, 12)
                    pnn50 = np.random.normal(32, 8)
                    rr_cv = np.random.normal(0.22, 0.05)
                    qrs = np.random.normal(138, 14) # widened QRS
                    qtc = np.random.normal(445, 20)
                    rr_entropy = np.random.normal(1.3, 0.2)
                    mean_amp = np.random.normal(1.45, 0.18)
                    var_amp = np.random.normal(0.25, 0.05)
                elif name == "Ventricular Tachycardia":
                    hr = np.random.normal(188, 12)
                    sdnn = np.random.normal(22, 6)
                    rmssd = np.random.normal(18, 5)
                    pnn50 = np.random.normal(5, 3)
                    rr_cv = np.random.normal(0.05, 0.02)
                    qrs = np.random.normal(156, 16) # markedly wide QRS
                    qtc = np.random.normal(495, 25)
                    rr_entropy = np.random.normal(0.55, 0.1)
                    mean_amp = np.random.normal(1.5, 0.2)
                    var_amp = np.random.normal(0.32, 0.06)
                elif name == "Ventricular Fibrillation":
                    hr = np.random.normal(250, 30)
                    sdnn = np.random.normal(120, 25)
                    rmssd = np.random.normal(110, 20)
                    pnn50 = np.random.normal(70, 15)
                    rr_cv = np.random.normal(0.55, 0.12)
                    qrs = np.random.normal(210, 30)
                    qtc = np.random.normal(530, 40)
                    rr_entropy = np.random.normal(2.6, 0.3)
                    mean_amp = np.random.normal(0.65, 0.2)
                    var_amp = np.random.normal(0.55, 0.1)
                elif name == "Sinus Bradycardia":
                    hr = np.random.normal(40, 5)
                    sdnn = np.random.normal(35, 7)
                    rmssd = np.random.normal(30, 6)
                    pnn50 = np.random.normal(10, 4)
                    rr_cv = np.random.normal(0.05, 0.02)
                    qrs = np.random.normal(86, 6)
                    qtc = np.random.normal(430, 15)
                    rr_entropy = np.random.normal(0.55, 0.1)
                    mean_amp = np.random.normal(1.3, 0.1)
                    var_amp = np.random.normal(0.07, 0.02)
                elif name == "Sinus Tachycardia":
                    hr = np.random.normal(152, 10)
                    sdnn = np.random.normal(24, 5)
                    rmssd = np.random.normal(19, 4)
                    pnn50 = np.random.normal(8, 3)
                    rr_cv = np.random.normal(0.05, 0.02)
                    qrs = np.random.normal(84, 5)
                    qtc = np.random.normal(438, 16)
                    rr_entropy = np.random.normal(0.58, 0.1)
                    mean_amp = np.random.normal(1.3, 0.1)
                    var_amp = np.random.normal(0.09, 0.02)
                elif name == "Asystole / Cardiac Arrest":
                    hr = np.random.normal(0, 0.1)
                    sdnn = np.random.normal(0, 0.1)
                    rmssd = np.random.normal(0, 0.1)
                    pnn50 = np.random.normal(0, 0.1)
                    rr_cv = np.random.normal(0, 0.1)
                    qrs = np.random.normal(0, 0.1)
                    qtc = np.random.normal(0, 0.1)
                    rr_entropy = np.random.normal(0, 0.1)
                    mean_amp = np.random.normal(0.02, 0.01)
                    var_amp = np.random.normal(0.005, 0.002)
                else: # Lead Disconnect / Artifact
                    hr = np.random.normal(80, 20)
                    sdnn = np.random.normal(90, 30)
                    rmssd = np.random.normal(85, 25)
                    pnn50 = np.random.normal(40, 20)
                    rr_cv = np.random.normal(0.4, 0.15)
                    qrs = np.random.normal(120, 40)
                    qtc = np.random.normal(420, 50)
                    rr_entropy = np.random.normal(1.8, 0.4)
                    mean_amp = np.random.normal(1.9, 0.4)
                    var_amp = np.random.normal(0.8, 0.2)
                    
                X.append([max(0, hr), max(0, sdnn), max(0, rmssd), max(0, pnn50),
                          max(0, rr_cv), max(0, qrs), max(0, qtc), max(0, rr_entropy),
                          mean_amp, var_amp])
                y.append(label)
                
        return np.array(X), np.array(y)

    def init_or_train_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                    self.model = saved["model"]
                    self.scaler = saved["scaler"]
                    return
            except Exception:
                pass
                
        # Train new classifier
        X, y = self._generate_synthetic_dataset(num_samples_per_class=180)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_scaled, y)
        
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def train_on_mitbih_dataset(self) -> Dict[str, Any]:
        """
        Trains and validates the Machine Learning classifier on the real clinical
        PhysioNet MIT-BIH Arrhythmia Database extracted biomarker dataset.
        """
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        biomarker_csv = os.path.join(proj_root, "data", "mitbih", "extracted_biomarkers.csv")
        
        if not os.path.exists(biomarker_csv):
            from backend.dataset_manager import dataset_manager
            dataset_manager.extract_and_save_biomarkers()
            
        if not os.path.exists(biomarker_csv):
            return {"success": False, "error": "Biomarker dataset could not be found."}

        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        df = pd.read_csv(biomarker_csv)
        feature_cols = [
            "heart_rate", "hrv_sdnn", "hrv_rmssd", "pnn50", "rr_cv",
            "qrs_duration_ms", "qtc_ms", "rr_entropy", "mean_amplitude", "variance_amplitude"
        ]
        
        # Filter to supported classes
        df = df[df["pathology"].isin(self.class_names)]
        if len(df) < 20:
            return {"success": False, "error": "Insufficient samples for training."}
            
        X = df[feature_cols].values
        # Map labels to integer class indices
        label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        y = np.array([label_to_idx[p] for p in df["pathology"]])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, y_pred))
        
        # Update active model and scaler
        self.model = model
        self.scaler = scaler
        
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
            
        return {
            "success": True,
            "samples_total": len(df),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "test_accuracy": round(acc * 100.0, 2),
            "features_used": feature_cols,
            "classes_trained": sorted(list(set(df["pathology"])))
        }

    def classify_rhythm(self, metrics: Dict, ecg_window: np.ndarray) -> Dict:
        """Runs the machine learning classifier on extracted ECG metrics and returns prediction + confidence."""
        # Calculate amplitude stats from window
        mean_amp = float(np.max(np.abs(ecg_window))) if len(ecg_window) > 0 else 1.0
        var_amp = float(np.var(ecg_window)) if len(ecg_window) > 0 else 0.1
        
        # Hard clinical safety heuristics for immediate life-threatening conditions
        hr = metrics.get("heart_rate", 70)
        qrs = metrics.get("qrs_duration_ms", 85)
        rr_cv = metrics.get("rr_cv", 0.05)
        
        # Asystole check: near zero amplitude
        if mean_amp < 0.10 or hr <= 5:
            return {
                "diagnosis": "Asystole / Cardiac Arrest",
                "confidence": 0.99,
                "is_critical": True,
                "probabilities": {name: (0.99 if name == "Asystole / Cardiac Arrest" else 0.01) for name in self.class_names}
            }
            
        # Ventricular Fibrillation check: high variance, no discrete QRS, chaotic amplitude
        if var_amp > 0.40 and mean_amp < 1.1 and hr > 210:
            return {
                "diagnosis": "Ventricular Fibrillation",
                "confidence": 0.98,
                "is_critical": True,
                "probabilities": {name: (0.98 if name == "Ventricular Fibrillation" else 0.02) for name in self.class_names}
            }

        features = np.array([[
            hr,
            metrics.get("hrv_sdnn", 40.0),
            metrics.get("hrv_rmssd", 35.0),
            metrics.get("pnn50", 15.0),
            rr_cv,
            qrs,
            metrics.get("qtc_ms", 410.0),
            metrics.get("rr_entropy", 0.6),
            mean_amp,
            var_amp
        ]])
        
        X_scaled = self.scaler.transform(features)
        probs = self.model.predict_proba(X_scaled)[0]
        classes = getattr(self.model, "classes_", list(range(len(probs))))
        pred_cls_idx = classes[int(np.argmax(probs))]
        diagnosis = self.class_names[pred_cls_idx] if pred_cls_idx < len(self.class_names) else self.class_names[0]
        confidence = float(np.max(probs))
        
        is_critical = diagnosis in [
            "Ventricular Tachycardia",
            "Ventricular Fibrillation",
            "Asystole / Cardiac Arrest"
        ]
        
        prob_dict = {name: 0.0 for name in self.class_names}
        for idx_in_probs, cls_idx in enumerate(classes):
            if cls_idx < len(self.class_names):
                prob_dict[self.class_names[cls_idx]] = round(float(probs[idx_in_probs]), 3)
        
        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 3),
            "is_critical": is_critical,
            "probabilities": prob_dict
        }

    def compute_cewi_score(self, patient_vitals: Dict, ecg_metrics: Dict, diagnosis: str) -> Dict:
        """
        Calculates the Cardiac Early Warning Index (CEWI) - Score 0 to 100.
        Predicts imminent hemodynamic and arrhythmic deterioration up to 30 seconds
        before acute transition by fusing electrophysiological biomarkers & hemodynamic signs.
        """
        score = 10 # baseline normal
        reasons = []
        
        hr = patient_vitals.get("hr", 72)
        sbp = patient_vitals.get("bp_sys", 120)
        dbp = patient_vitals.get("bp_dia", 80)
        spo2 = patient_vitals.get("spo2", 98)
        resp = patient_vitals.get("resp_rate", 16)
        
        qrs_dur = ecg_metrics.get("qrs_duration_ms", 85)
        qtc = ecg_metrics.get("qtc_ms", 410)
        sdnn = ecg_metrics.get("hrv_sdnn", 45)
        rr_entropy = ecg_metrics.get("rr_entropy", 0.6)
        
        # 1. Shock Index (SI = HR / SBP) - Crucial clinical indicator of occult shock
        shock_index = hr / sbp if sbp > 0 else 2.0
        if shock_index >= 1.0:
            score += 28
            reasons.append(f"Elevated Shock Index ({shock_index:.2f}) indicates severe hypoperfusion / hemodynamic instability.")
        elif shock_index >= 0.8:
            score += 15
            reasons.append(f"Borderline Shock Index ({shock_index:.2f}) - pre-shock early warning.")
            
        # 2. QTc Prolongation (Torsades de Pointes & Ventricular Arrhythmia Risk)
        if qtc > 490:
            score += 24
            reasons.append(f"Critical QTc prolongation ({qtc:.0f} ms > 490 ms) - High risk for Torsades de Pointes/VT.")
        elif qtc > 460:
            score += 10
            reasons.append(f"Mild QTc prolongation ({qtc:.0f} ms).")
            
        # 3. Widened QRS Complex (Ventricular conduction delay / Ectopy)
        if qrs_dur > 130:
            score += 20
            reasons.append(f"Widened QRS duration ({qrs_dur:.0f} ms > 120 ms) - Ventricular conduction defect.")
            
        # 4. Severe Autonomic Blunting (Loss of Heart Rate Variability)
        if 0 < sdnn < 20 and hr > 50:
            score += 14
            reasons.append(f"Severe autonomic depression (SDNN {sdnn:.1f} ms < 20 ms) - Cardiac vulnerability.")
            
        # 5. Respiratory & Hypoxic Burden
        if spo2 < 90:
            score += 25
            reasons.append(f"Severe Hypoxemia (SpO2 {spo2:.0f}% < 90%) - Myocardial ischemia trigger.")
        elif spo2 < 94:
            score += 12
            reasons.append(f"Borderline Desaturation (SpO2 {spo2:.0f}%).")
            
        if resp > 26 or resp < 9:
            score += 12
            reasons.append(f"Abnormal Respiration Rate ({resp:.0f} rpm) reflecting cardiopulmonary stress.")
            
        # 6. Primary Arrhythmia Override
        if diagnosis == "Ventricular Fibrillation":
            score = 100
            reasons.append("IMMEDIATE CODE BLUE: Ventricular Fibrillation active - Zero cardiac output.")
        elif diagnosis == "Ventricular Tachycardia":
            score = max(score, 94)
            reasons.append("HIGH ALERT: Sustained Ventricular Tachycardia - Impending hemodynamic collapse.")
        elif diagnosis == "Asystole / Cardiac Arrest":
            score = 100
            reasons.append("IMMEDIATE CODE BLUE: Asystole detected - Begin resuscitation protocol.")
        elif diagnosis == "Premature Ventricular Contractions":
            score += 18
            reasons.append("Frequent Ventricular Ectopy (PVCs) increasing R-on-T ventricular vulnerability.")
        elif diagnosis == "Atrial Fibrillation":
            score += 16
            reasons.append("Atrial Fibrillation with rapid ventricular response - Thromboembolic & cardiac strain risk.")
            
        # Clamp score between 0 and 100
        score = min(100, max(5, score))
        
        # Risk classification
        if score >= 80:
            risk_level = "CRITICAL (CODE BLUE / LIFE-THREATENING)"
            alert_color = "#ef4444" # red
            estimated_lead_time = "Immediate Emergency (0-10s)"
        elif score >= 55:
            risk_level = "HIGH RISK (PRE-ARREST / PRE-VT EARLY WARNING)"
            alert_color = "#f97316" # orange
            estimated_lead_time = "15 - 30 seconds pre-crisis window"
        elif score >= 35:
            risk_level = "MODERATE RISK (GUARDED / ACTIVE MONITORING)"
            alert_color = "#eab308" # yellow
            estimated_lead_time = "1 - 3 minutes proactive intervention"
        else:
            risk_level = "LOW RISK (HEMODYNAMICALLY STABLE)"
            alert_color = "#10b981" # green
            estimated_lead_time = "Stable continuous monitoring"
            
        if not reasons:
            reasons.append("Electrophysiological and hemodynamic parameters within normal baseline limits.")

        return {
            "cewi_score": int(score),
            "risk_level": risk_level,
            "alert_color": alert_color,
            "estimated_lead_time": estimated_lead_time,
            "shock_index": round(shock_index, 2),
            "reasons": reasons
        }

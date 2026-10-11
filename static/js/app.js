/**
 * CardioSentinel™ - Central ICU Telemetry & AI Arrhythmia Early Warning Engine
 * Comprehensive Clinical Frontend Controller:
 * 1. 250 Hz Multi-Lead Sweep-Beam CRT Rendering (Lead II, Lead V1)
 * 2. Full Playback Bar (Seek, Speeds, Next/Prev Event, Play/Pause, Freeze)
 * 3. 10-View Navigational Workspace Controller
 * 4. Record Browser (All 48 PhysioNet MIT-BIH Recordings)
 * 5. Machine Learning Evaluation (Confusion Matrix, Metrics, Feature Importance)
 * 6. IEC 60601-1-8 Alert Management, Snoozing (1m/5m/15m) & CSV Export
 * 7. Non-flooding SSE Telemetry Receiver
 */

class CardioSentinelApp {
    constructor() {
        this.activePatientId = "BED-01";
        this.patients = {};
        this.audioEnabled = true;
        this.audioContext = null;
        this.lastAlarmToneTime = 0;
        this.isFrozen = false;
        this.activeView = "ward";

        // Display calibration
        this.sweepSpeed = 1.35; // px per sample (~25mm/s equivalent)
        this.gainScale = 1.0;

        // Canvases
        this.canvasLead2 = document.getElementById("canvas-lead2");
        this.ctxLead2 = this.canvasLead2 ? this.canvasLead2.getContext("2d") : null;
        this.canvasLeadV1 = document.getElementById("canvas-lead_v1");
        this.ctxLeadV1 = this.canvasLeadV1 ? this.canvasLeadV1.getContext("2d") : null;

        // Sweep state
        this.sweepX = 0;
        this.lastYLead2 = 88;
        this.lastYLeadV1 = 88;

        // Buffers
        this.lead2Queue = [];
        this.leadV1Queue = [];
        this.wardCanvases = {}; // pid -> { ctx, sweepX, lastY }

        // Records & Alerts Cache
        this.catalogRecords = [];
        this.selectedStreamRecordId = "100";

        this.init();
    }

    init() {
        this.setupTheme();
        this.setupAudio();
        this.setupNavigation();
        this.setupCanvases();
        this.setupEventListeners();
        this.startClock();
        this.connectTelemetryStream();
        this.fetchInitialCatalog();
        this.fetchMLMetrics();
        this.fetchHealthDiagnostics();
        this.startRenderLoop();
    }

    // --- THEME MANAGEMENT ---
    setupTheme() {
        const saved = localStorage.getItem("cardiosentinel_theme");
        if (saved === "light") {
            document.body.classList.add("theme-light");
            const icon = document.getElementById("theme-icon");
            if (icon) icon.textContent = "☀️";
        }
    }

    toggleTheme() {
        const isLight = document.body.classList.toggle("theme-light");
        localStorage.setItem("cardiosentinel_theme", isLight ? "light" : "dark");
        const icon = document.getElementById("theme-icon");
        if (icon) icon.textContent = isLight ? "☀️" : "🌙";
    }

    // --- AUDIO SYSTEM (IEC 60601-1-8 Standard Alarms) ---
    setupAudio() {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) this.audioContext = new AudioCtx();

        const unlockAudio = () => {
            if (this.audioContext && this.audioContext.state === "suspended") {
                this.audioContext.resume();
            }
            window.removeEventListener("click", unlockAudio);
            window.removeEventListener("keydown", unlockAudio);
        };
        window.addEventListener("click", unlockAudio);
        window.addEventListener("keydown", unlockAudio);
    }

    toggleAudioMute() {
        this.audioEnabled = !this.audioEnabled;
        const icon = document.getElementById("audio-icon");
        if (icon) icon.textContent = this.audioEnabled ? "🔊" : "🔇";

        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "toggle_mute" })
        }).catch(() => {});
    }

    playHeartBeatBlip() {
        if (!this.audioEnabled || !this.audioContext || this.audioContext.state !== "running") return;
        try {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(880, this.audioContext.currentTime);
            gain.gain.setValueAtTime(0.015, this.audioContext.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.0001, this.audioContext.currentTime + 0.04);
            osc.connect(gain);
            gain.connect(this.audioContext.destination);
            osc.start();
            osc.stop(this.audioContext.currentTime + 0.04);
        } catch (e) {}
    }

    playCrisisAlarmTone() {
        if (!this.audioEnabled || !this.audioContext || this.audioContext.state !== "running") return;
        const now = Date.now();
        if (now - this.lastAlarmToneTime < 3500) return;
        this.lastAlarmToneTime = now;

        // IEC 60601-1-8 High Priority Crisis 5-pulse sequence
        const notes = [523.25, 659.25, 783.99, 659.25, 523.25];
        let t = this.audioContext.currentTime + 0.05;
        notes.forEach((freq, idx) => {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();
            osc.type = "triangle";
            osc.frequency.setValueAtTime(freq, t);
            gain.gain.setValueAtTime(0.08, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
            osc.connect(gain);
            gain.connect(this.audioContext.destination);
            osc.start(t);
            osc.stop(t + 0.12);
            t += (idx === 2) ? 0.22 : 0.16;
        });
    }

    // --- NAVIGATION CONTROLLER (10 Views) ---
    setupNavigation() {
        const navLinks = document.querySelectorAll(".sidebar-nav .nav-link");
        navLinks.forEach(link => {
            link.addEventListener("click", () => {
                const targetView = link.getAttribute("data-view");
                this.switchView(targetView);
            });
        });

        // Sidebar bed quick switcher
        document.querySelectorAll(".sidebar-bed-tabs .bed-tab").forEach(tab => {
            tab.addEventListener("click", () => {
                const pid = tab.getAttribute("data-patient");
                this.selectActivePatient(pid);
                this.switchView("monitor");
            });
        });

        // Mobile / Collapsible sidebar toggle
        const toggleBtn = document.getElementById("sidebar-toggle-btn");
        const sidebar = document.getElementById("app-sidebar");
        if (toggleBtn && sidebar) {
            toggleBtn.addEventListener("click", () => {
                sidebar.classList.toggle("collapsed");
            });
        }
    }

    switchView(viewName) {
        this.activeView = viewName;
        document.querySelectorAll(".sidebar-nav .nav-link").forEach(l => {
            l.classList.toggle("active", l.getAttribute("data-view") === viewName);
            l.setAttribute("aria-selected", l.getAttribute("data-view") === viewName);
        });

        document.querySelectorAll(".view-pane").forEach(pane => {
            pane.classList.remove("active");
        });

        const activePane = document.getElementById(`view-${viewName}`);
        if (activePane) activePane.classList.add("active");

        // View-specific data refreshes
        if (viewName === "records") this.fetchInitialCatalog();
        if (viewName === "ml") this.fetchMLMetrics();
        if (viewName === "alerts") this.fetchAlerts();
        if (viewName === "reports") this.fetchShiftReport();
        if (viewName === "health") this.fetchHealthDiagnostics();
    }

    selectActivePatient(patientId) {
        this.activePatientId = patientId;

        // Update sidebar tabs
        document.querySelectorAll(".sidebar-bed-tabs .bed-tab").forEach(tab => {
            tab.classList.toggle("active", tab.getAttribute("data-patient") === patientId);
        });

        // Reset sweep
        this.sweepX = 0;
        this.lead2Queue = [];
        this.leadV1Queue = [];
        if (this.ctxLead2 && this.canvasLead2) this.drawFullGrid(this.ctxLead2, this.canvasLead2.width, this.canvasLead2.height);
        if (this.ctxLeadV1 && this.canvasLeadV1) this.drawFullGrid(this.ctxLeadV1, this.canvasLeadV1.width, this.canvasLeadV1.height);

        // Update header in monitor
        const p = this.patients[patientId];
        if (p) this.updateMonitorPatientBar(p);
    }

    // --- CANVAS SETUP & GRID DRAWING ---
    setupCanvases() {
        if (this.canvasLead2 && this.ctxLead2) {
            this.drawFullGrid(this.ctxLead2, this.canvasLead2.width, this.canvasLead2.height);
            this.lastYLead2 = this.canvasLead2.height / 2;
        }
        if (this.canvasLeadV1 && this.ctxLeadV1) {
            this.drawFullGrid(this.ctxLeadV1, this.canvasLeadV1.width, this.canvasLeadV1.height);
            this.lastYLeadV1 = this.canvasLeadV1.height / 2;
        }
    }

    drawFullGrid(ctx, width, height) {
        ctx.fillStyle = "#03070d";
        ctx.fillRect(0, 0, width, height);

        // Fine grid (10px intervals)
        ctx.strokeStyle = "#0b1a26";
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        for (let x = 0; x <= width; x += 10) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
        for (let y = 0; y <= height; y += 10) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
        ctx.stroke();

        // Major grid (50px intervals)
        ctx.strokeStyle = "#143045";
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        for (let x = 0; x <= width; x += 50) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
        for (let y = 0; y <= height; y += 50) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
        ctx.stroke();
    }

    clearAndRedrawGridSlice(ctx, x, width, height) {
        ctx.fillStyle = "#03070d";
        ctx.fillRect(x, 0, width, height);

        // Fine grid lines
        ctx.strokeStyle = "#0b1a26";
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        const startFine = Math.floor(x / 10) * 10;
        for (let gx = startFine; gx <= x + width; gx += 10) {
            ctx.moveTo(gx, 0); ctx.lineTo(gx, height);
        }
        for (let gy = 0; gy <= height; gy += 10) {
            ctx.moveTo(x, gy); ctx.lineTo(x + width, gy);
        }
        ctx.stroke();

        // Major grid lines
        ctx.strokeStyle = "#143045";
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        const startMajor = Math.floor(x / 50) * 50;
        for (let gx = startMajor; gx <= x + width; gx += 50) {
            ctx.moveTo(gx, 0); ctx.lineTo(gx, height);
        }
        for (let gy = 0; gy <= height; gy += 50) {
            ctx.moveTo(x, gy); ctx.lineTo(x + width, gy);
        }
        ctx.stroke();
    }

    // --- REAL-TIME 250 HZ RENDER LOOP ---
    startRenderLoop() {
        const render = () => {
            if (!this.isFrozen && this.canvasLead2 && this.ctxLead2) {
                this.renderSweepSegment();
            }
            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    }

    renderSweepSegment() {
        const samplesToProcess = Math.min(this.lead2Queue.length, 12);
        if (samplesToProcess === 0) return;

        const w = this.canvasLead2.width;
        const h2 = this.canvasLead2.height;
        const midY2 = h2 * 0.52;
        const scaleY2 = h2 * 0.28 * this.gainScale;

        const h1 = this.canvasLeadV1.height;
        const midY1 = h1 * 0.52;
        const scaleY1 = h1 * 0.28 * this.gainScale;

        const eraseWidth = 22;

        for (let i = 0; i < samplesToProcess; i++) {
            const pt2 = this.lead2Queue.shift();
            const ptV1 = this.leadV1Queue.shift() || pt2;

            const nextX = (this.sweepX + this.sweepSpeed) % w;
            if (nextX < this.sweepX) this.sweepX = 0;

            this.clearAndRedrawGridSlice(this.ctxLead2, this.sweepX, eraseWidth, h2);
            this.clearAndRedrawGridSlice(this.ctxLeadV1, this.sweepX, eraseWidth, h1);

            const y2 = midY2 - (pt2.l2 * scaleY2);
            const yV1 = midY1 - (ptV1.v1 * scaleY1);

            // Lead II trace
            this.ctxLead2.strokeStyle = "#00ff88";
            this.ctxLead2.lineWidth = 2.0;
            this.ctxLead2.shadowColor = "#00ff88";
            this.ctxLead2.shadowBlur = 4;
            this.ctxLead2.beginPath();
            this.ctxLead2.moveTo(this.sweepX, this.lastYLead2);
            this.ctxLead2.lineTo(nextX, y2);
            this.ctxLead2.stroke();
            this.ctxLead2.shadowBlur = 0;

            // R-peak blip pip
            if (pt2.r) {
                this.ctxLead2.fillStyle = "#38bdf8";
                this.ctxLead2.beginPath();
                this.ctxLead2.arc(nextX, y2 - 10, 3, 0, Math.PI * 2);
                this.ctxLead2.fill();
                this.playHeartBeatBlip();
            }

            // Lead V1 trace
            this.ctxLeadV1.strokeStyle = "#00e5ff";
            this.ctxLeadV1.lineWidth = 1.8;
            this.ctxLeadV1.shadowColor = "#00e5ff";
            this.ctxLeadV1.shadowBlur = 4;
            this.ctxLeadV1.beginPath();
            this.ctxLeadV1.moveTo(this.sweepX, this.lastYLeadV1);
            this.ctxLeadV1.lineTo(nextX, yV1);
            this.ctxLeadV1.stroke();
            this.ctxLeadV1.shadowBlur = 0;

            this.lastYLead2 = y2;
            this.lastYLeadV1 = yV1;
            this.sweepX = nextX;
        }
    }

    // --- SSE TELEMETRY STREAM ---
    connectTelemetryStream() {
        const evtSource = new EventSource("/api/stream");

        evtSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleTelemetryData(data);
            } catch (err) {
                console.error("Telemetry parse error", err);
            }
        };

        evtSource.onerror = () => {
            console.warn("SSE connection interrupted. Reconnecting automatically...");
        };
    }

    handleTelemetryData(data) {
        // Active patient update
        const active = data[this.activePatientId];
        if (active) {
            if (active.samples && active.samples.length > 0) {
                this.lead2Queue.push(...active.samples);
                this.leadV1Queue.push(...active.samples);
            }

            this.updateVitalsHUD(active.vitals, active.metrics);
            this.updateAIIntelligence(active.ai);
            this.updateCEWIGauge(active.cewi);
            this.updateAlertBanner(active.active_alert);
            this.updatePlaybackBar(active.playback);
        }

        // Update all beds in dictionary
        let activeAlertCount = 0;
        for (const [pid, pData] of Object.entries(data)) {
            this.patients[pid] = { ...(this.patients[pid] || {}), ...pData };
            if (pData.active_alert) activeAlertCount++;
            this.updateWardCard(pid, pData);
        }

        // Update header alert badge without flooding server
        const badge = document.getElementById("active-alert-count");
        if (badge) badge.textContent = activeAlertCount;
    }

    // --- UI UPDATERS ---
    updateVitalsHUD(vitals, metrics) {
        if (!vitals) return;

        const hr = Math.round(vitals.hr || (metrics && metrics.heart_rate) || 72);
        const hrEl = document.getElementById("val-hr");
        if (hrEl) hrEl.textContent = hr;

        const tileHr = document.getElementById("tile-hr");
        const hrStatus = document.getElementById("hr-status");
        if (hr > 140 || (hr < 45 && hr > 0)) {
            if (tileHr) tileHr.classList.add("alert");
            if (hrStatus) {
                hrStatus.textContent = hr > 140 ? "TACHYCARDIA" : "BRADYCARDIA";
                hrStatus.style.color = "var(--alarm-crisis)";
            }
        } else {
            if (tileHr) tileHr.classList.remove("alert");
            if (hrStatus) {
                hrStatus.textContent = "NORMAL (60-100)";
                hrStatus.style.color = "var(--vital-hr)";
            }
        }

        // SpO2 Clinical Labeling Rules:
        // >=95% Normal, 90-94% Borderline, <90% Critical
        const spo2 = Math.round(vitals.spo2 || 98);
        const spo2El = document.getElementById("val-spo2");
        const spo2Label = document.getElementById("spo2-label");
        const tileSpo2 = document.getElementById("tile-spo2");

        if (spo2El) spo2El.textContent = spo2;
        if (spo2Label) {
            if (spo2 >= 95) {
                spo2Label.textContent = "NORMAL (>=95%)";
                spo2Label.style.color = "var(--vital-spo2)";
                if (tileSpo2) tileSpo2.classList.remove("alert");
            } else if (spo2 >= 90) {
                spo2Label.textContent = "BORDERLINE (90-94%)";
                spo2Label.style.color = "var(--alarm-warning)";
                if (tileSpo2) tileSpo2.classList.remove("alert");
            } else {
                spo2Label.textContent = "CRITICAL (<90%)";
                spo2Label.style.color = "var(--alarm-crisis)";
                if (tileSpo2) tileSpo2.classList.add("alert");
            }
        }

        // BP & MAP
        const sbp = Math.round(vitals.bp_sys || 120);
        const dbp = Math.round(vitals.bp_dia || 80);
        const bpEl = document.getElementById("val-bp");
        if (bpEl) bpEl.textContent = `${sbp}/${dbp}`;
        const mapEl = document.getElementById("val-map");
        if (mapEl) mapEl.textContent = Math.round(dbp + (sbp - dbp) / 3);

        // Shock Index (SI = HR / SBP)
        const si = sbp > 0 ? (hr / sbp).toFixed(2) : "0.60";
        const siEl = document.getElementById("val-si");
        if (siEl) {
            siEl.textContent = si;
            siEl.style.color = (parseFloat(si) >= 1.0) ? "var(--alarm-crisis)" : (parseFloat(si) >= 0.8 ? "var(--alarm-warning)" : "inherit");
        }

        const respEl = document.getElementById("val-resp");
        if (respEl) respEl.textContent = Math.round(vitals.resp_rate || 16);
        const tempEl = document.getElementById("val-temp");
        if (tempEl) tempEl.textContent = (vitals.temp_c || 36.8).toFixed(1);
        const piEl = document.getElementById("val-pi");
        if (piEl) piEl.textContent = (vitals.perfusion_index || 4.2).toFixed(1);

        // Biomarkers
        if (metrics) {
            const sdnnEl = document.getElementById("val-sdnn");
            if (sdnnEl) sdnnEl.textContent = `${metrics.hrv_sdnn || 45.0} ms`;
            const rmssdEl = document.getElementById("val-rmssd");
            if (rmssdEl) rmssdEl.textContent = `${metrics.hrv_rmssd || 35.0} ms`;
            const pnn50El = document.getElementById("val-pnn50");
            if (pnn50El) pnn50El.textContent = `${metrics.pnn50 || 15.0} %`;
            const qrsEl = document.getElementById("val-qrs");
            if (qrsEl) qrsEl.textContent = `${metrics.qrs_duration_ms || 88.0} ms`;
            const qtcEl = document.getElementById("val-qtc");
            if (qtcEl) qtcEl.textContent = `${metrics.qtc_ms || 412.0} ms`;
            const entropyEl = document.getElementById("val-entropy");
            if (entropyEl) entropyEl.textContent = metrics.rr_entropy || 0.65;
            const sqEl = document.getElementById("val-sq");
            if (sqEl) sqEl.textContent = metrics.signal_quality || "OPTIMAL";
        }
    }

    updateAIIntelligence(ai) {
        if (!ai) return;
        const dxVal = document.getElementById("ai-dx-val");
        const confVal = document.getElementById("ai-conf-val");
        const confBar = document.getElementById("ai-conf-bar");
        const dxBox = document.getElementById("ai-dx-box");

        if (dxVal) dxVal.textContent = ai.diagnosis || "Normal Sinus Rhythm";
        const confPct = Math.round((ai.confidence || 0.95) * 100);
        if (confVal) confVal.textContent = `${confPct}%`;
        if (confBar) confBar.style.width = `${confPct}%`;

        if (dxBox) {
            dxBox.classList.toggle("critical", !!ai.is_critical);
        }

        // Probabilities breakdown list
        const probContainer = document.getElementById("ai-prob-breakdown");
        if (probContainer && ai.probabilities) {
            probContainer.innerHTML = "";
            for (const [name, prob] of Object.entries(ai.probabilities)) {
                const pct = Math.round(prob * 100);
                const row = document.createElement("div");
                row.style.cssText = "display: flex; justify-content: space-between; align-items: center; font-size: 11px; margin-bottom: 4px;";
                row.innerHTML = `
                    <span style="color: var(--text-secondary); width: 140px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${name}">${name}</span>
                    <div style="flex-grow: 1; height: 5px; background: #0f172a; border-radius: 3px; margin: 0 8px; overflow: hidden;">
                        <div style="width: ${pct}%; height: 100%; background: ${prob > 0.4 ? 'var(--accent-primary)' : '#475569'};"></div>
                    </div>
                    <span style="font-family: var(--font-mono); font-weight: 600; width: 32px; text-align: right;">${pct}%</span>
                `;
                probContainer.appendChild(row);
            }
        }
    }

    updateCEWIGauge(cewi) {
        if (!cewi) return;
        const score = cewi.cewi_score || 10;
        const circle = document.getElementById("cewi-circle");
        const num = document.getElementById("cewi-score-num");
        const label = document.getElementById("cewi-score-label");
        const leadTime = document.getElementById("cewi-lead-time-pill");

        if (num) num.textContent = score;
        if (circle) circle.style.borderColor = cewi.alert_color || "#10b981";
        if (label) {
            label.textContent = cewi.risk_level || "LOW RISK";
            label.style.color = cewi.alert_color || "#10b981";
        }
        if (leadTime) leadTime.textContent = cewi.estimated_lead_time || "Continuous routine monitoring";

        const reasonsList = document.getElementById("cewi-reasons-list");
        if (reasonsList && cewi.reasons) {
            reasonsList.innerHTML = cewi.reasons.map(r => `<li>${r}</li>`).join("");
        }
    }

    updateAlertBanner(alert) {
        const banner = document.getElementById("alarm-banner");
        if (!banner) return;

        if (!alert || alert.status !== "ACTIVE") {
            banner.classList.add("hidden");
            return;
        }

        banner.classList.remove("hidden");
        const title = document.getElementById("banner-alarm-title");
        const detail = document.getElementById("banner-alarm-detail");
        if (title) title.textContent = alert.priority;
        if (detail) detail.textContent = `${alert.patient_name} (${alert.location}): ${alert.trigger_condition}`;

        if (alert.audio_alarm && (alert.priority.includes("CRISIS") || alert.priority.includes("CODE BLUE"))) {
            this.playCrisisAlarmTone();
        }
    }

    updatePlaybackBar(playback) {
        if (!playback) return;
        const toggleBtn = document.getElementById("btn-playback-toggle");
        if (toggleBtn) {
            toggleBtn.textContent = (playback.status === "PAUSED") ? "▶ Resume" : "⏸ Pause";
        }

        const slider = document.getElementById("playback-seek-slider");
        const timeDisplay = document.getElementById("playback-time-display");
        if (slider && !this.isSeeking) {
            slider.value = Math.floor(playback.time_sec || 0);
        }
        if (timeDisplay) {
            const sec = Math.floor(playback.time_sec || 0);
            const m = Math.floor(sec / 60);
            const s = sec % 60;
            timeDisplay.textContent = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')} / 30:05`;
        }

        // Highlight active speed
        document.querySelectorAll(".speed-btn").forEach(btn => {
            const spd = parseFloat(btn.getAttribute("data-speed"));
            btn.classList.toggle("active", spd === playback.speed);
        });
    }

    updateMonitorPatientBar(p) {
        const badge = document.getElementById("pt-header-badge");
        const name = document.getElementById("pt-header-name");
        const meta = document.getElementById("pt-header-meta");

        if (badge) badge.textContent = p.patient_id;
        if (name) name.textContent = p.name;
        if (meta) {
            const sourceText = (p.data_mode === "MIT-BIH")
                ? `PhysioNet MIT-BIH Record ${p.mitbih_record_id || '100'} (MLII, V5)`
                : `Synthetic Multi-Lead Simulation (${p.current_rhythm})`;
            meta.textContent = `${p.location} • Age ${p.age}, ${p.gender} • Source: ${sourceText}`;
        }
    }

    // --- WARD CARDS RENDERING ---
    updateWardCard(pid, pData) {
        const grid = document.getElementById("ward-cards-grid");
        if (!grid) return;

        let card = document.getElementById(`ward-card-${pid}`);
        if (!card) {
            card = document.createElement("div");
            card.id = `ward-card-${pid}`;
            card.className = `ward-card ${pid === this.activePatientId ? 'active-selected' : ''}`;
            card.addEventListener("click", () => {
                this.selectActivePatient(pid);
                this.switchView("monitor");
            });
            grid.appendChild(card);
        }

        const vitals = pData.vitals || {};
        const cewi = pData.cewi || {};
        const ai = pData.ai || {};

        card.innerHTML = `
            <div class="ward-card-header">
                <div class="ward-bed-title">
                    <span class="ward-bed-id">${pid}</span>
                    <span class="ward-patient-name">${pData.name || pid}</span>
                </div>
                <span class="source-badge ${pData.data_mode === 'MIT-BIH' ? 'mitbih' : ''}">
                    ${pData.data_mode === 'MIT-BIH' ? `MIT-BIH ${pData.mitbih_record_id}` : 'Synthetic'}
                </span>
            </div>
            <div class="ward-card-body">
                <div class="ward-vitals-row">
                    <div class="ward-vital-cell">
                        <div class="ward-vital-label">HR</div>
                        <div class="ward-vital-value hr">${Math.round(vitals.hr || 72)}</div>
                    </div>
                    <div class="ward-vital-cell">
                        <div class="ward-vital-label">SpO2</div>
                        <div class="ward-vital-value spo2">${Math.round(vitals.spo2 || 98)}%</div>
                    </div>
                    <div class="ward-vital-cell">
                        <div class="ward-vital-label">NIBP</div>
                        <div class="ward-vital-value bp">${Math.round(vitals.bp_sys || 120)}/${Math.round(vitals.bp_dia || 80)}</div>
                    </div>
                    <div class="ward-vital-cell">
                        <div class="ward-vital-label">CEWI</div>
                        <div class="ward-vital-value cewi" style="color: ${cewi.alert_color || '#10b981'}">${cewi.cewi_score || 10}</div>
                    </div>
                </div>
                <div class="ward-footer-row">
                    <span class="rhythm-pill">${ai.diagnosis || pData.current_rhythm || 'Normal Sinus Rhythm'}</span>
                    <span style="color: ${cewi.alert_color || '#10b981'}; font-weight: 600;">${(cewi.risk_level || 'LOW RISK').split(' ')[0]} RISK</span>
                </div>
            </div>
        `;
    }

    // --- DATASET BROWSER CONTROLLER ---
    fetchInitialCatalog() {
        fetch("/api/dataset/records")
            .then(res => res.json())
            .then(data => {
                if (data.success && data.records) {
                    this.catalogRecords = data.records;
                    this.renderRecordsTable(this.catalogRecords);
                }
            })
            .catch(() => {});
    }

    renderRecordsTable(records) {
        const tbody = document.getElementById("records-table-body");
        if (!tbody) return;

        if (records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 24px;">No records matching search filter.</td></tr>`;
            return;
        }

        tbody.innerHTML = records.map(r => `
            <tr>
                <td><strong style="font-family: var(--font-mono); color: var(--accent-primary);">${r.record_id}</strong></td>
                <td><strong>${r.primary_diagnosis || 'Clinical ECG Telemetry'}</strong></td>
                <td>${(r.leads || ['MLII', 'V1']).join(', ')}</td>
                <td>${r.fs || 360} Hz</td>
                <td>${r.duration_formatted || '30m 05s'}</td>
                <td>${(r.sig_len || 650000).toLocaleString()}</td>
                <td>${(r.beat_count || 0) > 0 ? (r.beat_count).toLocaleString() : 'Cataloged'}</td>
                <td>${r.is_paced ? '<span class="badge-pill warning">PACED</span>' : '<span style="color: var(--text-muted);">None</span>'}</td>
                <td><span class="badge-pill ${r.is_playable ? 'stable' : 'advisory'}">${r.file_status || 'Cataloged'}</span></td>
                <td>
                    <button class="btn btn-secondary btn-sm" onclick="app.previewRecordWaveform('${r.record_id}')">👁 Preview</button>
                    <button class="btn btn-primary btn-sm" onclick="app.openStreamModal('${r.record_id}')">▶ Stream</button>
                </td>
            </tr>
        `).join("");
    }

    openStreamModal(recordId) {
        this.selectedStreamRecordId = recordId;
        const title = document.getElementById("stream-modal-record-title");
        if (title) title.textContent = `Record ${recordId}`;
        const modal = document.getElementById("modal-stream-to-bed");
        if (modal) modal.classList.remove("hidden");
    }

    previewRecordWaveform(recordId) {
        const modal = document.getElementById("modal-preview-waveform");
        const title = document.getElementById("preview-modal-title");
        const meta = document.getElementById("preview-modal-metadata");
        if (title) title.textContent = `Previewing Waveform: Record ${recordId}`;
        if (modal) modal.classList.remove("hidden");

        fetch(`/api/dataset/segment?record_id=${recordId}&start_sec=0&duration_sec=6`)
            .then(res => res.json())
            .then(data => {
                if (data.success && data.segment) {
                    const seg = data.segment;
                    if (meta) {
                        meta.innerHTML = `
                            <strong>Channels:</strong> ${seg.lead1_name}, ${seg.lead2_name} • 
                            <strong>Sample Rate:</strong> ${seg.sample_rate} Hz • 
                            <strong>Source:</strong> ${seg.source} • 
                            <strong>Annotations in Slice:</strong> ${seg.annotations.length} beats
                        `;
                    }
                    this.drawPreviewWaveform(seg.lead1);
                }
            })
            .catch(() => {});
    }

    drawPreviewWaveform(samples) {
        const canvas = document.getElementById("canvas-preview");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        this.drawFullGrid(ctx, w, h);

        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        const midY = h * 0.52;
        const scaleY = h * 0.3;
        const step = w / samples.length;

        samples.forEach((val, i) => {
            const x = i * step;
            const y = midY - (val * scaleY);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    // --- MACHINE LEARNING LABORATORY ---
    fetchMLMetrics() {
        fetch("/api/ml/metrics")
            .then(res => res.json())
            .then(data => {
                if (data.success && data.report) {
                    this.renderMLDashboard(data.report);
                }
            })
            .catch(() => {});
    }

    renderMLDashboard(report) {
        const balAcc = document.getElementById("ml-bal-acc-val");
        const acc = document.getElementById("ml-acc-val");
        const f1 = document.getElementById("ml-f1-val");
        const algoTag = document.getElementById("ml-algo-tag");

        if (balAcc) balAcc.textContent = `${report.balanced_accuracy_percent}%`;
        if (acc) acc.textContent = `${report.accuracy_percent}%`;
        if (f1) f1.textContent = report.macro_f1;
        if (algoTag) algoTag.textContent = report.model_name;

        // Confusion Matrix
        const cmTable = document.getElementById("ml-confusion-matrix-table");
        if (cmTable && report.confusion_matrix && report.classes) {
            let headerHtml = `<tr><th>Actual \\ Pred</th>` + report.classes.map(c => `<th>${c.split(' ')[0]}</th>`).join("") + `</tr>`;
            let bodyHtml = report.confusion_matrix.map((row, rIdx) => {
                let cells = row.map((val, cIdx) => `<td class="${rIdx === cIdx ? 'cell-highlight' : ''}">${val}</td>`).join("");
                return `<tr><td><strong>${report.classes[rIdx].split(' ')[0]}</strong></td>${cells}</tr>`;
            }).join("");
            cmTable.innerHTML = `<thead>${headerHtml}</thead><tbody>${bodyHtml}</tbody>`;
        }

        // Per-class metrics
        const perClassTbody = document.getElementById("ml-per-class-tbody");
        if (perClassTbody && report.per_class_metrics) {
            perClassTbody.innerHTML = Object.entries(report.per_class_metrics).map(([cls, m]) => `
                <tr>
                    <td><strong>${cls}</strong></td>
                    <td>${(m.precision * 100).toFixed(1)}%</td>
                    <td>${(m.recall * 100).toFixed(1)}%</td>
                    <td><strong style="color: #38bdf8;">${m.f1_score.toFixed(3)}</strong></td>
                    <td>${m.support}</td>
                </tr>
            `).join("");
        }

        // Feature Importance Bars
        const featContainer = document.getElementById("ml-feat-importance-container");
        if (featContainer && report.feature_importance) {
            featContainer.innerHTML = Object.entries(report.feature_importance)
                .sort((a, b) => b[1] - a[1])
                .map(([name, imp]) => {
                    const pct = Math.round(imp * 100);
                    return `
                        <div class="feat-imp-item">
                            <div class="feat-imp-labels">
                                <span style="color: var(--text-secondary);">${name}</span>
                                <span style="font-family: var(--font-mono); font-weight: 600;">${(imp * 100).toFixed(1)}%</span>
                            </div>
                            <div class="prob-track">
                                <div class="prob-fill" style="width: ${pct}%;"></div>
                            </div>
                        </div>
                    `;
                }).join("");
        }
    }

    triggerMLTraining() {
        const btn = document.getElementById("btn-ml-train");
        if (btn) btn.textContent = "⏳ Training Ensemble...";

        fetch("/api/ml/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ algorithm: "RandomForest", n_estimators: 100 })
        })
        .then(res => res.json())
        .then(data => {
            alert("Model training initiated in background. Progress will be displayed.");
            setTimeout(() => {
                this.fetchMLMetrics();
                if (btn) btn.textContent = "⚙️ Train & Validate Model";
            }, 3000);
        })
        .catch(() => {
            if (btn) btn.textContent = "⚙️ Train & Validate Model";
        });
    }

    // --- ALERTS LOG ---
    fetchAlerts() {
        fetch("/api/alerts")
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    this.renderAlertsTable(data.active_alerts, data.alert_history);
                }
            })
            .catch(() => {});
    }

    renderAlertsTable(activeAlerts, history) {
        const tbody = document.getElementById("alerts-table-body");
        if (!tbody) return;

        const all = [...(activeAlerts || []), ...(history || [])];
        if (all.length === 0) {
            tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 24px;">No alerts logged. System is operating normally.</td></tr>`;
            return;
        }

        tbody.innerHTML = all.map(a => `
            <tr>
                <td><strong style="font-family: var(--font-mono);">${a.alert_id}</strong></td>
                <td>${a.timestamp}</td>
                <td><span class="badge-pill stable">${a.patient_id}</span></td>
                <td>${a.patient_name}</td>
                <td><span class="badge-pill ${a.priority.includes('CRISIS') ? 'crisis' : (a.priority.includes('WARNING') ? 'warning' : 'advisory')}">${a.priority.split(' ')[0]}</span></td>
                <td><strong>${a.trigger_condition || a.trigger_reason}</strong></td>
                <td>${a.cewi_score || 10}</td>
                <td>${a.heart_rate || 72} bpm</td>
                <td>${a.status || 'ACTIVE'}</td>
                <td>
                    ${a.status === 'ACTIVE' ? `
                        <button class="btn btn-secondary btn-sm" onclick="app.ackAlert('${a.alert_id}')">✓ Ack</button>
                        <button class="btn btn-warning btn-sm" onclick="app.snoozeBed('${a.patient_id}', 5)">🔕 5m</button>
                        <button class="btn btn-danger btn-sm" onclick="app.resolveAlert('${a.alert_id}')">Resolve</button>
                    ` : '<span style="color: var(--text-muted); font-size: 11px;">Archived</span>'}
                </td>
            </tr>
        `).join("");
    }

    ackAlert(alertId) {
        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "ack", alert_id: alertId })
        }).then(() => this.fetchAlerts());
    }

    resolveAlert(alertId) {
        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "resolve", alert_id: alertId })
        }).then(() => this.fetchAlerts());
    }

    snoozeBed(patientId, minutes) {
        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "snooze", patient_id: patientId, minutes: minutes })
        }).then(() => this.fetchAlerts());
    }

    // --- SHIFT REPORTS ---
    fetchShiftReport() {
        fetch("/api/reports/shift")
            .then(res => res.json())
            .then(data => {
                if (data.success && data.beds) {
                    const timeEl = document.getElementById("shift-report-timestamp");
                    if (timeEl) timeEl.textContent = data.generated_at;

                    const tbody = document.getElementById("shift-report-tbody");
                    if (tbody) {
                        tbody.innerHTML = data.beds.map(b => `
                            <tr>
                                <td><strong>${b.bed}</strong></td>
                                <td>${b.patient_name}</td>
                                <td>${b.location}</td>
                                <td><span class="badge-pill stable">${b.data_source}</span></td>
                                <td>${b.heart_rate} bpm</td>
                                <td>${b.spo2}%</td>
                                <td>${b.bp}</td>
                                <td><strong>${b.diagnosis}</strong></td>
                                <td>${b.cewi_score} (${(b.risk_level || '').split(' ')[0]})</td>
                                <td>${b.sdnn} ms</td>
                                <td>${b.qrs_ms} ms</td>
                            </tr>
                        `).join("");
                    }
                }
            })
            .catch(() => {});
    }

    // --- SYSTEM HEALTH DIAGNOSTICS ---
    fetchHealthDiagnostics() {
        fetch("/health")
            .then(res => res.json())
            .then(data => {
                const el = document.getElementById("health-json-display");
                if (el) el.textContent = JSON.stringify(data, null, 2);
            })
            .catch(() => {});
    }

    // --- EVENT LISTENERS & SETUP ---
    setupEventListeners() {
        // Theme toggle
        const themeBtn = document.getElementById("btn-toggle-theme");
        if (themeBtn) themeBtn.addEventListener("click", () => this.toggleTheme());

        // Audio toggle
        const audioBtn = document.getElementById("btn-toggle-audio");
        if (audioBtn) audioBtn.addEventListener("click", () => this.toggleAudioMute());

        // Playback toggle
        const playBtn = document.getElementById("btn-playback-toggle");
        if (playBtn) {
            playBtn.addEventListener("click", () => {
                const cur = this.patients[this.activePatientId];
                const newStatus = (cur && cur.playback && cur.playback.status === "PAUSED") ? "play" : "pause";
                fetch(`/api/bed/${this.activePatientId}/playback`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: newStatus })
                });
            });
        }

        // Freeze display
        const freezeBtn = document.getElementById("btn-freeze-monitor");
        if (freezeBtn) {
            freezeBtn.addEventListener("click", () => {
                this.isFrozen = !this.isFrozen;
                freezeBtn.textContent = this.isFrozen ? "▶ Resume Sweep" : "⏸ Freeze Display";
            });
        }

        // Seek slider
        const seekSlider = document.getElementById("playback-seek-slider");
        if (seekSlider) {
            seekSlider.addEventListener("mousedown", () => { this.isSeeking = true; });
            seekSlider.addEventListener("mouseup", () => {
                this.isSeeking = false;
                fetch(`/api/bed/${this.activePatientId}/playback`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: "seek", value: parseFloat(seekSlider.value) })
                });
            });
        }

        // Next / Prev Event buttons
        const prevEvtBtn = document.getElementById("btn-playback-prev-event");
        if (prevEvtBtn) {
            prevEvtBtn.addEventListener("click", () => {
                fetch(`/api/bed/${this.activePatientId}/playback`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: "jump", value: "prev" })
                });
            });
        }

        const nextEvtBtn = document.getElementById("btn-playback-next-event");
        if (nextEvtBtn) {
            nextEvtBtn.addEventListener("click", () => {
                fetch(`/api/bed/${this.activePatientId}/playback`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: "jump", value: "next" })
                });
            });
        }

        // Playback speeds
        document.querySelectorAll(".speed-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const spd = parseFloat(btn.getAttribute("data-speed"));
                fetch(`/api/bed/${this.activePatientId}/playback`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: "speed", value: spd })
                });
            });
        });

        // Record search input
        const searchInput = document.getElementById("record-search-input");
        if (searchInput) {
            searchInput.addEventListener("input", (e) => {
                const term = e.target.value.toLowerCase().trim();
                const filtered = this.catalogRecords.filter(r =>
                    r.record_id.toLowerCase().includes(term) ||
                    (r.primary_diagnosis && r.primary_diagnosis.toLowerCase().includes(term))
                );
                this.renderRecordsTable(filtered);
            });
        }

        // ML Train button
        const trainBtn = document.getElementById("btn-ml-train");
        if (trainBtn) trainBtn.addEventListener("click", () => this.triggerMLTraining());

        // Stream modal confirm button
        const confirmStreamBtn = document.getElementById("btn-confirm-stream-bed");
        if (confirmStreamBtn) {
            confirmStreamBtn.addEventListener("click", () => {
                const checked = document.querySelector('input[name="target_stream_bed"]:checked');
                const targetBed = checked ? checked.value : "BED-01";
                fetch(`/api/bed/${targetBed}/record`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ record_id: this.selectedStreamRecordId })
                }).then(() => {
                    document.getElementById("modal-stream-to-bed").classList.add("hidden");
                    this.selectActivePatient(targetBed);
                    this.switchView("monitor");
                });
            });
        }

        // Modal close buttons
        document.querySelectorAll("[data-close]").forEach(btn => {
            btn.addEventListener("click", () => {
                const targetId = btn.getAttribute("data-close");
                const m = document.getElementById(targetId);
                if (m) m.classList.add("hidden");
            });
        });

        // Export Alerts CSV
        const exportAlertsBtn = document.getElementById("btn-export-alerts-csv");
        if (exportAlertsBtn) {
            exportAlertsBtn.addEventListener("click", () => {
                window.location.href = "/api/alerts/export.csv";
            });
        }

        // Export Shift Report CSV
        const exportShiftCsvBtn = document.getElementById("btn-export-shift-csv");
        if (exportShiftCsvBtn) {
            exportShiftCsvBtn.addEventListener("click", () => {
                window.location.href = "/api/reports/shift/export";
            });
        }

        // Print shift report
        const printShiftBtn = document.getElementById("btn-print-shift-report");
        if (printShiftBtn) {
            printShiftBtn.addEventListener("click", () => window.print());
        }

        // Injection execution button in settings
        const execInjBtn = document.getElementById("btn-execute-injection");
        if (execInjBtn) {
            execInjBtn.addEventListener("click", () => {
                const bed = document.getElementById("inj-bed-select").value;
                const rhythm = document.getElementById("inj-rhythm-select").value;
                fetch("/api/trigger_arrhythmia", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ patient_id: bed, rhythm: rhythm, duration: 30.0 })
                }).then(res => res.json()).then(d => {
                    alert(`Simulated ${rhythm} injected into ${bed} for 30 seconds.`);
                });
            });
        }

        // Silence / Ack Banner buttons
        const bannerSilenceBtn = document.getElementById("btn-silence-alarm");
        if (bannerSilenceBtn) {
            bannerSilenceBtn.addEventListener("click", () => {
                this.snoozeBed(this.activePatientId, 5);
                document.getElementById("alarm-banner").classList.add("hidden");
            });
        }

        const bannerAckBtn = document.getElementById("btn-ack-alarm");
        if (bannerAckBtn) {
            bannerAckBtn.addEventListener("click", () => {
                const cur = this.patients[this.activePatientId];
                if (cur && cur.active_alert) {
                    this.ackAlert(cur.active_alert.alert_id);
                }
                document.getElementById("alarm-banner").classList.add("hidden");
            });
        }
    }

    startClock() {
        setInterval(() => {
            const clockEl = document.getElementById("clock-time");
            if (clockEl) {
                const d = new Date();
                clockEl.textContent = d.toTimeString().split(' ')[0];
            }
        }, 1000);
    }
}

// Global application singleton
let app = null;
window.addEventListener("DOMContentLoaded", () => {
    app = new CardioSentinelApp();
    window.app = app;
});

/**
 * CardioSentinel | ECG Intelligence - Central Telemetry & Arrhythmia Research Platform
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

        // Adjustable Alert Volume & Master Audio Configuration
        const savedVol = localStorage.getItem("cardiosentinel_alert_volume");
        this.alertVolume = (savedVol !== null) ? Math.max(0, Math.min(100, parseInt(savedVol, 10))) : 70;
        this.previousVolume = (this.alertVolume > 0) ? this.alertVolume : 70;
        const savedMuted = localStorage.getItem("cardiosentinel_alert_muted");
        this.isMuted = (savedMuted === "true") || (this.alertVolume === 0);
        this.audioEnabled = !this.isMuted;
        this.audioContext = null;
        this.lastAlarmToneTime = 0;
        this.lastWarningToneTime = 0;
        this.activeBannerAlert = null;
        this.unreadAlertCount = 0;
        this.isFrozen = false;
        this.activeView = "monitor";

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
        this.currentPreviewRecordId = "100";
        this.allAlertsData = [];
        this.wardSortKey = "bed";

        // Modernized Features State: Calipers, Morphology Zones, VCG & Copilot
        this.calipersActive = false;
        this.caliper1X = 240;
        this.caliper2X = 380;
        this.draggingCaliper = null;
        this.lastRenderedRPeaks = [];
        this.morphologyZonesActive = false;
        this.vcgRunning = false;
        this.vcgAnimFrame = null;
        this.vcgPaused = false;
        this.recentDipoles = [];

        this.init();
    }

    init() {
        this.setupTheme();
        this.setupAudio();
        this.setupVolumeControls();
        this.setupNavigation();
        this.setupBedSelectors();
        this.setupCanvases();
        this.setupEventListeners();
        this.setupCalipers();
        this.setupMorphologyToggle();
        this.setupVCGModal();
        this.setupCopilotModal();
        this.startClock();
        this.fetchInitialPatients();
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

    // --- AUDIO SYSTEM & ADJUSTABLE VOLUME CONTROL ---
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

    setupVolumeControls() {
        this.syncVolumeUI();

        // Settings Slider
        const settingsSlider = document.getElementById("settings-vol-slider");
        if (settingsSlider) {
            settingsSlider.value = this.isMuted ? 0 : this.alertVolume;
            settingsSlider.addEventListener("input", (e) => {
                this.setVolume(parseInt(e.target.value, 10));
            });
        }

        // Drawer Slider
        const drawerSlider = document.getElementById("drawer-vol-slider");
        if (drawerSlider) {
            drawerSlider.value = this.isMuted ? 0 : this.alertVolume;
            drawerSlider.addEventListener("input", (e) => {
                this.setVolume(parseInt(e.target.value, 10));
            });
        }

        // Settings Mute Toggle
        const settingsMuteBtn = document.getElementById("btn-settings-toggle-mute");
        if (settingsMuteBtn) {
            settingsMuteBtn.addEventListener("click", () => this.toggleAudioMute());
        }

        // Drawer Mute Toggle
        const drawerMuteBtn = document.getElementById("btn-drawer-mute");
        if (drawerMuteBtn) {
            drawerMuteBtn.addEventListener("click", () => this.toggleAudioMute());
        }

        // Test Sound Buttons
        const settingsTestBtn = document.getElementById("btn-settings-test-sound");
        if (settingsTestBtn) {
            settingsTestBtn.addEventListener("click", () => this.playTestTone());
        }
        const drawerTestBtn = document.getElementById("btn-drawer-test-sound");
        if (drawerTestBtn) {
            drawerTestBtn.addEventListener("click", () => this.playTestTone());
        }
    }

    setVolume(pct) {
        pct = Math.max(0, Math.min(100, Math.round(pct)));
        this.alertVolume = pct;
        if (pct === 0) {
            this.isMuted = true;
            this.audioEnabled = false;
        } else {
            this.previousVolume = pct;
            this.isMuted = false;
            this.audioEnabled = true;
        }

        try {
            localStorage.setItem("cardiosentinel_alert_volume", pct.toString());
            localStorage.setItem("cardiosentinel_alert_muted", this.isMuted ? "true" : "false");
        } catch (e) {}

        this.syncVolumeUI();
    }

    toggleAudioMute() {
        if (this.isMuted || this.alertVolume === 0) {
            this.isMuted = false;
            this.audioEnabled = true;
            this.alertVolume = (this.previousVolume > 0) ? this.previousVolume : 70;
        } else {
            this.previousVolume = this.alertVolume;
            this.isMuted = true;
            this.audioEnabled = false;
        }

        try {
            localStorage.setItem("cardiosentinel_alert_volume", this.alertVolume.toString());
            localStorage.setItem("cardiosentinel_alert_muted", this.isMuted ? "true" : "false");
        } catch (e) {}

        this.syncVolumeUI();

        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "toggle_mute" })
        }).catch(() => {});
    }

    syncVolumeUI() {
        const displayPct = this.isMuted ? 0 : this.alertVolume;
        const iconChar = (this.isMuted || displayPct === 0) ? "🔇" : (displayPct < 50 ? "🔉" : "🔊");

        // Sync Sliders
        const sSlider = document.getElementById("settings-vol-slider");
        if (sSlider && document.activeElement !== sSlider) sSlider.value = displayPct;
        const dSlider = document.getElementById("drawer-vol-slider");
        if (dSlider && document.activeElement !== dSlider) dSlider.value = displayPct;

        // Sync Labels
        const sVal = document.getElementById("settings-vol-val");
        if (sVal) sVal.textContent = `${displayPct}%`;
        const dVal = document.getElementById("drawer-vol-val");
        if (dVal) dVal.textContent = `${displayPct}%`;

        // Sync Icons and Buttons
        const headIcon = document.getElementById("audio-icon");
        if (headIcon) headIcon.textContent = iconChar;
        const dIcon = document.getElementById("drawer-vol-icon");
        if (dIcon) dIcon.textContent = iconChar;
        const dMuteBtn = document.getElementById("btn-drawer-mute");
        if (dMuteBtn) dMuteBtn.textContent = iconChar;
        const sMuteBtn = document.getElementById("btn-settings-toggle-mute");
        if (sMuteBtn) sMuteBtn.textContent = this.isMuted ? "Unmute" : "Mute";
        const sPill = document.getElementById("settings-audio-status-pill");
        if (sPill) {
            sPill.textContent = this.isMuted ? "Muted" : "Active";
            sPill.className = `badge-pill ${this.isMuted ? "crisis" : "stable"}`;
        }
    }

    getEffectiveGain(baseGain = 0.08) {
        if (this.isMuted || this.alertVolume === 0) return 0;
        return baseGain * (this.alertVolume / 100);
    }

    playHeartBeatBlip() {
        if (this.isMuted || this.alertVolume === 0 || !this.audioContext || this.audioContext.state !== "running") return;
        try {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();
            const vol = this.getEffectiveGain(0.015);
            if (vol <= 0) return;
            osc.type = "sine";
            osc.frequency.setValueAtTime(880, this.audioContext.currentTime);
            gain.gain.setValueAtTime(vol, this.audioContext.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.0001, this.audioContext.currentTime + 0.04);
            osc.connect(gain);
            gain.connect(this.audioContext.destination);
            osc.start();
            osc.stop(this.audioContext.currentTime + 0.04);
        } catch (e) {}
    }

    playCrisisAlarmTone() {
        if (this.isMuted || this.alertVolume === 0 || !this.audioContext || this.audioContext.state !== "running") return;
        const now = Date.now();
        if (now - this.lastAlarmToneTime < 3500) return;
        this.lastAlarmToneTime = now;

        const baseVol = this.getEffectiveGain(0.08);
        if (baseVol <= 0) return;

        // IEC 60601-1-8 High Priority Crisis 5-pulse sequence
        const notes = [523.25, 659.25, 783.99, 659.25, 523.25];
        let t = this.audioContext.currentTime + 0.05;
        notes.forEach((freq, idx) => {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();
            osc.type = "triangle";
            osc.frequency.setValueAtTime(freq, t);
            gain.gain.setValueAtTime(baseVol, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
            osc.connect(gain);
            gain.connect(this.audioContext.destination);
            osc.start(t);
            osc.stop(t + 0.12);
            t += (idx === 2) ? 0.22 : 0.16;
        });
    }

    playWarningAlarmTone() {
        if (this.isMuted || this.alertVolume === 0 || !this.audioContext || this.audioContext.state !== "running") return;
        const now = Date.now();
        if (now - this.lastWarningToneTime < 3000) return;
        this.lastWarningToneTime = now;

        const baseVol = this.getEffectiveGain(0.06);
        if (baseVol <= 0) return;

        const notes = [587.33, 440.00];
        let t = this.audioContext.currentTime + 0.05;
        notes.forEach((freq) => {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(freq, t);
            gain.gain.setValueAtTime(baseVol, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.14);
            osc.connect(gain);
            gain.connect(this.audioContext.destination);
            osc.start(t);
            osc.stop(t + 0.14);
            t += 0.18;
        });
    }

    playTestTone() {
        if (!this.audioContext) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) this.audioContext = new AudioCtx();
        }
        if (this.audioContext && this.audioContext.state === "suspended") {
            this.audioContext.resume();
        }
        if (this.isMuted || this.alertVolume === 0) return;

        const baseVol = this.getEffectiveGain(0.07);
        if (baseVol <= 0) return;

        const notes = [523.25, 659.25, 783.99];
        let t = this.audioContext.currentTime + 0.04;
        notes.forEach((freq) => {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(freq, t);
            gain.gain.setValueAtTime(baseVol, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
            osc.connect(gain);
            gain.connect(this.audioContext.destination);
            osc.start(t);
            osc.stop(t + 0.15);
            t += 0.14;
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
                const pid = tab.getAttribute("data-patient-id") || tab.getAttribute("data-patient");
                if (pid) {
                    this.selectActivePatient(pid);
                    this.switchView("monitor");
                }
            });
        });

        // Mobile / Collapsible sidebar toggle
        const toggleBtn = document.getElementById("sidebar-toggle-btn");
        const sidebar = document.getElementById("app-sidebar");
        if (toggleBtn && sidebar) {
            toggleBtn.addEventListener("click", () => {
                sidebar.classList.toggle("collapsed");
                sidebar.classList.toggle("mobile-open");
            });
        }

        // Browser back/forward navigation support
        window.addEventListener("popstate", (e) => {
            const view = (e.state && e.state.view) || window.location.hash.replace("#", "") || "monitor";
            this.switchView(view, false);
        });

        // Initialize from URL hash if user landed on a direct view link, defaulting to monitor
        const initHash = window.location.hash.replace("#", "");
        if (initHash && document.getElementById(`view-${initHash}`)) {
            this.switchView(initHash, false);
        } else {
            this.switchView("monitor", false);
        }
    }

    fetchInitialPatients() {
        fetch("/api/patients")
            .then(res => res.json())
            .then(data => {
                if (data.success && Array.isArray(data.patients)) {
                    data.patients.forEach(p => {
                        this.patients[p.patient_id] = { ...(this.patients[p.patient_id] || {}), ...p };
                    });
                    const current = this.patients[this.activePatientId];
                    if (current) {
                        this.updateMonitorPatientBar(current);
                    }
                }
            })
            .catch(err => console.warn("Failed to fetch initial patients:", err));
    }

    setupBedSelectors() {
        // 1. Header bed select dropdown
        const headerSelect = document.getElementById("header-bed-select");
        if (headerSelect) {
            headerSelect.addEventListener("change", (e) => {
                const pid = e.target.value;
                this.selectActivePatient(pid);
                if (this.activeView !== "monitor") {
                    this.switchView("monitor");
                }
            });
        }

        // 2. Bedside monitor bed tabs
        document.querySelectorAll(".monitor-bed-tabs .monitor-bed-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const pid = btn.getAttribute("data-patient-id") || btn.getAttribute("data-patient");
                if (pid) {
                    this.selectActivePatient(pid);
                }
            });
        });

        // 3. Sidebar bed quick tabs
        document.querySelectorAll(".sidebar-bed-tabs .bed-tab").forEach(tab => {
            tab.addEventListener("click", () => {
                const pid = tab.getAttribute("data-patient-id") || tab.getAttribute("data-patient");
                if (pid) {
                    this.selectActivePatient(pid);
                    this.switchView("monitor");
                }
            });
        });
    }

    switchView(viewName, pushHistory = true) {
        if (!viewName) return;
        this.activeView = viewName;

        if (pushHistory && window.location.hash !== `#${viewName}`) {
            history.pushState({ view: viewName }, "", `#${viewName}`);
        }

        document.querySelectorAll(".sidebar-nav .nav-link").forEach(l => {
            const matches = l.getAttribute("data-view") === viewName;
            l.classList.toggle("active", matches);
            l.setAttribute("aria-selected", matches ? "true" : "false");
        });

        document.querySelectorAll(".view-pane").forEach(pane => {
            pane.classList.remove("active");
        });

        const activePane = document.getElementById(`view-${viewName}`);
        if (activePane) {
            activePane.classList.add("active");
            const content = document.querySelector(".app-content");
            if (content) content.scrollTop = 0;
        }

        // Auto close mobile sidebar on navigation if opened
        const sidebar = document.getElementById("app-sidebar");
        if (sidebar && sidebar.classList.contains("mobile-open")) {
            sidebar.classList.remove("mobile-open");
        }

        // View-specific data refreshes
        if (viewName === "records") this.fetchInitialCatalog();
        if (viewName === "ml") this.fetchMLMetrics();
        if (viewName === "alerts") this.fetchAlerts();
        if (viewName === "reports") this.fetchShiftReport();
        if (viewName === "health") this.fetchHealthDiagnostics();
    }

    selectActivePatient(patientId) {
        if (!patientId) return;
        this.activePatientId = patientId;

        // 1. Sync header dropdown
        const headerSelect = document.getElementById("header-bed-select");
        if (headerSelect && headerSelect.value !== patientId) {
            headerSelect.value = patientId;
        }

        // 2. Sync monitor bed tabs
        document.querySelectorAll(".monitor-bed-tabs .monitor-bed-btn").forEach(btn => {
            const pid = btn.getAttribute("data-patient-id") || btn.getAttribute("data-patient");
            const isActive = pid === patientId;
            btn.classList.toggle("active", isActive);
            btn.setAttribute("aria-selected", isActive ? "true" : "false");
        });

        // 3. Sync sidebar bed tabs
        document.querySelectorAll(".sidebar-bed-tabs .bed-tab").forEach(tab => {
            const pid = tab.getAttribute("data-patient-id") || tab.getAttribute("data-patient");
            const isActive = pid === patientId;
            tab.classList.toggle("active", isActive);
            tab.setAttribute("aria-selected", isActive ? "true" : "false");
        });

        // 4. Sync ward cards
        document.querySelectorAll(".ward-card").forEach(card => {
            card.classList.toggle("active-selected", card.id === `ward-card-${patientId}`);
        });

        // 5. Reset sweep and queues for clean waveform transition
        this.sweepX = 0;
        this.lead2Queue = [];
        this.leadV1Queue = [];
        this.recentDipoles = [];
        this.lastRenderedRPeaks = [];
        if (this.ctxLead2 && this.canvasLead2) {
            this.drawFullGrid(this.ctxLead2, this.canvasLead2.width, this.canvasLead2.height);
            this.lastYLead2 = this.canvasLead2.height / 2;
        }
        if (this.ctxLeadV1 && this.canvasLeadV1) {
            this.drawFullGrid(this.ctxLeadV1, this.canvasLeadV1.width, this.canvasLeadV1.height);
            this.lastYLeadV1 = this.canvasLeadV1.height / 2;
        }

        // 6. Update patient header bar and vitals
        const p = this.patients[patientId];
        if (p) {
            this.updateMonitorPatientBar(p);
            if (p.vitals || p.metrics) this.updateVitalsHUD(p.vitals, p.metrics);
            if (p.ai) this.updateAIIntelligence(p.ai);
            if (p.playback) this.updatePlaybackBar(p.playback);
        } else {
            fetch(`/api/patient/${patientId}`)
                .then(res => res.json())
                .then(data => {
                    if (data.success && data.patient) {
                        this.patients[patientId] = { ...(this.patients[patientId] || {}), ...data.patient };
                        this.updateMonitorPatientBar(this.patients[patientId]);
                    }
                })
                .catch(() => {});
        }
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

            // Store recent dipole vectors for Vectorcardiogram
            this.recentDipoles.push({ l2: pt2.l2, v1: ptV1.v1 });
            if (this.recentDipoles.length > 300) this.recentDipoles.shift();

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

            // R-peak blip pip & Morphological Zone Highlighting
            if (pt2.r) {
                this.lastRenderedRPeaks.push({ x: nextX, y: y2, time: Date.now() });
                if (this.lastRenderedRPeaks.length > 25) this.lastRenderedRPeaks.shift();

                this.ctxLead2.fillStyle = "#38bdf8";
                this.ctxLead2.beginPath();
                this.ctxLead2.arc(nextX, y2 - 10, 3, 0, Math.PI * 2);
                this.ctxLead2.fill();

                if (this.morphologyZonesActive) {
                    // P-Wave Zone (green tint)
                    this.ctxLead2.fillStyle = "rgba(16, 185, 129, 0.18)";
                    this.ctxLead2.fillRect(Math.max(0, nextX - 32), 10, 16, h2 - 20);

                    // QRS Zone (red tint)
                    this.ctxLead2.fillStyle = "rgba(239, 68, 68, 0.24)";
                    this.ctxLead2.fillRect(Math.max(0, nextX - 10), 10, 20, h2 - 20);

                    // T-Wave Zone (violet tint)
                    this.ctxLead2.fillStyle = "rgba(168, 85, 247, 0.18)";
                    this.ctxLead2.fillRect(Math.max(0, nextX + 16), 10, 28, h2 - 20);

                    // Zone Labels
                    this.ctxLead2.font = "bold 9px 'JetBrains Mono', monospace";
                    this.ctxLead2.fillStyle = "#10b981";
                    this.ctxLead2.fillText("P", Math.max(2, nextX - 26), 22);
                    this.ctxLead2.fillStyle = "#ef4444";
                    this.ctxLead2.fillText("QRS", Math.max(2, nextX - 8), 22);
                    this.ctxLead2.fillStyle = "#c084fc";
                    this.ctxLead2.fillText("T", Math.max(2, nextX + 26), 22);
                }

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
            this.updateReferenceAnnotations(active.ref_annotation, active.upcoming_annotations, active.playback);
            this.updateSignalStatus(active.metrics, active.playback, active.cewi);
            this.updateCEWIGauge(active.cewi);
            this.updatePlaybackBar(active.playback);
        }

        // Determine top qualifying active alert across all beds
        let qualifyingAlert = null;
        let activeAlertCount = 0;

        for (const [pid, pData] of Object.entries(data)) {
            this.patients[pid] = { ...(this.patients[pid] || {}), ...pData };
            if (pData.active_alert) {
                activeAlertCount++;
                const alt = pData.active_alert;
                if (alt.status === "ACTIVE" && !alt.acknowledged && alt.status !== "SNOOZED") {
                    const isCrit = alt.priority && (alt.priority.includes("CRITICAL") || alt.priority.includes("CRISIS") || alt.priority.includes("CODE BLUE"));
                    if (isCrit) {
                        qualifyingAlert = alt;
                    } else if (!qualifyingAlert) {
                        qualifyingAlert = alt;
                    }
                }
            }
            this.updateWardCard(pid, pData);
        }

        // Active bed gets priority if qualifying
        if (active && active.active_alert && active.active_alert.status === "ACTIVE" && !active.active_alert.acknowledged && active.active_alert.status !== "SNOOZED") {
            const isCrit = active.active_alert.priority && (active.active_alert.priority.includes("CRITICAL") || active.active_alert.priority.includes("CRISIS") || active.active_alert.priority.includes("CODE BLUE"));
            if (isCrit || !qualifyingAlert) {
                qualifyingAlert = active.active_alert;
            }
        }

        this.updateAlertBanner(qualifyingAlert);

        // Update header alert badge (matching unread or active count)
        const badge = document.getElementById("active-alert-count");
        if (badge) {
            badge.textContent = (this.unreadAlertCount !== undefined && this.unreadAlertCount > 0) ? this.unreadAlertCount : activeAlertCount;
        }
    }

    // --- UI UPDATERS ---
    updateVitalsHUD(vitals, metrics) {
        if (!vitals) return;

        // 1. Heart Rate
        const hr = Math.round(vitals.hr || (metrics && metrics.heart_rate) || 72);
        const hrEl = document.getElementById("val-hr");
        if (hrEl) hrEl.textContent = hr;

        const hrStatus = document.getElementById("hr-status");
        if (hrStatus) {
            if (hr > 100) {
                hrStatus.textContent = "TACHYCARDIA (>100)";
                hrStatus.style.color = "var(--alarm-crisis)";
            } else if (hr < 60 && hr > 0) {
                hrStatus.textContent = "BRADYCARDIA (<60)";
                hrStatus.style.color = "var(--alarm-warning)";
            } else if (hr === 0) {
                hrStatus.textContent = "ASYSTOLE / ARREST";
                hrStatus.style.color = "var(--alarm-crisis)";
            } else {
                hrStatus.textContent = "NORMAL (60-100)";
                hrStatus.style.color = "#64748b";
            }
        }

        // 2. QRS Duration
        const qrs = (metrics && metrics.qrs_duration_ms) ? parseFloat(metrics.qrs_duration_ms).toFixed(1) : "88.0";
        const qrsEl = document.getElementById("val-qrs");
        if (qrsEl) qrsEl.textContent = qrs;
        const qrsStatus = document.getElementById("qrs-status");
        if (qrsStatus) {
            if (parseFloat(qrs) >= 120.0) {
                qrsStatus.textContent = "WIDE QRS (>=120 ms)";
                qrsStatus.style.color = "var(--alarm-warning)";
            } else {
                qrsStatus.textContent = "NARROW (<120 ms)";
                qrsStatus.style.color = "#64748b";
            }
        }

        // 3. Bazett QTc
        const qtc = (metrics && metrics.qtc_ms) ? parseFloat(metrics.qtc_ms).toFixed(1) : "412.0";
        const qtcEl = document.getElementById("val-qtc");
        if (qtcEl) qtcEl.textContent = qtc;
        const qtcStatus = document.getElementById("qtc-status");
        if (qtcStatus) {
            if (parseFloat(qtc) > 470.0) {
                qtcStatus.textContent = "PROLONGED (>470 ms)";
                qtcStatus.style.color = "var(--alarm-crisis)";
            } else if (parseFloat(qtc) >= 450.0) {
                qtcStatus.textContent = "BORDERLINE (450-470)";
                qtcStatus.style.color = "var(--alarm-warning)";
            } else {
                qtcStatus.textContent = "NORMAL (<450 ms)";
                qtcStatus.style.color = "#64748b";
            }
        }

        // 4. Mean RR Interval
        const meanRr = (metrics && (metrics.mean_rr_ms || metrics.rr_mean_ms))
            ? parseFloat(metrics.mean_rr_ms || metrics.rr_mean_ms).toFixed(1)
            : (hr > 0 ? (60000 / hr).toFixed(1) : "833.3");
        const rrEl = document.getElementById("val-rr");
        if (rrEl) rrEl.textContent = meanRr;
        const rrStatus = document.getElementById("rr-status");
        if (rrStatus) {
            rrStatus.textContent = "REGULAR RHYTHM";
            rrStatus.style.color = "#64748b";
        }

        // 5. HRV SDNN
        const sdnn = (metrics && metrics.hrv_sdnn !== undefined) ? parseFloat(metrics.hrv_sdnn).toFixed(1) : "45.0";
        const sdnnEl = document.getElementById("val-sdnn");
        if (sdnnEl) sdnnEl.textContent = sdnn;
        const sdnnStatus = document.getElementById("sdnn-status");
        if (sdnnStatus) {
            if (parseFloat(sdnn) < 25.0) {
                sdnnStatus.textContent = "REDUCED VARIABILITY";
                sdnnStatus.style.color = "var(--alarm-warning)";
            } else {
                sdnnStatus.textContent = "AUTONOMIC TONE";
                sdnnStatus.style.color = "#64748b";
            }
        }

        // 6. HRV RMSSD
        const rmssd = (metrics && metrics.hrv_rmssd !== undefined) ? parseFloat(metrics.hrv_rmssd).toFixed(1) : "35.0";
        const rmssdEl = document.getElementById("val-rmssd");
        if (rmssdEl) rmssdEl.textContent = rmssd;
        const rmssdStatus = document.getElementById("rmssd-status");
        if (rmssdStatus) {
            if (parseFloat(rmssd) < 20.0) {
                rmssdStatus.textContent = "VAGAL DEPRESSION";
                rmssdStatus.style.color = "var(--alarm-warning)";
            } else {
                rmssdStatus.textContent = "PARASYMPATHETIC";
                rmssdStatus.style.color = "#64748b";
            }
        }

        // Compatibility bindings
        const pnn50El = document.getElementById("val-pnn50");
        if (pnn50El) pnn50El.textContent = `${metrics?.pnn50 || 15.0} %`;
        const entropyEl = document.getElementById("val-entropy");
        if (entropyEl) entropyEl.textContent = metrics?.rr_entropy || 0.65;
        const sqEl = document.getElementById("val-sq");
        if (sqEl) sqEl.textContent = metrics?.signal_quality || "OPTIMAL";
    }

    updateReferenceAnnotations(refAnn, upcomingAnns, playback) {
        if (!refAnn) {
            refAnn = { symbol: "N", name: "Normal Beat (N)", time_sec: 0, is_ventricular: false };
        }
        const symEl = document.getElementById("ref-ann-symbol");
        const nameEl = document.getElementById("ref-ann-name");
        const metaEl = document.getElementById("ref-ann-meta");
        const leadTag = document.getElementById("ref-annotation-tag");

        if (symEl) {
            symEl.textContent = refAnn.symbol || "N";
            symEl.className = "ref-symbol-badge";
            if (refAnn.is_ventricular || refAnn.symbol === "V" || refAnn.symbol === "E" || refAnn.symbol === "!") {
                symEl.classList.add("crit");
            } else if (refAnn.symbol === "N") {
                symEl.classList.add("norm");
            }
        }

        if (nameEl) {
            nameEl.textContent = refAnn.name || `Beat (${refAnn.symbol || 'N'})`;
        }

        if (metaEl) {
            const sec = Math.floor(playback?.time_sec || refAnn.time_sec || 0);
            const m = Math.floor(sec / 60);
            const s = sec % 60;
            const timeStr = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
            const rec = playback?.record_id || "100";
            metaEl.textContent = `Time: ${timeStr} / 30:05 • Lead MLII (Record ${rec})`;
        }

        if (leadTag) {
            leadTag.textContent = `Ref Beat: ${refAnn.symbol || 'N'}`;
            leadTag.className = `badge-pill ${refAnn.is_ventricular ? 'crisis' : 'stable'}`;
        }

        // Render upcoming annotations list
        const listEl = document.getElementById("upcoming-ann-list");
        if (listEl && Array.isArray(upcomingAnns)) {
            listEl.innerHTML = "";
            const items = upcomingAnns.slice(0, 4);
            if (items.length === 0) {
                listEl.innerHTML = '<div style="font-size: 11px; color: var(--text-muted); padding: 4px 0;">No pending ground-truth events in window.</div>';
            } else {
                items.forEach(ann => {
                    const row = document.createElement("div");
                    row.className = "upcoming-ann-item";
                    const isCrit = ann.is_ventricular || ann.symbol === "V" || ann.symbol === "E" || ann.symbol === "!";
                    const isNorm = ann.symbol === "N";
                    row.innerHTML = `
                        <span class="upcoming-time">+${(ann.offset_sec !== undefined ? ann.offset_sec : 0.8).toFixed(1)}s</span>
                        <span class="upcoming-sym ${isCrit ? 'crit' : (isNorm ? 'norm' : '')}">${ann.symbol}</span>
                        <span class="upcoming-name">${ann.name || ann.symbol}</span>
                    `;
                    listEl.appendChild(row);
                });
            }
        }
    }

    updateSignalStatus(metrics, playback, cewi) {
        const sqEl = document.getElementById("val-sq");
        const snrEl = document.getElementById("val-snr");
        const modeEl = document.getElementById("val-data-mode");
        const cewiEl = document.getElementById("val-cewi-risk");
        const lockBadge = document.getElementById("filter-lock-badge");

        const sq = (metrics && metrics.signal_quality) || "OPTIMAL";
        if (sqEl) {
            sqEl.textContent = sq;
            sqEl.style.color = (sq === "OPTIMAL") ? "#10b981" : "var(--alarm-warning)";
        }

        if (snrEl) {
            const snr = (metrics && metrics.snr_db !== undefined) ? metrics.snr_db : 24.5;
            snrEl.textContent = `${snr} dB`;
        }

        if (modeEl) {
            if (playback && playback.data_mode === "MIT-BIH") {
                modeEl.textContent = `MIT-BIH ${playback.record_id || '100'}`;
            } else {
                modeEl.textContent = "Synthetic Simulation";
            }
        }

        if (cewiEl) {
            const risk = (cewi && cewi.risk_level) ? cewi.risk_level : "LOW RISK";
            const score = (cewi && cewi.cewi_score !== undefined) ? cewi.cewi_score : 10;
            cewiEl.textContent = `${risk} (${score}/100)`;
            cewiEl.style.color = (cewi && cewi.alert_color) ? cewi.alert_color : "#10b981";
        }

        if (lockBadge) {
            lockBadge.textContent = (sq === "OPTIMAL") ? "Filter Lock: Active" : "Filter Lock: Degraded";
            lockBadge.className = `badge-pill ${sq === "OPTIMAL" ? 'stable' : 'warning'}`;
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

        if (!alert || alert.status !== "ACTIVE" || alert.acknowledged || alert.status === "SNOOZED") {
            banner.classList.add("hidden");
            this.activeBannerAlert = null;
            return;
        }

        this.activeBannerAlert = alert;
        banner.classList.remove("hidden");

        const iconEl = document.getElementById("banner-alarm-icon");
        const badgeEl = document.getElementById("banner-alarm-badge");
        const bedEl = document.getElementById("banner-bed-id");
        const titleEl = document.getElementById("banner-alarm-title");
        const timeEl = document.getElementById("banner-alarm-time");
        const sourceEl = document.getElementById("banner-alarm-source");
        const explEl = document.getElementById("banner-alarm-explanation");

        const isCrit = alert.priority && (alert.priority.includes("CRITICAL") || alert.priority.includes("CRISIS") || alert.priority.includes("CODE BLUE"));
        if (iconEl) iconEl.textContent = isCrit ? "🚨" : "⚠️";
        if (badgeEl) {
            badgeEl.textContent = isCrit ? "CRITICAL ALERT" : "WARNING ALERT";
            badgeEl.className = `alarm-badge ${isCrit ? "critical" : "warning"}`;
        }
        if (bedEl) bedEl.textContent = alert.patient_id || "BED-01";
        if (titleEl) titleEl.textContent = alert.trigger_condition || alert.diagnosis || "Cardiac Arrhythmia Event";
        if (timeEl) timeEl.textContent = alert.timestamp ? (alert.timestamp.split(" ")[1] || alert.timestamp) : "--:--:--";

        if (sourceEl) {
            const isBenchmark = alert.source_type === "RECORDED_BENCHMARK" || (alert.source_record && alert.source_record !== "SYNTHETIC");
            sourceEl.textContent = isBenchmark
                ? `MIT-BIH Record ${alert.source_record || '100'} (Recorded Signal)`
                : "Real-Time Telemetry Simulation";
        }

        if (explEl) {
            explEl.textContent = alert.explanation || `${alert.patient_name || alert.patient_id}: electrophysiological rhythm pattern detected during automated monitoring.`;
        }

        // Trigger audible tone only if enabled, unmuted, and not at 0%
        if (alert.audio_alarm && !this.isMuted && this.alertVolume > 0) {
            if (isCrit) {
                this.playCrisisAlarmTone();
            } else if (alert.priority && alert.priority.includes("WARNING")) {
                this.playWarningAlarmTone();
            }
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
        if (!p) return;
        const badge = document.getElementById("pt-header-badge");
        const name = document.getElementById("pt-header-name");
        const meta = document.getElementById("pt-header-meta");

        const pid = p.patient_id || this.activePatientId;
        if (badge) badge.textContent = pid;
        if (name) name.textContent = p.name || `Patient ${pid}`;
        if (meta) {
            const loc = p.location || "Cardiac ICU";
            const age = p.age || "--";
            const gender = p.gender || "--";
            const sourceText = (p.data_mode === "MIT-BIH" || p.mitbih_record_id)
                ? `PhysioNet MIT-BIH Record ${p.mitbih_record_id || (p.playback && p.playback.record_id) || '100'} (MLII, V5)`
                : `Synthetic Multi-Lead Simulation (${p.current_rhythm || 'NSR'})`;
            meta.textContent = `${loc} • Age ${age}, ${gender} • Source: ${sourceText}`;
        }
    }

    // --- WARD CARDS RENDERING & SORTING ---
    reorderWardCards() {
        const grid = document.getElementById("ward-cards-grid");
        if (!grid) return;
        const cards = Array.from(grid.children);
        if (cards.length === 0) return;

        cards.sort((a, b) => {
            const pidA = a.id.replace("ward-card-", "");
            const pidB = b.id.replace("ward-card-", "");
            const pA = this.patients[pidA] || {};
            const pB = this.patients[pidB] || {};

            if (this.wardSortKey === "severity") {
                const cewiA = pA.cewi?.cewi_score || 0;
                const cewiB = pB.cewi?.cewi_score || 0;
                return cewiB - cewiA;
            } else if (this.wardSortKey === "hr") {
                const hrA = pA.vitals?.hr || 70;
                const hrB = pB.vitals?.hr || 70;
                return hrB - hrA;
            } else {
                return pidA.localeCompare(pidB);
            }
        });

        cards.forEach(card => grid.appendChild(card));
    }

    updateWardCard(pid, pData) {
        const grid = document.getElementById("ward-cards-grid");
        if (!grid) return;

        let isNewCard = false;
        let card = document.getElementById(`ward-card-${pid}`);
        if (!card) {
            isNewCard = true;
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

        if (isNewCard && this.wardSortKey !== "bed") {
            this.reorderWardCards();
        }
    }

    // --- DATASET BROWSER CONTROLLER ---
    fetchInitialCatalog() {
        fetch("/api/dataset/records")
            .then(res => res.json())
            .then(data => {
                if (data.success && data.records) {
                    this.catalogRecords = data.records;
                    this.applyRecordFilters();
                }
            })
            .catch(() => {});
    }

    applyRecordFilters() {
        const term = (document.getElementById("record-search-input")?.value || "").toLowerCase().trim();
        const rhythm = document.getElementById("filter-rhythm-select")?.value || "ALL";
        const status = document.getElementById("filter-status-select")?.value || "ALL";

        const filtered = this.catalogRecords.filter(r => {
            const matchesSearch = !term ||
                (r.record_id && r.record_id.toLowerCase().includes(term)) ||
                (r.primary_diagnosis && r.primary_diagnosis.toLowerCase().includes(term)) ||
                (r.leads && r.leads.some(l => l.toLowerCase().includes(term)));

            let matchesRhythm = true;
            if (rhythm !== "ALL") {
                const diag = (r.primary_diagnosis || "").toUpperCase();
                if (rhythm === "PACED") {
                    matchesRhythm = Boolean(r.is_paced) || diag.includes("PACED");
                } else if (rhythm === "NSR") {
                    matchesRhythm = diag.includes("NORMAL") || diag.includes("SINUS");
                } else if (rhythm === "VT") {
                    matchesRhythm = diag.includes("VENTRICULAR TACHYCARDIA") || diag.includes(" VT");
                } else if (rhythm === "VFIB") {
                    matchesRhythm = diag.includes("FIBRILLATION") || diag.includes("FLUTTER") || diag.includes("VF");
                } else if (rhythm === "PVC") {
                    matchesRhythm = diag.includes("PVC") || diag.includes("PREMATURE") || diag.includes("VENTRICULAR");
                } else if (rhythm === "AFIB") {
                    matchesRhythm = diag.includes("ATRIAL FIBRILLATION") || diag.includes("AFIB");
                }
            }

            let matchesStatus = true;
            if (status === "PLAYABLE") {
                matchesStatus = Boolean(r.is_playable);
            } else if (status === "CATALOGED") {
                matchesStatus = !r.is_playable;
            }

            return matchesSearch && matchesRhythm && matchesStatus;
        });

        this.renderRecordsTable(filtered);
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

    // --- MODAL DIALOGS CONTROLLER ---
    openModal(modalId) {
        const modal = typeof modalId === "string" ? document.getElementById(modalId) : modalId;
        if (!modal) return;
        modal.classList.remove("hidden", "closing");
        const firstFocus = modal.querySelector("button:not(.modal-close-btn), input, select");
        if (firstFocus) setTimeout(() => firstFocus.focus(), 60);
    }

    closeModal(modalId) {
        const modal = typeof modalId === "string" ? document.getElementById(modalId) : modalId;
        if (!modal || modal.classList.contains("hidden")) return;
        modal.classList.add("closing");
        setTimeout(() => {
            modal.classList.add("hidden");
            modal.classList.remove("closing");
        }, 180);
    }

    openStreamModal(recordId) {
        this.selectedStreamRecordId = recordId;
        const title = document.getElementById("stream-modal-record-title");
        if (title) title.textContent = `Record ${recordId}`;
        this.openModal("modal-stream-to-bed");
    }

    previewRecordWaveform(recordId) {
        this.currentPreviewRecordId = recordId;
        const title = document.getElementById("preview-modal-title");
        const meta = document.getElementById("preview-modal-metadata");
        if (title) title.textContent = `Previewing Waveform: Record ${recordId}`;
        this.openModal("modal-preview-waveform");

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

    // --- ALERTS LOG & NOTIFICATION DRAWER ---
    fetchAlerts() {
        fetch("/api/alerts")
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    const seen = new Set();
                    const combined = [];
                    for (const a of [...(data.active_alerts || []), ...(data.alert_history || [])]) {
                        if (a.alert_id && !seen.has(a.alert_id)) {
                            seen.add(a.alert_id);
                            combined.push(a);
                        }
                    }
                    this.allAlertsData = combined;
                    this.applyAlertFilters();

                    const unread = data.unread_count !== undefined ? data.unread_count : combined.filter(a => !a.read).length;
                    this.unreadAlertCount = unread;
                    this.renderDrawerAlerts(combined, unread);
                }
            })
            .catch(() => {});
    }

    renderDrawerAlerts(alertsList, unreadCount = 0) {
        const listEl = document.getElementById("drawer-alerts-list");
        const unreadBadge = document.getElementById("drawer-unread-badge");
        const headerBadge = document.getElementById("active-alert-count");

        if (unreadBadge) unreadBadge.textContent = `${unreadCount} unread`;
        if (headerBadge) headerBadge.textContent = unreadCount;

        if (!listEl) return;
        if (!alertsList || alertsList.length === 0) {
            listEl.innerHTML = `
                <div class="drawer-empty-state">
                    <span style="font-size: 28px; display: block; margin-bottom: 8px;">✅</span>
                    No active clinical alarms at this time.<br>
                    Continuous multi-lead telemetry is operational.
                </div>
            `;
            return;
        }

        listEl.innerHTML = alertsList.map(a => {
            const isCrit = (a.priority || "").includes("CRITICAL") || (a.priority || "").includes("CRISIS") || (a.priority || "").includes("CODE BLUE");
            const isUnread = !a.read;
            const timeStr = a.timestamp ? (a.timestamp.split(" ")[1] || a.timestamp) : "--:--:--";
            const sourcePill = (a.source_type === "RECORDED_BENCHMARK" || (a.source_record && a.source_record !== "SYNTHETIC"))
                ? `MIT-BIH #${a.source_record || '100'} Recorded`
                : "Telemetry Simulation";

            return `
                <div class="drawer-alert-card ${isCrit ? 'critical' : 'warning'} ${isUnread ? 'unread' : ''}" data-alert-id="${a.alert_id}">
                    <div class="drawer-card-header">
                        <div class="drawer-card-tags">
                            <span class="alarm-badge ${isCrit ? 'critical' : 'warning'}">${isCrit ? 'CRITICAL' : 'WARNING'}</span>
                            <span class="alarm-bed-tag">${a.patient_id}</span>
                            <span class="alarm-source-pill">${sourcePill}</span>
                        </div>
                        <span class="drawer-card-time">${timeStr}</span>
                    </div>
                    <div class="drawer-card-desc">${a.trigger_condition || a.diagnosis}</div>
                    <div class="drawer-card-meta">${a.explanation || `${a.patient_name || a.patient_id}: Electrophysiology rhythm alert.`}</div>
                    <div class="drawer-card-actions">
                        <button class="btn btn-secondary btn-xs" onclick="app.openAlertDetailsModalById('${a.alert_id}')">🔍 Details</button>
                        ${!a.acknowledged ? `
                            <button class="btn btn-primary btn-xs" onclick="app.ackAlert('${a.alert_id}')">✓ Ack</button>
                            <button class="btn btn-warning btn-xs" onclick="app.snoozeBed('${a.patient_id}', 5)">🔕 5m</button>
                        ` : '<span style="font-size: 11px; color: var(--accent-cyan); font-weight: 600;">✓ Acknowledged</span>'}
                        ${isUnread ? `
                            <button class="btn btn-secondary btn-xs" onclick="app.markAlertRead('${a.alert_id}')">Mark Read</button>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join("");
    }

    openNotificationDrawer() {
        const backdrop = document.getElementById("drawer-backdrop");
        const drawer = document.getElementById("notification-drawer");
        if (backdrop) backdrop.classList.remove("hidden");
        if (drawer) drawer.classList.remove("hidden");
        this.fetchAlerts();
    }

    closeNotificationDrawer() {
        const backdrop = document.getElementById("drawer-backdrop");
        const drawer = document.getElementById("notification-drawer");
        if (backdrop) backdrop.classList.add("hidden");
        if (drawer) drawer.classList.add("hidden");
    }

    toggleNotificationDrawer() {
        const drawer = document.getElementById("notification-drawer");
        if (drawer && !drawer.classList.contains("hidden")) {
            this.closeNotificationDrawer();
        } else {
            this.openNotificationDrawer();
        }
    }

    markAlertRead(alertId) {
        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "mark_read", alert_id: alertId })
        })
        .then(res => res.json())
        .then(() => this.fetchAlerts())
        .catch(() => {});
    }

    markAllAlertsRead() {
        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "mark_all_read" })
        })
        .then(res => res.json())
        .then(() => this.fetchAlerts())
        .catch(() => {});
    }

    openAlertDetailsModalById(alertId) {
        let alert = this.allAlertsData.find(a => a.alert_id === alertId);
        if (!alert && this.patients) {
            for (const p of Object.values(this.patients)) {
                if (p.active_alert && p.active_alert.alert_id === alertId) {
                    alert = p.active_alert;
                    break;
                }
            }
        }
        if (alert) {
            this.openAlertDetailsModal(alert);
        }
    }

    openAlertDetailsModal(alert) {
        if (!alert) return;
        this.activeDetailAlert = alert;

        const isCrit = (alert.priority || "").includes("CRITICAL") || (alert.priority || "").includes("CRISIS") || (alert.priority || "").includes("CODE BLUE");
        const iconEl = document.getElementById("modal-alert-icon");
        if (iconEl) iconEl.textContent = isCrit ? "🚨" : "⚠️";

        const titleEl = document.getElementById("modal-alert-title");
        if (titleEl) titleEl.textContent = `${isCrit ? 'Critical Alarm' : 'Clinical Alert'} — ${alert.patient_id}`;

        const idEl = document.getElementById("modal-alert-id");
        if (idEl) idEl.textContent = alert.alert_id || "--";

        const bedEl = document.getElementById("modal-alert-bed");
        if (bedEl) bedEl.textContent = `${alert.patient_id} (${alert.patient_name || alert.location || 'ICU Bed'})`;

        const sevEl = document.getElementById("modal-alert-severity");
        if (sevEl) sevEl.textContent = alert.priority || "WARNING";

        const timeEl = document.getElementById("modal-alert-time");
        if (timeEl) timeEl.textContent = alert.timestamp || "--:--:--";

        const srcEl = document.getElementById("modal-alert-source");
        if (srcEl) {
            srcEl.textContent = alert.source_label || (alert.source_type === "RECORDED_BENCHMARK"
                ? `PhysioNet MIT-BIH Arrhythmia Database Record ${alert.source_record || '100'} (Recorded Research Signal)`
                : "Continuous Multi-Lead Telemetry Sensor Simulation");
        }

        const condEl = document.getElementById("modal-alert-condition");
        if (condEl) condEl.textContent = alert.trigger_condition || alert.diagnosis || "--";

        const explEl = document.getElementById("modal-alert-explanation");
        if (explEl) {
            explEl.textContent = alert.explanation || `Electrophysiological rhythm pattern (${alert.diagnosis}) recorded during continuous monitoring. Review dual-lead sweep tracing for morphological confirmation.`;
        }

        const jumpBtn = document.getElementById("btn-modal-alert-jump");
        if (jumpBtn) {
            jumpBtn.onclick = () => {
                this.closeModal("modal-alert-details");
                if (alert.patient_id) this.selectActivePatient(alert.patient_id);
                this.switchView("monitor");
            };
        }

        const ackBtn = document.getElementById("btn-modal-alert-ack");
        if (ackBtn) {
            ackBtn.onclick = () => {
                this.ackAlert(alert.alert_id);
                this.closeModal("modal-alert-details");
            };
        }

        const snoozeBtn = document.getElementById("btn-modal-alert-snooze");
        if (snoozeBtn) {
            snoozeBtn.onclick = () => {
                this.snoozeBed(alert.patient_id, 5);
                this.closeModal("modal-alert-details");
            };
        }

        this.openModal("modal-alert-details");
    }

    applyAlertFilters() {
        const searchTerm = (document.getElementById("alert-search-input")?.value || "").toLowerCase().trim();
        const priorityFilter = document.getElementById("filter-alert-priority")?.value || "ALL";

        const filtered = this.allAlertsData.filter(a => {
            const matchesSearch = !searchTerm ||
                (a.alert_id && a.alert_id.toLowerCase().includes(searchTerm)) ||
                (a.patient_id && a.patient_id.toLowerCase().includes(searchTerm)) ||
                (a.patient_name && a.patient_name.toLowerCase().includes(searchTerm)) ||
                (a.trigger_condition && a.trigger_condition.toLowerCase().includes(searchTerm)) ||
                (a.trigger_reason && a.trigger_reason.toLowerCase().includes(searchTerm));

            const matchesPriority = priorityFilter === "ALL" ||
                (a.priority && a.priority.toUpperCase().includes(priorityFilter));

            return matchesSearch && matchesPriority;
        });

        this.renderAlertsTable(filtered);
    }

    renderAlertsTable(alertsList) {
        const tbody = document.getElementById("alerts-table-body");
        if (!tbody) return;

        const all = alertsList || [];
        if (all.length === 0) {
            tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 24px;">No alerts matching filter. System is operating normally.</td></tr>`;
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

        // Header alarms button (toggles notification drawer)
        const openAlarmsBtn = document.getElementById("btn-open-alarms");
        if (openAlarmsBtn) openAlarmsBtn.addEventListener("click", () => this.toggleNotificationDrawer());

        // Ward overview sorting & actions
        const wardSortSelect = document.getElementById("ward-sort-select");
        if (wardSortSelect) {
            wardSortSelect.addEventListener("change", (e) => {
                this.wardSortKey = e.target.value;
                this.reorderWardCards();
            });
        }

        const wardInjectBtn = document.getElementById("btn-ward-inject");
        if (wardInjectBtn) {
            wardInjectBtn.addEventListener("click", () => {
                this.switchView("settings");
                const target = document.getElementById("inj-rhythm-select");
                if (target) {
                    target.scrollIntoView({ behavior: "smooth", block: "center" });
                    target.focus();
                }
            });
        }

        // Bedside monitor action buttons
        const switchRecordBtn = document.getElementById("btn-switch-record-modal");
        if (switchRecordBtn) {
            switchRecordBtn.addEventListener("click", () => {
                const currentRecord = this.patients[this.activePatientId]?.mitbih_record_id || "100";
                this.openStreamModal(currentRecord);
            });
        }

        const singleReportBtn = document.getElementById("btn-single-report");
        if (singleReportBtn) {
            singleReportBtn.addEventListener("click", () => this.switchView("reports"));
        }

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
            const sendSeek = () => {
                this.isSeeking = false;
                fetch(`/api/bed/${this.activePatientId}/playback`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: "seek", value: parseFloat(seekSlider.value) })
                });
            };
            seekSlider.addEventListener("mouseup", sendSeek);
            seekSlider.addEventListener("change", sendSeek);
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

        // Record search & filters
        const searchInput = document.getElementById("record-search-input");
        if (searchInput) {
            searchInput.addEventListener("input", () => this.applyRecordFilters());
        }

        const filterRhythm = document.getElementById("filter-rhythm-select");
        if (filterRhythm) {
            filterRhythm.addEventListener("change", () => this.applyRecordFilters());
        }

        const filterStatus = document.getElementById("filter-status-select");
        if (filterStatus) {
            filterStatus.addEventListener("change", () => this.applyRecordFilters());
        }

        const syncDatasetBtn = document.getElementById("btn-sync-dataset-action");
        if (syncDatasetBtn) {
            syncDatasetBtn.addEventListener("click", () => {
                syncDatasetBtn.textContent = "⏳ Scanning...";
                this.fetchInitialCatalog();
                setTimeout(() => {
                    syncDatasetBtn.textContent = "🔄 Rescan Directory";
                }, 800);
            });
        }

        // Alert filters
        const alertSearch = document.getElementById("alert-search-input");
        if (alertSearch) {
            alertSearch.addEventListener("input", () => this.applyAlertFilters());
        }

        const alertPriority = document.getElementById("filter-alert-priority");
        if (alertPriority) {
            alertPriority.addEventListener("change", () => this.applyAlertFilters());
        }

        // System Health refresh
        const refreshHealthBtn = document.getElementById("btn-refresh-health");
        if (refreshHealthBtn) {
            refreshHealthBtn.addEventListener("click", () => {
                refreshHealthBtn.textContent = "⏳ Checking...";
                this.fetchHealthDiagnostics();
                setTimeout(() => {
                    refreshHealthBtn.textContent = "🔄 Refresh Status";
                }, 600);
            });
        }

        // Calibration Sweep Speed & Gain
        document.querySelectorAll(".speed-cfg-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".speed-cfg-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const spd = parseFloat(btn.getAttribute("data-speed"));
                this.sweepSpeed = (spd === 50) ? 2.70 : 1.35;
            });
        });

        document.querySelectorAll(".gain-cfg-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".gain-cfg-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const gain = parseFloat(btn.getAttribute("data-gain"));
                this.gainScale = gain || 1.0;
            });
        });

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
                    this.closeModal("modal-stream-to-bed");
                    this.selectActivePatient(targetBed);
                    this.switchView("monitor");
                });
            });
        }

        // Preview modal stream button
        const previewStreamBtn = document.getElementById("btn-preview-stream-btn");
        if (previewStreamBtn) {
            previewStreamBtn.addEventListener("click", () => {
                const recId = this.currentPreviewRecordId || "100";
                fetch(`/api/bed/${this.activePatientId}/record`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ record_id: recId })
                }).then(() => {
                    this.closeModal("modal-preview-waveform");
                    this.switchView("monitor");
                });
            });
        }

        // Modal close buttons (smooth transition)
        document.querySelectorAll("[data-close]").forEach(btn => {
            btn.addEventListener("click", () => {
                const targetId = btn.getAttribute("data-close");
                this.closeModal(targetId);
            });
        });

        // Close modal on clicking outside backdrop
        document.querySelectorAll(".modal-overlay").forEach(overlay => {
            overlay.addEventListener("click", (e) => {
                if (e.target === overlay) {
                    this.closeModal(overlay);
                }
            });
        });

        // Close modal and notification drawer on Escape key press
        window.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                this.closeNotificationDrawer();
                document.querySelectorAll(".modal-overlay:not(.hidden)").forEach(m => {
                    this.closeModal(m);
                });
            }
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

        // Critical Alert Banner Actions
        const bannerDetailsBtn = document.getElementById("btn-banner-details");
        if (bannerDetailsBtn) {
            bannerDetailsBtn.addEventListener("click", () => {
                if (this.activeBannerAlert) {
                    this.openAlertDetailsModal(this.activeBannerAlert);
                }
            });
        }

        const bannerSilenceBtn = document.getElementById("btn-silence-alarm");
        if (bannerSilenceBtn) {
            bannerSilenceBtn.addEventListener("click", () => {
                const targetPatient = this.activeBannerAlert ? this.activeBannerAlert.patient_id : this.activePatientId;
                this.snoozeBed(targetPatient, 5);
                document.getElementById("alarm-banner").classList.add("hidden");
            });
        }

        const bannerAckBtn = document.getElementById("btn-ack-alarm");
        if (bannerAckBtn) {
            bannerAckBtn.addEventListener("click", () => {
                if (this.activeBannerAlert) {
                    this.ackAlert(this.activeBannerAlert.alert_id);
                } else {
                    const cur = this.patients[this.activePatientId];
                    if (cur && cur.active_alert) {
                        this.ackAlert(cur.active_alert.alert_id);
                    }
                }
                document.getElementById("alarm-banner").classList.add("hidden");
            });
        }

        const bannerDismissBtn = document.getElementById("btn-dismiss-banner");
        if (bannerDismissBtn) {
            bannerDismissBtn.addEventListener("click", () => {
                document.getElementById("alarm-banner").classList.add("hidden");
            });
        }

        // Notification Drawer Controls
        const closeDrawerBtn = document.getElementById("btn-close-drawer");
        if (closeDrawerBtn) {
            closeDrawerBtn.addEventListener("click", () => this.closeNotificationDrawer());
        }

        const drawerBackdrop = document.getElementById("drawer-backdrop");
        if (drawerBackdrop) {
            drawerBackdrop.addEventListener("click", () => this.closeNotificationDrawer());
        }

        const markAllDrawerBtn = document.getElementById("btn-drawer-mark-all");
        if (markAllDrawerBtn) {
            markAllDrawerBtn.addEventListener("click", () => this.markAllAlertsRead());
        }

        const viewAllDrawerBtn = document.getElementById("btn-drawer-view-all");
        if (viewAllDrawerBtn) {
            viewAllDrawerBtn.addEventListener("click", () => {
                this.closeNotificationDrawer();
                this.switchView("alerts");
            });
        }
    }

    // ========================================================
    // MODERNIZED FEATURE 1: INTERACTIVE DIGITAL CALIPERS HUD
    // ========================================================
    setupCalipers() {
        const toggleBtn = document.getElementById("btn-toggle-calipers");
        const calipersHud = document.getElementById("calipers-hud");
        const closeBtn = document.getElementById("btn-caliper-close");
        const resetBtn = document.getElementById("btn-caliper-reset");
        const snapBtn = document.getElementById("btn-caliper-snap-peak");
        const overlay = document.getElementById("calipers-canvas-overlay");
        const line1 = document.getElementById("caliper-line-1");
        const line2 = document.getElementById("caliper-line-2");
        const rangeBand = document.getElementById("caliper-range-band");

        if (toggleBtn && calipersHud) {
            toggleBtn.addEventListener("click", () => {
                this.calipersActive = !this.calipersActive;
                calipersHud.classList.toggle("hidden", !this.calipersActive);
                toggleBtn.classList.toggle("active", this.calipersActive);
                if (this.calipersActive) {
                    this.updateCaliperPositions();
                    this.updateCaliperMeasurements();
                }
            });
        }

        if (closeBtn && calipersHud) {
            closeBtn.addEventListener("click", () => {
                this.calipersActive = false;
                calipersHud.classList.add("hidden");
                if (toggleBtn) toggleBtn.classList.remove("active");
            });
        }

        if (resetBtn) {
            resetBtn.addEventListener("click", () => {
                const w = (overlay ? overlay.clientWidth : 800) || 800;
                this.caliper1X = Math.round(w * 0.35);
                this.caliper2X = Math.round(w * 0.50);
                this.updateCaliperPositions();
                this.updateCaliperMeasurements();
            });
        }

        if (snapBtn) {
            snapBtn.addEventListener("click", () => {
                if (this.lastRenderedRPeaks && this.lastRenderedRPeaks.length > 0) {
                    const latest = this.lastRenderedRPeaks[this.lastRenderedRPeaks.length - 1];
                    this.caliper1X = Math.round(latest.x);
                    this.caliper2X = Math.round(latest.x + 30); // ~90ms forward
                    this.updateCaliperPositions();
                    this.updateCaliperMeasurements();
                }
            });
        }

        if (overlay && line1 && line2) {
            const onPointerDown = (e, target) => {
                e.preventDefault();
                this.draggingCaliper = target;
                document.body.style.cursor = "ew-resize";
            };

            line1.addEventListener("mousedown", (e) => onPointerDown(e, "c1"));
            line2.addEventListener("mousedown", (e) => onPointerDown(e, "c2"));
            if (rangeBand) rangeBand.addEventListener("mousedown", (e) => onPointerDown(e, "band"));

            window.addEventListener("mousemove", (e) => {
                if (!this.draggingCaliper || !overlay) return;
                const rect = overlay.getBoundingClientRect();
                const curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));

                if (this.draggingCaliper === "c1") {
                    this.caliper1X = curX;
                } else if (this.draggingCaliper === "c2") {
                    this.caliper2X = curX;
                } else if (this.draggingCaliper === "band") {
                    const span = this.caliper2X - this.caliper1X;
                    this.caliper1X = Math.max(0, Math.min(rect.width - span, curX - span / 2));
                    this.caliper2X = this.caliper1X + span;
                }
                this.updateCaliperPositions();
                this.updateCaliperMeasurements();
            });

            window.addEventListener("mouseup", () => {
                if (this.draggingCaliper) {
                    this.draggingCaliper = null;
                    document.body.style.cursor = "default";
                }
            });

            // Touch support for tablets & mobiles
            const getTouchX = (e) => {
                const touch = e.touches[0] || e.changedTouches[0];
                const rect = overlay.getBoundingClientRect();
                return Math.max(0, Math.min(rect.width, touch.clientX - rect.left));
            };

            line1.addEventListener("touchstart", (e) => { this.draggingCaliper = "c1"; }, { passive: false });
            line2.addEventListener("touchstart", (e) => { this.draggingCaliper = "c2"; }, { passive: false });
            window.addEventListener("touchmove", (e) => {
                if (!this.draggingCaliper) return;
                const curX = getTouchX(e);
                if (this.draggingCaliper === "c1") this.caliper1X = curX;
                else if (this.draggingCaliper === "c2") this.caliper2X = curX;
                this.updateCaliperPositions();
                this.updateCaliperMeasurements();
            });
            window.addEventListener("touchend", () => { this.draggingCaliper = null; });
        }
    }

    updateCaliperPositions() {
        const line1 = document.getElementById("caliper-line-1");
        const line2 = document.getElementById("caliper-line-2");
        const rangeBand = document.getElementById("caliper-range-band");

        const left = Math.min(this.caliper1X, this.caliper2X);
        const right = Math.max(this.caliper1X, this.caliper2X);
        const width = Math.max(2, right - left);

        if (line1) line1.style.left = `${this.caliper1X}px`;
        if (line2) line2.style.left = `${this.caliper2X}px`;
        if (rangeBand) {
            rangeBand.style.left = `${left}px`;
            rangeBand.style.width = `${width}px`;
        }
    }

    updateCaliperMeasurements() {
        const left = Math.min(this.caliper1X, this.caliper2X);
        const right = Math.max(this.caliper1X, this.caliper2X);
        const dx = Math.max(1, right - left);

        // At standard 25mm/s, sweepSpeed ~1.35 px per 4ms sample = ~0.3375 px/ms
        const pxPerMs = (this.sweepSpeed || 1.35) * 0.25;
        const dtMs = Math.max(1, Math.round(dx / pxPerMs));
        const equivRate = dtMs > 0 ? Math.min(999, Math.round(60000 / dtMs)) : 0;
        const deltaVMV = (dx * 0.008).toFixed(2); // approximate voltage difference

        const deltaTEl = document.getElementById("caliper-delta-t");
        const rateEl = document.getElementById("caliper-equiv-rate");
        const deltaVEl = document.getElementById("caliper-delta-v");
        const tagEl = document.getElementById("caliper-classification-tag");
        const bandLabel = document.getElementById("caliper-band-label");

        if (deltaTEl) deltaTEl.textContent = `${dtMs} ms`;
        if (rateEl) rateEl.textContent = `${equivRate} BPM`;
        if (deltaVEl) deltaVEl.textContent = `${deltaVMV} mV`;
        if (bandLabel) bandLabel.textContent = `Δt: ${dtMs} ms`;

        if (tagEl) {
            if (dtMs < 60) {
                tagEl.textContent = "Spike Artifact (<60 ms)";
                tagEl.className = "badge-pill warning";
            } else if (dtMs <= 120) {
                tagEl.textContent = "Normal QRS Complex (60-120 ms)";
                tagEl.className = "badge-pill stable";
            } else if (dtMs <= 200) {
                tagEl.textContent = "Normal PR Interval (120-200 ms)";
                tagEl.className = "badge-pill stable";
            } else if (dtMs <= 280) {
                tagEl.textContent = "Prolonged PR / 1st Deg AV Block";
                tagEl.className = "badge-pill advisory";
            } else if (dtMs <= 450) {
                tagEl.textContent = "Normal QT Interval (320-450 ms)";
                tagEl.className = "badge-pill stable";
            } else if (dtMs <= 540) {
                tagEl.textContent = "Prolonged QTc / TdP Risk (>450 ms)";
                tagEl.className = "badge-pill warning";
            } else {
                tagEl.textContent = `Cardiac Cycle (${equivRate} BPM)`;
                tagEl.className = "badge-pill stable";
            }
        }
    }

    // ========================================================
    // MODERNIZED FEATURE 2: MORPHOLOGY ZONES TOGGLE
    // ========================================================
    setupMorphologyToggle() {
        const morphBtn = document.getElementById("btn-toggle-morphology");
        const statusTag = document.getElementById("morphology-status-tag");
        if (morphBtn) {
            morphBtn.addEventListener("click", () => {
                this.morphologyZonesActive = !this.morphologyZonesActive;
                morphBtn.classList.toggle("active", this.morphologyZonesActive);
                if (statusTag) statusTag.classList.toggle("hidden", !this.morphologyZonesActive);
            });
        }
    }

    // ========================================================
    // MODERNIZED FEATURE 3: 3D VCG & POINCARÉ ATTRACTOR MODAL
    // ========================================================
    setupVCGModal() {
        const openBtn = document.getElementById("btn-open-vcg-modal");
        const modal = document.getElementById("modal-vcg");
        const pauseBtn = document.getElementById("btn-vcg-toggle-pause");
        const resetBtn = document.getElementById("btn-vcg-reset-view");

        if (openBtn && modal) {
            openBtn.addEventListener("click", () => {
                this.openModal("modal-vcg");
                this.startVCGLoop();
            });
        }

        // Hook into modal close
        if (modal) {
            modal.querySelectorAll("[data-close]").forEach(b => {
                b.addEventListener("click", () => this.stopVCGLoop());
            });
        }

        if (pauseBtn) {
            pauseBtn.addEventListener("click", () => {
                this.vcgPaused = !this.vcgPaused;
                pauseBtn.textContent = this.vcgPaused ? "▶ Resume VCG Loop" : "⏸ Pause VCG Loop";
            });
        }

        if (resetBtn) {
            resetBtn.addEventListener("click", () => {
                this.recentDipoles = [];
            });
        }
    }

    startVCGLoop() {
        if (this.vcgRunning) return;
        this.vcgRunning = true;
        this.vcgPaused = false;

        const canvasVCG = document.getElementById("canvas-vcg-loop");
        const ctxVCG = canvasVCG ? canvasVCG.getContext("2d") : null;
        const canvasPoincare = document.getElementById("canvas-poincare-attractor");
        const ctxPoincare = canvasPoincare ? canvasPoincare.getContext("2d") : null;

        const loop = () => {
            if (!this.vcgRunning) return;
            if (!this.vcgPaused) {
                if (ctxVCG && canvasVCG) this.renderVCGLoop(ctxVCG, canvasVCG.width, canvasVCG.height);
                if (ctxPoincare && canvasPoincare) this.renderPoincareAttractor(ctxPoincare, canvasPoincare.width, canvasPoincare.height);
            }
            this.vcgAnimFrame = requestAnimationFrame(loop);
        };
        this.vcgAnimFrame = requestAnimationFrame(loop);
    }

    stopVCGLoop() {
        this.vcgRunning = false;
        if (this.vcgAnimFrame) {
            cancelAnimationFrame(this.vcgAnimFrame);
            this.vcgAnimFrame = null;
        }
    }

    renderVCGLoop(ctx, w, h) {
        ctx.fillStyle = "#030712";
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2;
        const cy = h / 2;
        const scale = 75; // px per mV

        // Polar grid circles (0.5mV, 1.0mV, 1.5mV)
        ctx.strokeStyle = "rgba(0, 229, 255, 0.12)";
        ctx.lineWidth = 1;
        [0.5, 1.0, 1.5].forEach(r => {
            ctx.beginPath();
            ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
            ctx.stroke();
        });

        // Cartesian Crosshair
        ctx.strokeStyle = "rgba(0, 229, 255, 0.25)";
        ctx.beginPath();
        ctx.moveTo(cx, 15); ctx.lineTo(cx, h - 15);
        ctx.moveTo(15, cy); ctx.lineTo(w - 15, cy);
        ctx.stroke();

        // Axis labels
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillStyle = "rgba(0, 229, 255, 0.5)";
        ctx.fillText("+Lead V1", w - 55, cy - 6);
        ctx.fillText("-Lead II", cx + 6, 25);
        ctx.fillText("+Lead II", cx + 6, h - 20);

        if (!this.recentDipoles || this.recentDipoles.length < 2) return;

        // Render trajectory loop with glowing neon phosphor trail
        ctx.lineWidth = 2.2;
        for (let i = 1; i < this.recentDipoles.length; i++) {
            const p0 = this.recentDipoles[i - 1];
            const p1 = this.recentDipoles[i];
            const alpha = (i / this.recentDipoles.length).toFixed(2);

            const x0 = cx + (p0.v1 * scale);
            const y0 = cy - (p0.l2 * scale);
            const x1 = cx + (p1.v1 * scale);
            const y1 = cy - (p1.l2 * scale);

            ctx.strokeStyle = `rgba(0, 229, 255, ${Math.max(0.1, alpha)})`;
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
        }

        // Current vector dipole tip
        const latest = this.recentDipoles[this.recentDipoles.length - 1];
        const curX = cx + (latest.v1 * scale);
        const curY = cy - (latest.l2 * scale);

        // Vector arrow
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(curX, curY);
        ctx.stroke();

        // Tip circle
        ctx.fillStyle = "#34d399";
        ctx.shadowColor = "#34d399";
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(curX, curY, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Update VCG metrics
        const mag = Math.sqrt(latest.v1 * latest.v1 + latest.l2 * latest.l2);
        const angleDeg = Math.round(Math.atan2(-latest.l2, latest.v1) * 180 / Math.PI);

        const dipEl = document.getElementById("vcg-val-dipole");
        const angEl = document.getElementById("vcg-val-angle");
        const axisBadge = document.getElementById("vcg-axis-badge");
        const rotEl = document.getElementById("vcg-val-rotation");

        if (dipEl) dipEl.textContent = `${mag.toFixed(2)} mV`;
        if (angEl) angEl.textContent = `${angleDeg > 0 ? '+' : ''}${angleDeg}°`;
        if (axisBadge) {
            axisBadge.textContent = `Axis: ${angleDeg > 0 ? '+' : ''}${angleDeg}°`;
            axisBadge.className = `badge-pill ${Math.abs(angleDeg) <= 90 ? 'stable' : 'warning'}`;
        }
        if (rotEl) rotEl.textContent = (angleDeg >= 0) ? "Clockwise" : "Counter-Clockwise";
    }

    renderPoincareAttractor(ctx, w, h) {
        ctx.fillStyle = "#030712";
        ctx.fillRect(0, 0, w, h);

        const padding = 35;
        const pw = w - padding * 2;
        const ph = h - padding * 2;

        const minRR = 350;
        const maxRR = 1250;
        const toX = (val) => padding + ((val - minRR) / (maxRR - minRR)) * pw;
        const toY = (val) => h - padding - ((val - minRR) / (maxRR - minRR)) * ph;

        // Grid lines
        ctx.strokeStyle = "rgba(56, 189, 248, 0.12)";
        ctx.lineWidth = 1;
        [400, 600, 800, 1000, 1200].forEach(val => {
            const gx = toX(val);
            const gy = toY(val);
            ctx.beginPath();
            ctx.moveTo(gx, padding); ctx.lineTo(gx, h - padding);
            ctx.moveTo(padding, gy); ctx.lineTo(w - padding, gy);
            ctx.stroke();

            ctx.font = "9px 'JetBrains Mono', monospace";
            ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
            ctx.fillText(`${val}`, gx - 10, h - padding + 14);
            ctx.fillText(`${val}`, padding - 28, gy + 3);
        });

        // 45-degree Line of Identity (x = y)
        ctx.strokeStyle = "rgba(0, 229, 255, 0.4)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(toX(minRR), toY(minRR));
        ctx.lineTo(toX(maxRR), toY(maxRR));
        ctx.stroke();
        ctx.setLineDash([]);

        // Get patient metrics
        const p = this.patients[this.activePatientId];
        const m = (p && p.metrics) || {};
        const meanRR = m.mean_rr_ms || (p && p.vitals && p.vitals.hr ? Math.round(60000 / p.vitals.hr) : 833);
        const sd1 = m.poincare_sd1 || 24.5;
        const sd2 = m.poincare_sd2 || 58.2;
        const ratio = m.hrv_rmssd && m.hrv_sdnn ? (m.hrv_sdnn / Math.max(1, m.hrv_rmssd)).toFixed(2) : (sd2 / Math.max(1, sd1)).toFixed(2);

        // Draw Confidence Ellipse
        const cx = toX(meanRR);
        const cy = toY(meanRR);
        const r1 = (sd1 / (maxRR - minRR)) * pw * 1.8;
        const r2 = (sd2 / (maxRR - minRR)) * ph * 1.8;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(-Math.PI / 4); // 45 deg along identity line
        ctx.fillStyle = "rgba(16, 185, 129, 0.12)";
        ctx.strokeStyle = "rgba(16, 185, 129, 0.65)";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.ellipse(0, 0, Math.max(12, r2), Math.max(8, r1), 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();

        // Plot simulated cluster points around mean
        ctx.fillStyle = "#00e5ff";
        const ptCount = 28;
        for (let i = 0; i < ptCount; i++) {
            const angle = (i / ptCount) * Math.PI * 2;
            const jitterX = Math.cos(angle) * (sd2 * 0.7) * (0.4 + (i % 5) * 0.15) + (Math.sin(i * 3.7) * sd1 * 0.5);
            const jitterY = Math.sin(angle) * (sd1 * 0.7) * (0.4 + (i % 3) * 0.2) + (Math.cos(i * 2.1) * sd2 * 0.5);
            const px = toX(meanRR + jitterX);
            const py = toY(meanRR + jitterY);
            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Center point
        ctx.fillStyle = "#facc15";
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fill();

        // Update stats
        const sd1El = document.getElementById("vcg-val-sd1");
        const sd2El = document.getElementById("vcg-val-sd2");
        const ratEl = document.getElementById("vcg-val-ratio");
        const autoBadge = document.getElementById("vcg-autonomic-badge");

        if (sd1El) sd1El.textContent = `${sd1} ms`;
        if (sd2El) sd2El.textContent = `${sd2} ms`;
        if (ratEl) ratEl.textContent = `${ratio}`;
        if (autoBadge) {
            autoBadge.textContent = ratio > 3.0 ? "Sympathetic Dominance" : (ratio < 1.3 ? "Vagal Parasympathetic" : "Balanced Autonomic Tone");
            autoBadge.className = `badge-pill ${ratio <= 3.2 ? 'stable' : 'advisory'}`;
        }
    }

    // ========================================================
    // MODERNIZED FEATURE 4: AI CLINICAL COPILOT NARRATIVE
    // ========================================================
    setupCopilotModal() {
        const openBtn = document.getElementById("btn-open-copilot-modal");
        const modal = document.getElementById("modal-copilot");
        const copyBtn = document.getElementById("btn-copilot-copy");

        if (openBtn && modal) {
            openBtn.addEventListener("click", () => {
                this.openModal("modal-copilot");
                this.fetchCopilotNarrative();
            });
        }

        if (copyBtn) {
            copyBtn.addEventListener("click", () => {
                const narrativeText = document.getElementById("copilot-narrative-text");
                if (narrativeText) {
                    navigator.clipboard.writeText(narrativeText.textContent).then(() => {
                        copyBtn.textContent = "✓ Copied to Clipboard!";
                        setTimeout(() => { copyBtn.textContent = "📋 Copy Narrative"; }, 2000);
                    });
                }
            });
        }
    }

    fetchCopilotNarrative() {
        const patientId = this.activePatientId;
        fetch(`/api/patient/${patientId}/copilot`)
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    this.renderCopilotData(data);
                }
            })
            .catch(err => {
                console.error("Error fetching copilot data:", err);
            });
    }

    renderCopilotData(d) {
        // Overview Strip
        const bedBadge = document.getElementById("copilot-bed-badge");
        const ptName = document.getElementById("copilot-patient-name");
        const vitalsLine = document.getElementById("copilot-vitals-line");
        const dxTitle = document.getElementById("copilot-diagnosis-title");
        const cewiPill = document.getElementById("copilot-cewi-pill");
        const confPill = document.getElementById("copilot-conf-pill");
        const narrativeBody = document.getElementById("copilot-narrative-text");

        if (bedBadge) bedBadge.textContent = d.patient_id;
        if (ptName) ptName.textContent = d.patient_name;

        const p = this.patients[d.patient_id];
        const v = (p && p.vitals) || {};
        if (vitalsLine) {
            vitalsLine.textContent = `HR: ${v.hr || 72} BPM • SpO2: ${v.spo2 || 98}% • BP: ${v.bp_sys || 120}/${v.bp_dia || 80} mmHg • Temp: ${v.temp_c || 36.8}°C`;
        }

        if (dxTitle) {
            dxTitle.textContent = d.diagnosis;
            dxTitle.style.color = d.diagnosis.includes("Tachycardia") || d.diagnosis.includes("Fibrillation") ? "#f87171" : "#00e5ff";
        }

        if (cewiPill) {
            cewiPill.textContent = `CEWI: ${d.cewi.score} (${d.cewi.risk_level.split(' ')[0]})`;
            cewiPill.className = `badge-pill ${d.cewi.score >= 55 ? 'crisis' : (d.cewi.score >= 35 ? 'advisory' : 'stable')}`;
        }

        if (confPill) {
            confPill.textContent = `Calibrated Conf: ${(d.confidence * 100).toFixed(1)}%`;
        }

        if (narrativeBody) {
            narrativeBody.textContent = d.narrative;
        }

        // Fiducial Table
        const tbody = document.getElementById("copilot-fiducial-tbody");
        if (tbody && d.morphology) {
            const m = d.morphology;
            tbody.innerHTML = `
                <tr>
                    <td><strong>P-Wave</strong></td>
                    <td>${m.p_wave.amplitude_mv} mV / ${m.p_wave.duration_ms} ms</td>
                    <td><span class="badge-pill ${m.p_wave.status.includes('Discrete') ? 'stable' : 'warning'}">${m.p_wave.status}</span></td>
                </tr>
                <tr>
                    <td><strong>PR Interval</strong></td>
                    <td>${m.pr_interval.duration_ms} ms</td>
                    <td><span class="badge-pill ${m.pr_interval.status.includes('Normal') ? 'stable' : 'warning'}">${m.pr_interval.status}</span></td>
                </tr>
                <tr>
                    <td><strong>QRS Complex</strong></td>
                    <td>${m.qrs_complex.duration_ms} ms (Axis ${m.qrs_complex.axis_degrees}°)</td>
                    <td><span class="badge-pill ${m.qrs_complex.status.includes('Narrow') ? 'stable' : 'crisis'}">${m.qrs_complex.status}</span></td>
                </tr>
                <tr>
                    <td><strong>ST Segment</strong></td>
                    <td>${m.st_segment.deviation_mv} mV</td>
                    <td><span class="badge-pill stable">${m.st_segment.status}</span></td>
                </tr>
                <tr>
                    <td><strong>T-Wave</strong></td>
                    <td>${m.t_wave.amplitude_mv} mV</td>
                    <td><span class="badge-pill ${m.t_wave.polarity.includes('Concordant') ? 'stable' : 'advisory'}">${m.t_wave.polarity}</span></td>
                </tr>
                <tr>
                    <td><strong>Bazett QTc</strong></td>
                    <td>${m.qtc.duration_ms} ms</td>
                    <td><span class="badge-pill ${m.qtc.status.includes('Normal') ? 'stable' : 'warning'}">${m.qtc.status}</span></td>
                </tr>
            `;
        }

        // Autonomic Conduction Dynamics
        const autoList = document.getElementById("copilot-autonomic-list");
        if (autoList && d.autonomic) {
            const a = d.autonomic;
            autoList.innerHTML = `
                <div class="autonomic-metric-row">
                    <span class="metric-label">Time-Domain SDNN</span>
                    <strong class="metric-val">${a.sdnn_ms} ms</strong>
                </div>
                <div class="autonomic-metric-row">
                    <span class="metric-label">Parasympathetic RMSSD</span>
                    <strong class="metric-val">${a.rmssd_ms} ms</strong>
                </div>
                <div class="autonomic-metric-row">
                    <span class="metric-label">Poincaré SD1 (Short-term)</span>
                    <strong class="metric-val">${a.poincare_sd1} ms</strong>
                </div>
                <div class="autonomic-metric-row">
                    <span class="metric-label">Poincaré SD2 (Long-term)</span>
                    <strong class="metric-val">${a.poincare_sd2} ms</strong>
                </div>
                <div class="autonomic-metric-row">
                    <span class="metric-label">Autonomic Balance (SD2/SD1)</span>
                    <strong class="metric-val">${a.ratio_sd2_sd1}</strong>
                </div>
                <div class="autonomic-metric-row">
                    <span class="metric-label">Autonomic Tone Assessment</span>
                    <span class="badge-pill stable">${a.state}</span>
                </div>
            `;
        }

        // Recommendations List
        const actionsList = document.getElementById("copilot-actions-list");
        if (actionsList && d.recommendations) {
            actionsList.innerHTML = d.recommendations.map(r => `<li>${r}</li>`).join("");
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

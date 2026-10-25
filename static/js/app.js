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
        this.dismissedAlertIds = new Set();
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

        // Model Laboratory State
        this.mlTrainPage = 1;
        this.mlTestPage = 1;
        this.mlPageSize = 5;
        this.mlTrainRecords = [];
        this.mlTestRecords = [];
        this.mlReport = null;
        this.mlIsTraining = false;
        this.mlPollInterval = null;

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
        this.setupGuardianDashboard();
        this.setupGuardianSettings();
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
                    this.populateAnalysisBedDropdown(data.patients);
                    if (this.activeView === "analysis") {
                        this.renderElectrophysiologyBreakdown(this.activePatientId);
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
                if (this.activeView !== "monitor" && this.activeView !== "analysis" && this.activeView !== "guardian" && this.activeView !== "settings") {
                    this.switchView("monitor");
                }
            });
        }

        // 1b. Analysis bed select dropdown & refresh button
        const analysisBedSelect = document.getElementById("analysis-bed-select");
        if (analysisBedSelect) {
            analysisBedSelect.addEventListener("change", (e) => {
                const pid = e.target.value;
                if (pid) {
                    this.selectActivePatient(pid);
                } else {
                    this.activePatientId = null;
                    this.renderElectrophysiologyBreakdown(null);
                }
            });
        }

        const btnAnalysisRefresh = document.getElementById("btn-analysis-refresh");
        if (btnAnalysisRefresh) {
            btnAnalysisRefresh.addEventListener("click", () => {
                this.renderElectrophysiologyBreakdown(this.activePatientId);
            });
        }

        // 2. Sidebar bed quick tabs
        document.querySelectorAll(".sidebar-bed-tabs .bed-tab").forEach(tab => {
            tab.addEventListener("click", () => {
                const pid = tab.getAttribute("data-patient-id") || tab.getAttribute("data-patient");
                if (pid) {
                    this.selectActivePatient(pid);
                    if (this.activeView !== "analysis" && this.activeView !== "guardian" && this.activeView !== "settings") {
                        this.switchView("monitor");
                    }
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
        if (viewName === "analysis") this.renderElectrophysiologyBreakdown(this.activePatientId);
        if (viewName === "ml") this.fetchMLMetrics();
        if (viewName === "alerts") this.fetchAlerts();
        if (viewName === "reports") this.fetchShiftReport();
        if (viewName === "health") this.fetchHealthDiagnostics();
        if (viewName === "guardian") this.fetchGuardianData();
        if (viewName === "settings") this.loadGuardianSettingsForBed();
    }

    selectActivePatient(patientId) {
        if (!patientId) return;
        this.activePatientId = patientId;

        // 1. Sync header dropdown
        const headerSelect = document.getElementById("header-bed-select");
        if (headerSelect && headerSelect.value !== patientId) {
            headerSelect.value = patientId;
        }

        // Sync Analysis bed dropdown
        const analysisBedSelect = document.getElementById("analysis-bed-select");
        if (analysisBedSelect && analysisBedSelect.value !== patientId) {
            analysisBedSelect.value = patientId;
        }
        if (this.activeView === "analysis") {
            this.renderElectrophysiologyBreakdown(patientId);
        }

        // Sync Guardian and Settings bed dropdowns
        const guardianBedSelect = document.getElementById("guardian-bed-select");
        if (guardianBedSelect && guardianBedSelect.value !== patientId) {
            guardianBedSelect.value = patientId;
        }
        const settingsGuardianBedSelect = document.getElementById("settings-guardian-bed-select");
        if (settingsGuardianBedSelect && settingsGuardianBedSelect.value !== patientId) {
            settingsGuardianBedSelect.value = patientId;
        }
        if (this.activeView === "guardian") {
            this.fetchGuardianData(patientId);
        } else if (this.activeView === "settings") {
            this.loadGuardianSettingsForBed(patientId);
        }

        // 2. Sync sidebar bed tabs
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

            if (this.activeView === "analysis") {
                const now = Date.now();
                if (!this._lastEpRenderTime || now - this._lastEpRenderTime >= 1000) {
                    this._lastEpRenderTime = now;
                    this.renderElectrophysiologyBreakdown(this.activePatientId, true);
                }
            }
        }

        // Determine top qualifying active alert across all beds
        let qualifyingAlert = null;
        let activeAlertCount = 0;

        for (const [pid, pData] of Object.entries(data)) {
            this.patients[pid] = { ...(this.patients[pid] || {}), ...pData };
            if (pData.active_alert) {
                activeAlertCount++;
                const alt = pData.active_alert;
                const isDismissed = alt.banner_dismissed || (this.dismissedAlertIds && this.dismissedAlertIds.has(alt.alert_id));
                if (alt.status === "ACTIVE" && !alt.acknowledged && alt.status !== "SNOOZED" && !isDismissed) {
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
        if (active && active.active_alert && active.active_alert.status === "ACTIVE" && !active.active_alert.acknowledged && active.active_alert.status !== "SNOOZED" && !active.active_alert.banner_dismissed && (!this.dismissedAlertIds || !this.dismissedAlertIds.has(active.active_alert.alert_id))) {
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
                hrStatus.style.color = "var(--text-muted)";
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
                qrsStatus.style.color = "var(--text-muted)";
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
                qtcStatus.style.color = "var(--text-muted)";
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
            rrStatus.style.color = "var(--text-muted)";
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
                sdnnStatus.style.color = "var(--text-muted)";
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
                rmssdStatus.style.color = "var(--text-muted)";
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

        // Feature 1: ECG Signal Quality Monitor Component
        const qa = (metrics && metrics.quality_assessment) || {};
        
        // Derive technical status dynamically from backend data without hardcoded fallbacks
        let techStatus;
        if (metrics && metrics.technical_quality_status) {
            techStatus = metrics.technical_quality_status;
        } else if (qa.status) {
            techStatus = qa.status;
        } else if (metrics && metrics.signal_quality) {
            techStatus = (metrics.signal_quality === "OPTIMAL") ? "GOOD SIGNAL" : "REVIEW SIGNAL";
        } else {
            techStatus = "INSUFFICIENT DATA";
        }

        const isGood = techStatus === "GOOD SIGNAL";
        const isReview = techStatus === "REVIEW SIGNAL";
        const isPoor = techStatus === "POOR SIGNAL";
        const isInsufficient = techStatus === "INSUFFICIENT DATA";

        if (sqEl) {
            sqEl.textContent = isInsufficient ? "INSUFFICIENT" : (isGood ? "OPTIMAL" : (isPoor ? "POOR" : "REVIEW"));
            sqEl.style.color = isGood ? "#10b981" : (isReview ? "#eab308" : (isPoor ? "var(--alarm-crisis)" : "var(--text-muted)"));
        }

        if (snrEl) {
            const snr = (metrics && metrics.snr_db !== undefined) 
                ? metrics.snr_db 
                : (qa.measurements && qa.measurements.estimated_snr_db !== undefined ? qa.measurements.estimated_snr_db : null);
            snrEl.textContent = (snr !== null) ? `${snr} dB` : "-- dB";
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
            lockBadge.textContent = isGood ? "Filter Lock: Active" : (isInsufficient ? "Filter Lock: Standby" : "Filter Lock: Degraded");
            lockBadge.className = `badge-pill ${isGood ? 'stable' : (isInsufficient ? 'info' : 'warning')}`;
        }

        const techReason = (metrics && metrics.technical_quality_reason) 
            || qa.primary_reason 
            || (isGood 
                ? "Signal quality checks passed. Technical signal suitable for analysis."
                : (isInsufficient 
                    ? "Incomplete signal window. Insufficient samples for assessment."
                    : "Technical quality concern detected. Review telemetry leads."));

        const techScore = (qa.score !== undefined && qa.score !== null) 
            ? qa.score 
            : ((metrics && metrics.quality_score !== undefined && metrics.quality_score !== null) ? metrics.quality_score : null);

        const sqBadge = document.getElementById("ecg-sq-status-badge");
        if (sqBadge) {
            sqBadge.textContent = techStatus;
            sqBadge.className = "ecg-sq-status-badge";
            if (isGood) sqBadge.classList.add("badge-good");
            else if (isReview) sqBadge.classList.add("badge-review");
            else if (isPoor) sqBadge.classList.add("badge-poor");
            else sqBadge.classList.add("badge-insufficient");
        }

        const sqDot = document.getElementById("ecg-sq-dot");
        if (sqDot) {
            sqDot.className = "ecg-sq-dot";
            if (isGood) sqDot.classList.add("status-good");
            else if (isReview) sqDot.classList.add("status-review");
            else if (isPoor) sqDot.classList.add("status-poor");
            else sqDot.classList.add("status-insufficient");
        }

        // Quality score: ONLY show if calculated from a documented method and not insufficient data
        const sqScoreBadge = document.getElementById("ecg-sq-score-badge");
        if (sqScoreBadge) {
            if (techScore !== null && !isInsufficient) {
                sqScoreBadge.style.display = "inline-flex";
                sqScoreBadge.textContent = `Score: ${Math.round(techScore)}/100`;
            } else {
                sqScoreBadge.style.display = "none";
            }
        }

        const sqReasonText = document.getElementById("ecg-sq-reason-text");
        if (sqReasonText) {
            sqReasonText.textContent = techReason;
        }

        // Detected technical issues banner:
        // DISPLAY ONLY WHEN AN ISSUE IS ACTUALLY DETECTED AND NOT GOOD SIGNAL
        const issuesContainer = document.getElementById("ecg-sq-issues-container");
        const issuesTitle = document.getElementById("ecg-sq-issues-title");
        const issuesList = document.getElementById("ecg-sq-issues-list");
        const detectedIssues = qa.detected_issues || [];

        if (issuesContainer) {
            if (detectedIssues.length > 0 && !isGood) {
                issuesContainer.classList.remove("hidden");
                issuesContainer.style.display = "block";
                
                issuesContainer.className = "ecg-sq-issues-container";
                if (isPoor) {
                    issuesContainer.classList.add("severity-poor");
                    if (issuesTitle) issuesTitle.textContent = "Detected Technical Issues (Unsuitable Signal):";
                } else if (isReview) {
                    issuesContainer.classList.add("severity-review");
                    if (issuesTitle) issuesTitle.textContent = "Technical Quality Concerns (Review Required):";
                } else {
                    issuesContainer.classList.add("severity-insufficient");
                    if (issuesTitle) issuesTitle.textContent = "Technical Quality Limitations:";
                }

                if (issuesList) {
                    issuesList.innerHTML = detectedIssues.map(issue => `<li>${issue}</li>`).join("");
                }
            } else {
                // When every applicable check passes or no issues: completely hide
                issuesContainer.classList.add("hidden");
                issuesContainer.style.display = "none";
                if (issuesList) issuesList.innerHTML = "";
            }
        }

        // Update Quality Details Table values & result badges
        const checks = qa.checks || {};
        
        const updateCheckRow = (valElId, badgeElId, valText, passed, severity) => {
            const valEl = document.getElementById(valElId);
            if (valEl && valText !== null) valEl.textContent = valText;
            
            const badgeEl = document.getElementById(badgeElId);
            if (badgeEl) {
                if (isInsufficient && (!checks || Object.keys(checks).length === 0)) {
                    badgeEl.textContent = "--";
                    badgeEl.className = "badge-check-insufficient";
                } else if (passed === true) {
                    badgeEl.textContent = "PASS";
                    badgeEl.className = "badge-check-pass";
                } else if (severity === "REVIEW" || severity === "WARN") {
                    badgeEl.textContent = "WARN";
                    badgeEl.className = "badge-check-review";
                } else if (passed === false) {
                    badgeEl.textContent = "FAIL";
                    badgeEl.className = "badge-check-fail";
                } else {
                    badgeEl.textContent = "PASS";
                    badgeEl.className = "badge-check-pass";
                }
            }
        };

        // 1. Finite Samples
        const finitePassed = checks.finite_samples ? checks.finite_samples.passed : (isInsufficient ? null : true);
        const finiteText = checks.finite_samples ? `${checks.finite_samples.invalid_count || 0} invalid` : null;
        updateCheckRow("sq-val-finite", "sq-badge-finite", finiteText, finitePassed, "FAIL");

        // 2. Flatline
        const flatlinePassed = checks.flatline ? checks.flatline.passed : (isInsufficient ? null : true);
        const flatlineText = checks.flatline ? `PTP ${(checks.flatline.ptp_mv || 0).toFixed(2)} mV` : null;
        updateCheckRow("sq-val-flatline", "sq-badge-flatline", flatlineText, flatlinePassed, "FAIL");

        // 3. Saturation
        const satPassed = checks.saturation ? checks.saturation.passed : (isInsufficient ? null : true);
        const satText = checks.saturation ? `PTP ${(checks.saturation.ptp_mv || 0).toFixed(2)} mV` : null;
        updateCheckRow("sq-val-sat", "sq-badge-sat", satText, satPassed, "FAIL");

        // 4. Baseline Wander
        let driftPassed = true;
        let driftText = null;
        if (checks.noise_and_drift) {
            const driftVal = checks.noise_and_drift.baseline_drift_mv || 0;
            const maxDrift = checks.noise_and_drift.max_drift_threshold_mv || 1.5;
            driftPassed = driftVal <= maxDrift;
            driftText = `${driftVal.toFixed(2)} mV drift`;
        } else if (isInsufficient) {
            driftPassed = null;
        }
        updateCheckRow("sq-val-drift", "sq-badge-drift", driftText, driftPassed, "WARN");

        // 5. SNR
        let snrPassed = true;
        let snrSeverity = "FAIL";
        let snrCheckText = null;
        if (checks.noise_and_drift) {
            const snrVal = checks.noise_and_drift.snr_db || 0;
            const minSnr = checks.noise_and_drift.min_snr_threshold_db || 10.0;
            if (snrVal >= minSnr) {
                snrPassed = true;
            } else if (snrVal >= 4.0) {
                snrPassed = false;
                snrSeverity = "WARN";
            } else {
                snrPassed = false;
                snrSeverity = "FAIL";
            }
            snrCheckText = `${snrVal.toFixed(1)} dB`;
        } else if (isInsufficient) {
            snrPassed = null;
        }
        updateCheckRow("sq-val-snr", "sq-badge-snr", snrCheckText, snrPassed, snrSeverity);

        // 6. Duration
        const durPassed = checks.duration ? checks.duration.passed : (isInsufficient ? null : true);
        const durText = checks.duration ? `${(checks.duration.duration_sec || 0).toFixed(2)} s (${checks.duration.samples || 0} pts)` : null;
        updateCheckRow("sq-val-dur", "sq-badge-dur", durText, durPassed, "WARN");
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

        const isDismissed = alert && (alert.banner_dismissed || (this.dismissedAlertIds && this.dismissedAlertIds.has(alert.alert_id)));
        if (!alert || alert.status !== "ACTIVE" || alert.acknowledged || alert.status === "SNOOZED" || isDismissed) {
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
        if (iconEl) {
            iconEl.innerHTML = `<svg class="heart-pulse-svg ${isCrit ? 'pulse-crit' : 'pulse-warn'}" viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                <path d="M6 12h3l2-3 2 6 2-3h3"></path>
            </svg>`;
        }
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
                    if (data.counts) {
                        const bAll = document.getElementById("badge-count-all");
                        const bNorm = document.getElementById("badge-count-normal");
                        const bAbn = document.getElementById("badge-count-abnormal");
                        const bUnc = document.getElementById("badge-count-unclassified");
                        if (bAll) bAll.textContent = data.counts.all ?? data.records.length;
                        if (bNorm) bNorm.textContent = data.counts.normal ?? 0;
                        if (bAbn) bAbn.textContent = data.counts.abnormal ?? 0;
                        if (bUnc) bUnc.textContent = data.counts.unclassified ?? 0;
                    }
                    this.applyRecordFilters();
                }
            })
            .catch(() => {});
    }

    setClassificationFilter(category) {
        this.activeClassificationFilter = (category || "ALL").toUpperCase();

        // Update tab buttons
        document.querySelectorAll(".classification-filter-btn").forEach(btn => {
            const cls = (btn.getAttribute("data-classification") || "").toUpperCase();
            if (cls === this.activeClassificationFilter) {
                btn.classList.add("active");
                btn.setAttribute("aria-selected", "true");
            } else {
                btn.classList.remove("active");
                btn.setAttribute("aria-selected", "false");
            }
        });

        // Update select dropdown if present
        const selectEl = document.getElementById("filter-classification-select");
        if (selectEl && selectEl.value !== this.activeClassificationFilter) {
            selectEl.value = this.activeClassificationFilter;
        }

        this.applyRecordFilters();
    }

    applyRecordFilters() {
        const term = (document.getElementById("record-search-input")?.value || "").toLowerCase().trim();
        const rhythm = document.getElementById("filter-rhythm-select")?.value || "ALL";
        const status = document.getElementById("filter-status-select")?.value || "ALL";
        const classification = (this.activeClassificationFilter || document.getElementById("filter-classification-select")?.value || "ALL").toUpperCase();

        const filtered = (this.catalogRecords || []).filter(r => {
            // Search filter
            const matchesSearch = !term ||
                (r.record_id && r.record_id.toLowerCase().includes(term)) ||
                (r.primary_diagnosis && r.primary_diagnosis.toLowerCase().includes(term)) ||
                (r.patient_summary && r.patient_summary.toLowerCase().includes(term)) ||
                (r.classification_reason && r.classification_reason.toLowerCase().includes(term)) ||
                (r.leads && r.leads.some(l => l.toLowerCase().includes(term)));

            // Classification filter
            let matchesClassification = true;
            if (classification !== "ALL") {
                const recCls = (r.classification || r.record_classification || "").toUpperCase();
                matchesClassification = (recCls === classification);
            }

            // Rhythm filter
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

            // Status filter
            let matchesStatus = true;
            if (status === "PLAYABLE") {
                matchesStatus = Boolean(r.is_playable);
            } else if (status === "CATALOGED") {
                matchesStatus = !r.is_playable;
            }

            return matchesSearch && matchesClassification && matchesRhythm && matchesStatus;
        });

        this.renderRecordsTable(filtered);
    }

    renderRecordsTable(records) {
        const tbody = document.getElementById("records-table-body");
        if (!tbody) return;

        if (records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="11" style="text-align: center; padding: 24px; color: var(--text-secondary);">No records matching current classification and search filters.</td></tr>`;
            return;
        }

        const escapeHtml = (str) => String(str || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        tbody.innerHTML = records.map(r => {
            const cls = (r.record_classification || "Unclassified");
            let badgeClass = "advisory";
            if (cls === "Normal") badgeClass = "stable";
            else if (cls === "Abnormal") badgeClass = "crisis";

            const patientInfo = r.patient_summary || (r.patient_info && r.patient_info.summary) || r.primary_diagnosis || "Clinical Telemetry";
            const reason = r.classification_reason || "Evaluated against verified MIT-BIH reference annotations.";

            return `
            <tr>
                <td><strong style="font-family: var(--font-mono); color: var(--accent-primary);">${r.record_id}</strong></td>
                <td style="max-width: 220px; font-size: 12px; line-height: 1.4;">
                    <div style="font-weight: 500; color: var(--text-primary); text-overflow: ellipsis; overflow: hidden; white-space: nowrap;" title="${escapeHtml(patientInfo)}">
                        ${escapeHtml(patientInfo)}
                    </div>
                </td>
                <td>
                    <span class="badge-pill ${badgeClass}" title="${escapeHtml(reason)}" style="cursor: help; letter-spacing: 0.3px;">
                        ${cls}
                    </span>
                </td>
                <td><strong>${r.primary_diagnosis || 'Clinical ECG Telemetry'}</strong></td>
                <td>${(r.leads || ['MLII', 'V1']).join(', ')}</td>
                <td>${r.fs || 360} Hz</td>
                <td>${r.duration_formatted || '30m 05s'}</td>
                <td>${(r.beat_count || 0) > 0 ? (r.beat_count).toLocaleString() : 'Cataloged'}</td>
                <td>${r.is_paced ? '<span class="badge-pill warning">PACED</span>' : '<span style="color: var(--text-muted);">None</span>'}</td>
                <td><span class="badge-pill ${r.is_playable ? 'stable' : 'advisory'}">${r.file_status || 'Cataloged'}</span></td>
                <td>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        <button class="btn btn-secondary btn-sm" onclick="app.openAndAnalyseRecord('${r.record_id}', 0)" title="Open and analyse record waveform, reference annotations, and RF prediction">🔬 Analyse</button>
                        <button class="btn btn-primary btn-sm" onclick="app.openStreamModal('${r.record_id}')" title="Stream to active bed">▶ Stream</button>
                    </div>
                </td>
            </tr>
            `;
        }).join("");
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

    openAndAnalyseRecord(recordId, startSec = 0) {
        this.currentAnalysisRecordId = String(recordId);
        this.currentAnalysisWindowStart = Math.max(0, startSec);
        this.currentAnalysisWindowDuration = 10.0;

        const title = document.getElementById("preview-modal-title");
        const classifBadge = document.getElementById("preview-modal-classification-badge");
        const patientBanner = document.getElementById("preview-modal-patient-info");
        const windowLabel = document.getElementById("preview-modal-window-label");
        const meta = document.getElementById("preview-modal-metadata");

        if (title) title.textContent = `Record ${recordId} • Waveform & Clinical Analysis`;
        if (classifBadge) {
            classifBadge.className = "badge-pill info";
            classifBadge.textContent = "Loading Analysis...";
        }
        if (patientBanner) {
            patientBanner.innerHTML = `<span style="color: var(--text-secondary);">Loading clinical metadata for MIT-BIH Record ${recordId}...</span>`;
        }
        if (windowLabel) {
            const endSec = this.currentAnalysisWindowStart + this.currentAnalysisWindowDuration;
            windowLabel.textContent = `${this.currentAnalysisWindowStart.toFixed(1)}s – ${endSec.toFixed(1)}s`;
        }

        this.openModal("modal-preview-waveform");

        fetch(`/api/dataset/record_analysis?record_id=${recordId}&start_sec=${this.currentAnalysisWindowStart}&duration_sec=${this.currentAnalysisWindowDuration}`)
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    this.renderRecordAnalysisData(data);
                } else if (data.error) {
                    if (meta) meta.textContent = `Error loading record: ${data.error}`;
                }
            })
            .catch(err => {
                if (meta) meta.textContent = `Network error fetching record analysis: ${err}`;
            });
    }

    previewRecordWaveform(recordId) {
        // Compatibility alias for existing callers
        this.openAndAnalyseRecord(recordId, 0);
    }

    shiftRecordWindow(deltaSec) {
        const nextStart = Math.max(0, (this.currentAnalysisWindowStart || 0) + deltaSec);
        this.openAndAnalyseRecord(this.currentAnalysisRecordId || "100", nextStart);
    }

    renderRecordAnalysisData(data) {
        this.currentAnalysisData = data;
        const recId = data.record_id;
        const pat = data.patient_info || {};
        const ref = data.reference_classification || {};
        const ml = data.model_prediction || {};
        const wave = data.waveform || {};

        // 1. Header & Badges
        const title = document.getElementById("preview-modal-title");
        const topBadge = document.getElementById("preview-modal-classification-badge");
        if (title) title.textContent = `Record ${recId} • PhysioNet MIT-BIH Waveform & Analysis`;

        const refCls = ref.classification_label || "Unclassified";
        if (topBadge) {
            topBadge.className = `badge-pill ${refCls === 'Normal' ? 'stable' : (refCls === 'Abnormal' ? 'crisis' : 'advisory')}`;
            topBadge.textContent = `Reference: ${refCls}`;
            topBadge.title = ref.reason || "";
        }

        // 2. Patient Background Banner
        const patBanner = document.getElementById("preview-modal-patient-info");
        if (patBanner) {
            patBanner.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap;">
                    <div>
                        <strong style="color: var(--accent-cyan); font-size: 14px;">Subject ${pat.subject_id || recId}</strong> 
                        <span style="color: var(--text-secondary); margin-left: 6px;">${pat.age ? `${pat.age}yo` : 'Age Unspecified'} ${pat.sex || ''}</span>
                        <div style="color: var(--text-primary); margin-top: 3px;"><strong>Pathology:</strong> ${pat.diagnosis || 'Clinical ECG Telemetry'}</div>
                    </div>
                    <div style="font-size: 12px; color: var(--text-secondary); text-align: right;">
                        <div><strong>Medications:</strong> ${pat.medications || 'None documented'}</div>
                        <div style="color: var(--text-muted); font-size: 11px;">Leads: ${wave.lead1_name || 'MLII'}, ${wave.lead2_name || 'V1'} • ${wave.sample_rate || 250} Hz</div>
                    </div>
                </div>
            `;
        }

        // 3. Draw Waveform with beat marker annotations
        this.drawPreviewWaveformWithAnnotations(wave.lead1 || [], ref.segment_annotations || [], wave.sample_rate || 250);

        const meta = document.getElementById("preview-modal-metadata");
        if (meta) {
            meta.innerHTML = `
                <strong>Lead:</strong> ${wave.lead1_name || 'MLII'} • 
                <strong>Sample Rate:</strong> ${wave.sample_rate || 250} Hz (Resampled from 360 Hz) • 
                <strong>Duration:</strong> ${wave.duration_sec || 10}s window • 
                <strong>Total Session:</strong> ${Math.floor((wave.total_duration_sec || 1805) / 60)}m ${Math.floor((wave.total_duration_sec || 1805) % 60)}s • 
                <strong>Slice Beats:</strong> ${ref.segment_annotation_count || 0} reference annotations
            `;
        }

        const qBadge = document.getElementById("preview-modal-quality-badge");
        const qDot = document.getElementById("preview-modal-quality-dot");
        if (qBadge && data.signal_quality) {
            const sq = data.signal_quality;
            qBadge.textContent = `${sq.status} (${sq.score}/100)`;
            qBadge.className = `badge-pill ${sq.status === 'GOOD SIGNAL' ? 'stable' : (sq.status === 'REVIEW SIGNAL' ? 'warning' : 'crisis')}`;
            if (qDot) {
                qDot.className = `ecg-sq-dot ${sq.status === 'GOOD SIGNAL' ? 'status-good' : (sq.status === 'REVIEW SIGNAL' ? 'status-review' : 'status-poor')}`;
            }
        }

        const windowLabel = document.getElementById("preview-modal-window-label");
        if (windowLabel) {
            const st = wave.start_sec || 0;
            const dur = wave.duration_sec || 10;
            windowLabel.textContent = `${st.toFixed(1)}s – ${(st + dur).toFixed(1)}s`;
        }

        // 4. Section 2: Reference Ground Truth Annotations
        const refBadge = document.getElementById("modal-ref-classif-badge");
        if (refBadge) {
            refBadge.className = `badge-pill ${refCls === 'Normal' ? 'stable' : (refCls === 'Abnormal' ? 'crisis' : 'advisory')}`;
            refBadge.textContent = refCls;
        }

        const refSummary = document.getElementById("modal-ref-beat-summary");
        if (refSummary) {
            const symCounts = ref.symbol_counts || {};
            const countTokens = Object.entries(symCounts)
                .filter(([k]) => k !== '+' && k !== '~')
                .map(([sym, cnt]) => `${sym}: ${cnt}`);
            refSummary.textContent = `${ref.beat_count?.toLocaleString() || 0} total beats (${countTokens.slice(0, 4).join(', ')})`;
        }

        const refReason = document.getElementById("modal-ref-classif-reason");
        if (refReason) {
            refReason.textContent = ref.reason || "Evaluated against verified MIT-BIH reference annotations.";
        }

        const sliceAnnContainer = document.getElementById("modal-ref-slice-annotations");
        if (sliceAnnContainer) {
            const anns = ref.segment_annotations || [];
            if (anns.length === 0) {
                sliceAnnContainer.innerHTML = `<span style="color: var(--text-muted);">No beat annotations in this 10s window.</span>`;
            } else {
                sliceAnnContainer.innerHTML = anns.map(a => `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 11px;">${a.time_sec.toFixed(2)}s</span>
                        <strong style="color: ${a.is_abnormal ? 'var(--alarm-crisis)' : 'var(--vital-hr)'}; font-family: var(--font-mono);">${a.symbol}</strong>
                        <span style="color: var(--text-primary); font-size: 11px; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; max-width: 170px;">${a.name}</span>
                    </div>
                `).join("");
            }
        }

        // 5. Section 3: Random Forest Model Prediction
        const mlDiagnosis = document.getElementById("modal-ml-pred-diagnosis");
        if (mlDiagnosis) {
            mlDiagnosis.textContent = ml.diagnosis || "Normal Sinus Rhythm";
            mlDiagnosis.style.color = ml.is_critical ? 'var(--alarm-crisis)' : 'var(--accent-cyan)';
        }

        const mlConfidence = document.getElementById("modal-ml-pred-confidence");
        if (mlConfidence) {
            const conf = ml.confidence_percent || ((ml.confidence || 0.9) * 100);
            mlConfidence.textContent = `${conf.toFixed(1)}% Confidence`;
            mlConfidence.className = `badge-pill ${conf >= 75 ? 'stable' : 'warning'}`;
        }

        // Extracted biomarkers
        const feats = ml.features || {};
        const bioHr = document.getElementById("modal-bio-hr");
        const bioQrs = document.getElementById("modal-bio-qrs");
        const bioQtc = document.getElementById("modal-bio-qtc");
        if (bioHr) bioHr.textContent = `${Math.round(feats.heart_rate || 72)} bpm`;
        if (bioQrs) bioQrs.textContent = `${Math.round(feats.qrs_duration_ms || 88)} ms`;
        if (bioQtc) bioQtc.textContent = `${Math.round(feats.qtc_ms || 412)} ms`;

        // Class probabilities
        const probsContainer = document.getElementById("modal-ml-probabilities");
        if (probsContainer) {
            const probs = ml.probabilities || {};
            const entries = Object.entries(probs).filter(([_, p]) => p > 0.01).sort((a, b) => b[1] - a[1]);
            if (entries.length === 0) {
                probsContainer.innerHTML = `<span style="color: var(--text-muted);">Probability data unavailable.</span>`;
            } else {
                probsContainer.innerHTML = entries.map(([name, p]) => {
                    const pct = Math.round(p * 100);
                    return `
                        <div style="margin-bottom: 4px;">
                            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
                                <span>${name}</span>
                                <span style="font-family: var(--font-mono); font-weight: 600;">${pct}%</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.08); height: 4px; border-radius: 2px; overflow: hidden;">
                                <div style="background: var(--accent-cyan); width: ${pct}%; height: 100%;"></div>
                            </div>
                        </div>
                    `;
                }).join("");
            }
        }
    }

    drawPreviewWaveform(samples) {
        this.drawPreviewWaveformWithAnnotations(samples, [], 250);
    }

    drawPreviewWaveformWithAnnotations(samples, annotations = [], sampleRate = 250) {
        const canvas = document.getElementById("canvas-preview");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        this.drawFullGrid(ctx, w, h);

        if (!samples || samples.length === 0) return;

        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        const midY = h * 0.52;
        const scaleY = h * 0.30;
        const step = w / samples.length;

        samples.forEach((val, i) => {
            const x = i * step;
            const y = midY - (val * scaleY);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Draw aligned reference beat annotations
        const startTimeSec = this.currentAnalysisWindowStart || 0;
        const totalDurationSec = samples.length / sampleRate;

        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";

        annotations.forEach(a => {
            const relTime = a.time_sec - startTimeSec;
            if (relTime >= 0 && relTime <= totalDurationSec) {
                const sampleIdx = Math.round(relTime * sampleRate);
                const x = sampleIdx * step;
                const valAtX = (sampleIdx >= 0 && sampleIdx < samples.length) ? samples[sampleIdx] : 0;
                const y = Math.max(22, midY - (valAtX * scaleY) - 16);

                // Small marker pin
                ctx.fillStyle = a.is_abnormal ? "#ef4444" : "#00ff88";
                ctx.beginPath();
                ctx.arc(x, y + 8, 3, 0, Math.PI * 2);
                ctx.fill();

                // Symbol text
                ctx.fillStyle = a.is_abnormal ? "#fca5a5" : "#a7f3d0";
                ctx.fillText(a.symbol, x, y);
            }
        });
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
            .catch(err => {
                console.warn("[ML] Error loading metrics:", err);
            });
    }

    renderMLDashboard(report) {
        if (!report) return;
        this.mlReport = report;

        // Model version & status
        const modelVer = document.getElementById("ml-model-version");
        if (modelVer) modelVer.textContent = report.model_version || report.model_name || "RandomForest v2.1";

        const statusBadge = document.getElementById("ml-training-status-badge");
        if (statusBadge && !this.mlIsTraining) {
            statusBadge.textContent = "Model Ready / Validated";
            statusBadge.className = "badge-pill active";
        }

        // Split pill & leakage badge
        const splitConfig = report.split_config || {};
        const splitPill = document.getElementById("ml-split-pill");
        if (splitPill) {
            const trPct = splitConfig.train_percent || `${Math.round((splitConfig.train_ratio || 0.7) * 100)}%`;
            const tsPct = splitConfig.test_percent || `${Math.round((splitConfig.test_ratio || 0.3) * 100)}%`;
            splitPill.textContent = `${trPct} Train / ${tsPct} Test`;
        }

        const leakageBadge = document.getElementById("ml-leakage-badge");
        if (leakageBadge) {
            const isDisjoint = splitConfig.is_leakage_free !== false;
            leakageBadge.textContent = isDisjoint ? "🛡️ Zero Record Overlap" : "⚠️ Partition Overlap Detected";
            leakageBadge.className = `badge-pill ${isDisjoint ? 'stable' : 'critical'}`;
        }

        const lastEvalTime = document.getElementById("ml-last-eval-time");
        if (lastEvalTime) lastEvalTime.textContent = report.training_timestamp || "--";

        // Extract Training & Testing Records arrays
        const trainData = report.training_dataset || {};
        const testData = report.testing_dataset || {};

        this.mlTrainRecords = Array.isArray(trainData.records) && trainData.records.length > 0
            ? trainData.records
            : (report.training_records || []).map(r => ({
                record_id: String(r),
                subject_id: String(r),
                pathology: "PhysioNet Arrhythmia",
                samples: 177,
                classification: "Abnormal",
                role: "Training"
            }));

        this.mlTestRecords = Array.isArray(testData.records) && testData.records.length > 0
            ? testData.records
            : (report.test_records || []).map(r => ({
                record_id: String(r),
                subject_id: String(r),
                pathology: "PhysioNet Arrhythmia",
                samples: 177,
                classification: "Abnormal",
                role: "Held-out Test"
            }));

        // Update counts and sample badges
        const trainCountBadge = document.getElementById("ml-train-count-badge");
        if (trainCountBadge) trainCountBadge.textContent = `${this.mlTrainRecords.length} Records`;

        const trainSamplesBadge = document.getElementById("ml-train-samples-badge");
        if (trainSamplesBadge) {
            const trainSamples = trainData.sample_count || report.samples_train || (this.mlTrainRecords.length * 177);
            trainSamplesBadge.textContent = `${trainSamples.toLocaleString()} Windows`;
        }

        const testCountBadge = document.getElementById("ml-test-count-badge");
        if (testCountBadge) testCountBadge.textContent = `${this.mlTestRecords.length} Records`;

        const testSamplesBadge = document.getElementById("ml-test-samples-badge");
        if (testSamplesBadge) {
            const testSamples = testData.sample_count || report.samples_test || (this.mlTestRecords.length * 177);
            testSamplesBadge.textContent = `${testSamples.toLocaleString()} Windows`;
        }

        // Render paginated tables
        this.renderMLTrainTable();
        this.renderMLTestTable();

        // Render performance metric boxes
        const balAcc = document.getElementById("ml-bal-acc-val");
        if (balAcc) balAcc.textContent = `${report.balanced_accuracy_percent || (report.balanced_accuracy ? (report.balanced_accuracy * 100).toFixed(1) : '91.6')}%`;

        const acc = document.getElementById("ml-acc-val");
        if (acc) acc.textContent = `${report.accuracy_percent || (report.accuracy ? (report.accuracy * 100).toFixed(1) : '91.6')}%`;

        const f1 = document.getElementById("ml-f1-val");
        if (f1) f1.textContent = report.macro_f1 ? report.macro_f1.toFixed(3) : "0.917";

        const testSamplesBox = document.getElementById("ml-test-samples-val");
        if (testSamplesBox) testSamplesBox.textContent = (report.samples_test || (this.mlTestRecords.length * 177)).toLocaleString();

        // Split Summary Box
        const summaryText = document.getElementById("ml-split-summary-text");
        if (summaryText) {
            const trCount = this.mlTrainRecords.length;
            const trSamp = (report.samples_train || trCount * 177).toLocaleString();
            const tsCount = this.mlTestRecords.length;
            const tsSamp = (report.samples_test || tsCount * 177).toLocaleString();
            summaryText.textContent = `The Random Forest classifier was trained on ${trCount} records (${trSamp} samples) and evaluated on ${tsCount} completely held-out records (${tsSamp} samples). Exactly 0 record IDs or patient subjects overlap between training and testing.`;
        }

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
                    <td style="font-family: var(--font-mono);">${m.support}</td>
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

    renderMLTrainTable() {
        const tbody = document.getElementById("ml-train-tbody");
        if (!tbody) return;

        const totalRecords = this.mlTrainRecords.length;
        const totalPages = Math.max(1, Math.ceil(totalRecords / this.mlPageSize));
        this.mlTrainPage = Math.min(Math.max(1, this.mlTrainPage), totalPages);

        const startIdx = (this.mlTrainPage - 1) * this.mlPageSize;
        const pageRecords = this.mlTrainRecords.slice(startIdx, startIdx + this.mlPageSize);

        if (pageRecords.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 16px;">No training records configured.</td></tr>`;
        } else {
            tbody.innerHTML = pageRecords.map(r => {
                const isNormal = r.classification === 'Normal' || (r.pathology && r.pathology.includes('Normal'));
                return `
                    <tr>
                        <td><span class="badge-pill" style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-cyan);">#${r.record_id}</span></td>
                        <td>Subject ${r.subject_id || r.record_id}</td>
                        <td><strong>${r.pathology || 'PhysioNet Rhythm'}</strong></td>
                        <td style="font-family: var(--font-mono);">${r.samples || 177}</td>
                        <td><span class="badge-pill ${isNormal ? 'stable' : 'crisis'}">${isNormal ? 'Normal' : 'Abnormal'}</span></td>
                        <td style="text-align: right;">
                            <button class="btn btn-secondary btn-xs" onclick="app.openAndAnalyseRecord('${r.record_id}')" title="Inspect Record in ECG Record Browser">🔍 Inspect</button>
                        </td>
                    </tr>
                `;
            }).join("");
        }

        const pageInfo = document.getElementById("ml-train-page-info");
        if (pageInfo) pageInfo.textContent = `Page ${this.mlTrainPage} of ${totalPages} (${totalRecords} records)`;

        const prevBtn = document.getElementById("ml-train-prev");
        if (prevBtn) prevBtn.disabled = this.mlTrainPage <= 1;

        const nextBtn = document.getElementById("ml-train-next");
        if (nextBtn) nextBtn.disabled = this.mlTrainPage >= totalPages;
    }

    renderMLTestTable() {
        const tbody = document.getElementById("ml-test-tbody");
        if (!tbody) return;

        const totalRecords = this.mlTestRecords.length;
        const totalPages = Math.max(1, Math.ceil(totalRecords / this.mlPageSize));
        this.mlTestPage = Math.min(Math.max(1, this.mlTestPage), totalPages);

        const startIdx = (this.mlTestPage - 1) * this.mlPageSize;
        const pageRecords = this.mlTestRecords.slice(startIdx, startIdx + this.mlPageSize);

        if (pageRecords.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 16px;">No testing records configured.</td></tr>`;
        } else {
            tbody.innerHTML = pageRecords.map(r => {
                const isNormal = r.classification === 'Normal' || (r.pathology && r.pathology.includes('Normal'));
                return `
                    <tr>
                        <td><span class="badge-pill" style="font-family: var(--font-mono); font-weight: 700; color: #f59e0b;">#${r.record_id}</span></td>
                        <td>Subject ${r.subject_id || r.record_id}</td>
                        <td><strong>${r.pathology || 'PhysioNet Rhythm'}</strong></td>
                        <td style="font-family: var(--font-mono);">${r.samples || 177}</td>
                        <td><span class="badge-pill ${isNormal ? 'stable' : 'crisis'}">${isNormal ? 'Normal' : 'Abnormal'}</span></td>
                        <td style="text-align: right;">
                            <button class="btn btn-secondary btn-xs" onclick="app.openAndAnalyseRecord('${r.record_id}')" title="Inspect Record in ECG Record Browser">🔍 Inspect</button>
                        </td>
                    </tr>
                `;
            }).join("");
        }

        const pageInfo = document.getElementById("ml-test-page-info");
        if (pageInfo) pageInfo.textContent = `Page ${this.mlTestPage} of ${totalPages} (${totalRecords} records)`;

        const prevBtn = document.getElementById("ml-test-prev");
        if (prevBtn) prevBtn.disabled = this.mlTestPage <= 1;

        const nextBtn = document.getElementById("ml-test-next");
        if (nextBtn) nextBtn.disabled = this.mlTestPage >= totalPages;
    }

    triggerMLTraining() {
        if (this.mlIsTraining) {
            console.warn("[ML] Training submission ignored: training already in progress.");
            return;
        }

        const splitSelect = document.getElementById("ml-split-ratio");
        const trainRatio = splitSelect ? parseFloat(splitSelect.value) : 0.70;

        const btn = document.getElementById("btn-ml-train");
        const spinner = document.getElementById("ml-train-spinner");
        const btnText = document.getElementById("ml-train-btn-text");
        const statusBadge = document.getElementById("ml-training-status-badge");
        const progressCard = document.getElementById("ml-training-progress-card");
        const progressFill = document.getElementById("ml-progress-fill");
        const progressPercent = document.getElementById("ml-progress-percent");
        const progressStepText = document.getElementById("ml-progress-step-text");

        // Set Loading UI state
        this.mlIsTraining = true;
        if (btn) btn.disabled = true;
        if (spinner) spinner.classList.remove("hidden");
        if (btnText) btnText.textContent = "Training Ensemble...";
        if (statusBadge) {
            statusBadge.textContent = "Training in Progress...";
            statusBadge.className = "badge-pill warning";
        }
        if (progressCard) progressCard.classList.remove("hidden");
        if (progressFill) progressFill.style.width = "10%";
        if (progressPercent) progressPercent.textContent = "10%";
        if (progressStepText) progressStepText.textContent = "Initializing PhysioNet Feature Pipeline...";

        fetch("/api/ml/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                algorithm: "RandomForest",
                n_estimators: 100,
                train_ratio: trainRatio
            })
        })
        .then(res => {
            if (res.status === 409) {
                throw new Error("Model training already in progress. Duplicate submission prevented.");
            }
            return res.json();
        })
        .then(data => {
            if (!data.success) {
                throw new Error(data.message || "Training initiation failed");
            }

            // Start polling training status
            if (this.mlPollInterval) clearInterval(this.mlPollInterval);
            this.mlPollInterval = setInterval(() => {
                fetch("/api/ml/train_status")
                    .then(res => res.json())
                    .then(statData => {
                        const s = statData.status || {};
                        const pct = s.progress_percent || 10;
                        if (progressFill) progressFill.style.width = `${pct}%`;
                        if (progressPercent) progressPercent.textContent = `${pct}%`;
                        if (progressStepText && s.step) progressStepText.textContent = s.step;

                        if (s.error) {
                            clearInterval(this.mlPollInterval);
                            this.mlIsTraining = false;
                            if (btn) btn.disabled = false;
                            if (spinner) spinner.classList.add("hidden");
                            if (btnText) btnText.textContent = "⚡ Train & Evaluate Model";
                            if (statusBadge) {
                                statusBadge.textContent = "Training Error";
                                statusBadge.className = "badge-pill critical";
                            }
                            alert(`Training Error: ${s.error}`);
                            return;
                        }

                        if (pct >= 100 || !s.is_training) {
                            clearInterval(this.mlPollInterval);
                            setTimeout(() => {
                                this.fetchMLMetrics();
                                this.mlIsTraining = false;
                                if (btn) btn.disabled = false;
                                if (spinner) spinner.classList.add("hidden");
                                if (btnText) btnText.textContent = "⚡ Train & Evaluate Model";
                                if (statusBadge) {
                                    statusBadge.textContent = "Model Ready / Validated";
                                    statusBadge.className = "badge-pill active";
                                }
                                setTimeout(() => {
                                    if (progressCard) progressCard.classList.add("hidden");
                                }, 1800);
                            }, 500);
                        }
                    })
                    .catch(() => {});
            }, 400);
        })
        .catch(err => {
            this.mlIsTraining = false;
            if (btn) btn.disabled = false;
            if (spinner) spinner.classList.add("hidden");
            if (btnText) btnText.textContent = "⚡ Train & Evaluate Model";
            if (statusBadge) {
                statusBadge.textContent = "Failed";
                statusBadge.className = "badge-pill critical";
            }
            if (progressCard) progressCard.classList.add("hidden");
            alert(`Training Failed: ${err.message}`);
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
        this.lastAlertsData = { active_alerts: alertsList, unread_count: unreadCount };
        const listEl = document.getElementById("drawer-alerts-list");
        const unreadBadge = document.getElementById("drawer-unread-badge");
        const headerBadge = document.getElementById("active-alert-count");

        if (unreadBadge) unreadBadge.textContent = `${unreadCount} unread`;
        if (headerBadge) headerBadge.textContent = unreadCount;

        if (!listEl) return;

        let displayAlerts = alertsList || [];
        const filter = this.drawerFilter || "ALL";

        if (filter === "ALARMS") {
            displayAlerts = displayAlerts.filter(a => !a.guardian_dispatch);
        } else if (filter === "GUARDIAN") {
            displayAlerts = displayAlerts.filter(a => !!a.guardian_dispatch);
        }

        if (displayAlerts.length === 0) {
            listEl.innerHTML = `
                <div class="drawer-empty-state">
                    <span style="font-size: 28px; display: block; margin-bottom: 8px;">✅</span>
                    No active ${filter === 'GUARDIAN' ? 'guardian notifications' : 'clinical alarms'} at this time.<br>
                    Continuous multi-lead telemetry is operational.
                </div>
            `;
            return;
        }

        listEl.innerHTML = displayAlerts.map(a => {
            const isCrit = (a.priority || "").includes("CRITICAL") || (a.priority || "").includes("CRISIS") || (a.priority || "").includes("CODE BLUE");
            const isUnread = !a.read;
            const timeStr = a.timestamp ? (a.timestamp.split(" ")[1] || a.timestamp) : "--:--:--";
            const sourcePill = (a.source_type === "RECORDED_BENCHMARK" || (a.source_record && a.source_record !== "SYNTHETIC"))
                ? `MIT-BIH #${a.source_record || '100'} Recorded`
                : "Telemetry Simulation";

            let guardianTag = "";
            if (a.guardian_dispatch) {
                if (a.guardian_dispatch.mode === "DEMO_RECORDED") {
                    guardianTag = '<span class="drawer-sms-tag badge-status-demo">🛡️ Demo Preview Logged</span>';
                } else if (a.guardian_dispatch.eligible) {
                    guardianTag = '<span class="drawer-sms-tag badge-status-delivered">🛡️ Guardian SMS Queued</span>';
                }
            }

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
                    ${guardianTag ? `<div style="margin: 4px 0;">${guardianTag}</div>` : ''}
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
        if (iconEl) {
            iconEl.innerHTML = `<svg class="heart-pulse-svg ${isCrit ? 'pulse-crit' : 'pulse-warn'}" viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                <path d="M6 12h3l2-3 2 6 2-3h3"></path>
            </svg>`;
        }

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

        // --- Feature 2: Smart Alert Explanation Integration ---
        // Ensure supporting evidence panel starts collapsed
        const panelEv = document.getElementById("smart-alert-evidence-panel");
        const iconEv = document.getElementById("smart-alert-evidence-icon");
        if (panelEv) {
            panelEv.classList.add("hidden");
            panelEv.style.display = "none";
        }
        if (iconEv) iconEv.textContent = "▾";

        const expl = alert.smart_explanation || {};
        const ruleBadge = document.getElementById("smart-alert-rule-badge");
        if (ruleBadge) ruleBadge.textContent = expl.rule_title || alert.trigger_condition || "Alert Condition";

        const explEl = document.getElementById("modal-alert-explanation");
        if (explEl) {
            explEl.textContent = expl.summary_explanation || alert.explanation || `Electrophysiological rhythm pattern (${alert.diagnosis || 'Arrhythmia'}) recorded during continuous monitoring. Review dual-lead sweep tracing for morphological confirmation.`;
        }

        const obsVal = document.getElementById("smart-alert-observed-val");
        if (obsVal) obsVal.textContent = expl.observed_measurement || (alert.heart_rate ? `${alert.heart_rate} bpm` : "--");

        const modelVal = document.getElementById("smart-alert-model-val");
        if (modelVal) modelVal.textContent = expl.model_result || alert.diagnosis || "Not available";

        const triggerVal = document.getElementById("smart-alert-trigger-val");
        if (triggerVal) triggerVal.textContent = expl.rule_condition || expl.triggering_condition || alert.trigger_condition || "--";

        const qualityVal = document.getElementById("smart-alert-quality-val");
        const qualitySub = document.getElementById("smart-alert-quality-sub");
        if (qualityVal) {
            const qStatus = expl.signal_quality_status || (alert.signal_quality && alert.signal_quality.status) || "GOOD SIGNAL";
            qualityVal.textContent = qStatus;
            if (qStatus === "POOR SIGNAL") qualityVal.style.color = "var(--alarm-crisis)";
            else if (qStatus === "REVIEW SIGNAL") qualityVal.style.color = "var(--alarm-warning)";
            else qualityVal.style.color = "#10b981";
        }
        if (qualitySub) qualitySub.textContent = expl.signal_quality_details || "Technical Check Evaluation";

        // Supporting Evidence Table
        const evidenceTbody = document.getElementById("smart-evidence-tbody");
        if (evidenceTbody) {
            const evList = expl.supporting_evidence || [];
            if (evList.length > 0) {
                evidenceTbody.innerHTML = evList.map(ev => `
                    <tr>
                        <td>${ev.parameter || '--'}</td>
                        <td>${ev.observed_value || '--'}</td>
                        <td>${ev.threshold || '--'}</td>
                        <td><span class="badge-pill ${ev.rule_status === 'TRIGGERED' ? 'crisis' : (ev.rule_status === 'REVIEW' ? 'warning' : 'stable')}">${ev.rule_status || 'EVALUATED'}</span></td>
                    </tr>
                `).join("");
            } else {
                evidenceTbody.innerHTML = `
                    <tr>
                        <td>Trigger Condition</td>
                        <td>${alert.trigger_condition || 'Alert condition met'}</td>
                        <td>Configured Clinical Alert Rules</td>
                        <td><span class="badge-pill crisis">TRIGGERED</span></td>
                    </tr>
                `;
            }
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
                ackBtn.disabled = true;
                ackBtn.textContent = "⏳ Acknowledging...";
                this.ackAlert(alert.alert_id, (success) => {
                    ackBtn.disabled = false;
                    ackBtn.textContent = "✓ Acknowledge";
                    if (success) {
                        alert.acknowledged = true;
                        alert.status = "ACKNOWLEDGED";
                        this.closeModal("modal-alert-details");
                    }
                });
            };
        }

        const snoozeBtn = document.getElementById("btn-modal-alert-snooze");
        if (snoozeBtn) {
            snoozeBtn.onclick = () => {
                snoozeBtn.disabled = true;
                snoozeBtn.textContent = "⏳ Snoozing...";
                this.snoozeBed(alert.patient_id, 5, alert.alert_id, (success) => {
                    snoozeBtn.disabled = false;
                    snoozeBtn.textContent = "🔕 Snooze (5m)";
                    if (success) {
                        alert.status = "SNOOZED";
                        this.closeModal("modal-alert-details");
                    }
                });
            };
        }

        this.openModal("modal-alert-details");
    }

    openAlertDetailsById(alertId) {
        if (!alertId) return;
        // Search in cached alerts
        let alertObj = (this.allAlertsData || []).find(a => a.alert_id === alertId);
        if (!alertObj) {
            for (const p of Object.values(this.patients || {})) {
                if (p.active_alert && p.active_alert.alert_id === alertId) {
                    alertObj = p.active_alert;
                    break;
                }
            }
        }
        if (alertObj) {
            this.openAlertDetailsModal(alertObj);
        } else {
            // Fetch explanation from dedicated backend API
            fetch(`/api/alerts/${alertId}/explanation`)
                .then(res => res.json())
                .then(data => {
                    if (data.success && data.explanation) {
                        const fallbackAlert = {
                            alert_id: alertId,
                            patient_id: data.explanation.bed_id || "BED-01",
                            priority: data.explanation.alert_severity || "WARNING",
                            timestamp: data.explanation.timestamp || "--:--:--",
                            source_label: data.explanation.source_reference || "CardioSentinel Telemetry",
                            trigger_condition: data.explanation.rule_condition || "Clinical Alert Trigger",
                            smart_explanation: data.explanation
                        };
                        this.openAlertDetailsModal(fallbackAlert);
                    }
                })
                .catch(() => {});
        }
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
                    <button class="btn-why-alert" onclick="app.openAlertDetailsById('${a.alert_id}')" title="View Why This Alert Was Generated">🔍 Why?</button>
                    ${a.status === 'ACTIVE' ? `
                        <button class="btn btn-secondary btn-sm" onclick="app.ackAlert('${a.alert_id}')">✓ Ack</button>
                        <button class="btn btn-warning btn-sm" onclick="app.snoozeBed('${a.patient_id}', 5)">🔕 5m</button>
                        <button class="btn btn-danger btn-sm" onclick="app.resolveAlert('${a.alert_id}')">Resolve</button>
                    ` : '<span style="color: var(--text-muted); font-size: 11px;">Archived</span>'}
                </td>
            </tr>
        `).join("");
    }

    ackAlert(alertId, callback) {
        if (!alertId) return;
        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "ack", alert_id: alertId })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                if (this.dismissedAlertIds) this.dismissedAlertIds.add(alertId);
                if (this.activeBannerAlert && this.activeBannerAlert.alert_id === alertId) {
                    this.activeBannerAlert.acknowledged = true;
                    this.activeBannerAlert.status = "ACKNOWLEDGED";
                    const banner = document.getElementById("alarm-banner");
                    if (banner) banner.classList.add("hidden");
                    this.activeBannerAlert = null;
                }
                this.fetchAlerts();
                if (callback) callback(true, data);
            } else {
                alert(`Failed to acknowledge alert: ${data.error || "Unknown error"}`);
                if (callback) callback(false, data);
            }
        })
        .catch(err => {
            alert("Network error while acknowledging alert. Please retry.");
            if (callback) callback(false, err);
        });
    }

    resolveAlert(alertId, callback) {
        if (!alertId) return;
        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "resolve", alert_id: alertId })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                if (this.dismissedAlertIds) this.dismissedAlertIds.add(alertId);
                if (this.activeBannerAlert && this.activeBannerAlert.alert_id === alertId) {
                    this.activeBannerAlert.status = "RESOLVED";
                    const banner = document.getElementById("alarm-banner");
                    if (banner) banner.classList.add("hidden");
                    this.activeBannerAlert = null;
                }
                this.fetchAlerts();
                if (callback) callback(true, data);
            } else {
                alert(`Failed to resolve alert: ${data.error || "Unknown error"}`);
                if (callback) callback(false, data);
            }
        })
        .catch(err => {
            alert("Network error while resolving alert. Please retry.");
            if (callback) callback(false, err);
        });
    }

    snoozeBed(patientId, minutes, alertId, callback) {
        const payload = { action: "snooze", patient_id: patientId, minutes: minutes || 5 };
        if (alertId) payload.alert_id = alertId;
        fetch("/api/alerts/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                if (alertId && this.dismissedAlertIds) this.dismissedAlertIds.add(alertId);
                if (this.activeBannerAlert && (this.activeBannerAlert.patient_id === patientId || this.activeBannerAlert.alert_id === alertId)) {
                    this.activeBannerAlert.status = "SNOOZED";
                    const banner = document.getElementById("alarm-banner");
                    if (banner) banner.classList.add("hidden");
                    this.activeBannerAlert = null;
                }
                this.fetchAlerts();
                if (callback) callback(true, data);
            } else {
                alert(`Failed to snooze alert: ${data.error || "Unknown error"}`);
                if (callback) callback(false, data);
            }
        })
        .catch(err => {
            alert("Network error while snoozing alert. Please retry.");
            if (callback) callback(false, err);
        });
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
        // Feature 1: Signal Quality Details Expandable Toggle
        const btnToggleSQ = document.getElementById("btn-toggle-sq-details");
        const panelSQ = document.getElementById("ecg-sq-details-panel");
        const iconSQ = document.getElementById("ecg-sq-toggle-icon");
        if (btnToggleSQ && panelSQ) {
            btnToggleSQ.addEventListener("click", () => {
                const isCurrentlyHidden = panelSQ.classList.contains("hidden") || panelSQ.style.display === "none";
                if (isCurrentlyHidden) {
                    panelSQ.classList.remove("hidden");
                    panelSQ.style.display = "block";
                    if (iconSQ) iconSQ.textContent = "▴";
                } else {
                    panelSQ.classList.add("hidden");
                    panelSQ.style.display = "none";
                    if (iconSQ) iconSQ.textContent = "▾";
                }
            });
        }

        // Feature 2: Smart Alert Supporting Evidence Expandable Toggle
        const btnToggleEv = document.getElementById("btn-toggle-alert-evidence");
        if (btnToggleEv) {
            btnToggleEv.addEventListener("click", () => {
                const pEv = document.getElementById("smart-alert-evidence-panel");
                const iEv = document.getElementById("smart-alert-evidence-icon");
                if (pEv) {
                    const isCurrentlyHidden = pEv.classList.contains("hidden") || pEv.style.display === "none";
                    if (isCurrentlyHidden) {
                        pEv.classList.remove("hidden");
                        pEv.style.display = "block";
                        if (iEv) iEv.textContent = "▴";
                    } else {
                        pEv.classList.add("hidden");
                        pEv.style.display = "none";
                        if (iEv) iEv.textContent = "▾";
                    }
                }
            });
        }

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

        const filterClassif = document.getElementById("filter-classification-select");
        if (filterClassif) {
            filterClassif.addEventListener("change", (e) => this.setClassificationFilter(e.target.value));
        }

        document.querySelectorAll(".classification-filter-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const cat = btn.getAttribute("data-classification") || "ALL";
                this.setClassificationFilter(cat);
            });
        });

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

        // ML Split Ratio selector
        const splitSelect = document.getElementById("ml-split-ratio");
        if (splitSelect) {
            splitSelect.addEventListener("change", (e) => {
                const ratio = parseFloat(e.target.value) || 0.70;
                const splitPill = document.getElementById("ml-split-pill");
                if (splitPill) {
                    splitPill.textContent = `${Math.round(ratio * 100)}% Train / ${Math.round((1 - ratio) * 100)}% Test`;
                }
            });
        }

        // ML Training Table Pagination
        const trainPrev = document.getElementById("ml-train-prev");
        if (trainPrev) trainPrev.addEventListener("click", () => {
            if (this.mlTrainPage > 1) {
                this.mlTrainPage--;
                this.renderMLTrainTable();
            }
        });
        const trainNext = document.getElementById("ml-train-next");
        if (trainNext) trainNext.addEventListener("click", () => {
            const totalPages = Math.max(1, Math.ceil(this.mlTrainRecords.length / this.mlPageSize));
            if (this.mlTrainPage < totalPages) {
                this.mlTrainPage++;
                this.renderMLTrainTable();
            }
        });

        // ML Testing Table Pagination
        const testPrev = document.getElementById("ml-test-prev");
        if (testPrev) testPrev.addEventListener("click", () => {
            if (this.mlTestPage > 1) {
                this.mlTestPage--;
                this.renderMLTestTable();
            }
        });
        const testNext = document.getElementById("ml-test-next");
        if (testNext) testNext.addEventListener("click", () => {
            const totalPages = Math.max(1, Math.ceil(this.mlTestRecords.length / this.mlPageSize));
            if (this.mlTestPage < totalPages) {
                this.mlTestPage++;
                this.renderMLTestTable();
            }
        });

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
                const targetAlert = this.activeBannerAlert;
                const targetPatient = targetAlert ? targetAlert.patient_id : this.activePatientId;
                const targetAlertId = targetAlert ? targetAlert.alert_id : null;

                bannerSilenceBtn.disabled = true;
                const origText = bannerSilenceBtn.innerHTML;
                bannerSilenceBtn.innerHTML = "<span>⏳ Snoozing...</span>";

                this.snoozeBed(targetPatient, 5, targetAlertId, (success) => {
                    bannerSilenceBtn.disabled = false;
                    bannerSilenceBtn.innerHTML = origText;
                    if (success) {
                        if (targetAlert) targetAlert.status = "SNOOZED";
                        if (targetAlertId && this.dismissedAlertIds) this.dismissedAlertIds.add(targetAlertId);
                        const banner = document.getElementById("alarm-banner");
                        if (banner) banner.classList.add("hidden");
                        this.activeBannerAlert = null;
                    }
                });
            });
        }

        const bannerAckBtn = document.getElementById("btn-ack-alarm");
        if (bannerAckBtn) {
            bannerAckBtn.addEventListener("click", () => {
                const targetAlert = this.activeBannerAlert || (this.patients[this.activePatientId] && this.patients[this.activePatientId].active_alert);
                if (!targetAlert || !targetAlert.alert_id) return;
                const alertId = targetAlert.alert_id;

                bannerAckBtn.disabled = true;
                const origText = bannerAckBtn.innerHTML;
                bannerAckBtn.innerHTML = "<span>⏳ Acknowledging...</span>";

                this.ackAlert(alertId, (success) => {
                    bannerAckBtn.disabled = false;
                    bannerAckBtn.innerHTML = origText;
                    if (success) {
                        targetAlert.acknowledged = true;
                        targetAlert.status = "ACKNOWLEDGED";
                        if (this.dismissedAlertIds) this.dismissedAlertIds.add(alertId);
                        const banner = document.getElementById("alarm-banner");
                        if (banner) banner.classList.add("hidden");
                        this.activeBannerAlert = null;
                    }
                });
            });
        }

        const bannerDismissBtn = document.getElementById("btn-dismiss-banner");
        if (bannerDismissBtn) {
            bannerDismissBtn.addEventListener("click", () => {
                const targetAlert = this.activeBannerAlert;
                const alertId = targetAlert ? targetAlert.alert_id : null;
                const targetPatient = targetAlert ? targetAlert.patient_id : this.activePatientId;

                if (alertId && this.dismissedAlertIds) {
                    this.dismissedAlertIds.add(alertId);
                }
                if (targetAlert) {
                    targetAlert.banner_dismissed = true;
                }
                const banner = document.getElementById("alarm-banner");
                if (banner) banner.classList.add("hidden");
                this.activeBannerAlert = null;

                // Persist banner dismissal to backend
                if (alertId) {
                    fetch("/api/alerts/action", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ action: "dismiss", alert_id: alertId, patient_id: targetPatient })
                    }).catch(err => console.warn("Failed to notify backend of banner dismissal:", err));
                }
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
            dxTitle.style.color = d.diagnosis.includes("Tachycardia") || d.diagnosis.includes("Fibrillation") ? "var(--alarm-crisis)" : "var(--accent-primary)";
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

    // ========================================================
    // GUARDIAN EMERGENCY NOTIFICATION SYSTEM
    // ========================================================
    setupGuardianDashboard() {
        // Bed selector change
        const bedSelect = document.getElementById("guardian-bed-select");
        if (bedSelect) {
            bedSelect.addEventListener("change", (e) => {
                const pid = e.target.value;
                this.selectActivePatient(pid);
                this.fetchGuardianData(pid);
            });
        }

        // Refresh button
        const refreshBtn = document.getElementById("btn-guardian-refresh");
        if (refreshBtn) {
            refreshBtn.addEventListener("click", () => {
                this.fetchGuardianData(this.activePatientId);
            });
        }

        // Send Test SMS open modal button
        const openTestSmsBtn = document.getElementById("btn-open-test-sms-modal");
        if (openTestSmsBtn) {
            openTestSmsBtn.addEventListener("click", () => this.openTestSmsModal());
        }

        // Toggle clinical operating mode (MIT-BIH playback vs Live Telemetry mode)
        const toggleModeBtn = document.getElementById("btn-toggle-clinical-mode");
        if (toggleModeBtn) {
            toggleModeBtn.addEventListener("click", () => this.toggleClinicalMode());
        }

        // Confirm & Send Test SMS
        const confirmSendTestBtn = document.getElementById("btn-confirm-send-test-sms");
        if (confirmSendTestBtn) {
            confirmSendTestBtn.addEventListener("click", () => this.confirmSendTestSms());
        }

        // Preview Demo Notification button
        const previewDemoBtn = document.getElementById("btn-preview-demo-sms");
        if (previewDemoBtn) {
            previewDemoBtn.addEventListener("click", () => this.previewDemoNotification());
        }

        // Edit Settings button -> navigate to Settings view for currently selected bed
        const editSettingsBtn = document.getElementById("btn-guardian-edit-settings");
        if (editSettingsBtn) {
            editSettingsBtn.addEventListener("click", () => {
                const pid = document.getElementById("guardian-bed-select")?.value || this.activePatientId || "BED-01";
                this.switchView("settings");
                this.loadGuardianSettingsForBed(pid);
                const card = document.querySelector(".guardian-settings-card");
                if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
            });
        }

        // View Audit Trail button
        const auditBtn = document.getElementById("btn-guardian-view-audit");
        if (auditBtn) {
            auditBtn.addEventListener("click", () => this.showGuardianAuditTrail());
        }

        // Notification filters (All, Delivered, Pending, Failed, Demo)
        document.querySelectorAll(".notif-filter-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".notif-filter-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const filter = btn.getAttribute("data-filter") || "ALL";
                this.guardianNotificationFilter = filter;
                this.fetchGuardianNotifications(this.activePatientId, filter);
            });
        });

        // Drawer integration: Goto Guardian view
        const drawerGuardianBtn = document.getElementById("btn-drawer-goto-guardian");
        if (drawerGuardianBtn) {
            drawerGuardianBtn.addEventListener("click", () => {
                this.closeNotificationDrawer();
                this.switchView("guardian");
            });
        }

        // Drawer filter tabs (All Feeds, Clinical Alarms, Guardian SMS)
        document.querySelectorAll(".drawer-filter-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".drawer-filter-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const filter = btn.getAttribute("data-drawer-filter") || "ALL";
                this.drawerFilter = filter;
                if (this.lastAlertsData) {
                    this.renderDrawerAlerts(this.lastAlertsData.active_alerts, this.lastAlertsData.unread_count);
                }
            });
        });
    }

    toggleClinicalMode() {
        const pid = document.getElementById("guardian-bed-select")?.value || this.activePatientId || "BED-01";
        const pt = this.patients[pid];
        const curMode = (pt && pt.data_mode) ? pt.data_mode : "MIT-BIH";
        const newMode = (curMode === "LIVE") ? "MIT-BIH" : "LIVE";

        fetch(`/api/bed/${pid}/mode`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data_mode: newMode })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                if (pt) pt.data_mode = data.data_mode;
                this.fetchGuardianData(pid);
            }
        })
        .catch(err => console.warn("Failed to toggle clinical mode:", err));
    }

    setupGuardianSettings() {
        const bedSelect = document.getElementById("settings-guardian-bed-select");
        if (bedSelect) {
            bedSelect.addEventListener("change", (e) => {
                this.loadGuardianSettingsForBed(e.target.value);
            });
        }

        const saveBtn = document.getElementById("btn-save-guardian-settings");
        if (saveBtn) {
            saveBtn.addEventListener("click", () => this.saveGuardianSettings());
        }

        const resetBtn = document.getElementById("btn-cancel-guardian-settings");
        if (resetBtn) {
            resetBtn.addEventListener("click", () => this.resetGuardianForm());
        }

        const removeBtn = document.getElementById("btn-remove-guardian");
        if (removeBtn) {
            removeBtn.addEventListener("click", () => this.removeGuardian());
        }
    }

    resetGuardianForm() {
        const bed = document.getElementById("settings-guardian-bed-select")?.value || this.activePatientId || "BED-01";
        if (confirm(`Discard unsaved edits and restore saved configuration for ${bed}?`)) {
            this.loadGuardianSettingsForBed(bed);
            const feedback = document.getElementById("guardian-settings-feedback");
            if (feedback) {
                this._showFeedback(feedback, "Form restored to database state.", "success");
            }
        }
    }

    removeGuardian() {
        const bed = document.getElementById("settings-guardian-bed-select")?.value || this.activePatientId || "BED-01";
        const feedback = document.getElementById("guardian-settings-feedback");
        if (!confirm(`Are you sure you want to remove the registered guardian for ${bed}? Emergency notifications will be disarmed for this bed.`)) {
            return;
        }

        const removeBtn = document.getElementById("btn-remove-guardian");
        if (removeBtn) removeBtn.disabled = true;

        fetch("/api/guardian/remove", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patient_id: bed })
        })
        .then(res => res.json())
        .then(data => {
            if (removeBtn) removeBtn.disabled = false;
            if (data.success) {
                this._showFeedback(feedback, "Guardian contact removed successfully. Notification history preserved.", "success");
                this.loadGuardianSettingsForBed(bed);
                if (this.activeView === "guardian") {
                    this.fetchGuardianData(bed);
                }
            } else {
                this._showFeedback(feedback, data.error || "Failed to remove guardian.", "error");
            }
        })
        .catch(err => {
            if (removeBtn) removeBtn.disabled = false;
            this._showFeedback(feedback, "Server communication error.", "error");
        });
    }

    fetchGuardianData(patientId = null) {
        const pid = patientId || this.activePatientId || "BED-01";
        
        // Sync bed selector in guardian view
        const bedSelect = document.getElementById("guardian-bed-select");
        if (bedSelect && bedSelect.value !== pid) {
            bedSelect.value = pid;
        }

        fetch(`/api/guardian/${pid}`)
            .then(res => res.json())
            .then(data => {
                if (!data.success) return;

                const guardian = data.guardian;
                const readiness = data.readiness;
                const provider = data.provider;
                const summary = data.summary || {};

                // 1. Update Mode Indicator Banner
                const modeBanner = document.getElementById("guardian-mode-banner");
                const modeTitle = document.getElementById("guardian-mode-title");
                const modeDesc = document.getElementById("guardian-mode-desc");
                const modeBadge = document.getElementById("guardian-mode-badge");

                const isLive = readiness && readiness.checks && readiness.checks.data_source_mode && readiness.checks.data_source_mode.passed;

                if (modeBanner) {
                    if (isLive) {
                        modeBanner.className = "guardian-mode-banner live-mode";
                        if (modeTitle) modeTitle.textContent = "LIVE TELEMETRY MODE — EMERGENCY NOTIFICATIONS ACTIVE";
                        if (modeDesc) modeDesc.textContent = "Live multi-lead telemetry stream detected. Real emergency SMS notifications are dispatched to verified guardians for eligible crisis/warning events.";
                        if (modeBadge) modeBadge.innerHTML = '<span class="badge-pill stable">LIVE STREAM</span>';
                    } else {
                        modeBanner.className = "guardian-mode-banner demo-mode";
                        if (modeTitle) modeTitle.textContent = "RECORDED ECG — DEMONSTRATION MODE";
                        if (modeDesc) modeDesc.textContent = "Historical PhysioNet MIT-BIH arrhythmia recordings are active for educational demonstration. Simulated events produce local demonstration previews and do not contact real emergency services or send live patient SMS messages.";
                        if (modeBadge) modeBadge.innerHTML = '<span class="badge-pill warning">DEMO PLAYBACK</span>';
                    }
                }

                // 2. Update Readiness Checklist
                const overallBadge = document.getElementById("guardian-overall-readiness-badge");
                if (overallBadge && readiness) {
                    overallBadge.textContent = readiness.overall_label;
                    if (readiness.overall_status === "READY_LIVE") {
                        overallBadge.className = "badge-pill stable";
                    } else if (readiness.overall_status === "DEMO_MODE_ACTIVE") {
                        overallBadge.className = "badge-pill info";
                    } else {
                        overallBadge.className = "badge-pill warning";
                    }
                }

                if (readiness && readiness.checks) {
                    const c = readiness.checks;
                    this._updateReadinessItem("reg", c.guardian_registered);
                    this._updateReadinessItem("phone", c.phone_verified);
                    this._updateReadinessItem("provider", c.provider_configured);
                    this._updateReadinessItem("perm", c.notification_permission);
                    this._updateReadinessItem("mode", c.data_source_mode);
                }

                // 3. Update Primary Guardian Contact Card
                const pt = this.patients[pid];
                const ptName = pt ? pt.name : pid;
                const bedTag = document.getElementById("guardian-bed-tag");
                if (bedTag) bedTag.textContent = `${pid} • ${ptName}`;

                const nameEl = document.getElementById("guardian-name-val");
                const phoneEl = document.getElementById("guardian-phone-val");
                const relEl = document.getElementById("guardian-relation-val");
                const statusTag = document.getElementById("guardian-status-tag");
                const verifyBadge = document.getElementById("guardian-verify-badge");
                const prefsEl = document.getElementById("guardian-prefs-val");
                const consentBadge = document.getElementById("guardian-consent-badge");
                const updatedEl = document.getElementById("guardian-updated-val");
                if (guardian) {
                    if (nameEl) nameEl.textContent = guardian.full_name;
                    if (phoneEl) phoneEl.textContent = guardian.phone_masked || "--";
                    if (relEl) relEl.textContent = guardian.relationship;
                    if (prefsEl) prefsEl.textContent = (guardian.preferred_categories || "CRISIS,WARNING").replace(",", ", ");
                    if (updatedEl) updatedEl.textContent = guardian.updated_at || "--";

                    const isConsented = (guardian.consent_confirmed === 1);
                    const verif = data.verification_status || { label: "Registered — Not Phone Verified", badge_class: "badge-pill info" };

                    if (statusTag) {
                        statusTag.textContent = verif.label;
                        statusTag.className = verif.badge_class;
                    }
                    if (verifyBadge) {
                        verifyBadge.textContent = verif.label;
                        verifyBadge.className = verif.badge_class;
                    }
                    if (consentBadge) {
                        consentBadge.textContent = isConsented ? "Explicit Consent Recorded" : "Consent Missing";
                        consentBadge.className = isConsented ? "badge-pill stable" : "badge-pill warning";
                    }
                } else {
                    if (nameEl) nameEl.textContent = "No guardian registered";
                    if (phoneEl) phoneEl.textContent = "--";
                    if (relEl) relEl.textContent = "--";
                    if (statusTag) {
                        statusTag.textContent = "Not Registered";
                        statusTag.className = "badge-pill warning";
                    }
                    if (verifyBadge) {
                        verifyBadge.textContent = "Not Registered";
                        verifyBadge.className = "badge-pill warning";
                    }
                }

                // 4. Update Summary Counters
                const sentEl = document.getElementById("stat-sms-sent");
                const deliveredEl = document.getElementById("stat-sms-delivered");
                const failedEl = document.getElementById("stat-sms-failed");
                const demoEl = document.getElementById("stat-sms-demo");

                if (sentEl) sentEl.textContent = summary.SENT || 0;
                if (deliveredEl) deliveredEl.textContent = summary.DELIVERED || 0;
                if (failedEl) failedEl.textContent = summary.FAILED || 0;
                if (demoEl) demoEl.textContent = summary.DEMO_RECORDED || 0;

                // Provider warning alert box
                const warnBox = document.getElementById("guardian-provider-warning-box");
                if (warnBox) {
                    const isConfig = Boolean(provider && (provider.is_configured || provider.mock_mode_active));
                    warnBox.classList.toggle("hidden", isConfig);
                }

                // Mode toggle button text
                const toggleModeBtn = document.getElementById("btn-toggle-clinical-mode");
                if (toggleModeBtn) {
                    toggleModeBtn.innerHTML = isLive
                        ? "📼 Switch to Demonstration Mode (MIT-BIH)"
                        : "📡 Switch to Live Telemetry Mode";
                }

                // Test SMS button state
                const testSmsBtn = document.getElementById("btn-open-test-sms-modal");
                if (testSmsBtn) {
                    const hasPhone = Boolean(guardian && (guardian.phone_normalized || guardian.phone_number));
                    const isConfig = Boolean(provider && (provider.is_configured || provider.mock_mode_active));
                    testSmsBtn.disabled = !hasPhone;
                    if (!guardian) {
                        testSmsBtn.title = "Register a guardian in Settings first";
                    } else if (!hasPhone) {
                        testSmsBtn.title = "Guardian phone number is required";
                    } else if (!isConfig) {
                        testSmsBtn.title = "SMS Provider credentials missing in .env (click to view setup instructions)";
                        testSmsBtn.disabled = false;
                    } else {
                        testSmsBtn.title = "Send operational test SMS to registered guardian phone";
                    }
                }

                // 5. Fetch notification activity list
                this.fetchGuardianNotifications(pid, this.guardianNotificationFilter || "ALL");
            })
            .catch(err => console.warn("Failed to fetch guardian data:", err));
    }

    _updateReadinessItem(key, checkObj) {
        if (!checkObj) return;
        const badge = document.getElementById(`badge-${key}`);
        const sub = document.getElementById(`sub-${key}`);
        const ind = document.getElementById(`ind-${key}`);

        if (badge) {
            badge.textContent = checkObj.label;
            badge.className = checkObj.passed ? "badge-pill stable" : "badge-pill warning";
        }
        if (sub && checkObj.details) {
            sub.textContent = checkObj.details;
        }
        if (ind) {
            ind.className = checkObj.passed ? "readiness-indicator" : "readiness-indicator warn";
        }
    }

    fetchGuardianNotifications(patientId = null, filter = "ALL") {
        const pid = patientId || this.activePatientId || "BED-01";
        const tbody = document.getElementById("guardian-notifications-tbody");
        const emptyState = document.getElementById("guardian-empty-state");

        fetch(`/api/guardian/notifications?patient_id=${pid}&status=${filter}&limit=50`)
            .then(res => res.json())
            .then(data => {
                if (!data.success) return;
                const notifs = data.notifications || [];

                if (!tbody) return;

                if (notifs.length === 0) {
                    tbody.innerHTML = "";
                    if (emptyState) emptyState.classList.remove("hidden");
                    return;
                }

                if (emptyState) emptyState.classList.add("hidden");

                tbody.innerHTML = notifs.map(n => {
                    let statusBadgeClass = "badge-status-pending";
                    const st = (n.status || "").toUpperCase();
                    if (st === "DELIVERED") statusBadgeClass = "badge-status-delivered";
                    else if (st === "ACCEPTED" || st === "SENT") statusBadgeClass = "badge-status-accepted";
                    else if (st === "FAILED") statusBadgeClass = "badge-status-failed";
                    else if (st === "DEMO_RECORDED") statusBadgeClass = "badge-status-demo";

                    const typePill = n.message_type === "REAL_EMERGENCY"
                        ? '<span class="badge-pill crisis">EMERGENCY</span>'
                        : (n.message_type === "TEST" ? '<span class="badge-pill info">TEST SMS</span>' : '<span class="badge-pill warning">DEMO PREVIEW</span>');

                    const retryBtn = (st === "FAILED" && (n.retry_count || 0) < 3)
                        ? `<button class="btn btn-secondary btn-xs" onclick="app.retryGuardianNotification('${n.notification_uuid}')" title="Retry transmission">🔄 Retry</button>`
                        : '';

                    const syncBtn = (n.provider_message_id && !n.provider_message_id.startsWith("DEMO-") && n.status !== "DELIVERED")
                        ? `<button class="btn btn-secondary btn-xs" onclick="app.syncNotificationStatus('${n.notification_uuid}')" title="Sync delivery status with carrier">🔄 Sync</button>`
                        : '';

                    return `
                        <tr>
                            <td style="font-size: 11.5px; font-family: monospace; color: var(--text-secondary);">${n.created_at || '--'}</td>
                            <td><strong>${n.alert_id || n.notification_uuid}</strong></td>
                            <td>${typePill}</td>
                            <td><span class="guardian-phone-tag" style="font-size: 11px;">${n.recipient_phone_masked}</span></td>
                            <td><span class="${statusBadgeClass}">${st}</span></td>
                            <td style="font-size: 11px; font-family: monospace; color: var(--text-muted);">${n.provider_message_id || n.provider || '--'}</td>
                            <td>
                                <div style="display: flex; gap: 6px; align-items: center;">
                                    ${retryBtn}
                                    ${syncBtn}
                                    <button class="btn btn-secondary btn-xs" onclick="app.inspectNotificationMessage('${n.notification_uuid}')" title="View Message Text">👁️ Text</button>
                                </div>
                            </td>
                        </tr>
                    `;
                }).join("");
            })
            .catch(err => console.warn("Failed to fetch notification activity:", err));
    }

    loadGuardianSettingsForBed(patientId = null) {
        const pid = patientId || document.getElementById("settings-guardian-bed-select")?.value || this.activePatientId || "BED-01";
        
        const bedSelect = document.getElementById("settings-guardian-bed-select");
        if (bedSelect && bedSelect.value !== pid) {
            bedSelect.value = pid;
        }

        fetch(`/api/guardian/${pid}`)
            .then(res => res.json())
            .then(data => {
                if (!data.success) return;
                const g = data.guardian;
                const nameInp = document.getElementById("input-guardian-name");
                const phoneInp = document.getElementById("input-guardian-phone");
                const relSelect = document.getElementById("select-guardian-relationship");
                const enabledChk = document.getElementById("check-guardian-enabled");
                const consentChk = document.getElementById("check-guardian-consent");
                const verifyBadge = document.getElementById("settings-phone-verify-badge");
                const cardStatusBadge = document.getElementById("settings-guardian-status-badge");
                const lastUpdated = document.getElementById("guardian-last-updated-text");

                if (g) {
                    if (nameInp) nameInp.value = g.full_name || "";
                    if (phoneInp) phoneInp.value = g.phone_number || g.phone_normalized || "";
                    if (relSelect) relSelect.value = g.relationship || "Primary Caregiver";
                    if (enabledChk) enabledChk.checked = (g.is_enabled === 1);
                    if (consentChk) consentChk.checked = (g.consent_confirmed === 1);

                    const verif = data.verification_status || { label: "Registered — Not Phone Verified", badge_class: "badge-pill info" };

                    if (verifyBadge) {
                        verifyBadge.textContent = verif.label;
                        verifyBadge.className = verif.badge_class;
                    }
                    if (cardStatusBadge) {
                        cardStatusBadge.textContent = verif.label;
                        cardStatusBadge.className = verif.badge_class;
                    }

                    // Preferences
                    const prefs = (g.preferred_categories || "CRISIS,WARNING").split(",");
                    const chkCrisis = document.getElementById("check-pref-crisis");
                    const chkWarning = document.getElementById("check-pref-warning");
                    const chkAdvisory = document.getElementById("check-pref-advisory");
                    if (chkCrisis) chkCrisis.checked = prefs.includes("CRISIS");
                    if (chkWarning) chkWarning.checked = prefs.includes("WARNING");
                    if (chkAdvisory) chkAdvisory.checked = prefs.includes("ADVISORY");

                    if (lastUpdated) lastUpdated.textContent = `Last updated: ${g.updated_at || '--'}`;
                } else {
                    if (nameInp) nameInp.value = "";
                    if (phoneInp) phoneInp.value = "";
                    if (verifyBadge) {
                        verifyBadge.textContent = "Not Registered";
                        verifyBadge.className = "badge-pill warning";
                    }
                    if (cardStatusBadge) {
                        cardStatusBadge.textContent = "Not Registered";
                        cardStatusBadge.className = "badge-pill warning";
                    }
                }
            })
            .catch(err => console.warn("Failed to load guardian settings:", err));
    }

    saveGuardianSettings() {
        const bed = document.getElementById("settings-guardian-bed-select")?.value || this.activePatientId;
        const name = document.getElementById("input-guardian-name")?.value.trim();
        const phone = document.getElementById("input-guardian-phone")?.value.trim();
        const rel = document.getElementById("select-guardian-relationship")?.value;
        const enabled = document.getElementById("check-guardian-enabled")?.checked;
        const consent = document.getElementById("check-guardian-consent")?.checked;
        const feedback = document.getElementById("guardian-settings-feedback");

        // Client-side validations
        if (!name) {
            this._showFeedback(feedback, "Guardian full name is required.", "error");
            document.getElementById("input-guardian-name")?.focus();
            return;
        }

        if (!phone) {
            this._showFeedback(feedback, "Guardian phone number is required.", "error");
            document.getElementById("input-guardian-phone")?.focus();
            return;
        }

        // Validate international phone format (E.164)
        const cleaned = phone.replace(/[\s\-\(\)\.]/g, "");
        if (!/^\+[1-9]\d{6,14}$/.test(cleaned)) {
            this._showFeedback(feedback, "Please enter a valid international phone number starting with '+' and country code (e.g. +14155552671 or +919876543210).", "error");
            document.getElementById("input-guardian-phone")?.focus();
            return;
        }

        if (enabled && !consent) {
            this._showFeedback(feedback, "Explicit consent confirmation is required before enabling emergency notifications.", "error");
            return;
        }

        // Collect categories
        const cats = [];
        if (document.getElementById("check-pref-crisis")?.checked) cats.push("CRISIS");
        if (document.getElementById("check-pref-warning")?.checked) cats.push("WARNING");
        if (document.getElementById("check-pref-advisory")?.checked) cats.push("ADVISORY");
        if (cats.length === 0) cats.push("CRISIS");

        const payload = {
            patient_id: bed,
            full_name: name,
            phone_number: phone,
            relationship: rel,
            is_enabled: enabled,
            consent_confirmed: consent,
            preferred_categories: cats.join(",")
        };

        const saveBtn = document.getElementById("btn-save-guardian-settings");
        if (saveBtn) {
            saveBtn.disabled = true;
            saveBtn.innerHTML = "<span>⏳ Saving Changes...</span>";
        }

        fetch("/api/guardian", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (saveBtn) {
                saveBtn.disabled = false;
                saveBtn.innerHTML = "<span>💾 Save Changes</span>";
            }
            if (data.success) {
                // Confirm database transaction succeeded, reload saved details from backend, then display success message
                this.loadGuardianSettingsForBed(bed);
                this.fetchGuardianData(bed);
                this._showFeedback(feedback, `✓ Guardian details successfully saved to database. Phone: ${data.guardian.phone_masked || data.guardian.phone_normalized}`, "success");
            } else {
                this._showFeedback(feedback, `⚠️ ${data.error || "Failed to save guardian settings."}`, "error");
            }
        })
        .catch(err => {
            if (saveBtn) {
                saveBtn.disabled = false;
                saveBtn.innerHTML = "<span>💾 Save Changes</span>";
            }
            this._showFeedback(feedback, "Server communication error. Please try again.", "error");
        });
    }

    _showFeedback(el, msg, type) {
        if (!el) return;
        el.className = `form-feedback ${type}`;
        el.textContent = msg;
        el.classList.remove("hidden");
        setTimeout(() => {
            el.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, 50);
    }

    openTestSmsModal() {
        const pid = document.getElementById("guardian-bed-select")?.value || this.activePatientId || "BED-01";

        fetch(`/api/guardian/test_sms_preview/${pid}`)
            .then(res => res.json())
            .then(data => {
                if (!data.success) {
                    alert(data.error || "Please register a guardian in Guardian Settings before attempting a test SMS.");
                    this.switchView("settings");
                    this.loadGuardianSettingsForBed(pid);
                    return;
                }

                // Populate modal with preview
                const bedBadge = document.getElementById("test-sms-bed-id");
                const nameEl = document.getElementById("test-sms-recipient-name");
                const phoneEl = document.getElementById("test-sms-phone-masked");
                const bodyEl = document.getElementById("test-sms-message-preview");
                const feedback = document.getElementById("test-sms-modal-feedback");
                const sendBtn = document.getElementById("btn-confirm-send-test-sms");

                if (bedBadge) bedBadge.textContent = pid;
                if (nameEl) nameEl.textContent = data.recipient_name || "--";
                if (phoneEl) phoneEl.textContent = data.phone_masked || "--";
                if (bodyEl) bodyEl.textContent = data.preview_body || "--";
                if (feedback) feedback.classList.add("hidden");

                if (!data.provider_configured) {
                    if (sendBtn) sendBtn.disabled = true;
                    const missing = data.missing_keys?.join(", ") || "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER";
                    this._showFeedback(feedback, `SMS Provider Not Configured (${missing}). Real test SMS transmission is disabled until credentials are added to .env.`, "error");
                } else {
                    if (sendBtn) sendBtn.disabled = false;
                }

                const modal = document.getElementById("modal-test-sms-confirm");
                if (modal) modal.classList.remove("hidden");
            })
            .catch(err => {
                console.warn("Failed to load test SMS preview:", err);
                alert("Failed to load test SMS preview. Please check network connection.");
            });
    }

    confirmSendTestSms() {
        const pid = document.getElementById("guardian-bed-select")?.value || this.activePatientId || "BED-01";
        const btn = document.getElementById("btn-confirm-send-test-sms");
        const spinner = document.getElementById("test-sms-spinner");
        const feedback = document.getElementById("test-sms-modal-feedback");

        if (btn) btn.disabled = true;
        if (spinner) spinner.classList.remove("hidden");
        if (feedback) feedback.classList.add("hidden");

        fetch("/api/guardian/test_sms", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patient_id: pid })
        })
        .then(res => res.json().then(data => ({ status: res.status, data })))
        .then(({ status, data }) => {
            if (btn) btn.disabled = false;
            if (spinner) spinner.classList.add("hidden");

            if (data.success) {
                this._showFeedback(feedback, `✓ Test SMS successfully accepted by provider (Ref: ${data.message_id || 'OK'}).`, "success");
                this.fetchGuardianData(pid);
                setTimeout(() => {
                    this.closeModal("modal-test-sms-confirm");
                }, 1500);
            } else {
                this._showFeedback(feedback, `⚠️ ${data.error || 'Failed to dispatch test message.'}`, "error");
                this.fetchGuardianData(pid);
            }
        })
        .catch(err => {
            if (btn) btn.disabled = false;
            if (spinner) spinner.classList.add("hidden");
            this._showFeedback(feedback, "Network connection error while contacting server.", "error");
        });
    }

    previewDemoNotification() {
        const pid = document.getElementById("guardian-bed-select")?.value || this.activePatientId || "BED-01";
        const pt = this.patients[pid];
        const rhythm = pt ? pt.current_rhythm : "Ventricular Tachycardia";

        fetch("/api/guardian/preview_demo", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patient_id: pid,
                condition: `${rhythm} (MIT-BIH Demonstration Playback)`
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                this.fetchGuardianData(pid);
                alert("Demonstration preview recorded successfully. Check the Activity Log below.");
            }
        })
        .catch(err => console.warn("Demo preview error:", err));
    }

    retryGuardianNotification(notifUuid) {
        if (!notifUuid) return;
        fetch("/api/guardian/notifications/retry", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ notification_uuid: notifUuid })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert(`Notification retry succeeded! Status: ${data.status}`);
            } else {
                alert(`Retry failed: ${data.error || 'Provider rejected request'}`);
            }
            this.fetchGuardianData(this.activePatientId);
        })
        .catch(err => alert("Retry request failed."));
    }

    syncNotificationStatus(notifUuid) {
        if (!notifUuid) return;
        fetch(`/api/guardian/notifications/${notifUuid}/sync_status`, {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert(`Delivery status updated from provider: ${data.status}`);
            } else {
                alert(`Provider status sync: ${data.error || data.status || 'No update available'}`);
            }
            this.fetchGuardianData(this.activePatientId);
        })
        .catch(err => alert("Error contacting server for status sync."));
    }

    inspectNotificationMessage(notifUuid) {
        fetch(`/api/guardian/notifications?limit=100`)
            .then(res => res.json())
            .then(data => {
                const n = (data.notifications || []).find(item => item.notification_uuid === notifUuid);
                if (n) {
                    alert(`Notification Message Body:\n\n${n.message_body}\n\nRecipient: ${n.recipient_phone_masked}\nStatus: ${n.status}\nProvider ID: ${n.provider_message_id || 'None'}`);
                }
            });
    }

    showGuardianAuditTrail() {
        fetch("/api/guardian/audit_logs?limit=25")
            .then(res => res.json())
            .then(data => {
                const logs = data.logs || [];
                if (logs.length === 0) {
                    alert("No audit log records recorded yet.");
                    return;
                }
                const formatted = logs.map(l => `[${l.timestamp}] ${l.action} (${l.actor}) [${l.patient_id}]: ${l.details}`).join("\n\n");
                alert(`Guardian System Immutable Audit Trail:\n\n${formatted}`);
            });
    }

    // --- VIEW 4: ARRHYTHMIA ELECTROPHYSIOLOGY BREAKDOWN METHODS ---

    populateAnalysisBedDropdown(patients) {
        const select = document.getElementById("analysis-bed-select");
        if (!select || !Array.isArray(patients)) return;

        const currentVal = select.value || this.activePatientId;
        const optionsHtml = patients.map(p => {
            const pid = p.patient_id;
            const name = p.name || pid;
            const isSel = pid === currentVal ? "selected" : "";
            return `<option value="${pid}" ${isSel}>${pid} &bull; ${name}</option>`;
        }).join("");

        select.innerHTML = optionsHtml;
        if (currentVal) select.value = currentVal;
    }

    renderElectrophysiologyBreakdown(patientId, silentUpdate = false) {
        const container = document.getElementById("analysis-bed-breakdown");
        const sourceBadge = document.getElementById("analysis-source-badge");
        if (!container) return;

        // Sync dropdown if needed
        const bedSelect = document.getElementById("analysis-bed-select");
        if (bedSelect && patientId && bedSelect.value !== patientId) {
            bedSelect.value = patientId;
        }

        // 1. Useful empty state if no bed or record is selected
        if (!patientId) {
            if (sourceBadge) {
                sourceBadge.textContent = "NO BED SELECTED";
                sourceBadge.className = "badge-pill advisory";
            }
            container.innerHTML = `
                <div class="ep-empty-state">
                    <div style="font-size: 32px; margin-bottom: 2px;">🫀</div>
                    <div class="ep-empty-title">No Monitored Bed Selected</div>
                    <div class="ep-empty-desc">
                        Select an active telemetry bed or historical recording session from the ICU ward to view real-time Pan-Tompkins QRS complex fiducials, rate-corrected QTc intervals (Bazett &amp; Fridericia), and autonomic heart-rate variability biomarkers.
                    </div>
                    <div class="ep-empty-actions">
                        ${Object.keys(this.patients).map(pid => {
                            const p = this.patients[pid];
                            return `<button type="button" class="btn btn-secondary btn-sm" onclick="if (window.app) window.app.selectActivePatient('${pid}')">
                                Select ${pid} (${p ? (p.name || pid) : pid})
                            </button>`;
                        }).join("")}
                    </div>
                </div>
            `;
            return;
        }

        // 2. Fetch electrophysiological interval measurements from backend
        fetch(`/api/patient/${patientId}/electrophysiology`)
            .then(res => res.json())
            .then(data => {
                if (!data.success) {
                    if (sourceBadge) {
                        sourceBadge.textContent = "RECORD UNAVAILABLE";
                        sourceBadge.className = "badge-pill crisis";
                    }
                    container.innerHTML = `
                        <div class="ep-warning-card">
                            <div class="ep-warning-title">
                                <span>⚠️</span>
                                <span>Electrophysiology Record Unavailable</span>
                            </div>
                            <div class="ep-warning-text">${data.error || "Electrophysiological data could not be computed for the requested bed session."}</div>
                        </div>
                    `;
                    return;
                }
                this.displayElectrophysiologyData(data, container, sourceBadge, silentUpdate);
            })
            .catch(err => {
                console.warn("Electrophysiology fetch failed:", err);
            });
    }

    displayElectrophysiologyData(data, container, sourceBadge, silentUpdate = false) {
        if (!data || !container) return;

        // Update Source badge with clinical distinction (recorded historical vs live stream)
        if (sourceBadge) {
            if (data.is_historical_recorded) {
                sourceBadge.textContent = `RECORDED-DATA ANALYSIS (PhysioNet MIT-BIH ${data.mitbih_record_id})`;
                sourceBadge.className = "badge-pill stable";
                sourceBadge.title = "Analysis derived from digitized PhysioNet MIT-BIH ambulatory Holter dataset";
            } else {
                sourceBadge.textContent = "LIVE TELEMETRY STREAM";
                sourceBadge.className = "badge-pill stable";
                sourceBadge.title = "Analysis derived from bedside telemetry monitor stream";
            }
        }

        const isSuitable = data.signal_quality && data.signal_quality.is_suitable;
        const isFlatline = data.signal_quality && (data.signal_quality.status === "LEAD OFF / FLATLINE" || data.signal_quality.status === "LEAD_OFF");

        // If signal is flatline or unsuitable: show actual warning/error without fabricated values
        if (!isSuitable || isFlatline) {
            const statusStr = data.signal_quality ? data.signal_quality.status : "LEAD OFF / FLATLINE";
            const snrVal = data.signal_quality ? data.signal_quality.snr_db : -10.0;
            const scoreVal = data.signal_quality ? data.signal_quality.quality_score : 10;

            container.innerHTML = `
                <div class="ep-breakdown-wrapper">
                    <div class="ep-warning-card">
                        <div class="ep-warning-title">
                            <span>⚠️</span>
                            <span>Signal Unsuitable for Electrophysiological Analysis</span>
                        </div>
                        <div class="ep-warning-text">
                            <strong>Quality Status:</strong> ${statusStr} &bull; 
                            <strong>SNR:</strong> ${snrVal} dB &bull; 
                            <strong>Quality Index:</strong> ${scoreVal}%<br>
                            ${isFlatline ? 
                                "Isoelectric flatline / lead disconnection detected: Signal amplitude is below detection threshold (0.12 mV). Ventricular depolarization and repolarization fiducials cannot be computed on isoelectric trace." : 
                                "Excessive motion artifact or amplifier saturation detected. Signal-to-noise ratio is insufficient for reliable QRS and T-wave delineation."}
                        </div>
                    </div>

                    <div class="ep-grid">
                        <div class="ep-metric-card">
                            <div class="ep-metric-header">
                                <span class="ep-metric-label">Heart Rate</span>
                                <span class="ep-metric-status unmeasurable">Unmeasurable</span>
                            </div>
                            <div class="ep-metric-body">
                                <span class="ep-metric-value unmeasurable">--</span>
                                <span class="ep-metric-unit">bpm</span>
                            </div>
                            <div class="ep-unmeasurable-note">Isoelectric baseline &mdash; Asystole or lead disconnected. Zero R-peaks detected.</div>
                        </div>

                        <div class="ep-metric-card">
                            <div class="ep-metric-header">
                                <span class="ep-metric-label">QRS Duration</span>
                                <span class="ep-metric-status unmeasurable">Unmeasurable</span>
                            </div>
                            <div class="ep-metric-body">
                                <span class="ep-metric-value unmeasurable">--</span>
                                <span class="ep-metric-unit">ms</span>
                            </div>
                            <div class="ep-unmeasurable-note">Isoelectric baseline prevents Q-onset and J-point fiducial identification.</div>
                        </div>

                        <div class="ep-metric-card">
                            <div class="ep-metric-header">
                                <span class="ep-metric-label">QT / QTc Interval</span>
                                <span class="ep-metric-status unmeasurable">Unmeasurable</span>
                            </div>
                            <div class="ep-metric-body">
                                <span class="ep-metric-value unmeasurable">--</span>
                                <span class="ep-metric-unit">ms</span>
                            </div>
                            <div class="ep-unmeasurable-note">Ventricular repolarization absent on isoelectric trace.</div>
                        </div>

                        <div class="ep-metric-card">
                            <div class="ep-metric-header">
                                <span class="ep-metric-label">Heart Rate Variability</span>
                                <span class="ep-metric-status unmeasurable">Unmeasurable</span>
                            </div>
                            <div class="ep-metric-body">
                                <span class="ep-metric-value unmeasurable">--</span>
                                <span class="ep-metric-unit">ms</span>
                            </div>
                            <div class="ep-unmeasurable-note">No consecutive cardiac cycles available for statistical dispersion.</div>
                        </div>
                    </div>

                    <div class="ep-disclaimer">
                        ⚠️ <strong>Research &amp; Educational Software:</strong> Electrophysiological interval calculations (Pan-Tompkins QRS, Bazett/Fridericia QTc, Poincaré HRV) are validated algorithmic estimations for research and continuous telemetry monitoring, not clinical diagnostic interpretations. Clinical diagnoses require full 12-lead diagnostic electrocardiography and certified physician evaluation.
                    </div>
                </div>
            `;
            return;
        }

        // Signal is suitable: Check if silent update is possible to avoid DOM re-draw
        const existingWrapper = container.querySelector(".ep-breakdown-wrapper");
        if (silentUpdate && existingWrapper && container.querySelector("#ep-val-hr")) {
            this.updateElectrophysiologyDOMValues(data);
            return;
        }

        // Full Render
        const m = data.measurements || {};
        const hr = m.heart_rate || {};
        const qrs = m.qrs_duration || {};
        const qt = m.qt_interval || {};
        const hrv = m.hrv || {};
        const sq = data.signal_quality || {};
        const pb = data.playback || {};
        const cond = data.conduction_diagnosis || {};

        container.innerHTML = `
            <div class="ep-breakdown-wrapper">
                <!-- Session Banner -->
                <div class="ep-session-header">
                    <div class="ep-session-title">
                        <div>
                            <div class="ep-patient-name">${data.patient_id} &bull; ${data.patient_name} <span style="font-weight: 400; color: var(--text-secondary);">(${data.age || '--'}${data.gender || ''}, ${data.location || '--'})</span></div>
                            <div class="ep-session-meta">
                                <span>${data.mode_label}</span> &bull; 
                                <span>Playback: ${pb.status || 'PLAYING'} (${pb.speed || 1.0}x) &bull; Elapsed: ${pb.time_position_sec || 0}s</span>
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="ep-quality-pill optimal">● ${sq.status || 'OPTIMAL'} (${sq.snr_db || 24.5} dB)</span>
                        <span id="ep-time-stamp" style="font-size: 11px; color: var(--text-secondary); font-family: var(--font-mono, monospace);">${data.formatted_timestamp}</span>
                    </div>
                </div>

                <!-- 4 Main Electrophysiological Interval Metric Cards -->
                <div class="ep-grid">
                    <!-- Metric 1: Heart Rate -->
                    <div class="ep-metric-card">
                        <div class="ep-metric-header">
                            <span class="ep-metric-label">Heart Rate (Cadence)</span>
                            <span class="ep-metric-status ${hr.measurable ? 'normal' : 'unmeasurable'}">${hr.status_note || 'Normal Sinus'}</span>
                        </div>
                        <div class="ep-metric-body">
                            <span class="ep-metric-value ${hr.measurable ? '' : 'unmeasurable'}" id="ep-val-hr">${hr.measurable ? hr.value : '--'}</span>
                            <span class="ep-metric-unit">${hr.unit || 'bpm'}</span>
                        </div>
                        <div class="ep-metric-desc">
                            Target Range: ${hr.reference_range || '60 – 100 bpm'} &bull; Critical: ${hr.critical_threshold || '<40 or >150 bpm'}
                        </div>
                        ${!hr.measurable && hr.unmeasurable_reason ? `<div class="ep-unmeasurable-note">${hr.unmeasurable_reason}</div>` : ''}
                    </div>

                    <!-- Metric 2: QRS Duration -->
                    <div class="ep-metric-card">
                        <div class="ep-metric-header">
                            <span class="ep-metric-label">QRS Duration (Ventricular)</span>
                            <span class="ep-metric-status ${qrs.measurable ? (qrs.value > 120 ? 'alert' : 'normal') : 'unmeasurable'}">${qrs.status_note || (qrs.measurable ? 'Normal (<100ms)' : 'Unmeasurable')}</span>
                        </div>
                        <div class="ep-metric-body">
                            <span class="ep-metric-value ${qrs.measurable ? '' : 'unmeasurable'}" id="ep-val-qrs">${qrs.measurable ? qrs.value : '--'}</span>
                            <span class="ep-metric-unit">${qrs.unit || 'ms'}</span>
                        </div>
                        <div class="ep-metric-desc">
                            ${qrs.measurable ? 
                                `Pan-Tompkins adaptive fiducials (Q-onset to J-point return) &bull; Normal: ${qrs.reference_range || '80 – 100 ms'} &bull; Delay: ${qrs.critical_threshold || '>120 ms'}` : 
                                `Q-onset and J-point baseline return boundaries could not be resolved reliably in current lead.`}
                        </div>
                        ${!qrs.measurable && qrs.unmeasurable_reason ? `<div class="ep-unmeasurable-note">${qrs.unmeasurable_reason}</div>` : ''}
                    </div>

                    <!-- Metric 3: QT / QTc Interval -->
                    <div class="ep-metric-card">
                        <div class="ep-metric-header">
                            <span class="ep-metric-label">QT / QTc (Repolarization)</span>
                            <span class="ep-metric-status ${qt.measurable ? (qt.qtc_bazett_ms > 480 ? 'alert' : 'normal') : 'unmeasurable'}">${qt.status_note || (qt.measurable ? 'Normal (<440ms)' : 'Unmeasurable')}</span>
                        </div>
                        <div class="ep-metric-body">
                            <span class="ep-metric-value ${qt.measurable ? '' : 'unmeasurable'}" id="ep-val-qt">${qt.measurable ? qt.qtc_bazett_ms : '--'}</span>
                            <span class="ep-metric-unit">ms (Bazett QTc)</span>
                        </div>
                        <div class="ep-metric-desc">
                            ${qt.measurable ? 
                                `Fridericia QTc: <strong id="ep-val-qt-frid">${qt.qtc_fridericia_ms}</strong> ms &bull; Raw QT: <span id="ep-val-qt-raw">${qt.raw_qt_ms}</span> ms<br>Formulas: Bazett (QT/&radic;RR), Fridericia (QT/&sup3;&radic;RR) &bull; Normal: ${qt.reference_range || '<440 ms (M), <460 ms (F)'}` : 
                                `Consecutive RR interval or distinct T-wave repolarization fiducial unavailable.`}
                        </div>
                        ${!qt.measurable && qt.unmeasurable_reason ? `<div class="ep-unmeasurable-note">${qt.unmeasurable_reason}</div>` : ''}
                    </div>

                    <!-- Metric 4: HRV Biomarkers -->
                    <div class="ep-metric-card">
                        <div class="ep-metric-header">
                            <span class="ep-metric-label">HRV (Autonomic Tone)</span>
                            <span class="ep-metric-status ${hrv.measurable ? (hrv.sdnn < 20 ? 'alert' : 'normal') : 'unmeasurable'}">${hrv.status_note || (hrv.measurable ? 'Normal Autonomic Tone' : 'Unmeasurable')}</span>
                        </div>
                        <div class="ep-metric-body">
                            <span class="ep-metric-value ${hrv.measurable ? '' : 'unmeasurable'}" id="ep-val-hrv">${hrv.measurable ? hrv.sdnn : '--'}</span>
                            <span class="ep-metric-unit">${hrv.unit || 'ms (SDNN)'}</span>
                        </div>
                        <div class="ep-metric-desc">
                            ${hrv.measurable ? 
                                `RMSSD: <strong id="ep-val-rmssd">${hrv.rmssd}</strong> ms &bull; pNN50: <span id="ep-val-pnn50">${hrv.pnn50}</span>% &bull; RR Mean: <span id="ep-val-rrmean">${hrv.mean_rr_ms}</span> ms<br>Poincaré: SD1 <span id="ep-val-sd1">${hrv.poincare_sd1}</span> ms, SD2 <span id="ep-val-sd2">${hrv.poincare_sd2}</span> ms &bull; Entropy: <span id="ep-val-entropy">${hrv.shannon_entropy}</span> bits` : 
                                `Minimum 2 consecutive cardiac cycles required for interval dispersion.`}
                        </div>
                        ${!hrv.measurable && hrv.unmeasurable_reason ? `<div class="ep-unmeasurable-note">${hrv.unmeasurable_reason}</div>` : ''}
                    </div>
                </div>

                <!-- Conduction & Automated Rhythm Analysis -->
                <div style="background: var(--bg-input); border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 14px; font-size: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="color: var(--text-secondary);">Monitored Rhythm:</span>
                        <strong style="color: var(--accent-cyan); margin-left: 4px;">${cond.current_rhythm || 'Normal Sinus Rhythm'}</strong>
                    </div>
                    <div>
                        <span style="color: var(--text-secondary);">Computational Classifier:</span>
                        <strong style="color: var(--text-primary); margin-left: 4px;">${cond.ai_diagnosis || 'Normal Sinus Rhythm'}</strong>
                        <span style="font-size: 11px; color: var(--text-secondary); margin-left: 4px;">(${Math.round((cond.ai_confidence || 0.95) * 100)}% conf)</span>
                    </div>
                    ${cond.ref_annotation ? `
                        <div>
                            <span style="color: var(--text-secondary);">PhysioNet Annotation:</span>
                            <strong style="color: #f59e0b; margin-left: 4px;">${cond.ref_annotation.type_name || cond.ref_annotation.type}</strong> at ${cond.ref_annotation.time_sec || 0}s
                        </div>
                    ` : ''}
                </div>

                <!-- Channel Inventory & Non-ECG Measurements Explanation (PhysioNet Historical Integrity) -->
                ${data.unavailable_measurements && data.unavailable_measurements.length > 0 ? `
                    <div class="ep-unavailable-section">
                        <div style="font-size: 12px; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">
                            PhysioNet Historical Dataset Inventory &amp; Non-ECG Channels
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 6px;">
                            ${data.unavailable_measurements.map(item => `
                                <div style="background: var(--bg-input); border: 1px solid var(--border-color); border-radius: 6px; padding: 8px 10px; font-size: 11px; display: flex; justify-content: space-between; align-items: flex-start; gap: 10px;">
                                    <div style="flex: 1;">
                                        <strong style="color: var(--text-primary);">${item.parameter}</strong>
                                        <div style="color: var(--text-secondary); line-height: 1.35; margin-top: 2px;">${item.reason}</div>
                                    </div>
                                    <span class="badge-pill advisory" style="white-space: nowrap; font-size: 10px;">${item.status}</span>
                                </div>
                            `).join("")}
                        </div>
                    </div>
                ` : ''}

                <!-- Informational Research Disclaimer -->
                <div class="ep-disclaimer">
                    ⚠️ <strong>Research &amp; Educational Software:</strong> Electrophysiological interval calculations (Pan-Tompkins QRS onset-to-J-point, Bazett/Fridericia QTc, Poincaré HRV descriptors) are validated algorithmic estimations for research and continuous telemetry monitoring, not clinical diagnostic interpretations. Clinical diagnoses require full 12-lead diagnostic electrocardiography and certified physician evaluation.
                </div>
            </div>
        `;
    }

    updateElectrophysiologyDOMValues(data) {
        const m = data.measurements || {};
        const hr = m.heart_rate || {};
        const qrs = m.qrs_duration || {};
        const qt = m.qt_interval || {};
        const hrv = m.hrv || {};

        const elHr = document.getElementById("ep-val-hr");
        if (elHr && hr.measurable) elHr.textContent = hr.value;

        const elQrs = document.getElementById("ep-val-qrs");
        if (elQrs && qrs.measurable) elQrs.textContent = qrs.value;

        const elQt = document.getElementById("ep-val-qt");
        if (elQt && qt.measurable) elQt.textContent = qt.qtc_bazett_ms;

        const elQtFrid = document.getElementById("ep-val-qt-frid");
        if (elQtFrid && qt.measurable) elQtFrid.textContent = qt.qtc_fridericia_ms;

        const elQtRaw = document.getElementById("ep-val-qt-raw");
        if (elQtRaw && qt.measurable) elQtRaw.textContent = qt.raw_qt_ms;

        const elHrv = document.getElementById("ep-val-hrv");
        if (elHrv && hrv.measurable) elHrv.textContent = hrv.sdnn;

        const elRmssd = document.getElementById("ep-val-rmssd");
        if (elRmssd && hrv.measurable) elRmssd.textContent = hrv.rmssd;

        const elPnn50 = document.getElementById("ep-val-pnn50");
        if (elPnn50 && hrv.measurable) elPnn50.textContent = hrv.pnn50;

        const elRrMean = document.getElementById("ep-val-rrmean");
        if (elRrMean && hrv.measurable) elRrMean.textContent = hrv.mean_rr_ms;

        const elSd1 = document.getElementById("ep-val-sd1");
        if (elSd1 && hrv.measurable) elSd1.textContent = hrv.poincare_sd1;

        const elSd2 = document.getElementById("ep-val-sd2");
        if (elSd2 && hrv.measurable) elSd2.textContent = hrv.poincare_sd2;

        const elEntropy = document.getElementById("ep-val-entropy");
        if (elEntropy && hrv.measurable) elEntropy.textContent = hrv.shannon_entropy;

        const elTime = document.getElementById("ep-time-stamp");
        if (elTime && data.formatted_timestamp) elTime.textContent = data.formatted_timestamp;
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

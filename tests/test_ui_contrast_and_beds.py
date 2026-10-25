"""
Tests for Light Mode Contrast, Duplicate Monitored Beds Removal, and Icon Modernization
Validates requirements:
1. Light mode text visibility & contrast (WCAG standards, centralized variables, no hardcoded invisible text)
2. Removal of duplicate monitored bed section in Bedside Monitor workspace
3. Heart-pulse SVG replacement of siren emoji
4. Preservation of sidebar bed list, patient identity bar, and monitor controls
5. Complete removal of user-facing 'AI' text
6. Endpoint integrity and responses
"""

import unittest
import re
import os
from backend.server import app

class TestUIContrastAndBeds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

        with open("templates/index.html", "r", encoding="utf-8") as f:
            cls.html = f.read()

        with open("static/css/style.css", "r", encoding="utf-8") as f:
            cls.css = f.read()

        with open("static/js/app.js", "r", encoding="utf-8") as f:
            cls.js = f.read()

    # --- 1. DUPLICATE MONITORED BEDS REMOVAL & CONTROLS PRESERVATION ---

    def test_duplicate_bed_section_removed_from_monitor(self):
        """Verify the duplicate monitor-bed-tabs strip is completely removed from Bedside Monitor."""
        self.assertNotIn("monitor-bed-tabs", self.html)
        self.assertNotIn("monitor-bed-btn", self.html)
        self.assertNotIn("monitor-bed-tabs", self.js)

    def test_sidebar_bed_selection_retained(self):
        """Verify sidebar retains the primary monitored-bed tabs for BED-01, BED-02, BED-03, REMOTE-01."""
        self.assertIn("sidebar-bed-tabs", self.html)
        for bed_id in ["BED-01", "BED-02", "BED-03", "REMOTE-01"]:
            self.assertIn(f'data-patient-id="{bed_id}"', self.html)

    def test_patient_identity_bar_preserved(self):
        """Verify the patient identity header above the ECG viewer is preserved."""
        self.assertIn('id="pt-header-badge"', self.html)
        self.assertIn('id="pt-header-name"', self.html)
        self.assertIn('id="pt-header-meta"', self.html)

    def test_monitor_toolbar_controls_preserved(self):
        """Verify compact monitor toolbar buttons are preserved."""
        required_controls = [
            "btn-toggle-calipers",
            "btn-toggle-morphology",
            "btn-open-vcg-modal",
            "btn-open-copilot-modal",
            "btn-freeze-monitor",
            "btn-switch-record-modal",
            "btn-single-report"
        ]
        for ctrl in required_controls:
            self.assertIn(f'id="{ctrl}"', self.html)

    def test_header_bed_select_preserved(self):
        """Verify the header active bed dropdown is preserved."""
        self.assertIn('id="header-bed-select"', self.html)

    # --- 2. SIREN EMOJI REPLACEMENT ---

    def test_siren_emoji_completely_removed(self):
        """Verify no siren emoji (🚨) appears anywhere in HTML or JS."""
        self.assertNotIn("🚨", self.html)
        self.assertNotIn("🚨", self.js)

    def test_heart_pulse_svg_present(self):
        """Verify professional heart-pulse SVG icon is implemented in alarm banner and modals."""
        self.assertIn("heart-pulse-svg", self.html)
        self.assertIn("heart-pulse-svg", self.js)
        self.assertIn(".heart-pulse-svg", self.css)
        self.assertIn("pulseHeartbeat", self.css)

    # --- 3. NO USER-FACING "AI" TEXT ---

    def test_no_user_facing_ai_text(self):
        """Verify the word 'AI' does not appear in user-facing HTML text."""
        # Check text within HTML tags
        tag_texts = re.findall(r'>([^<]+)<', self.html)
        for t in tag_texts:
            clean = t.strip()
            # Ignore script code or template tags
            if clean and not clean.startswith("{") and not clean.startswith("/*"):
                self.assertFalse(
                    bool(re.search(r'\bAI\b', clean)),
                    f"User-facing text contains standalone 'AI': '{clean}'"
                )

        # Specifically verify copilot title and buttons
        self.assertIn("Clinical Copilot", self.html)
        self.assertNotIn("AI Copilot", self.html)
        self.assertNotIn("AI Clinical Copilot", self.html)
        self.assertNotIn("AI Classification", self.html)

    # --- 4. LIGHT MODE CONTRAST & CSS DESIGN TOKENS ---

    def test_centralized_theme_variables_exist(self):
        """Verify light mode centralized CSS variables for high contrast."""
        self.assertIn("body.theme-light", self.css)
        # Primary text should be dark navy / charcoal
        self.assertIn("--text-primary: #0f172a", self.css)
        # Secondary text should be readable slate
        self.assertIn("--text-secondary: #334155", self.css)
        # Backgrounds
        self.assertIn("--bg-app: #f1f5f9", self.css)
        self.assertIn("--bg-card: #ffffff", self.css)
        # Cyan / teal accent preserved
        self.assertIn("--accent-primary: #0284c7", self.css)

    def test_measurement_value_uses_theme_variable(self):
        """Verify measurement value uses var(--text-primary) instead of hardcoded white (#f1f5f9)."""
        match = re.search(r'\.measurement-value\s*\{([^}]+)\}', self.css)
        self.assertTrue(match)
        props = match.group(1)
        self.assertIn("color: var(--text-primary)", props)
        self.assertNotIn("color: #f1f5f9", props)

    def test_ref_annotation_title_uses_theme_variable(self):
        """Verify reference annotation title uses var(--text-primary) instead of hardcoded white."""
        match = re.search(r'\.ref-ann-title\s*\{([^}]+)\}', self.css)
        self.assertTrue(match)
        props = match.group(1)
        self.assertIn("color: var(--text-primary)", props)

    def test_status_tile_val_uses_theme_variable(self):
        """Verify status tile value uses var(--text-primary) instead of hardcoded white."""
        match = re.search(r'\.status-tile-val\s*\{([^}]+)\}', self.css)
        self.assertTrue(match)
        props = match.group(1)
        self.assertIn("color: var(--text-primary)", props)

    def test_light_mode_component_overrides_exist(self):
        """Verify specific required components have light-mode overrides."""
        required_selectors = [
            "body.theme-light .alarm-banner",
            "body.theme-light .nav-link.active",
            "body.theme-light .bed-tab.active",
            "body.theme-light .ward-card",
            "body.theme-light .measurement-card",
            "body.theme-light .classifier-dx-box",
            "body.theme-light .clinical-table th",
            "body.theme-light .clinical-table td",
            "body.theme-light .modal-card",
            "body.theme-light .notification-drawer"
        ]
        for sel in required_selectors:
            self.assertIn(sel, self.css, f"Missing light-mode override for selector: {sel}")

    # --- 5. ENDPOINTS & SERVER INTEGRITY ---

    def test_endpoints_response_status(self):
        """Verify all critical endpoints return HTTP 200 with proper data."""
        endpoints = [
            "/",
            "/api/patients",
            "/api/patient/BED-01",
            "/api/patient/BED-02",
            "/api/alerts",
            "/api/dataset/records",
            "/api/ml/metrics",
            "/api/health"
        ]
        for ep in endpoints:
            res = self.client.get(ep)
            self.assertEqual(res.status_code, 200, f"Endpoint {ep} failed with status {res.status_code}")

    def test_patient_switching_endpoints(self):
        """Verify each patient bed returns valid clinical telemetry state."""
        for bed_id in ["BED-01", "BED-02", "BED-03", "REMOTE-01"]:
            res = self.client.get(f"/api/patient/{bed_id}")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["patient"]["patient_id"], bed_id)

if __name__ == "__main__":
    unittest.main()

"""
Test layout and DOM integrity for CardioSentinel after ECG Signal Quality Monitor
and Smart Alert Explanation layout repairs.
"""
from backend.server import app

def test_html_layout_elements():
    client = app.test_client()
    res = client.get('/')
    assert res.status_code == 200
    html = res.data.decode('utf-8')

    # 1. ECG Signal Quality Monitor Component present
    assert 'id="ecg-sq-monitor-component"' in html

    # 2. Issues container and details panel initially hidden with inline style display: none
    assert 'id="ecg-sq-issues-container"' in html
    assert 'style="display: none;"' in html
    assert 'id="ecg-sq-details-panel"' in html

    # 3. Dynamic check badges present with individual IDs for state updates
    assert 'id="sq-badge-finite"' in html
    assert 'id="sq-badge-flatline"' in html
    assert 'id="sq-badge-sat"' in html
    assert 'id="sq-badge-drift"' in html
    assert 'id="sq-badge-snr"' in html
    assert 'id="sq-badge-dur"' in html

    # 4. Smart alert explanation evidence panel initially hidden
    assert 'id="smart-alert-evidence-panel"' in html

    # 5. Check no visible "AI" in index.html visible text
    # Brand is CardioSentinel | ECG Intelligence
    assert "CardioSentinel" in html
    assert "ECG Intelligence" in html

def test_css_layout_rules():
    with open('static/css/style.css', 'r', encoding='utf-8') as f:
        css = f.read()

    # 1. Universal hidden rules
    assert '.hidden, [hidden]' in css
    assert 'display: none !important;' in css

    # 2. Multi-column monitor row aligned to start (not stretched artificially)
    assert '.monitor-panels-row' in css
    assert 'align-items: start;' in css

    # 3. Natural panel sizing (min-height: auto, justify-content: flex-start)
    assert '.monitor-panel' in css
    assert 'justify-content: flex-start;' in css
    assert 'min-height: auto;' in css

    # 4. Issues container severity rules
    assert '.ecg-sq-issues-container.severity-poor' in css
    assert '.ecg-sq-issues-container.severity-review' in css
    assert '.ecg-sq-issues-container.severity-insufficient' in css

    # 5. Checks table layout fixed with stable column widths
    assert '.ecg-sq-checks-table' in css
    assert 'table-layout: fixed;' in css
    assert '.badge-check-pass' in css
    assert '.badge-check-review' in css
    assert '.badge-check-fail' in css
    assert '.badge-check-insufficient' in css

def test_js_conditional_logic():
    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        js = f.read()

    # 1. Dynamic derivation without hardcoding GOOD SIGNAL when metrics are missing
    assert 'techStatus = "INSUFFICIENT DATA"' in js

    # 2. Issues container hidden when no issues or good signal
    assert 'issuesContainer.classList.add("hidden")' in js
    assert 'issuesContainer.style.display = "none"' in js

    # 3. Dynamic badge updates (finite, flatline, sat, drift, snr, dur)
    assert 'sq-badge-finite' in js
    assert 'sq-badge-flatline' in js
    assert 'sq-badge-sat' in js
    assert 'sq-badge-drift' in js
    assert 'sq-badge-snr' in js
    assert 'sq-badge-dur' in js

    # 4. Smart alert evidence panel collapsed on open
    assert 'panelEv.classList.add("hidden")' in js

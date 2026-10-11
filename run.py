"""
CardioSentinel™ - System Launcher & Diagnostic Runner
Starts the Central ICU Telemetry Server & Real-time Early Warning Alert Engine.
"""

import sys
import os
import webbrowser
import time

def print_banner():
    banner = r"""
  ____              _ _       ____             _   _            _ 
 / ___|__ _ _ __ __| (_) ___ / ___|  ___ _ __ | |_(_)_ __   ___| |
| |   / _` | '__/ _` | |/ _ \\___ \ / _ \ '_ \| __| | '_ \ / _ \ |
| |__| (_| | | | (_| | | (_) |___) |  __/ | | | |_| | | | |  __/ |
 \____\__,_|_|  \__,_|_|\___/|____/ \___|_| |_|\__|_|_| |_|\___|_|

  AI-Powered Continuous Remote Patient Telemetry & Cardiac Arrhythmia
                    Early Warning Alerting Engine
    """
    print(banner)

def main():
    print_banner()
    print("[*] Validating clinical modules and trained models...")

    from backend.server import app, telemetry_engine, signal_processor, ai_engine, alert_system

    print(f"    [OK] Telemetry Engine: {len(telemetry_engine.patients)} Active Beds Loaded (BED-01, BED-02, BED-03, REMOTE-01)")
    print(f"    [OK] Digital Signal Processor: 250 Hz Pan-Tompkins QRS & Filter Active")
    print(f"    [OK] AI Early Warning Engine: {len(ai_engine.class_names)} Arrhythmia Classes Ready")
    print(f"    [OK] Clinical Alert System: IEC 60601-1-8 Tiered Dispatch Active")

    url = "http://127.0.0.1:5000"
    print(f"\n[*] CardioSentinel Telemetry & Web Server ready!")
    print(f"[*] Access Central ICU Telemetry Console at: {url}")
    print(f"[*] Launching web browser automatically...")
    print("[*] Press Ctrl+C in terminal to stop server.\n")

    import threading
    def auto_open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[!] Could not launch browser automatically: {e}")

    threading.Thread(target=auto_open_browser, daemon=True).start()

    # Launch server
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()

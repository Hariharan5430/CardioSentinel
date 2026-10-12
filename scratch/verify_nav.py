import urllib.request

with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=5) as resp:
    html = resp.read().decode("utf-8")

nav_items = [
    "Ward Overview",
    "Bedside Monitor",
    "Record Browser",
    "Arrhythmia Analysis",
    "Model Laboratory",
    "Alert Log",
    "Shift Reports",
    "Settings",
    "System Health",
    "About"
]

print("--- Navigation Items in Served HTML ---")
all_ok = True
for item in nav_items:
    has_icon = f'title="{item}"' in html
    has_text = f'class="nav-text">{item}</span>' in html
    status = "OK" if (has_icon and has_text) else "FAIL"
    if not (has_icon and has_text):
        all_ok = False
    print(f"[{status}] {item}: Icon={has_icon}, Text={has_text}")

print("\n--- Active Link Check ---")
active_ok = 'class="nav-link active" data-view="monitor"' in html
print(f"[{'OK' if active_ok else 'FAIL'}] Bedside Monitor Active: {active_ok}")

print("\n--- Brand Card Check ---")
b1 = 'class="sidebar-brand-title">CardioSentinel</div>' in html
b2 = 'class="sidebar-brand-subtitle">ECG Intelligence</div>' in html
b3 = 'v2.1.0' in html
b4 = 'Research Edition' in html
b5 = 'For educational and research use only.' in html
brand_ok = all([b1, b2, b3, b4, b5])
print(f"[{'OK' if brand_ok else 'FAIL'}] Sidebar Brand Card: {brand_ok}")

if all_ok and active_ok and brand_ok:
    print("\n>>> ALL 10 SIDEBAR ITEMS AND BRAND FOOTER MATCH THE IMAGE PERFECTLY! <<<")

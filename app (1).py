from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)
model = joblib.load(Path(__file__).with_name("model.pkl"))

NUMERIC = [
    ("age", "Age", "30"),
    ("bmi", "BMI", "24.5"),
    ("hours_sleep_per_night", "Hours of sleep per night", "7"),
    ("stress_level", "Stress level", "3"),
    ("fasting_blood_sugar", "Fasting blood sugar", "95"),
    ("hba1c_level", "HbA1c level", "5.4"),
    ("blood_pressure_systolic", "Systolic blood pressure", "120"),
    ("blood_pressure_diastolic", "Diastolic blood pressure", "80"),
    ("waist_circumference_cm", "Waist circumference (cm)", "80"),
]
CATEGORICAL = [
    ("gender", "Gender", "Female"),
    ("city", "City", "Pune"),
    ("family_history_diabetes", "Family history of diabetes", "No"),
    ("physical_activity_level", "Physical activity level", "Moderate"),
    ("diet_type", "Diet type", "Balanced"),
    ("smoking_status", "Smoking status", "Never"),
    ("alcohol_consumption", "Alcohol consumption", "Never"),
    ("income_bracket", "Income bracket", "Middle"),
]
FEATURES = [x[0] for x in NUMERIC + CATEGORICAL]

PAGE = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diabetes Risk Prediction</title>
<style>
:root{color-scheme:light dark;--bg:#f2f5fb;--card:#fff;--txt:#172033;--muted:#68738a;--field:#fff;--line:#d9e0ed;--accent:#4967f5}
body.dark{--bg:#101522;--card:#1b2435;--txt:#edf2ff;--muted:#abb6cc;--field:#121a29;--line:#354158;--accent:#8296ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:15px system-ui,Arial,sans-serif}
main{max-width:980px;margin:32px auto;padding:0 16px}.head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
h1{font-size:clamp(27px,4vw,38px);margin:0 0 8px}.muted{color:var(--muted);margin:0 0 22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;box-shadow:0 12px 32px #0000000b}
h2{font-size:18px;margin:4px 0 15px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}
label{display:block;font-size:13px;font-weight:700;color:var(--muted);margin-bottom:6px}
input{width:100%;padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--field);color:var(--txt);font:inherit}
button{padding:12px 16px;border-radius:11px;border:1px solid var(--line);font-weight:700;cursor:pointer;background:var(--card);color:var(--txt)}
.primary{background:var(--accent);color:white;border:0}.actions{display:flex;gap:10px;margin-top:22px;flex-wrap:wrap}
.section{margin-top:24px}.result{padding:15px;border-radius:12px;background:#e6f7ed;color:#15552f;margin-bottom:18px}
.error{background:#fff0f0;color:#9b1c1c}.note{font-size:12px;color:var(--muted);margin-top:18px}
@media(max-width:600px){.grid{grid-template-columns:1fr}.card{padding:16px}}
</style></head><body><main>
<div class="head"><div><h1>Diabetes Risk Prediction</h1></div><button type="button" id="theme">Dark / Light mode</button></div>
<p class="muted">Enter the patient details to get a model prediction.</p><div class="card">
{% if prediction is not none %}<div class="result"><b>Model prediction:</b> {{prediction}}</div>{% endif %}
{% if error %}<div class="result error">{{error}}</div>{% endif %}
<form method="post"><h2>Numerical details</h2><div class="grid">
{% for key,label,default in numeric %}<div><label for="{{key}}">{{label}}</label><input id="{{key}}" name="{{key}}" type="number" step="any" required value="{{values.get(key,default)}}"></div>{% endfor %}
</div><div class="section"><h2>Categorical details</h2><div class="grid">
{% for key,label,default in categorical %}<div><label for="{{key}}">{{label}}</label><input id="{{key}}" name="{{key}}" required value="{{values.get(key,default)}}"></div>{% endfor %}
</div></div><div class="actions"><button class="primary" type="submit">Predict risk</button><button type="reset">Reset</button></div></form>
<p class="note">Educational demo only; this is not a medical diagnosis. Consult a qualified healthcare professional for medical advice.</p>
</div></main>
<script>const b=document.body,t=document.getElementById('theme');if(localStorage.getItem('dr-theme')==='dark')b.classList.add('dark');t.onclick=()=>{b.classList.toggle('dark');localStorage.setItem('dr-theme',b.classList.contains('dark')?'dark':'light')};</script>
</body></html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction, error, values = None, None, {}
    if request.method == "POST":
        try:
            row = {}
            for key, label, default in NUMERIC:
                raw = request.form.get(key, "").strip()
                if not raw:
                    raise ValueError(f"Enter {label}.")
                row[key] = float(raw)
                values[key] = raw
            for key, label, default in CATEGORICAL:
                raw = request.form.get(key, "").strip()
                if not raw:
                    raise ValueError(f"Enter {label}.")
                row[key] = raw
                values[key] = raw
            prediction = str(model.predict(pd.DataFrame([row], columns=FEATURES))[0])
        except Exception as exc:
            error = f"Prediction failed: {exc}"
    return render_template_string(PAGE, numeric=NUMERIC, categorical=CATEGORICAL,
                                  values=values, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

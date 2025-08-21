# import os
# port = int(os.environ.get("PORT", 8501))
# os.environ["STREAMLIT_SERVER_PORT"] = str(port)
# import numpy as np
# import pandas as pd
# from PIL import Image
# import qrcode
# from qrcode.constants import ERROR_CORRECT_H
# import io
# import streamlit as st
# import joblib
# import shap
# import matplotlib.pyplot as plt

# try:
#     import joblib
# except Exception:
#     joblib = None
# try:
#     import gspread
# except Exception:
#     gspread = None

# # ──────────────────────────────
# # PAGE CONFIG & GLOBAL STYLES
# # ──────────────────────────────
# st.set_page_config(
#     page_title="Delhi AQI – Prediction & Insights",
#     page_icon="🌫️",
#     layout="wide"
# )
# st.markdown(
#     """
#     <style>
#     .badge { padding: 0.35rem 0.7rem; border-radius: 999px; font-weight: 600; display: inline-block; }
#     .badge.good { background:#e7f5e9; color:#1e7e34; }
#     .badge.moderate { background:#fff3cd; color:#856404; }
#     .badge.poor { background:#ffe5d0; color:#a1490c; }
#     .badge.verypoor { background:#fde2e1; color:#9b1c1c; }
#     .badge.severe { background:#f8d7da; color:#721c24; }
#     .card { border-radius: 18px; padding: 16px; border: 1px solid #eee; background: white; box-shadow: 0 2px 12px rgba(0,0,0,0.04); height: 100%; }
#     .qr-box { text-align:center; }
#     .qr-title { font-weight:700; margin-bottom:0.3rem; }
#     small.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace; color:#666; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ──────────────────────────────
# # CONSTANTS & HELPERS
# # ──────────────────────────────
# APP_URL = "https://pollutionappcreatedbyalok.streamlit.app/"

# COLUMNS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]

# PRESENTS = {
#     "Good":       [30,  40,  20,  5,  0.4, 10],
#     "Moderate":   [90, 110,  40, 10,  1.2, 30],
#     "Poor":       [200, 250, 90, 20,  2.0, 50],
#     "Very Poor":  [300, 350, 120, 30,  3.5, 70],
#     "Severe":     [400, 500, 150, 40,  4.5, 90],
# }

# WHO_LIMITS = {
#     "PM2.5": 15,
#     "PM10": 45,
#     "NO2": 25,
#     "SO2": 40,
#     "CO": 4.0,    # mg/m3 (8-hour guideline)
#     "Ozone": 100,
# }

# DELHI_AVG = {
#     "PM2.5": 120,
#     "PM10": 200,
#     "NO2": 45,
#     "SO2": 12,
#     "CO": 1.7,
#     "Ozone": 60,
# }
# from typing import Dict, List, Tuple, Optional


# POLLUTANT_INFO: Dict[str, str] = {
#     "PM2.5": "Fine particles (≤2.5μm) penetrate deep into lungs; linked to heart & lung disease.",
#     "PM10": "Coarse particles (≤10μm) irritate airways; worsen asthma and bronchitis.",
#     "NO2":  "Traffic/industrial gas; inflames airways; reduces lung function over time.",
#     "SO2":  "From coal/oil burning; triggers wheezing, coughing; forms secondary PM.",
#     "CO":   "Colorless gas; reduces oxygen delivery in body; dangerous in high doses.",
#     "Ozone":"Formed in sunlight; irritates airways; causes chest pain & coughing.",
# }
# def ensure_session_defaults():
#     if "values" not in st.session_state:
#         st.session_state.values = {k: float(v) for k, v in zip(COLUMNS, PRESENTS["Moderate"])}
#     if "last_prediction" not in st.session_state:
#         st.session_state.last_prediction = None  # (aqi_value:int, aqi_label:str)
#     if "scenario_applied" not in st.session_state:
#         st.session_state.scenario_applied = ""
#     if "nav" not in st.session_state:
#         st.session_state.nav = "1) Understand + Share"
        
# def normalize_values(values: Dict[str, float]) -> Dict[str, float]:
#     """Ensure we always have all COLUMNS with float values. Prevents shape/key errors."""
#     safe = {}
#     if isinstance(values, dict):
#         for c in COLUMNS:
#             try:
#                 safe[c] = float(values.get(c, 0.0))
#             except Exception:
#                 safe[c] = 0.0
#     else:
#         # if someone accidentally put a list/tuple in session_state.values
#         for i, c in enumerate(COLUMNS):
#             try:
#                 safe[c] = float(values[i])
#             except Exception:
#                 safe[c] = 0.0
#     return safe
    
# def load_model_and_encoder():
#     """Load RF model + label encoder. Safe fallback if missing."""
#     model, encoder = None, None
#     try:
#         if joblib is not None and os.path.exists("aqi_rf_model.joblib"):
#             model = joblib.load("aqi_rf_model.joblib")
#     except Exception:
#         model = None
#     try:
#         # Use the most common filename first
#         if joblib is not None and os.path.exists("label_encoder.joblib"):
#             encoder = joblib.load("label_encoder.joblib")
#         elif joblib is not None and os.path.exists("label_encoder_.joblib"):
#             encoder = joblib.load("label_encoder_.joblib")
#     except Exception:
#         encoder = None
#     return model, encoder
    
# def simple_category_from_aqi(aqi: int) -> str:
#     if aqi <= 50: return "Good"
#     if aqi <= 100: return "Satisfactory"
#     if aqi <= 200: return "Moderate"
#     if aqi <= 300: return "Poor"
#     if aqi <= 400: return "Very Poor"
#     return "Severe"
    
# def badge_class(label: str) -> str:
#     key = label.replace(" ", "").lower()
#     if key in ["good"]: return "good"
#     if key in ["moderate", "satisfactory"]: return "moderate"
#     if key == "poor": return "poor"
#     if key in ["verypoor", "verybad"]: return "verypoor"
#     return "severe"
    
# def predict_aqi(pm25, pm10, no2, so2, co, ozone):
#     """
#     Predicts AQI value & category based on pollutant levels.
#     Returns (aqi_value, aqi_category).
#     """
#     try:
#         # Load model + encoder only once
#         if "MODEL" not in st.session_state:
#             st.session_state.MODEL, st.session_state.ENCODER = load_model()

#         MODEL = st.session_state.MODEL
#         ENCODER = st.session_state.ENCODER

#         # Prepare input array
#         X = np.array([[pm25, pm10, no2, so2, co, ozone]])

#         # Predict AQI value (regression output)
#         aqi_val = MODEL.predict(X)[0]

#         # Predict AQI category (classification output if available)
#         if hasattr(MODEL, "predict_proba"):  
#             # If classification model with label encoding
#             y_class = MODEL.predict(X)
#             aqi_label = ENCODER.inverse_transform(y_class)[0]
#         else:
#             # Fallback: manually classify based on AQI range
#             if aqi_val <= 50:
#                 aqi_label = "Good"
#             elif aqi_val <= 100:
#                 aqi_label = "Satisfactory"
#             elif aqi_val <= 200:
#                 aqi_label = "Moderate"
#             elif aqi_val <= 300:
#                 aqi_label = "Poor"
#             elif aqi_val <= 400:
#                 aqi_label = "Very Poor"
#             else:
#                 aqi_label = "Severe"

#         return round(aqi_val, 2), aqi_label

#     except Exception as e:
#         st.error(f"⚠️ Prediction error: {e}")
#         return None, None
        
# def make_qr_bytes(content: str, size_px: int = 160) -> bytes:
#     qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_H, box_size=10, border=2)
#     qr.add_data(content)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
#     img = img.resize((size_px, size_px), resample=Image.Resampling.LANCZOS)
#     buf = io.BytesIO()
#     img.save(buf, format="PNG", optimize=True)
#     return buf.getvalue()
    
# def try_log_to_sheets(values: Dict[str, float], aqi_val: int, aqi_label: str):
#     if gspread is None: return
#     if "gspread" not in st.secrets: return
#     try:
#         gc = gspread.service_account_from_dict(st.secrets["gspread"])
#         sheet = gc.open("Delhi AQI Predictions").sheet1
#         now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         values = normalize_values(values)
#         row = [
#             now,
#             float(values["PM2.5"]), float(values["PM10"]), float(values["NO2"]),
#             float(values["SO2"]), float(values["CO"]), float(values["Ozone"]),
#             int(aqi_val), str(aqi_label),
#         ]
#         sheet.append_row(row)
#         st.toast("Logged to Google Sheets.", icon="☁️")
#     except Exception as e:
#         st.info(f"Sheets logging skipped: {e}")
        
# def log_to_csv(values: Dict[str, float], aqi_val: int, aqi_label: str):
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     values = normalize_values(values)
#     row = {
#         "Timestamp": now,
#         "PM2.5": float(values["PM2.5"]),
#         "PM10": float(values["PM10"]),
#         "NO2": float(values["NO2"]),
#         "SO2": float(values["SO2"]),
#         "CO": float(values["CO"]),
#         "Ozone": float(values["Ozone"]),
#         "PredictedAQI": int(aqi_val),
#         "AQICategory": str(aqi_label),
#     }
#     df_row = pd.DataFrame([row])
#     if os.path.exists("aqi_logs.csv"):
#         try:
#             df_old = pd.read_csv("aqi_logs.csv")
#             df_new = pd.concat([df_old, df_row], ignore_index=True)
#         except Exception:
#             df_new = df_row
#     else:
#         df_new = df_row
#     df_new.to_csv("aqi_logs.csv", index=False)

# def values_table(values: Dict[str, float]) -> pd.DataFrame:
#     values = normalize_values(values)
#     return pd.DataFrame([[values[c] for c in COLUMNS]], columns=COLUMNS)

# def comparison_frame(values: Dict[str, float]) -> pd.DataFrame:
#     values = normalize_values(values)
#     rows = []
#     for p in COLUMNS:
#         rows.append({
#             "Pollutant": p,
#             "Your Level": float(values.get(p, 0.0)),
#             "Delhi Avg": float(DELHI_AVG[p]),
#             "WHO Limit": float(WHO_LIMITS[p]),
#         })
#     return pd.DataFrame(rows)

# # ──────────────────────────────
# # INIT & LOAD MODEL ONCE
# # ──────────────────────────────
# ensure_session_defaults()
# MODEL, ENCODER = load_model_and_encoder()

# # ──────────────────────────────
# # SIDEBAR NAVIGATION — SINGLE ROUTER
# # (Fixes: no duplicate imports, no stray pages outside conditions)
# # ──────────────────────────────
# # ✅ Ensure default session state
# options_list = [
#     "1) Understand + Share",
#     "2) Learn About AQI & Health Tips",
#     "3) Try a Sample AQI Scenario",
#     "4) Preset or Custom Inputs",
#     "5) Predict Delhi AQI Category",
#     "6) Compare with Delhi Avg & WHO",
# ]

# if "nav" not in st.session_state or st.session_state.nav not in options_list:
#     # default to the prediction page if you want to be at the end of the project,
#     # otherwise set to options_list[0]
#     st.session_state.nav = "5) Predict Delhi AQI Category"

# # default pollutant values (safe defaults so UI doesn't break)
# if "values" not in st.session_state:
#     st.session_state.values = {
#         "pm25": 80.0,
#         "pm10": 120.0,
#         "no2": 40.0,
#         "so2": 10.0,
#         "co": 1.0,
#         "ozone": 50.0,
#     }

# with st.sidebar:
#     st.image("https://img.icons8.com/?size=100&id=12448&format=png&color=000000", width=32)
#     st.markdown("### Delhi AQI App")
#     # safe index lookup (we already ensured st.session_state.nav is valid)
#     page = st.radio(
#         "Navigation",
#         options=options_list,
#         index=options_list.index(st.session_state.nav),
#         key="nav",
#     )
#     st.caption("Made with ❤️ for Delhi air quality. Follow the pages in order.")

# def ensure_session_defaults():
#     if "values" not in st.session_state:
#         st.session_state.values = {
#             "PM2.5": 40.0,
#             "PM10": 80.0,
#             "NO2": 25.0,
#             "SO2": 15.0,
#             "CO": 0.8,
#             "Ozone": 30.0,
#         }
#     if "last_prediction" not in st.session_state:
#         st.session_state.last_prediction = None
#     if "scenario_applied" not in st.session_state:
#         st.session_state.scenario_applied = False
#     if "last_present" not in st.session_state:
#         st.session_state.last_present = None

# # --- CALL HERE ---
# ensure_session_defaults()

# # --- Load model, encoders, and continue with pages ---
# MODEL, ENCODER = load_model_and_encoder()


# # ──────────────────────────────
# # INIT & LOAD MODEL ONCE
# # ──────────────────────────────
# ensure_session_defaults()  # ✅ now works, since defined above
# MODEL, ENCODER = load_model_and_encoder()

# # ──────────────────────────────
# # SIDEBAR NAVIGATION — SINGLE ROUTER
# # ──────────────────────────────
# # --- Ensure nav default exists ---
# if "nav" not in st.session_state:
#     st.session_state.nav = "1) Understand + Share"
    
# if page.startswith("1)"):
#     # Nice title + thin brand line
#     st.markdown(
#         "<h1 style='text-align:center; color:#2E86C1;'>🌍 Delhi AQI Prediction Dashboard</h1>",
#         unsafe_allow_html=True
#     )
#     st.markdown("<hr style='border:2px solid #2E86C1; margin-top:4px;'>", unsafe_allow_html=True)

#     # Two-column layout: Left = intro + features + pollutant cards | Right = QR
#     c1, c2 = st.columns([2, 1], vertical_alignment="top")

#     with c1:
#         st.subheader("✨ Welcome!")
#         # Intro + features line (your request: pollutants section comes right below this)
#         st.markdown(
#             """
#             <div style="font-size:18px; line-height:1.6; color:#444;">
#               This interactive dashboard helps you understand and predict
#               <b>Delhi's Air Quality Index (AQI)</b> 📊.
#               <br><br>
#               ✅ Real-time like predictions &nbsp;&nbsp; ✅ Pollutant-wise insights &nbsp;&nbsp; ✅ Health recommendations 🩺
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#         # 🔻 Right below features: Key Pollutants Tracked
#         st.markdown("### 🌟 Key Pollutants Tracked")

#         # Small helper for pretty colored cards
#         def _card(bg_hex: str, title_html: str, subtitle: str) -> str:
#             return (
#                 f"<div style='background:#{bg_hex}; padding:16px; border-radius:16px;"
#                 f"box-shadow:0 2px 10px rgba(0,0,0,.05); border:1px solid rgba(0,0,0,.04);'>"
#                 f"{title_html}<br><small style='color:#333;'>{subtitle}</small></div>"
#             )

#         # Row 1
#         r1c1, r1c2, r1c3 = st.columns(3)
#         with r1c1:
#             st.markdown(_card("FADBD8", "🌫️ <b>PM2.5</b>", "Fine particulate matter"), unsafe_allow_html=True)
#         with r1c2:
#             st.markdown(_card("D6EAF8", "🌪️ <b>PM10</b>", "Coarse particles"), unsafe_allow_html=True)
#         with r1c3:
#             st.markdown(_card("E8DAEF", "🌬️ <b>NO₂</b>", "Nitrogen dioxide"), unsafe_allow_html=True)

#         # Row 2
#         r2c1, r2c2, r2c3 = st.columns(3)
#         with r2c1:
#             st.markdown(_card("FCF3CF", "🔥 <b>SO₂</b>", "Sulfur dioxide"), unsafe_allow_html=True)
#         with r2c2:
#             st.markdown(_card("D5F5E3", "🟢 <b>CO</b>", "Carbon monoxide"), unsafe_allow_html=True)
#         with r2c3:
#             st.markdown(_card("FDEDEC", "☀️ <b>O₃</b>", "Ozone"), unsafe_allow_html=True)

#         st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

#     with c2:
#         # Your QR remains exactly as before
#         st.image(
#             make_qr_bytes(APP_URL),
#             caption="📱 Scan to open on mobile",
#             use_container_width=True
#         )

#     # Subtle footer hint
#     st.markdown("<hr style='margin:18px 0 6px 0;'>", unsafe_allow_html=True)
#     st.markdown(
#         "<div style='text-align:center; color:#666;'>Use the sidebar to navigate the app</div>",
#         unsafe_allow_html=True
#     )

# # ──────────────────────────────
# # 2) LEARN ABOUT AQI & HEALTH TIPS (Download)
# # ──────────────────────────────
# elif page.startswith("2)"):
#     st.title("📚 Learn About AQI & Health Tips")

#     st.markdown("""
#     **AQI Categories (India - simplified):**
#     - **Good (0–50):** Enjoy outdoor activities.
#     - **Satisfactory/Moderate (51–100):** Sensitive groups take care.
#     - **Moderate (101–200):** Reduce prolonged outdoor exertion.
#     - **Poor (201–300):** Consider masks; limit outdoor time.
#     - **Very Poor (301–400):** Avoid outdoor exertion; use purifiers.
#     - **Severe (401–500):** Stay indoors; seek medical advice for symptoms.

#     **General Health Tips:**
#     - Track AQI daily and plan outdoor tasks on lower-AQI hours.
#     - Use N95/FFP2 masks during poor days.
#     - Keep windows closed during peak pollution; ventilate when cleaner.
#     - Use HEPA purifiers indoors.
#     - Stay hydrated; saline/nasal rinse after heavy exposure.
#     """)





# # ──────────────────────────────
# # 3) TRY A SAMPLE AQI SCENARIO
# # ──────────────────────────────
# elif page.startswith("3)"):
#     st.title("🧪 Try a Sample AQI Scenario")

#     scenarios = {
#         "Winter Smog Morning": [280, 360, 95, 18, 2.2, 45],
#         "Post-Diwali Night":   [420, 520, 130, 30, 3.2, 70],
#         "Summer Breeze Day":   [55,  75,  22,  6,  0.6, 25],
#     }

#     chosen = st.selectbox("Select a scenario", list(scenarios.keys()))
#     vals = scenarios[chosen]
#     df_preview = pd.DataFrame([vals], columns=COLUMNS)
#     st.dataframe(df_preview, use_container_width=True)

#     if st.button("✅ Apply This Scenario", use_container_width=True):
#         st.session_state.values = {k: float(v) for k, v in zip(COLUMNS, vals)}
#         st.session_state.scenario_applied = chosen
#         st.success(f"Scenario applied: {chosen}. Go to page 4 to review/edit values.")

# # ──────────────────────────────
# # 4) PRESET OR CUSTOM INPUTS
# # ──────────────────────────────

# elif page.startswith("4)"):
#     st.title("🎛️ Present or Custom Inputs")

#     # Present selector
#     present = st.selectbox("Choose Present AQI Level", list(PRESENTS.keys()))

#     # If a new present is selected, reset scenario_applied
#     if "last_present" not in st.session_state or st.session_state.last_present != present:
#         st.session_state.scenario_applied = False
#         st.session_state.last_present = present

#     # Defaults from present
#     defaults = list(map(float, PRESENTS[present]))

#     # If scenario was applied earlier, override with those values
#     if st.session_state.get("scenario_applied", False):
#         defaults = [normalize_values(st.session_state.values)[c] for c in COLUMNS]

#     c1, c2 = st.columns(2)
#     with c1:
#         pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=float(defaults[0]))
#         no2  = st.number_input("NO2 (µg/m³)",   min_value=0.0, value=float(defaults[2]))
#         co   = st.number_input("CO (mg/m³)",    min_value=0.0, value=float(defaults[4]))
#     with c2:
#         pm10 = st.number_input("PM10 (µg/m³)",  min_value=0.0, value=float(defaults[1]))
#         so2  = st.number_input("SO2 (µg/m³)",   min_value=0.0, value=float(defaults[3]))
#         o3   = st.number_input("Ozone (µg/m³)", min_value=0.0, value=float(defaults[5]))

#     st.session_state.values = {
#     "PM2.5": pm25,
#     "PM10": pm10,
#     "NO2": no2,
#     "SO2": so2,
#     "CO": co,
#     "Ozone": o3,
# }

# # -------------------------
# # Page 5: Predict Delhi AQI Category (isolated)
# # -------------------------
# elif page.startswith("5)"):

#     st.title("🔮 Predict Delhi AQI Category (Random Forest)")

#     # --- Inputs for all 6 pollutants ---
#     pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=80.0, step=1.0)
#     pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=120.0, step=1.0)
#     no2  = st.number_input("NO2 (µg/m³)", min_value=0.0, value=40.0, step=1.0)
#     so2  = st.number_input("SO2 (µg/m³)", min_value=0.0, value=10.0, step=1.0)
#     co   = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0, step=0.1)
#     o3   = st.number_input("O3 (µg/m³)", min_value=0.0, value=50.0, step=1.0)

#     if st.button("🚀 Predict AQI"):

#         # Convert input into array
#         X_input = np.array([[pm25, pm10, no2, so2, co, o3]])

#         try:
#             # Load trained Random Forest model & label encoder
#             MODEL = joblib.load("aqi_rf_model.joblib")
#             ENCODER = joblib.load("label_encoder.joblib")

#             # Predict category
#             y_pred_class = MODEL.predict(X_input)[0]
#             y_pred_label = ENCODER.inverse_transform([y_pred_class])[0]

#             # Save predicted values & category to session_state
#             st.session_state["predicted_values"] = {
#                 "PM2.5": pm25,
#                 "PM10": pm10,
#                 "NO2": no2,
#                 "SO2": so2,
#                 "CO": co,
#                 "O3": o3
#             }
#             st.session_state["predicted_label"] = y_pred_label

#             st.success(f"Predicted AQI Category: **{y_pred_label}**")
#             st.write("✅ Values stored! Now go to **Page 6** for comparison with Delhi Avg & WHO.")

#         except Exception as e:
#             st.error(f"Prediction failed: {e}")



# #  ──────────────────────────────
# # 6) COMPARE WITH DELHI AVERAGES & WHO LIMITS
# #  ──────────────────────────────
# elif page.startswith("6)"):

#     import pandas as pd
#     import numpy as np
#     import altair as alt
#     import datetime

#     st.title("📊 Compare Predicted AQI with Delhi Avg & WHO Limits")

#     # 1) Load Page-5 prediction from session_state
#     predicted_values = st.session_state.get("predicted_values")
#     predicted_label = st.session_state.get("predicted_label")
#     prediction_time = st.session_state.get("prediction_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

#     if not predicted_values or not predicted_label:
#         st.warning("⚠️ No predicted values found. Please complete Page 5 (Predict AQI) first.")
#         st.stop()

#     st.success(f"**Predicted AQI Category (from Step 5): {predicted_label}**")
#     st.caption(f"Prediction made at: {prediction_time}")

#     # 2) Build comparison DataFrame
#     cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
#     DELHI_AVG = {"PM2.5": 95, "PM10": 180, "NO2": 60, "SO2": 20, "CO": 1.2, "O3": 50}
#     WHO_LIMITS = {"PM2.5": 25, "PM10": 50, "NO2": 40, "SO2": 20, "CO": 4, "O3": 100}

#     df_cmp = pd.DataFrame([
#         {
#             "Pollutant": p,
#             "Predicted Level": float(predicted_values.get(p, 0.0)),
#             "Delhi Avg": float(DELHI_AVG.get(p, np.nan)),
#             "WHO Limit": float(WHO_LIMITS.get(p, np.nan))
#         } for p in cols
#     ])

#     # 3) Display table
#     st.dataframe(df_cmp.set_index("Pollutant"), use_container_width=True)

#     # 4) Visualization using Altair
#     st.markdown("#### Visual Comparison: Predicted vs Delhi Avg vs WHO")
#     df_long = df_cmp.melt(id_vars="Pollutant", var_name="Metric", value_name="Level")

#     chart = alt.Chart(df_long).mark_bar().encode(
#         x=alt.X('Pollutant:N', title='Pollutant'),
#         y=alt.Y('Level:Q', title='Concentration'),
#         color=alt.Color('Metric:N', scale=alt.Scale(scheme='set1')),
#         tooltip=['Pollutant', 'Metric', 'Level']
#     ).properties(width=700, height=400)

#     st.altair_chart(chart, use_container_width=True)

#     # 5) Quick interpretation: check predicted vs WHO and Delhi Avg
#     st.markdown("### ⚠️ Quick Interpretation")
#     any_exceed = False
#     for _, row in df_cmp.iterrows():
#         p = row["Pollutant"]
#         pred = row["Predicted Level"]
#         delhi = row["Delhi Avg"]
#         who = row["WHO Limit"]

#         if not np.isnan(who) and pred > who:
#             st.error(f"**{p}** — Predicted level {pred} exceeds WHO limit {who}. Take precautions!")
#             any_exceed = True
#         elif not np.isnan(delhi) and pred > delhi:
#             st.warning(f"**{p}** — Predicted level {pred} is above Delhi average ({delhi}).")
#         else:
#             st.success(f"**{p}** — Predicted level {pred} is within Delhi Avg and WHO limit.")

#     if not any_exceed:
#         st.info("✅ None of the predicted pollutant levels exceed WHO limits.")

#     # 6) Download CSV
#     csv_bytes = df_cmp.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "⬇️ Download Predicted vs Delhi Avg & WHO (CSV)",
#         data=csv_bytes,
#         file_name="predicted_vs_delhi_who.csv",
#         mime="text/csv"
#     )



# ──────────────────────────────
# DELHI AQI PREDICTION – UPGRADED STREAMLIT APP
# ──────────────────────────────
import os
import io
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import qrcode
from qrcode.constants import ERROR_CORRECT_H
import streamlit as st
import joblib

# Optional imports
try:
    import shap
except Exception:
    shap = None

import plotly.express as px

# ──────────────────────────────
# APP CONFIG
# ──────────────────────────────
port = int(os.environ.get("PORT", 8501))
os.environ["STREAMLIT_SERVER_PORT"] = str(port)

st.set_page_config(
    page_title="Delhi AQI – Prediction & Insights",
    page_icon="🌫️",
    layout="wide"
)

# ──────────────────────────────
# STYLES
# ──────────────────────────────
st.markdown("""
<style>
.badge { padding: 0.35rem 0.7rem; border-radius: 999px; font-weight: 600; display: inline-block; }
.badge.good { background:#e7f5e9; color:#1e7e34; }
.badge.moderate { background:#fff3cd; color:#856404; }
.badge.poor { background:#ffe5d0; color:#a1490c; }
.badge.verypoor { background:#fde2e1; color:#9b1c1c; }
.badge.severe { background:#f8d7da; color:#721c24; }
.card { border-radius: 18px; padding: 16px; border: 1px solid #eee; background: white; box-shadow: 0 2px 12px rgba(0,0,0,0.04); height: 100%; }
.qr-box { text-align:center; }
.qr-title { font-weight:700; margin-bottom:0.3rem; }
small.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace; color:#666; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────
# CONSTANTS
# ──────────────────────────────
APP_URL = "https://pollutionappcreatedbyalok.streamlit.app/"

COLUMNS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

PRESENTS = {
    "Good":       [30,  40,  20,  5,  0.4, 10],
    "Moderate":   [90, 110,  40, 10,  1.2, 30],
    "Poor":       [200, 250, 90, 20,  2.0, 50],
    "Very Poor":  [300, 350, 120, 30,  3.5, 70],
    "Severe":     [400, 500, 150, 40,  4.5, 90],
}

DELHI_AVG = {
    "PM2.5": 95,
    "PM10": 180,
    "NO2": 60,
    "SO2": 20,
    "CO": 1.2,
    "O3": 50,
}

WHO_LIMITS = {
    "PM2.5": 25,
    "PM10": 50,
    "NO2": 40,
    "SO2": 20,
    "CO": 4,
    "O3": 100,
}

# ──────────────────────────────
# SESSION DEFAULTS
# ──────────────────────────────
def ensure_session_defaults():
    if "values" not in st.session_state:
        st.session_state.values = {k: float(v) for k, v in zip(COLUMNS, PRESENTS["Moderate"])}
    if "predicted_values" not in st.session_state:
        st.session_state.predicted_values = None
    if "predicted_label" not in st.session_state:
        st.session_state.predicted_label = None
    if "nav" not in st.session_state:
        st.session_state.nav = "1) Understand + Share"
    if "MODEL" not in st.session_state or "ENCODER" not in st.session_state:
        st.session_state.MODEL, st.session_state.ENCODER = load_model_and_encoder()
        
# ──────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────
def load_model_and_encoder():
    model, encoder = None, None
    try:
        if os.path.exists("aqi_rf_model.joblib"):
            model = joblib.load("aqi_rf_model.joblib")
        if os.path.exists("label_encoder.joblib"):
            encoder = joblib.load("label_encoder.joblib")
    except Exception:
        model, encoder = None, None
    return model, encoder

def badge_class(label: str) -> str:
    key = label.replace(" ", "").lower()
    if key in ["good"]: return "good"
    if key in ["moderate","satisfactory"]: return "moderate"
    if key == "poor": return "poor"
    if key in ["verypoor","verybad"]: return "verypoor"
    return "severe"

def make_qr_bytes(content: str, size_px: int = 160) -> bytes:
    qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_H, box_size=10, border=2)
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((size_px, size_px), resample=Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def predict_aqi(values: dict):
    MODEL = st.session_state.MODEL
    ENCODER = st.session_state.ENCODER
    X = np.array([[values[c] for c in COLUMNS]])
    try:
        aqi_val = MODEL.predict(X)[0]
        if ENCODER:
            y_class = MODEL.predict(X)
            aqi_label = ENCODER.inverse_transform(y_class)[0]
        else:
            # fallback based on AQI ranges
            aqi_label = "Good" if aqi_val <=50 else "Moderate" if aqi_val <=200 else "Poor" if aqi_val<=300 else "Very Poor" if aqi_val<=400 else "Severe"
        return round(aqi_val,2), aqi_label
    except:
        return None, None

def comparison_dataframe(values: dict):
    df = pd.DataFrame([
        {"Pollutant": p, "Predicted": values.get(p,0),
         "Delhi Avg": DELHI_AVG.get(p,0), "WHO": WHO_LIMITS.get(p,0)}
        for p in COLUMNS
    ])
    return df

# ──────────────────────────────
# ENSURE SESSION
# ──────────────────────────────
ensure_session_defaults()

# ──────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────
options_list = [
    "1) Understand + Share",
    "2) Learn About AQI & Health Tips",
    "3) Try a Sample AQI Scenario",
    "4) Preset or Custom Inputs",
    "5) Predict Delhi AQI Category",
    "6) Compare with Delhi Avg & WHO",
]
with st.sidebar:
    st.image(
        "https://img.icons8.com/?size=100&id=12448&format=png&color=000000", 
        width=32
    )
    st.markdown("### Delhi AQI App")

    # Ensure nav in session_state is valid
    current_nav = st.session_state.get("nav", options_list[0])
    if current_nav not in options_list:
        current_nav = options_list[0]

    # Radio for navigation
    page = st.radio(
        "Navigation",
        options_list,
        index=options_list.index(current_nav)
    )

    st.caption("Made with ❤️ for Delhi air quality. Follow pages in order.")

    # Update session state
    st.session_state.nav = page


# ──────────────────────────────
# PAGE 1: WELCOME + QR
# ──────────────────────────────
if page.startswith("1)"):
    st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🌍 Delhi AQI Prediction Dashboard</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("✨ Welcome!")
        st.markdown("""
        This interactive dashboard helps you understand and predict
        <b>Delhi's Air Quality Index (AQI)</b> 📊.<br>
        ✅ Real-time like predictions &nbsp; ✅ Pollutant-wise insights &nbsp; ✅ Health recommendations 🩺
        """, unsafe_allow_html=True)
        st.markdown("### 🌟 Key Pollutants Tracked")
        def _card(bg_hex, title, subtitle):
            return f"<div style='background:#{bg_hex}; padding:16px; border-radius:16px;box-shadow:0 2px 10px rgba(0,0,0,.05); border:1px solid rgba(0,0,0,.04);'>{title}<br><small style='color:#333;'>{subtitle}</small></div>"
        r1c1,r1c2,r1c3 = st.columns(3)
        r1c1.markdown(_card("FADBD8","🌫️ <b>PM2.5</b>","Fine particulate matter"), unsafe_allow_html=True)
        r1c2.markdown(_card("D6EAF8","🌪️ <b>PM10</b>","Coarse particles"), unsafe_allow_html=True)
        r1c3.markdown(_card("E8DAEF","🌬️ <b>NO₂</b>","Nitrogen dioxide"), unsafe_allow_html=True)
        r2c1,r2c2,r2c3 = st.columns(3)
        r2c1.markdown(_card("FCF3CF","🔥 <b>SO₂</b>","Sulfur dioxide"), unsafe_allow_html=True)
        r2c2.markdown(_card("D5F5E3","🟢 <b>CO</b>","Carbon monoxide"), unsafe_allow_html=True)
        r2c3.markdown(_card("FDEDEC","☀️ <b>O₃</b>","Ozone"), unsafe_allow_html=True)
    with c2:
        st.image(make_qr_bytes(APP_URL), caption="📱 Scan to open on mobile", use_container_width=True)

# ──────────────────────────────
# PAGE 2: AQI & HEALTH TIPS
# ──────────────────────────────
elif page.startswith("2)"):
    st.title("📚 Learn About AQI & Health Tips")
    st.markdown("""
    **AQI Categories (India - simplified):**
    - **Good (0–50):** Enjoy outdoor activities.
    - **Moderate (51–200):** Reduce prolonged outdoor exertion.
    - **Poor (201–300):** Consider masks; limit outdoor time.
    - **Very Poor (301–400):** Avoid outdoor exertion; use purifiers.
    - **Severe (401–500):** Stay indoors; seek medical advice.
    """)

# ──────────────────────────────
# PAGE 3: SAMPLE SCENARIO
# ──────────────────────────────
elif page.startswith("3)"):
    st.title("🧪 Try a Sample AQI Scenario")
    scenarios = {
        "Winter Smog Morning": [280, 360, 95, 18, 2.2, 45],
        "Post-Diwali Night":   [420, 520, 130, 30, 3.2, 70],
        "Summer Breeze Day":   [55, 75, 22, 6, 0.6, 25],
    }
    chosen = st.selectbox("Select a scenario", list(scenarios.keys()))
    df_preview = pd.DataFrame([scenarios[chosen]], columns=COLUMNS)
    st.dataframe(df_preview, use_container_width=True)
    if st.button("✅ Apply This Scenario"):
        st.session_state.values = {k: float(v) for k,v in zip(COLUMNS, scenarios[chosen])}
        st.success(f"Scenario applied: {chosen}. Go to Page 4 to review/edit values.")

# ──────────────────────────────
# PAGE 4: CUSTOM INPUTS
# ──────────────────────────────
elif page.startswith("4)"):
    st.title("🎛️ Present or Custom Inputs")
    present = st.selectbox("Choose Present AQI Level", list(PRESENTS.keys()))
    defaults = PRESENTS[present]
    sliders = []
    for i, col in enumerate(COLUMNS):
        val = st.slider(f"{col}", 0, 500, int(defaults[i]), step=1)
        sliders.append(val)
    st.session_state.values = {col: sliders[i] for i,col in enumerate(COLUMNS)}

# ──────────────────────────────
# PAGE 5: PREDICT AQI
# ──────────────────────────────
# ──────────────────────────────
# 5) Predict Delhi AQI Category
# ──────────────────────────────
elif page.startswith("5)"):

    st.title("🔮 Predict Delhi AQI Category (Random Forest)")

    # --- Inputs for all 6 pollutants ---
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=80.0, step=1.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=120.0, step=1.0)
    no2  = st.number_input("NO2 (µg/m³)", min_value=0.0, value=40.0, step=1.0)
    so2  = st.number_input("SO2 (µg/m³)", min_value=0.0, value=10.0, step=1.0)
    co   = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0, step=0.1)
    o3   = st.number_input("O3 (µg/m³)", min_value=0.0, value=50.0, step=1.0)

    if st.button("🚀 Predict AQI"):

        # Convert input into array
        X_input = np.array([[pm25, pm10, no2, so2, co, o3]])

        try:
            # Load trained Random Forest model & label encoder
            MODEL = joblib.load("aqi_rf_model.joblib")
            ENCODER = joblib.load("label_encoder.joblib")

            # Predict category
            y_pred_class = MODEL.predict(X_input)[0]
            y_pred_label = ENCODER.inverse_transform([y_pred_class])[0]

            # Save predicted values & category to session_state
            st.session_state["predicted_values"] = {
                "PM2.5": pm25,
                "PM10": pm10,
                "NO2": no2,
                "SO2": so2,
                "CO": co,
                "O3": o3
            }
            st.session_state["predicted_label"] = y_pred_label

            st.success(f"Predicted AQI Category: **{y_pred_label}**")
            st.write("✅ Values stored! Now go to **Page 6** for comparison with Delhi Avg & WHO.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ──────────────────────────────
# PAGE 6: COMPARE WITH DELHI AVG
# ──────────────────────────────
elif page.startswith("6)"):
    st.title("📊 Compare with Delhi Avg & WHO")
    df_cmp = comparison_dataframe(st.session_state.predicted_values or st.session_state.values)
    fig = px.bar(
        df_cmp.melt(id_vars="Pollutant", value_vars=["Predicted","Delhi Avg","WHO"]),
        x="Pollutant", y="value", color="variable", barmode="group",
        labels={"value":"Concentration (µg/m³ / ppm)", "variable":"Source"}
    )
    st.plotly_chart(fig, use_container_width=True)
    csv_bytes = df_cmp.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv_bytes, file_name="aqi_comparison.csv", mime="text/csv")

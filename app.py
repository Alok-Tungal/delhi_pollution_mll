import os
port = int(os.environ.get("PORT", 8501))
os.environ["STREAMLIT_SERVER_PORT"] = str(port)
import streamlit as st
from streamlit_option_menu import option_menu
import io  
import seaborn as sns
# Set page config 
st.set_page_config(page_title="🌫️ Delhi AQI Dashboard", layout="wide")
st.markdown("---")
st.markdown("### 🧠 Understand the Pollutants & Their Impact")

pollutant_info = {
    "PM2.5": {
        "emoji": "🌫️",
        "source": "Combustion engines, factories, stubble burning",
        "effect": "Can penetrate deep into lungs and enter bloodstream, causing heart and lung issues.",
    },
    "PM10": {
        "emoji": "🌪️",
        "source": "Dust, construction, roads",
        "effect": "Irritates nose, throat, and lungs. Can trigger asthma.",
    },
    "NO₂": {
        "emoji": "🛻",
        "source": "Vehicle emissions, industrial activities",
        "effect": "Aggravates respiratory diseases like asthma. Increases hospital visits.",
    },
    "SO₂": {
        "emoji": "🏭",
        "source": "Coal burning, thermal power plants",
        "effect": "Affects lungs, causes wheezing, shortness of breath.",
    },
    "CO": {
        "emoji": "🚗",
        "source": "Incomplete combustion in vehicles, stoves",
        "effect": "Reduces oxygen supply to body organs. Dangerous in enclosed areas.",
    },
    "Ozone": {
        "emoji": "☀️",
        "source": "Formed by sunlight reacting with pollutants (secondary pollutant)",
        "effect": "Causes chest pain, coughing, worsens bronchitis & asthma.",
    }
}

for pollutant, details in pollutant_info.items():
    st.markdown(f"""
**{details['emoji']} {pollutant}**
- **Source:** {details['source']}
- **Health Effect:** {details['effect']}
    """)
    
# ✅ STEP 7: AQI Knowledge Hub 🧠💨
with st.expander("📚 Learn About AQI & Health Tips"):
    st.markdown("### 💡 What Do These Pollutants Mean?")
    
    st.markdown("""
- **🟤 PM2.5 (Fine Particles):** Penetrates deep into lungs. Sources: dust, smoke.
- **🟠 PM10 (Coarse Particles):** Irritates eyes, nose, and throat.
- **🟣 NO₂ (Nitrogen Dioxide):** Increases asthma risk, especially in children.
- **🔵 SO₂ (Sulfur Dioxide):** Causes coughing, shortness of breath.
- **⚫ CO (Carbon Monoxide):** Reduces oxygen to brain; very dangerous at high levels.
- **🟢 Ozone (O₃):** Harmful at ground level — affects lung function.
""")

    st.markdown("### 📈 AQI Historical Meaning:")
    st.info("""
- AQI below **100** = Generally safe for most people.
- AQI above **200** = Can be dangerous for sensitive groups.
- AQI **above 300** = Public health emergency levels!
    """)

    st.markdown("### 🧘 Health Tips for High AQI Days:")
    st.success("""
- ✅ Stay indoors & use air purifiers
- ✅ Wear N95 masks outdoors
- ✅ Drink water to stay hydrated
- ✅ Avoid morning walks on high-pollution days
""")

    # ✅ Fixed Download Button (text string instead of StringIO)
    education_text = """
Air Quality & You 🌍

Pollutants Explained:
- PM2.5, PM10 → Lung irritants
- NO2, SO2 → Harmful to respiratory system
- CO → Oxygen blocker
- Ozone → Triggers asthma

Stay safe:
✔ Stay indoors on high AQI days
✔ Use masks, purifiers, and hydrate often

Made with ❤️ by Alok Tungal
    """
    st.download_button(
        label="📥 Download AQI Safety Guide",
        data=education_text,  # 🛠️ Send string instead of StringIO
        file_name="aqi_safety_guide.txt",
        mime="text/plain",
        key="download_guide_education"
    )




# Inject custom CSS for cleaner, modern look
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif; 
        }
        .main, .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 20px;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1F2937;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
# with st.sidebar:
#     selected = option_menu(
#         menu_title="🌫️ Delhi AQI App",
#         options=["Live AQI Dashboard", "Predict AQI", "AQI History", "Pollutant Info", "About"],
#         icons=["cloud-fog2", "graph-up", "bar-chart-line", "info-circle", "person-circle"],
#         menu_icon="cast",
#         default_index=0,
#     )

# # Placeholder Pages (will be filled in future steps)
# if selected == "AQI Dashboard":
#     st.title("📡Delhi AQI Dashboard")
#     st.info("We will integrate live AQI from OpenAQ API here.")

# elif selected == "Predict AQI":
#     st.title("🤖 Predict AQI Category")
#     st.warning("This will use your trained ML model with SHAP analysis.")

# elif selected == "AQI History":
#     st.title("📈 AQI History & Trends")
#     st.info("Time series line chart & heatmap coming soon.")

# elif selected == "Pollutant Info":
#     st.title("🧪 Pollutant Information")
#     st.success("Will display health impact & limits of PM2.5, NO2, etc.")

# elif selected == "About":
# #     st.title("ℹ️ About This App")
#     st.markdown("""
#     **Creator**: Alok Tungal  
#     **Purpose**: Predict and analyze Delhi's air quality using AI and real-time data.  
#     **Tech Used**: Python, Streamlit, scikit-learn, SHAP, OpenAQ API
#     """)




import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and encoder
model = joblib.load("aqi_rf_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Page title
st.title("🔮 **Predict Delhi AQI Category**")
st.markdown("Enter the pollutant levels below to predict the **Air Quality Index (AQI)** category.")

# Input form
with st.form("aqi_form"):
    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 120.0)
        no2 = st.number_input("NO₂ (µg/m³)", 0.0, 1000.0, 40.0)
        co = st.number_input("CO (mg/m³)", 0.0, 50.0, 1.2)
    with col2:
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 180.0)
        so2 = st.number_input("SO₂ (µg/m³)", 0.0, 1000.0, 10.0)
        ozone = st.number_input("Ozone (µg/m³)", 0.0, 1000.0, 20.0)

    submitted = st.form_submit_button("🔍 Predict AQI")

# 🧠 Predict
if st.button("🔮 Predict AQI Category"):
    input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # 🟨 AQI Emoji Map
    emoji_map = {
        "Good": "🟢",
        "Satisfactory": "🟡",
        "Moderate": "🟠",
        "Poor": "🔴",
        "Very Poor": "🟣",
        "Severe": "⚫️"
    }
    emoji = emoji_map.get(pred_label, "❓")

    # ✅ Beautiful Output - Light & Dark mode compatible
    st.success(f"📌 Predicted AQI Category: {emoji} **{pred_label}**")


    st.markdown("---")
    st.markdown("📊 **SHAP Explainability**")

    try:
        explainer = shap.Explainer(model, feature_names=["PM2.5", "PM10", "NO₂", "SO₂", "CO", "Ozone"])
        shap_values = explainer(input_data)

        if len(shap_values.values.shape) == 3:  # Multiclass
            class_index = pred_encoded
            class_shap = shap_values.values[0][class_index]

            fig1, ax1 = plt.subplots(figsize=(10, 4))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[class_index],
                class_shap,
                feature_names=explainer.feature_names,
                features=input_data[0]
            )
            st.pyplot(fig1)
            plt.clf()

        else:
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values.values[0],
                feature_names=explainer.feature_names,
                features=input_data[0]
            )
            st.pyplot(fig1)
            plt.clf()

    except Exception as e:
        st.warning(f"⚠️ SHAP explanation failed: {e}")


st.markdown("### 🧪 Try a Sample AQI Scenario")
selected_category = st.selectbox(
    "Pick Target AQI Category to Auto-Fill Inputs:",
    ["-- Select --", "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
)

preset_values = {
    "Good": [25.0, 40.0, 20.0, 5.0, 0.8, 10.0],
    "Satisfactory": [60.0, 70.0, 30.0, 8.0, 1.0, 15.0],
    "Moderate": [110.0, 150.0, 50.0, 15.0, 1.5, 25.0],
    "Poor": [180.0, 250.0, 80.0, 25.0, 2.0, 35.0],
    "Very Poor": [310.0, 400.0, 110.0, 40.0, 2.5, 60.0],
    "Severe": [420.0, 500.0, 150.0, 60.0, 3.0, 90.0]
}

# Set default values
default_values = preset_values.get(selected_category, [120.0, 180.0, 40.0, 10.0, 1.2, 20.0])


col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=default_values[0], key="pm25_input")
    no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=default_values[2], key="no2_input")
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=default_values[4], key="co_input")
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=default_values[1], key="pm10_input")
    so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=default_values[3], key="so2_input")
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=default_values[5], key="ozone_input")




st.markdown("#### 🔁 Choose a Preset AQI Level or Enter Custom Values")

preset_values = {
    "Good": [30, 40, 20, 5, 0.4, 10],
    "Moderate": [90, 110, 40, 10, 1.2, 30],
    "Poor": [200, 250, 90, 20, 2.0, 50],
    "Very Poor": [300, 350, 120, 30, 3.5, 70],
    "Severe": [400, 500, 150, 40, 4.5, 90],
}

selected_level = st.selectbox("Choose Preset AQI Level", list(preset_values.keys()))
default_values = preset_values[selected_level]
default_values = list(map(float, default_values))  # Fix type mismatch


col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=default_values[0])
    no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=default_values[2])
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=default_values[4])
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=default_values[1])
    so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=default_values[3])
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=default_values[5])


# 🌍 Show Pollution Summary (Step 3.2)
st.markdown("### 📋 Your Entered Pollution Levels:")
st.info(f"""
- **PM2.5:** {pm25} µg/m³
- **PM10:** {pm10} µg/m³
- **NO₂:** {no2} µg/m³
- **SO₂:** {so2} µg/m³
- **CO:** {co} mg/m³
- **Ozone:** {ozone} µg/m³
""")

# 🎯 Show PM-based Air Quality Advisory
if pm25 > 250 or pm10 > 300:
    st.warning("⚠️ High levels of PM detected. Stay indoors if possible.")
elif pm25 < 50 and pm10 < 50:
    st.success("✅ Air looks clean today! Great time for a walk.")


# step 4

# Step 4: Predict AQI
if st.button("🔮 Predict AQI Category", key="predict_aqi"):
    input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    color_map = {
        "Good": "🟢",
        "Satisfactory": "🟡",
        "Moderate": "🟠",
        "Poor": "🔴",
        "Very Poor": "🟣",
        "Severe": "⚫️"
    }
    emoji = color_map.get(pred_label, "❓")

    # ✅ Show Prediction Result
    st.markdown(f"### 📌 AQI Category: {emoji} **{pred_label}**")

    # ✅ Step 5: Health Tips & Recommendations
    st.markdown("---")
    st.markdown("🩺 **Health Impact & Recommendations:**")

    aqi_health_tips = {
        "Good": {
            "impact": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
            "tip": "Enjoy your day! It’s a great time for outdoor activities. 😊"
        },
        "Satisfactory": {
            "impact": "Air quality is acceptable. However, there may be a risk for some sensitive individuals.",
            "tip": "If you have asthma or allergies, keep medications handy. 🤧"
        },
        "Moderate": {
            "impact": "Air quality is okay for most, but may cause minor irritation to sensitive groups.",
            "tip": "Avoid intense outdoor activities. Hydrate well. 💧"
        },
        "Poor": {
            "impact": "Everyone may begin to experience health effects; sensitive individuals may experience serious effects.",
            "tip": "Limit outdoor exposure. Use a mask if necessary. 😷"
        },
        "Very Poor": {
            "impact": "Health warnings of emergency conditions. Serious effects on everyone's health.",
            "tip": "Avoid going out. Stay indoors with air filters. ❌🌫️"
        },
        "Severe": {
            "impact": "Serious health effects even for healthy people.",
            "tip": "Emergency! Remain indoors and avoid all physical exertion. 🚨"
        }
    }

    if pred_label in aqi_health_tips:
        info = aqi_health_tips[pred_label]
        st.error(f"**Impact:** {info['impact']}")
        st.info(f"**Tip:** {info['tip']}")
    else:
        st.warning("No health tips available for this AQI category.")


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("### 📊 Compare Your Pollution Levels with Delhi Averages and WHO Safe Limits")

# Reference data
historical_avg = {
    "PM2.5": 90,
    "PM10": 160,
    "NO₂": 35,
    "SO₂": 12,
    "CO": 1.0,
    "Ozone": 25
}

who_limits = {
    "PM2.5": 25,
    "PM10": 50,
    "NO₂": 40,
    "SO₂": 20,
    "CO": 4.0,
    "Ozone": 50
}

# User inputs
pollutants = ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "Ozone"]
your_values = [pm25, pm10, no2, so2, co, ozone]
delhi_avg = [historical_avg[p] for p in pollutants]
who_safe = [who_limits[p] for p in pollutants]

# Create DataFrame
df_compare = pd.DataFrame({
    "Pollutant": pollutants,
    "Your Input": your_values,
    "Delhi Avg": delhi_avg,
    "WHO Limit": who_safe
})

# Melt DataFrame for seaborn
df_melt = df_compare.melt(id_vars="Pollutant", var_name="Type", value_name="Value")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df_melt, x="Pollutant", y="Value", hue="Type", ax=ax)
plt.title("📉 Your Pollution Levels vs Delhi Avg vs WHO Safe Limits")
plt.ylabel("Concentration")
plt.xticks(rotation=0)
plt.grid(axis="y")

# Display in Streamlit
st.pyplot(fig)
plt.clf()


# step 6
# Step 6: Show Recent AQI Trend (Static Sample Data for Demo)

import pandas as pd
import random
st.markdown("---")
st.markdown("📈 **Recent AQI Trends (Simulated)**")

# Sample dummy data for past 7 days
trend_data = {
    "Date": pd.date_range(end=pd.Timestamp.today(), periods=7).strftime("%Y-%m-%d"),
    "AQI": [random.randint(80, 450) for _ in range(7)]
}
df_trend = pd.DataFrame(trend_data)

# Plot the AQI line chart
st.line_chart(df_trend.set_index("Date"), use_container_width=True)

# Add a mini table below
st.dataframe(df_trend.rename(columns={"Date": "📅 Date", "AQI": "🌫️ AQI Value"}), use_container_width=True)


import qrcode
from PIL import Image
import streamlit as st
from io import BytesIO
import urllib.parse
import os

input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
pred_encoded = model.predict(input_data)[0]
pred_label = label_encoder.inverse_transform([pred_encoded])[0]
paste_url = "https://alokdelhiairqualityml.streamlit.app/"

color_map = {
    "Good": "🟢",
    "Satisfactory": "🟡",
    "Moderate": "🟠",
    "Poor": "🔴",
    "Very Poor": "🟣",
    "Severe": "⚫️"
}
emoji = color_map.get(pred_label, "❓")

# ✅ Optional social media share
tweet_text = f"Delhi AQI today is {pred_label} {emoji}. Check pollution levels here: {paste_url} #AQI #AirQuality"
tweet_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(tweet_text)}"

st.markdown("### 📤 Share on Social Media")
st.markdown(f"[🐦 Tweet This Report]({tweet_url})", unsafe_allow_html=True)

# Generate QR Code with high box_size for clarity
qr = qrcode.QRCode(
    version=1,
    box_size=10,
    border=4
)
qr.add_data(paste_url)
qr.make(fit=True)

# Create and resize image for laptop viewing
img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
img = img.resize((300, 300), Image.LANCZOS)  # Clear and sharp

# Display QR code with updated Streamlit parameter
st.image(img, caption="📱 Scan to open the report", use_container_width=False)

# ✅ Show in Streamlit
st.markdown("### 📲 Share This AQI Summary via QR Code")
# st.image(qr_path, caption="🔗 Scan to open AQI Report", use_container_width=True)

# Optional: Download QR Code
buf = BytesIO()
img.save(buf, format="PNG")
byte_im = buf.getvalue()

st.download_button(
    label="📥 Download QR Code",
    data=byte_im,
    file_name="Delhi_AQI_QR_Code.png",
    mime="image/png"
)


inputs = {
    "PM2.5": pm25,
    "PM10": pm10,
    "NO2": no2,
    "SO2": so2,
    "CO": co,
    "Ozone": ozone
}

# 4. 🧠 Use preset label as AQI category (simulate ML prediction here)
aqi_category = selected_level  # You can replace this with your ML model's output if needed

# 5. 🚦 Risk Badge Generator
def get_risk_badge(aqi_category, inputs):
    main_pollutant = max(inputs, key=inputs.get)
    risk = {
        "Good": "LOW",
        "Satisfactory": "LOW",
        "Moderate": "MEDIUM",
        "Poor": "HIGH",
        "Very Poor": "HIGH",
        "Severe": "CRITICAL"
    }.get(aqi_category, "UNKNOWN")

    emoji = {
        "LOW": "🟢",
        "MEDIUM": "🟠",
        "HIGH": "🔴",
        "CRITICAL": "🚨"
    }.get(risk, "❓")

    return main_pollutant, risk, emoji

# 6. 🎯 Display Summary
main_pollutant, risk, emoji = get_risk_badge(aqi_category, inputs)

st.markdown("---")
st.markdown(f"""
### {emoji} Pollution Risk Summary
- **Risk Level:** `{risk}`
- **Main Pollutant:** `{main_pollutant}`
- **AQI Category:** `{aqi_category}`
""")
st.markdown("---")

import csv
import os
from datetime import datetime

def log_prediction(inputs, aqi_category, main_pollutant, risk):
    log_file = "aqi_logs.csv"
    file_exists = os.path.exists(log_file)

    try:
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "AQI Category", "Main Pollutant", "Risk Level"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                inputs["PM2.5"],
                inputs["PM10"],
                inputs["NO2"],
                inputs["SO2"],
                inputs["CO"],
                inputs["Ozone"],
                aqi_category,
                main_pollutant,
                risk
            ])
        
        st.success("✅ Prediction logged successfully to `aqi_logs.csv`.")
    except Exception as e:
        st.error(f"❌ Failed to log prediction: {e}")

log_prediction(inputs, aqi_category, main_pollutant, risk)



import pandas as pd
import os
import streamlit as st
from io import BytesIO

st.sidebar.markdown("### 🧠 Admin Dashboard")

# Load log file
if os.path.exists("aqi_logs.csv"):
    df_log = pd.read_csv("aqi_logs.csv")
    if df_log.shape[1] == 9:
        df_log.columns = ["Timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "AQI", "Category"]
    elif df_log.shape[1] == 10:
        df_log.columns = ["Timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "AQI", "Category", "Extra"]
    else:
        st.error("Unexpected number of columns in log file. Please check aqi_logs.csv")

    # Optional: Convert timestamp column to datetime
    df_log["Timestamp"] = pd.to_datetime(df_log["Timestamp"])

    # Filters
    with st.expander("🔍 Filter Logs"):
        selected_category = st.multiselect("Filter by AQI Category", options=df_log["Category"].unique())
        start_date = st.date_input("Start Date", value=df_log["Timestamp"].min().date())
        end_date = st.date_input("End Date", value=df_log["Timestamp"].max().date())

        # Apply filters
        filtered_df = df_log[
            (df_log["Timestamp"].dt.date >= start_date) &
            (df_log["Timestamp"].dt.date <= end_date)
        ]
        if selected_category:
            filtered_df = filtered_df[filtered_df["Category"].isin(selected_category)]

    # Show filtered data
    st.markdown("### 📋 Filtered AQI Logs")
    st.dataframe(filtered_df, use_container_width=True)

    # Download filtered data as Excel
    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='AQI Logs')
        processed_data = output.getvalue()
        return processed_data

    excel_data = convert_df_to_excel(filtered_df)

    st.download_button(
        label="📥 Download Logs as Excel",
        data=excel_data,
        file_name="filtered_aqi_logs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.warning("No AQI log file found yet.")




# # ──────────────────────────────
# # Hugging Face / Cloud friendly
# # ──────────────────────────────
# import os
# os.environ["XDG_CONFIG_HOME"] = os.path.join(os.getcwd(), ".streamlit")
# os.environ["STREAMLIT_METRICS_ENABLED"] = "false"
# os.environ["STREAMLIT_TELEMETRY_ENABLED"] = "false"

# # ──────────────────────────────
# # Imports
# # ──────────────────────────────
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import datetime
# import csv

# # Optional (uncomment if you use these sections)
# # import qrcode
# # import shap
# # import matplotlib.pyplot as plt

# # ──────────────────────────────
# # Page config
# # ──────────────────────────────
# st.set_page_config(
#     page_title="Delhi AQI – ML Dashboard",
#     page_icon="🌏",
#     layout="wide",
# )

# # ──────────────────────────────
# # Load model & encoder (robust)
# # ──────────────────────────────
# @st.cache_resource
# def load_model_and_encoder():
#     model_paths = ["aqi_rf_model.joblib", "model.joblib"]
#     enc_paths   = ["label_encoder_.joblib", "label_encoder.joblib"]

#     model, le = None, None
#     for p in model_paths:
#         if os.path.exists(p):
#             try:
#                 model = joblib.load(p)
#                 break
#             except Exception:
#                 pass
#     for p in enc_paths:
#         if os.path.exists(p):
#             try:
#                 le = joblib.load(p)
#                 break
#             except Exception:
#                 pass
#     return model, le

# model, label_encoder = load_model_and_encoder()

# # ──────────────────────────────
# # Helpers
# # ──────────────────────────────
# FEATURE_COLUMNS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]

# def predict_aqi(pm25, pm10, no2, so2, co, ozone):
#     """
#     Returns (predicted_aqi_label_int, aqi_category_str)
#     """
#     if model is None or label_encoder is None:
#         raise RuntimeError("Model or label encoder not loaded. Check files in repo.")

#     # Build DataFrame with exact training feature names
#     X = pd.DataFrame([[pm25, pm10, no2, so2, co, ozone]], columns=FEATURE_COLUMNS)

#     # If your trained model expects columns NO2/SO2 (no unicode), this matches already.
#     # If your model expects NO₂/SO₂ (unicode), map here:
#     # X.rename(columns={"NO2": "NO₂", "SO2": "SO₂"}, inplace=True)

#     pred_int = int(np.asarray(model.predict(X))[0])
#     category = label_encoder.inverse_transform([pred_int])[0]
#     return pred_int, str(category)

# def append_csv_row(row):
#     """
#     Appends to aqi_logs.csv (creates if missing) with a header on first write.
#     """
#     path = "aqi_logs.csv"
#     file_exists = os.path.exists(path)
#     with open(path, "a", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         if not file_exists:
#             w.writerow(["Timestamp"] + FEATURE_COLUMNS + ["PredictedAQI", "AQICategory"])
#         w.writerow(row)

# def log_to_google_sheets(row):
#     """
#     Append to Google Sheet if secrets are configured.
#     st.secrets["gspread"] must contain your service account dict.
#     Spreadsheet must exist and be shared with that service account email.
#     """
#     try:
#         import gspread
#         gc = gspread.service_account_from_dict(st.secrets["gspread"])
#         sh = gc.open("Delhi AQI Predictions")
#         ws = sh.sheet1
#         ws.append_row(row)
#         return True, None
#     except Exception as e:
#         return False, str(e)

# def load_logs_df():
#     if os.path.exists("aqi_logs.csv"):
#         try:
#             df = pd.read_csv("aqi_logs.csv")
#             # Ensure consistent types
#             if "Timestamp" in df.columns:
#                 df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce") 
#             return df
#         except Exception:
#             pass
#     return pd.DataFrame(columns=["Timestamp"] + FEATURE_COLUMNS + ["PredictedAQI", "AQICategory"])

# # ──────────────────────────────
# # Sidebar Navigation
# # ──────────────────────────────
# st.sidebar.title("🌏 Delhi AQI Dashboard")
# page = st.sidebar.radio(
#     "Go to",
#     [
#         "🔮 Prediction",
#         "📊 Recent AQI Trends",
#         "📈 Compare with Delhi & WHO Limits",
#         "🧪 Try a Sample Scenario",
#         "🛠 Admin",
#         "ℹ️ Information",
#     ],
#     index=0,
# )

# # ──────────────────────────────
# # PREDICTION PAGE
# # ──────────────────────────────
# if page == "🔮 Prediction":
#     st.title("🔮 AQI Prediction")
#     st.caption("Enter pollutant levels to predict AQI category.")

#     col1, col2, col3 = st.columns(3)
#     pm25 = col1.number_input("PM2.5 (µg/m³)", min_value=0.0, value=80.0, step=1.0)
#     pm10 = col2.number_input("PM10 (µg/m³)", min_value=0.0, value=120.0, step=1.0)
#     no2  = col3.number_input("NO2 (µg/m³)",  min_value=0.0, value=40.0,  step=1.0)

#     col4, col5, col6 = st.columns(3)
#     so2  = col4.number_input("SO2 (µg/m³)",  min_value=0.0, value=10.0,  step=1.0)
#     co   = col5.number_input("CO (mg/m³)",   min_value=0.0, value=1.0,   step=0.1)
#     ozone= col6.number_input("Ozone (µg/m³)",min_value=0.0, value=50.0,  step=1.0)

#     if st.button("🚀 Predict", use_container_width=True):
#         try:
#             predicted_aqi, aqi_category = predict_aqi(pm25, pm10, no2, so2, co, ozone)
#             st.success(f"**Predicted AQI Category:** {aqi_category}")

#             # ⬇️ OPTIONAL: add your SHAP/feature-importance visualizations here
#             # e.g., shap.force_plot / shap.summary_plot

#             # Prepare row for logs
#             now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             row = [
#                 now,
#                 float(pm25), float(pm10), float(no2), float(so2), float(co), float(ozone),
#                 int(predicted_aqi), str(aqi_category),
#             ]

#             # CSV log (local)
#             try:
#                 append_csv_row(row)
#                 st.toast("✅ Logged to CSV (aqi_logs.csv)", icon="✅")
#             except Exception as e_csv:
#                 st.warning(f"CSV log failed: {e_csv}")

#             # Google Sheets log (cloud)
#             gs_ok, gs_err = log_to_google_sheets(row)
#             if gs_ok:
#                 st.toast("✅ Logged to Google Sheets", icon="✅")
#             else:
#                 st.info(f"Google Sheets skipped/failed: {gs_err}")

#             # ⬇️ OPTIONAL: add your QR code/report download here
#             # e.g., generate a QR to the app URL or a sharable result
#             # qr = qrcode.make("https://your-app-url")
#             # st.image(qr, caption="Scan to share", use_container_width=True)

#         except Exception as e:
#             st.error(f"Prediction failed: {e}")

# # ──────────────────────────────
# # RECENT AQI TRENDS
# # ──────────────────────────────
# elif page == "📊 Recent AQI Trends":
#     st.title("📊 Recent AQI Trends")
#     st.caption("Visualize recent predictions from your logs. If no data is found, simulated values will be shown.")

#     # Load logs
#     df_logs = load_logs_df()

#     required_cols = {"Timestamp", "PredictedAQI"}
#     if df_logs is not None and required_cols.issubset(df_logs.columns) and len(df_logs) >= 2:
#         st.subheader("📈 Recent Prediction Trends (From Logs)")

#         # Convert Timestamp to datetime if needed
#         df_logs["Timestamp"] = pd.to_datetime(df_logs["Timestamp"], errors="coerce")

#         # Drop any rows with invalid timestamps
#         df_logs = df_logs.dropna(subset=["Timestamp"])

#         # Plot AQI trend
#         st.line_chart(
#             df_logs.set_index("Timestamp")["PredictedAQI"],
#             use_container_width=True
#         )

#         # Show last 20 records with gradient formatting
#         st.write("📋 **Last 20 Predictions**")
#         styled_df = df_logs.tail(20).style.background_gradient(
#             subset=["PredictedAQI"], cmap="RdYlGn_r"
#         )
#         st.dataframe(styled_df, use_container_width=True)

#     else:
#         # Simulated fallback
#         st.warning("⚠️ No valid logs found. Showing simulated AQI trend for demo purposes.")

#         dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
#         sim = pd.DataFrame({
#             "Timestamp": dates,
#             "PredictedAQI": np.random.randint(50, 300, size=30)
#         })

#         st.subheader("📈 Simulated AQI Trend")
#         st.line_chart(sim.set_index("Timestamp"), use_container_width=True)


# # ──────────────────────────────
# # COMPARISON
# # ──────────────────────────────
# elif page == "📈 Compare with Delhi & WHO Limits":
#     st.title("📈 Compare Your Pollution Levels")
#     st.caption("Compare your entered values with Delhi averages & WHO recommended limits.")

#     # You can also pull last-entered values from session_state if desired
#     col1, col2, col3 = st.columns(3)
#     pm25 = col1.number_input("Your PM2.5", min_value=0.0, value=80.0, step=1.0, key="cmp_pm25")
#     pm10 = col2.number_input("Your PM10",  min_value=0.0, value=120.0, step=1.0, key="cmp_pm10")
#     no2  = col3.number_input("Your NO2",   min_value=0.0, value=40.0,  step=1.0, key="cmp_no2")

#     col4, col5, col6 = st.columns(3)
#     so2  = col4.number_input("Your SO2",   min_value=0.0, value=10.0,  step=1.0, key="cmp_so2")
#     co   = col5.number_input("Your CO",    min_value=0.0, value=1.0,   step=0.1, key="cmp_co")
#     ozone= col6.number_input("Your Ozone", min_value=0.0, value=50.0,  step=1.0, key="cmp_o3")

#     who_limits = {"PM2.5": 25, "PM10": 50, "NO2": 40, "SO2": 20, "CO": 4, "Ozone": 100}
#     delhi_avg  = {"PM2.5": 95, "PM10": 180, "NO2": 60, "SO2": 25, "CO": 1.2, "Ozone": 80}
#     yours      = {"PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, "CO": co, "Ozone": ozone}

#     df_cmp = pd.DataFrame([yours, delhi_avg, who_limits], index=["Your Input", "Delhi Avg", "WHO Limit"])
#     st.bar_chart(df_cmp.T, use_container_width=True)
#     st.dataframe(df_cmp.T, use_container_width=True)

# # ──────────────────────────────
# # SCENARIOS
# # ──────────────────────────────
# elif page == "🧪 Try a Sample Scenario":
#     st.title("🧪 AQI Sample Scenarios")
#     st.caption("Pick a preset level or enter custom values.")

#     preset_values = {
#         "Good":       [30, 40, 20, 5, 0.4, 10],
#         "Moderate":   [90, 110, 40, 10, 1.2, 30],
#         "Poor":       [200, 250, 90, 20, 2.0, 50],
#         "Very Poor":  [300, 350, 120, 30, 3.5, 70],
#         "Severe":     [400, 500, 150, 40, 4.5, 90],
#     }

#     level = st.selectbox("Choose Preset AQI Level", list(preset_values.keys()))
#     vals = list(map(float, preset_values[level]))

#     col1, col2 = st.columns(2)
#     with col1:
#         pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=vals[0], key="sc_pm25")
#         no2  = st.number_input("NO2 (µg/m³)",   min_value=0.0, value=vals[2], key="sc_no2")
#         co   = st.number_input("CO (mg/m³)",    min_value=0.0, value=vals[4], key="sc_co")
#     with col2:
#         pm10 = st.number_input("PM10 (µg/m³)",  min_value=0.0, value=vals[1], key="sc_pm10")
#         so2  = st.number_input("SO2 (µg/m³)",   min_value=0.0, value=vals[3], key="sc_so2")
#         ozone= st.number_input("Ozone (µg/m³)", min_value=0.0, value=vals[5], key="sc_o3")

#     if st.button("🎯 Predict Scenario", use_container_width=True):
#         try:
#             predicted_aqi, aqi_category = predict_aqi(pm25, pm10, no2, so2, co, ozone)
#             st.info(f"Scenario **{level}** → Predicted AQI Category: **{aqi_category}**")

#             # Log scenario (optional)
#             now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             row = [now, float(pm25), float(pm10), float(no2), float(so2), float(co), float(ozone), int(predicted_aqi), str(aqi_category)]
#             try:
#                 append_csv_row(row)
#                 st.toast("✅ Logged to CSV", icon="✅")
#             except Exception as e_csv:
#                 st.warning(f"CSV log failed: {e_csv}")
#         except Exception as e:
#             st.error(f"Prediction failed: {e}")

# # ──────────────────────────────
# # ADMIN
# # ──────────────────────────────
# elif page == "🛠 Admin":
#     st.title("🛠 Admin Dashboard")
#     st.caption("View logs, filter, and download. (Add box plots/heatmaps here if you like.)")

#     df = load_logs_df()
#     if df.empty:
#         st.warning("No logs yet.")
#     else:
#         colf1, colf2 = st.columns(2)
#         with colf1:
#             start = st.date_input("Start date", value=df["Timestamp"].min().date() if df["Timestamp"].notna().any() else datetime.date.today())
#         with colf2:
#             end   = st.date_input("End date", value=df["Timestamp"].max().date() if df["Timestamp"].notna().any() else datetime.date.today())

#         # Filter
#         if df["Timestamp"].notna().any():
#             mask = (df["Timestamp"].dt.date >= start) & (df["Timestamp"].dt.date <= end)
#             df_f = df.loc[mask].copy()
#         else:
#             df_f = df.copy()

#         st.subheader("📄 Filtered Logs")
#         st.dataframe(df_f, use_container_width=True)

#         # Download as Excel
#         from io import BytesIO
#         fname = f"aqi_logs_{start}_to_{end}.xlsx"
#         buffer = BytesIO()
#         with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
#             df_f.to_excel(writer, index=False, sheet_name="Logs")
#         st.download_button("⬇️ Download Excel", data=buffer.getvalue(), file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

#         # OPTIONAL: Box plots (matplotlib) — uncomment if you want
#         # import matplotlib.pyplot as plt
#         # st.subheader("📦 Pollutant Distributions (Box Plots)")
#         # keep_cols = [c for c in FEATURE_COLUMNS if c in df_f.columns]
#         # if keep_cols:
#         #     fig, ax = plt.subplots(figsize=(10, 5))
#         #     ax.boxplot([df_f[c].dropna() for c in keep_cols], labels=keep_cols, showmeans=True)
#         #     ax.set_ylabel("Concentration")
#         #     st.pyplot(fig, use_container_width=True)

# # ──────────────────────────────
# # INFO
# # ──────────────────────────────
# elif page == "ℹ️ Information":
#     st.title("ℹ️ Project Information")
#     st.markdown(
#         """
# **Model**: Random Forest classifier  
# **Features**: PM2.5, PM10, NO2, SO2, CO, Ozone  
# **Capabilities**: Prediction, scenarios, comparison, trends, CSV & Google Sheets logging, admin tools  
# **Notes**:
# - Ensure `aqi_rf_model.joblib` and `label_encoder_.joblib` are present.
# - For Google Sheets, add your service account JSON **as `st.secrets["gspread"]`** in the cloud and share the sheet with that service account email.
#         """
#     )

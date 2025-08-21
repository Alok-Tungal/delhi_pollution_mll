import os
port = int(os.environ.get("PORT", 8501))
os.environ["STREAMLIT_SERVER_PORT"] = str(port)
# import streamlit as st
# from streamlit_option_menu import option_menu 
# import io  
# import seaborn as sns
# # Set page config 
# st.set_page_config(page_title="🌫️ Delhi AQI Dashboard", layout="wide") 
# st.markdown("---")   
# st.markdown("### 🧠 Understand the Pollutants & Their Impact")

# pollutant_info = {
#     "PM2.5": {
#         "emoji": "🌫️",
#         "source": "Combustion engines, factories, stubble burning",
#         "effect": "Can penetrate deep into lungs and enter bloodstream, causing heart and lung issues.",
#     },
#     "PM10": {
#         "emoji": "🌪️",
#         "source": "Dust, construction, roads",
#         "effect": "Irritates nose, throat, and lungs. Can trigger asthma.",
#     },
#     "NO₂": {
#         "emoji": "🛻",
#         "source": "Vehicle emissions, industrial activities",
#         "effect": "Aggravates respiratory diseases like asthma. Increases hospital visits.",
#     },
#     "SO₂": {
#         "emoji": "🏭",
#         "source": "Coal burning, thermal power plants",
#         "effect": "Affects lungs, causes wheezing, shortness of breath.",
#     },
#     "CO": {
#         "emoji": "🚗",
#         "source": "Incomplete combustion in vehicles, stoves",
#         "effect": "Reduces oxygen supply to body organs. Dangerous in enclosed areas.",
#     },
#     "Ozone": {
#         "emoji": "☀️",
#         "source": "Formed by sunlight reacting with pollutants (secondary pollutant)",
#         "effect": "Causes chest pain, coughing, worsens bronchitis & asthma.",
#     }
# }

# for pollutant, details in pollutant_info.items():
#     st.markdown(f"""
# **{details['emoji']} {pollutant}**
# - **Source:** {details['source']}
# - **Health Effect:** {details['effect']}
#     """)
    
# # ✅ STEP 7: AQI Knowledge Hub 🧠💨
# with st.expander("📚 Learn About AQI & Health Tips"):
#     st.markdown("### 💡 What Do These Pollutants Mean?")
    
#     st.markdown("""
# - **🟤 PM2.5 (Fine Particles):** Penetrates deep into lungs. Sources: dust, smoke.
# - **🟠 PM10 (Coarse Particles):** Irritates eyes, nose, and throat.
# - **🟣 NO₂ (Nitrogen Dioxide):** Increases asthma risk, especially in children.
# - **🔵 SO₂ (Sulfur Dioxide):** Causes coughing, shortness of breath.
# - **⚫ CO (Carbon Monoxide):** Reduces oxygen to brain; very dangerous at high levels.
# - **🟢 Ozone (O₃):** Harmful at ground level — affects lung function.
# """)

#     st.markdown("### 📈 AQI Historical Meaning:")
#     st.info("""
# - AQI below **100** = Generally safe for most people.
# - AQI above **200** = Can be dangerous for sensitive groups.
# - AQI **above 300** = Public health emergency levels!
#     """)

#     st.markdown("### 🧘 Health Tips for High AQI Days:")
#     st.success("""
# - ✅ Stay indoors & use air purifiers
# - ✅ Wear N95 masks outdoors
# - ✅ Drink water to stay hydrated
# - ✅ Avoid morning walks on high-pollution days
# """)

#     # ✅ Fixed Download Button (text string instead of StringIO)
#     education_text = """
# Air Quality & You 🌍

# Pollutants Explained:
# - PM2.5, PM10 → Lung irritants
# - NO2, SO2 → Harmful to respiratory system
# - CO → Oxygen blocker
# - Ozone → Triggers asthma

# Stay safe:
# ✔ Stay indoors on high AQI days
# ✔ Use masks, purifiers, and hydrate often

# Made with ❤️ by Alok Tungal
#     """
#     st.download_button(
#         label="📥 Download AQI Safety Guide",
#         data=education_text,  # 🛠️ Send string instead of StringIO
#         file_name="aqi_safety_guide.txt",
#         mime="text/plain",
#         key="download_guide_education"
#     )




# # Inject custom CSS for cleaner, modern look
# st.markdown("""
#     <style>
#         body {
#             background-color: #f9f9f9;
#             font-family: 'Segoe UI', sans-serif; 
#         }
#         .main, .block-container {
#             padding-top: 1rem;
#             padding-bottom: 1rem;
#         }
#         .stButton>button {
#             background-color: #4CAF50;
#             color: white;
#             border: none;
#             border-radius: 8px;
#             font-size: 16px;
#             padding: 8px 20px;
#         }
#         .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
#             color: #1F2937;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Sidebar Navigation
# # with st.sidebar:
# #     selected = option_menu(
# #         menu_title="🌫️ Delhi AQI App",
# #         options=["Live AQI Dashboard", "Predict AQI", "AQI History", "Pollutant Info", "About"],
# #         icons=["cloud-fog2", "graph-up", "bar-chart-line", "info-circle", "person-circle"],
# #         menu_icon="cast",
# #         default_index=0,
# #     )

# # # Placeholder Pages (will be filled in future steps)
# # if selected == "AQI Dashboard":
# #     st.title("📡Delhi AQI Dashboard")
# #     st.info("We will integrate live AQI from OpenAQ API here.")

# # elif selected == "Predict AQI":
# #     st.title("🤖 Predict AQI Category")
# #     st.warning("This will use your trained ML model with SHAP analysis.")

# # elif selected == "AQI History":
# #     st.title("📈 AQI History & Trends")
# #     st.info("Time series line chart & heatmap coming soon.")

# # elif selected == "Pollutant Info":
# #     st.title("🧪 Pollutant Information")
# #     st.success("Will display health impact & limits of PM2.5, NO2, etc.")

# # elif selected == "About":
# # #     st.title("ℹ️ About This App")
# #     st.markdown("""
# #     **Creator**: Alok Tungal  
# #     **Purpose**: Predict and analyze Delhi's air quality using AI and real-time data.  
# #     **Tech Used**: Python, Streamlit, scikit-learn, SHAP, OpenAQ API
# #     """)




# import streamlit as st
# import numpy as np
# import joblib
# import shap
# import matplotlib.pyplot as plt

# # Load model and encoder
# model = joblib.load("aqi_rf_model.joblib")
# label_encoder = joblib.load("label_encoder.joblib")

# # Page title
# st.title("🔮 **Predict Delhi AQI Category**")
# st.markdown("Enter the pollutant levels below to predict the **Air Quality Index (AQI)** category.")

# # Input form
# with st.form("aqi_form"):
#     col1, col2 = st.columns(2)
#     with col1:
#         pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 120.0)
#         no2 = st.number_input("NO₂ (µg/m³)", 0.0, 1000.0, 40.0)
#         co = st.number_input("CO (mg/m³)", 0.0, 50.0, 1.2)
#     with col2:
#         pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 180.0)
#         so2 = st.number_input("SO₂ (µg/m³)", 0.0, 1000.0, 10.0)
#         ozone = st.number_input("Ozone (µg/m³)", 0.0, 1000.0, 20.0)

#     submitted = st.form_submit_button("🔍 Predict AQI")

# # 🧠 Predict
# if st.button("🔮 Predict AQI Category"):
#     input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
#     pred_encoded = model.predict(input_data)[0]
#     pred_label = label_encoder.inverse_transform([pred_encoded])[0]

#     # 🟨 AQI Emoji Map
#     emoji_map = {
#         "Good": "🟢",
#         "Satisfactory": "🟡",
#         "Moderate": "🟠",
#         "Poor": "🔴",
#         "Very Poor": "🟣",
#         "Severe": "⚫️"
#     }
#     emoji = emoji_map.get(pred_label, "❓")

#     # ✅ Beautiful Output - Light & Dark mode compatible
#     st.success(f"📌 Predicted AQI Category: {emoji} **{pred_label}**")


#     st.markdown("---")
#     st.markdown("📊 **SHAP Explainability**")

#     try:
#         explainer = shap.Explainer(model, feature_names=["PM2.5", "PM10", "NO₂", "SO₂", "CO", "Ozone"])
#         shap_values = explainer(input_data)

#         if len(shap_values.values.shape) == 3:  # Multiclass
#             class_index = pred_encoded
#             class_shap = shap_values.values[0][class_index]

#             fig1, ax1 = plt.subplots(figsize=(10, 4))
#             shap.plots._waterfall.waterfall_legacy(
#                 explainer.expected_value[class_index],
#                 class_shap,
#                 feature_names=explainer.feature_names,
#                 features=input_data[0]
#             )
#             st.pyplot(fig1)
#             plt.clf()

#         else:
#             fig1, ax1 = plt.subplots(figsize=(10, 4))
#             shap.plots._waterfall.waterfall_legacy(
#                 explainer.expected_value,
#                 shap_values.values[0],
#                 feature_names=explainer.feature_names,
#                 features=input_data[0]
#             )
#             st.pyplot(fig1)
#             plt.clf()

#     except Exception as e:
#         st.warning(f"⚠️ SHAP explanation failed: {e}")


# st.markdown("### 🧪 Try a Sample AQI Scenario")
# selected_category = st.selectbox(
#     "Pick Target AQI Category to Auto-Fill Inputs:",
#     ["-- Select --", "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
# )

# preset_values = {
#     "Good": [25.0, 40.0, 20.0, 5.0, 0.8, 10.0],
#     "Satisfactory": [60.0, 70.0, 30.0, 8.0, 1.0, 15.0],
#     "Moderate": [110.0, 150.0, 50.0, 15.0, 1.5, 25.0],
#     "Poor": [180.0, 250.0, 80.0, 25.0, 2.0, 35.0],
#     "Very Poor": [310.0, 400.0, 110.0, 40.0, 2.5, 60.0],
#     "Severe": [420.0, 500.0, 150.0, 60.0, 3.0, 90.0]
# }

# # Set default values
# default_values = preset_values.get(selected_category, [120.0, 180.0, 40.0, 10.0, 1.2, 20.0])


# col1, col2 = st.columns(2)
# with col1:
#     pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=default_values[0], key="pm25_input")
#     no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=default_values[2], key="no2_input")
#     co = st.number_input("CO (mg/m³)", min_value=0.0, value=default_values[4], key="co_input")
# with col2:
#     pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=default_values[1], key="pm10_input")
#     so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=default_values[3], key="so2_input")
#     ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=default_values[5], key="ozone_input")




# st.markdown("#### 🔁 Choose a Preset AQI Level or Enter Custom Values")

# preset_values = {
#     "Good": [30, 40, 20, 5, 0.4, 10],
#     "Moderate": [90, 110, 40, 10, 1.2, 30],
#     "Poor": [200, 250, 90, 20, 2.0, 50],
#     "Very Poor": [300, 350, 120, 30, 3.5, 70],
#     "Severe": [400, 500, 150, 40, 4.5, 90],
# }

# selected_level = st.selectbox("Choose Preset AQI Level", list(preset_values.keys()))
# default_values = preset_values[selected_level]
# default_values = list(map(float, default_values))  # Fix type mismatch


# col1, col2 = st.columns(2)
# with col1:
#     pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=default_values[0])
#     no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=default_values[2])
#     co = st.number_input("CO (mg/m³)", min_value=0.0, value=default_values[4])
# with col2:
#     pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=default_values[1])
#     so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=default_values[3])
#     ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=default_values[5])


# # 🌍 Show Pollution Summary (Step 3.2)
# st.markdown("### 📋 Your Entered Pollution Levels:")
# st.info(f"""
# - **PM2.5:** {pm25} µg/m³
# - **PM10:** {pm10} µg/m³
# - **NO₂:** {no2} µg/m³
# - **SO₂:** {so2} µg/m³
# - **CO:** {co} mg/m³
# - **Ozone:** {ozone} µg/m³
# """)

# # 🎯 Show PM-based Air Quality Advisory
# if pm25 > 250 or pm10 > 300:
#     st.warning("⚠️ High levels of PM detected. Stay indoors if possible.")
# elif pm25 < 50 and pm10 < 50:
#     st.success("✅ Air looks clean today! Great time for a walk.")


# # step 4

# # Step 4: Predict AQI
# if st.button("🔮 Predict AQI Category", key="predict_aqi"):
#     input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
#     pred_encoded = model.predict(input_data)[0]
#     pred_label = label_encoder.inverse_transform([pred_encoded])[0]

#     color_map = {
#         "Good": "🟢",
#         "Satisfactory": "🟡",
#         "Moderate": "🟠",
#         "Poor": "🔴",
#         "Very Poor": "🟣",
#         "Severe": "⚫️"
#     }
#     emoji = color_map.get(pred_label, "❓")

#     # ✅ Show Prediction Result
#     st.markdown(f"### 📌 AQI Category: {emoji} **{pred_label}**")

#     # ✅ Step 5: Health Tips & Recommendations
#     st.markdown("---")
#     st.markdown("🩺 **Health Impact & Recommendations:**")

#     aqi_health_tips = {
#         "Good": {
#             "impact": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
#             "tip": "Enjoy your day! It’s a great time for outdoor activities. 😊"
#         },
#         "Satisfactory": {
#             "impact": "Air quality is acceptable. However, there may be a risk for some sensitive individuals.",
#             "tip": "If you have asthma or allergies, keep medications handy. 🤧"
#         },
#         "Moderate": {
#             "impact": "Air quality is okay for most, but may cause minor irritation to sensitive groups.",
#             "tip": "Avoid intense outdoor activities. Hydrate well. 💧"
#         },
#         "Poor": {
#             "impact": "Everyone may begin to experience health effects; sensitive individuals may experience serious effects.",
#             "tip": "Limit outdoor exposure. Use a mask if necessary. 😷"
#         },
#         "Very Poor": {
#             "impact": "Health warnings of emergency conditions. Serious effects on everyone's health.",
#             "tip": "Avoid going out. Stay indoors with air filters. ❌🌫️"
#         },
#         "Severe": {
#             "impact": "Serious health effects even for healthy people.",
#             "tip": "Emergency! Remain indoors and avoid all physical exertion. 🚨"
#         }
#     }

#     if pred_label in aqi_health_tips:
#         info = aqi_health_tips[pred_label]
#         st.error(f"**Impact:** {info['impact']}")
#         st.info(f"**Tip:** {info['tip']}")
#     else:
#         st.warning("No health tips available for this AQI category.")


# import pandas as pd
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# st.markdown("### 📊 Compare Your Pollution Levels with Delhi Averages and WHO Safe Limits")

# # Reference data
# historical_avg = {
#     "PM2.5": 90,
#     "PM10": 160,
#     "NO₂": 35,
#     "SO₂": 12,
#     "CO": 1.0,
#     "Ozone": 25
# }

# who_limits = {
#     "PM2.5": 25,
#     "PM10": 50,
#     "NO₂": 40,
#     "SO₂": 20,
#     "CO": 4.0,
#     "Ozone": 50
# }

# # User inputs
# pollutants = ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "Ozone"]
# your_values = [pm25, pm10, no2, so2, co, ozone]
# delhi_avg = [historical_avg[p] for p in pollutants]
# who_safe = [who_limits[p] for p in pollutants]

# # Create DataFrame
# df_compare = pd.DataFrame({
#     "Pollutant": pollutants,
#     "Your Input": your_values,
#     "Delhi Avg": delhi_avg,
#     "WHO Limit": who_safe
# })

# # Melt DataFrame for seaborn
# df_melt = df_compare.melt(id_vars="Pollutant", var_name="Type", value_name="Value")

# # Plot
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(data=df_melt, x="Pollutant", y="Value", hue="Type", ax=ax)
# plt.title("📉 Your Pollution Levels vs Delhi Avg vs WHO Safe Limits")
# plt.ylabel("Concentration")
# plt.xticks(rotation=0)
# plt.grid(axis="y")

# # Display in Streamlit
# st.pyplot(fig)
# plt.clf()


# # step 6
# # Step 6: Show Recent AQI Trend (Static Sample Data for Demo)

# import pandas as pd
# import random
# st.markdown("---")
# st.markdown("📈 **Recent AQI Trends (Simulated)**")

# # Sample dummy data for past 7 days
# trend_data = {
#     "Date": pd.date_range(end=pd.Timestamp.today(), periods=7).strftime("%Y-%m-%d"),
#     "AQI": [random.randint(80, 450) for _ in range(7)]
# }
# df_trend = pd.DataFrame(trend_data)

# # Plot the AQI line chart
# st.line_chart(df_trend.set_index("Date"), use_container_width=True)

# # Add a mini table below
# st.dataframe(df_trend.rename(columns={"Date": "📅 Date", "AQI": "🌫️ AQI Value"}), use_container_width=True)


# import qrcode
# from PIL import Image
# import streamlit as st
# from io import BytesIO
# import urllib.parse
# import os

# input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
# pred_encoded = model.predict(input_data)[0]
# pred_label = label_encoder.inverse_transform([pred_encoded])[0]
# paste_url = "https://alokdelhiairqualityml.streamlit.app/"

# color_map = {
#     "Good": "🟢",
#     "Satisfactory": "🟡",
#     "Moderate": "🟠",
#     "Poor": "🔴",
#     "Very Poor": "🟣",
#     "Severe": "⚫️"
# }
# emoji = color_map.get(pred_label, "❓")

# # ✅ Optional social media share
# tweet_text = f"Delhi AQI today is {pred_label} {emoji}. Check pollution levels here: {paste_url} #AQI #AirQuality"
# tweet_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(tweet_text)}"

# st.markdown("### 📤 Share on Social Media")
# st.markdown(f"[🐦 Tweet This Report]({tweet_url})", unsafe_allow_html=True)

# # Generate QR Code with high box_size for clarity
# qr = qrcode.QRCode(
#     version=1,
#     box_size=10,
#     border=4
# )
# qr.add_data(paste_url)
# qr.make(fit=True)

# # Create and resize image for laptop viewing
# img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
# img = img.resize((300, 300), Image.LANCZOS)  # Clear and sharp

# # Display QR code with updated Streamlit parameter
# st.image(img, caption="📱 Scan to open the report", use_container_width=False)

# # ✅ Show in Streamlit
# st.markdown("### 📲 Share This AQI Summary via QR Code")
# # st.image(qr_path, caption="🔗 Scan to open AQI Report", use_container_width=True)

# # Optional: Download QR Code
# buf = BytesIO()
# img.save(buf, format="PNG")
# byte_im = buf.getvalue()

# st.download_button(
#     label="📥 Download QR Code",
#     data=byte_im,
#     file_name="Delhi_AQI_QR_Code.png",
#     mime="image/png"
# )


# inputs = {
#     "PM2.5": pm25,
#     "PM10": pm10,
#     "NO2": no2,
#     "SO2": so2,
#     "CO": co,
#     "Ozone": ozone
# }

# # 4. 🧠 Use preset label as AQI category (simulate ML prediction here)
# aqi_category = selected_level  # You can replace this with your ML model's output if needed

# # 5. 🚦 Risk Badge Generator
# def get_risk_badge(aqi_category, inputs):
#     main_pollutant = max(inputs, key=inputs.get)
#     risk = {
#         "Good": "LOW",
#         "Satisfactory": "LOW",
#         "Moderate": "MEDIUM",
#         "Poor": "HIGH",
#         "Very Poor": "HIGH",
#         "Severe": "CRITICAL"
#     }.get(aqi_category, "UNKNOWN")

#     emoji = {
#         "LOW": "🟢",
#         "MEDIUM": "🟠",
#         "HIGH": "🔴",
#         "CRITICAL": "🚨"
#     }.get(risk, "❓")

#     return main_pollutant, risk, emoji

# # 6. 🎯 Display Summary
# main_pollutant, risk, emoji = get_risk_badge(aqi_category, inputs)

# st.markdown("---")
# st.markdown(f"""
# ### {emoji} Pollution Risk Summary
# - **Risk Level:** `{risk}`
# - **Main Pollutant:** `{main_pollutant}`
# - **AQI Category:** `{aqi_category}`
# """)
# st.markdown("---")

# import csv
# import os
# from datetime import datetime

# def log_prediction(inputs, aqi_category, main_pollutant, risk):
#     log_file = "aqi_logs.csv"
#     file_exists = os.path.exists(log_file)

#     try:
#         with open(log_file, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             if not file_exists:
#                 writer.writerow(["Timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "AQI Category", "Main Pollutant", "Risk Level"])
#             writer.writerow([
#                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 inputs["PM2.5"],
#                 inputs["PM10"],
#                 inputs["NO2"],
#                 inputs["SO2"],
#                 inputs["CO"],
#                 inputs["Ozone"],
#                 aqi_category,
#                 main_pollutant,
#                 risk
#             ])
        
#         st.success("✅ Prediction logged successfully to `aqi_logs.csv`.")
#     except Exception as e:
#         st.error(f"❌ Failed to log prediction: {e}")

# log_prediction(inputs, aqi_category, main_pollutant, risk)



# import pandas as pd
# import os
# import streamlit as st
# from io import BytesIO

# st.sidebar.markdown("### 🧠 Admin Dashboard")

# # Load log file
# if os.path.exists("aqi_logs.csv"):
#     df_log = pd.read_csv("aqi_logs.csv")
#     if df_log.shape[1] == 9:
#         df_log.columns = ["Timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "AQI", "Category"]
#     elif df_log.shape[1] == 10:
#         df_log.columns = ["Timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "AQI", "Category", "Extra"]
#     else:
#         st.error("Unexpected number of columns in log file. Please check aqi_logs.csv")

#     # Optional: Convert timestamp column to datetime
#     df_log["Timestamp"] = pd.to_datetime(df_log["Timestamp"])

#     # Filters
#     with st.expander("🔍 Filter Logs"):
#         selected_category = st.multiselect("Filter by AQI Category", options=df_log["Category"].unique())
#         start_date = st.date_input("Start Date", value=df_log["Timestamp"].min().date())
#         end_date = st.date_input("End Date", value=df_log["Timestamp"].max().date())

#         # Apply filters
#         filtered_df = df_log[
#             (df_log["Timestamp"].dt.date >= start_date) &
#             (df_log["Timestamp"].dt.date <= end_date)
#         ]
#         if selected_category:
#             filtered_df = filtered_df[filtered_df["Category"].isin(selected_category)]

#     # Show filtered data
#     st.markdown("### 📋 Filtered AQI Logs")
#     st.dataframe(filtered_df, use_container_width=True)

#     # Download filtered data as Excel
#     def convert_df_to_excel(df):
#         output = BytesIO()
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             df.to_excel(writer, index=False, sheet_name='AQI Logs')
#         processed_data = output.getvalue()
#         return processed_data

#     excel_data = convert_df_to_excel(filtered_df)

#     st.download_button(
#         label="📥 Download Logs as Excel",
#         data=excel_data,
#         file_name="filtered_aqi_logs.xlsx",
#         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#     )
# else:
#     st.warning("No AQI log file found yet.")




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
# 🔮 Prediction — PASTE THIS WHOLE BLOCK
# if page == "🔮 Prediction":
#     st.title("🔮 AQI Prediction")
#     st.caption("Enter pollutant levels to predict AQI category.")

#     # Inputs (good non-zero defaults)
#     col1, col2, col3 = st.columns(3)
#     pm25 = col1.number_input("PM2.5 (µg/m³)", min_value=0.0, value=80.0, step=1.0)
#     pm10 = col2.number_input("PM10 (µg/m³)", min_value=0.0, value=120.0, step=1.0)
#     no2  = col3.number_input("NO2 (µg/m³)",  min_value=0.0, value=40.0,  step=1.0)

#     col4, col5, col6 = st.columns(3)
#     so2   = col4.number_input("SO2 (µg/m³)",  min_value=0.0, value=10.0, step=1.0)
#     co    = col5.number_input("CO (mg/m³)",   min_value=0.0, value=1.0,  step=0.1, format="%.2f")
#     ozone = col6.number_input("Ozone (µg/m³)",min_value=0.0, value=50.0, step=1.0)

#     # Build a values dict from current inputs
#     values = {
#         "PM2.5": float(pm25),
#         "PM10":  float(pm10),
#         "NO2":   float(no2),
#         "SO2":   float(so2),
        # "CO":    float(co),
        # "Ozone": float(ozone),
    # }

    # Try to normalize if your app has normalize_values(); else use as-is
    # try:
    #     norm_values = normalize_values(values)  # your function (if defined)
    # except Exception:
    #     norm_values = values

    # Helper: local fallback category (if your simple_category_from_aqi not available)
    # def _fallback_cat(aqi: int) -> str:
    #     if aqi <= 50:  return "Good"
    #     if aqi <= 100: return "Satisfactory"
    #     if aqi <= 200: return "Moderate"
    #     if aqi <= 300: return "Poor"
    #     if aqi <= 400: return "Very Poor"
    #     return "Severe"

    # if st.button("🚀 Predict", use_container_width=True):

    #     # Try multiple predict_aqi signatures safely
    #     aqi_val, aqi_label = None, None
    #     try:
    #         # Most common in your project: dict + MODEL + ENCODER
        #     aqi_val, aqi_label = predict_aqi(norm_values, MODEL, ENCODER)
        # except Exception:
        #     try:
        #         # Sometimes only dict + MODEL
        #         aqi_val, aqi_label = predict_aqi(norm_values, MODEL)
        #     except Exception:
        #         try:
        #             # Some versions expect raw numeric args
        #             aqi_val, aqi_label = predict_aqi(pm25, pm10, no2, so2, co, ozone)
        #         except Exception:
                    # Final safe fallback: compute a heuristic AQI + label
                    # w = {"PM2.5": 0.35, "PM10": 0.25, "NO2": 0.20, "SO2": 0.07, "CO": 0.05, "Ozone": 0.08}
                    # try:
                    #     score = sum(norm_values[k] * w[k] for k in w) / sum(w.values())
                    # except Exception:
                    #     score = sum(values[k] * w[k] for k in w) / sum(w.values())
                    # aqi_val = int(max(0, min(500, score)))
                    # try:
                    #     aqi_label = simple_category_from_aqi(aqi_val)  # if your function exists
                    # except Exception:
                    #     aqi_label = _fallback_cat(aqi_val)

        # Display result (styled if your badge_class exists)
        # try:
        #     bc = badge_class(aqi_label)
        #     st.markdown(
        #         f"""
        #         <div class="card" style="text-align:center">
        #             <div style="font-size:46px; font-weight:800; line-height:1">AQI {aqi_val}</div>
        #             <div class="badge {bc}" style="margin-top:8px; font-size:16px">{aqi_label}</div>
        #             <div style="margin-top:6px"><small class="mono">Model: Random Forest (+safe fallback)</small></div>
        #         </div>
        #         """,
        #         unsafe_allow_html=True,
        #     )
        # except Exception:
        #     # Plain fallback if CSS helpers absent
        #     st.success(f"🌍 Predicted AQI: **{aqi_val}** → Category: **{aqi_label}**")

        # # Optional: remember last prediction (no logs)
        # try:
        #     st.session_state.last_prediction = (int(aqi_val), str(aqi_label))
        # except Exception:
        #     pass


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


# # app.py
# import os
# import io
# import base64
# import datetime
# from typing import Tuple, Dict

# import numpy as np
# import pandas as pd
# from PIL import Image
# import qrcode
# from qrcode.constants import ERROR_CORRECT_H

# import streamlit as st

# # Optional: external deps (guarded)
# try:
#     import joblib
# except Exception:
#     joblib = None

# try:
#     import gspread
# except Exception:
#     gspread = None


# # ──────────────────────────────
# # PAGE CONFIG
# # ──────────────────────────────
# st.set_page_config(
#     page_title="Delhi AQI – Prediction & Insights",
#     page_icon="🌫️",
#     layout="wide"
# )

# # Small CSS touch for polished cards / QR container
# st.markdown("""
# <style>
# .badge {
#   padding: 0.35rem 0.7rem; border-radius: 999px; font-weight: 600; display: inline-block;
# }
# .badge.good { background:#e7f5e9; color:#1e7e34; }
# .badge.moderate { background:#fff3cd; color:#856404; }
# .badge.poor { background:#ffe5d0; color:#a1490c; }
# .badge.verypoor { background:#fde2e1; color:#9b1c1c; }
# .badge.severe { background:#f8d7da; color:#721c24; }

# .card {
#   border-radius: 18px; padding: 16px; border: 1px solid #eee; background: white;
#   box-shadow: 0 2px 12px rgba(0,0,0,0.04); height: 100%;
# }
# .qr-box { text-align:center; }
# .qr-title { font-weight:700; margin-bottom:0.3rem; }
# small.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace; color:#666; }
# </style>
# """, unsafe_allow_html=True)

# # ──────────────────────────────
# # HELPERS
# # ──────────────────────────────

# APP_URL = "https://pollutionappcreatedbyalok.streamlit.app/"


# COLUMNS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]

# PRESETS = {
#     "Good":       [30,  40,  20,  5,  0.4, 10],
#     "Moderate":   [90, 110,  40, 10,  1.2, 30],
#     "Poor":       [200, 250, 90, 20,  2.0, 50],
#     "Very Poor":  [300, 350, 120, 30, 3.5, 70],
#     "Severe":     [400, 500, 150, 40, 4.5, 90],
# }

# WHO_LIMITS = {  # µg/m3 or mg/m3 as commonly communicated; simplified set
#     "PM2.5": 15,
#     "PM10": 45,
#     "NO2": 25,
#     "SO2": 40,
#     "CO": 4.0,         # mg/m3 (8-hour guideline)
#     "Ozone": 100
# }

# DELHI_AVG = {   # Example anchors; adjust if you have better baselines
#     "PM2.5": 120,
#     "PM10": 200,
#     "NO2": 45,
#     "SO2": 12,
#     "CO": 1.7,
#     "Ozone": 60
# }

# POLLUTANT_INFO: Dict[str, str] = {
#     "PM2.5": "Fine particles (≤2.5μm) penetrate deep into lungs; linked to heart & lung disease.",
#     "PM10": "Coarse particles (≤10μm) irritate airways; worsen asthma and bronchitis.",
#     "NO2":  "Traffic/industrial gas; inflames airways; reduces lung function over time.",
#     "SO2":  "From coal/oil burning; triggers wheezing, coughing; forms secondary PM.",
#     "CO":   "Colorless gas; reduces oxygen delivery in body; dangerous in high doses.",
#     "Ozone":"Formed in sunlight; irritates airways; causes chest pain & coughing."
# }


# def ensure_session_defaults():
#     if "values" not in st.session_state:
#         st.session_state.values = dict(zip(COLUMNS, PRESETS["Moderate"]))
#     if "last_prediction" not in st.session_state:
#         st.session_state.last_prediction = None  # (aqi_value:int, aqi_label:str)
#     if "scenario_applied" not in st.session_state:
#         st.session_state.scenario_applied = ""


# def load_model_and_encoder():
#     """Load RF model + label encoder. Safe fallback if missing."""
#     model, encoder = None, None
#     try:
#         if joblib is not None and os.path.exists("aqi_rf_model.joblib"):
#             model = joblib.load("aqi_rf_model.joblib")
#     except Exception:
#         model = None
#     try:
#         if joblib is not None and os.path.exists("label_encoder_.joblib"):
#             encoder = joblib.load("label_encoder_.joblib")
#     except Exception:
#         encoder = None
#     return model, encoder


# def simple_category_from_aqi(aqi: int) -> str:
#     # Generic Indian AQI buckets (simplified)
#     if aqi <= 50: return "Good"
#     if aqi <= 100: return "Satisfactory"  # or Moderate per your encoding
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


# def predict_aqi(values: Dict[str, float], model, encoder) -> Tuple[int, str]:
#     """Predict AQI (int) and label (str) from values dict."""
#     row = pd.DataFrame([[values[c] for c in COLUMNS]], columns=COLUMNS)

#     if model is not None:
#         try:
#             pred_raw = model.predict(row)[0]
#             # If model returns numpy type, coerce to python int
#             aqi_val = int(np.array(pred_raw).item())
#         except Exception:
#             # If model.predict yields class labels directly (encoded)
#             try:
#                 aqi_val = int(pred_raw)
#             except Exception:
#                 aqi_val = int(np.clip(np.average(list(values.values())), 0, 500))
#     else:
#         # Fallback heuristic if model missing
#         # Weighted sum emphasizing PMs and NO2
#         w = {"PM2.5": 0.35, "PM10": 0.25, "NO2": 0.2, "SO2": 0.07, "CO": 0.05, "Ozone": 0.08}
#         aqi_val = int(
#             sum(values[k] * w[k] for k in COLUMNS) / (sum(w.values()) or 1.0)
#         )
#         aqi_val = int(np.clip(aqi_val, 0, 500))

#     # Decode label
#     if encoder is not None:
#         try:
#             label = encoder.inverse_transform([aqi_val])[0]
#             if isinstance(label, (np.generic, np.integer)):  # weird encoders
#                 label = simple_category_from_aqi(int(label))
#         except Exception:
#             label = simple_category_from_aqi(aqi_val)
#     else:
#         label = simple_category_from_aqi(aqi_val)

#     return aqi_val, label


# def make_qr_bytes(content: str, size_px: int = 160) -> bytes:
#     qr = qrcode.QRCode(
#         version=None,
#         error_correction=ERROR_CORRECT_H,
#         box_size=10,
#         border=2
#     )
#     qr.add_data(content)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
#     # Resize to crisp "passport size"
#     img = img.resize((size_px, size_px), resample=Image.Resampling.LANCZOS)
#     buf = io.BytesIO()
#     img.save(buf, format="PNG", optimize=True)
#     return buf.getvalue()


# def try_log_to_sheets(values: Dict[str, float], aqi_val: int, aqi_label: str):
#     """Optional Google Sheets logging (only if secrets and gspread are available)."""
#     if gspread is None:
#         return
#     if "gspread" not in st.secrets:
#         return
#     try:
#         gc = gspread.service_account_from_dict(st.secrets["gspread"])
#         sheet = gc.open("Delhi AQI Predictions").sheet1
#         now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         row = [
#             now,
#             float(values["PM2.5"]), float(values["PM10"]), float(values["NO2"]),
#             float(values["SO2"]), float(values["CO"]), float(values["Ozone"]),
#             int(aqi_val), str(aqi_label)
#         ]
#         sheet.append_row(row)
#         st.toast("Logged to Google Sheets.", icon="☁️")
#     except Exception as e:
#         st.info(f"Sheets logging skipped: {e}")


# def log_to_csv(values: Dict[str, float], aqi_val: int, aqi_label: str):
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
#     return pd.DataFrame([values], columns=COLUMNS)


# def comparison_frame(values: Dict[str, float]) -> pd.DataFrame:
#     rows = []
#     for p in COLUMNS:
#         rows.append({
#             "Pollutant": p,
#             "Your Level": float(values[p]),
#             "Delhi Avg": float(DELHI_AVG[p]),
#             "WHO Limit": float(WHO_LIMITS[p])
#         })
#     return pd.DataFrame(rows)


# # ──────────────────────────────
# # LOAD MODEL ONCE
# # ──────────────────────────────
# ensure_session_defaults()
# MODEL, ENCODER = load_model_and_encoder()

# # ──────────────────────────────
# # SIDEBAR NAVIGATION (ORDER EXACT)
# # ──────────────────────────────
# with st.sidebar:
#     st.image("https://img.icons8.com/?size=100&id=12448&format=png&color=000000", width=32)
#     st.markdown("### Delhi AQI App")
#     page = st.radio(
#         "Navigation",
#         options=[
#             "1) Understand Pollutants + Share",
#             "2) Learn About AQI & Health Tips",
#             "3) Try a Sample AQI Scenario",
#             "4) Preset or Custom Inputs + Your Levels",
#             "5) Predict Delhi AQI Category",
#             "6) Compare with Delhi Avg & WHO"
#         ],
#         index=0
#     )
#     st.caption("Made with ❤️ for Delhi air quality.\nUse the pages in order for the best flow.")




# import streamlit as st
# import qrcode
# from io import BytesIO

# # ---------------------------
# # CONFIGURE APP URL
# # ---------------------------
# APP_URL = "https://pollutionappcreatedbyalok.streamlit.app/"  # Your real app link

# # ---------------------------
# # QR CODE GENERATOR FUNCTION
# # ---------------------------
# def make_qr_image(url):
#     qr = qrcode.QRCode(
#         version=1, box_size=10, border=2
#     )
#     qr.add_data(url)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color="black", back_color="white")
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     buf.seek(0)
#     return buf

# # ---------------------------
# # PAGE 1
# # ---------------------------
# st.title("🌍 Delhi AQI Prediction App")

# st.markdown("📌 Scan the QR code to open this app on your mobile!")

# # Two-column layout (QR on right, intro + button on left)
# c1, c2 = st.columns([2, 1])

# with c1:
#     st.subheader("Welcome!")
#     st.write("This app predicts **Delhi's Air Quality Index (AQI)** and provides health recommendations.")
#     st.write("👉 Click below to continue with your analysis.")
    
#     # Place button here so it’s front-level under intro
#     if st.button("➡️ Take Analysis"):
#         st.session_state["page"] = "2) AQI Prediction"

# with c2:
#     qr_buf = make_qr_image(APP_URL)
#     st.image(qr_buf, caption="📱 Scan to open the app", use_container_width=True)

# # ---------------------------
# # PAGE 2 placeholder
# # ---------------------------
# if st.session_state.get("page") == "2) AQI Prediction":
#     st.header("📊 AQI Prediction Page")
#     st.write("This is where your pollutant input form + prediction model will run.")




# # # ──────────────────────────────
# # # 2) LEARN ABOUT AQI & HEALTH TIPS (Download)
# # # ──────────────────────────────
# # elif page.startswith("2)"):
# #     st.title("📚 Learn About AQI & Health Tips")

#     st.markdown("""
# **AQI Categories (India - simplified):**
# - **Good (0–50):** Enjoy outdoor activities.
# - **Satisfactory/Moderate (51–100):** Sensitive groups take care.
# - **Moderate (101–200):** Reduce prolonged outdoor exertion.
# - **Poor (201–300):** Consider masks; limit outdoor time.
# - **Very Poor (301–400):** Avoid outdoor exertion; use purifiers.
# - **Severe (401–500):** Stay indoors; medical advice for symptoms.

# **General Health Tips:**
# - Track AQI daily and plan outdoor tasks on lower-AQI hours.
# - Use N95/FFP2 masks during poor days.
# - Keep windows closed during peak pollution hours; ventilate in cleaner windows.
# - Use HEPA filters/purifiers indoors.
# - Stay hydrated; consider nasal irrigation after heavy exposure.
# """)

#     latest_txt = ""
#     if st.session_state.last_prediction is not None:
#         aqi_val, aqi_label = st.session_state.last_prediction
#         latest_txt = f"\nLatest Prediction: {aqi_val} ({aqi_label})"

#     tips_md = f"# Delhi AQI – Quick Guide\n{latest_txt}\n\n" + \
#               "• AQI buckets and what they mean\n" \
#               "• Tips for masks, purifiers, and timing your outdoor activities\n" \
#               "• Monitor pollutants: PM2.5, PM10, NO2, SO2, CO, Ozone\n" \
#               f"\nApp: {APP_URL}\n"

#     st.download_button(
#         "⬇️ Download This Guide (Markdown)",
#         data=tips_md.encode(),
#         file_name="Delhi_AQI_Guide.md",
#         mime="text/markdown",
#         use_container_width=True
#     )



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
#         st.session_state.values = dict(zip(COLUMNS, map(float, vals)))
#         st.session_state.scenario_applied = chosen
#         st.success(f"Scenario applied: {chosen}. Go to page 4 to review/edit values.")


# # ──────────────────────────────
# # 4) PRESET OR CUSTOM INPUTS + YOUR LEVELS
# # ──────────────────────────────
# elif page.startswith("4)"):
#     st.title("🎛️ Choose a Preset AQI Level or Enter Custom Values")

#     # Preset selector
#     preset = st.selectbox("Choose Preset AQI Level", list(PRESETS.keys()))
#     defaults = list(map(float, PRESETS[preset]))

#     # If scenario was applied earlier, override defaults with the scenario
#     if st.session_state.scenario_applied:
#         defaults = [st.session_state.values[c] for c in COLUMNS]

#     c1, c2 = st.columns(2)
#     with c1:
#         pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=defaults[0])
#         no2  = st.number_input("NO2 (µg/m³)",   min_value=0.0, value=defaults[2])
#         co   = st.number_input("CO (mg/m³)",    min_value=0.0, value=defaults[4])
#     with c2:
#         pm10 = st.number_input("PM10 (µg/m³)",  min_value=0.0, value=defaults[1])
#         so2  = st.number_input("SO2 (µg/m³)",   min_value=0.0, value=defaults[3])
#         o3   = st.number_input("Ozone (µg/m³)", min_value=0.0, value=defaults[5])

#     # Update session
#     st.session_state.values = {
#         "PM2.5": float(pm25), "PM10": float(pm10), "NO2": float(no2),
#         "SO2": float(so2), "CO": float(co), "Ozone": float(o3)
#     }

#     st.markdown("### 📋 Your Entered Pollution Levels")
#     st.dataframe(values_table(st.session_state.values), use_container_width=True)


# # ──────────────────────────────
# # 5) PREDICT DELHI AQI CATEGORY
# # ──────────────────────────────
# elif page.startswith("5)"):
#     st.title("🔮 Predict Delhi AQI Category")

#     values = st.session_state.values
#     st.markdown("Review your inputs before predicting:")
#     st.dataframe(values_table(values), use_container_width=True)

#     if st.button("🚀 Run Prediction", use_container_width=True):
#         aqi_val, aqi_label = predict_aqi(values, MODEL, ENCODER)
#         st.session_state.last_prediction = (aqi_val, aqi_label)

#         # Pretty badge
#         bc = badge_class(aqi_label)
#         st.markdown(f"""
#         <div class="card" style="text-align:center">
#             <div style="font-size:46px; font-weight:800; line-height:1">AQI {aqi_val}</div>
#             <div class="badge {bc}" style="margin-top:8px; font-size:16px">{aqi_label}</div>
#             <div style="margin-top:6px"><small class="mono">Model: Random Forest (+fallback safe)</small></div>
#         </div>
#         """, unsafe_allow_html=True)

#         # Logging
#         try:
#             log_to_csv(values, aqi_val, aqi_label)
#             st.success("✅ Prediction logged to aqi_logs.csv")
#         except Exception as e:
#             st.info(f"CSV logging skipped: {e}")

#         try:
#             try_log_to_sheets(values, aqi_val, aqi_label)
#         except Exception:
#             pass

#         # Simple success toast
#         st.toast("Prediction done!", icon="✅")

#     # If we have a previous prediction, show it compactly
#     if st.session_state.last_prediction:
#         aqi_val, aqi_label = st.session_state.last_prediction
#         st.caption(f"Last prediction: **AQI {aqi_val} ({aqi_label})**")


# # ──────────────────────────────
# # 6) COMPARE WITH DELHI AVERAGES & WHO LIMITS
# # ──────────────────────────────
# elif page.startswith("6)"):
#     st.title("📊 Compare Your Pollution Levels with Delhi Averages & WHO Limits")

#     values = st.session_state.values
#     df_cmp = comparison_frame(values)

#     # Show table
#     st.dataframe(df_cmp, use_container_width=True)

#     st.markdown("#### Visual Comparison")
#     # Melt long for plotting with Streamlit
#     df_long = df_cmp.melt(id_vars="Pollutant", var_name="Metric", value_name="Level")
#     # Use built-in chart per pollutant for clarity
#     for p in COLUMNS:
#         sub = df_long[df_long["Pollutant"] == p].set_index("Metric")["Level"]
#         st.markdown(f"**{p}**")
#         st.bar_chart(sub, use_container_width=True)

#     st.info("Tip: Aim to keep each pollutant at or below the WHO guideline when possible.")


# # ──────────────────────────────
# # FOOTER
# # ──────────────────────────────
# st.markdown("---")
# st.caption("© 2025 Delhi AQI App • Built with Streamlit •")





# import os
# import io
# import base64
# import datetime
# from typing import Tuple, Dict


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




import numpy as np
import pandas as pd
from PIL import Image
import qrcode
from qrcode.constants import ERROR_CORRECT_H
import io

import streamlit as st

# Optional: external deps (guarded)
try:
    import joblib
except Exception:
    joblib = None

try:
    import gspread
except Exception:
    gspread = None

# ──────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ──────────────────────────────
st.set_page_config(
    page_title="Delhi AQI – Prediction & Insights",
    page_icon="🌫️",
    layout="wide"
)

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────
# CONSTANTS & HELPERS
# ──────────────────────────────
APP_URL = "https://pollutionappcreatedbyalok.streamlit.app/"

COLUMNS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]

PRESENTS = {
    "Good":       [30,  40,  20,  5,  0.4, 10],
    "Moderate":   [90, 110,  40, 10,  1.2, 30],
    "Poor":       [200, 250, 90, 20,  2.0, 50],
    "Very Poor":  [300, 350, 120, 30,  3.5, 70],
    "Severe":     [400, 500, 150, 40,  4.5, 90],
}

WHO_LIMITS = {
    "PM2.5": 15,
    "PM10": 45,
    "NO2": 25,
    "SO2": 40,
    "CO": 4.0,    # mg/m3 (8-hour guideline)
    "Ozone": 100,
}

DELHI_AVG = {
    "PM2.5": 120,
    "PM10": 200,
    "NO2": 45,
    "SO2": 12,
    "CO": 1.7,
    "Ozone": 60,
}


from typing import Dict, List, Tuple, Optional


POLLUTANT_INFO: Dict[str, str] = {
    "PM2.5": "Fine particles (≤2.5μm) penetrate deep into lungs; linked to heart & lung disease.",
    "PM10": "Coarse particles (≤10μm) irritate airways; worsen asthma and bronchitis.",
    "NO2":  "Traffic/industrial gas; inflames airways; reduces lung function over time.",
    "SO2":  "From coal/oil burning; triggers wheezing, coughing; forms secondary PM.",
    "CO":   "Colorless gas; reduces oxygen delivery in body; dangerous in high doses.",
    "Ozone":"Formed in sunlight; irritates airways; causes chest pain & coughing.",
}


def ensure_session_defaults():
    if "values" not in st.session_state:
        st.session_state.values = {k: float(v) for k, v in zip(COLUMNS, PRESENTS["Moderate"])}
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None  # (aqi_value:int, aqi_label:str)
    if "scenario_applied" not in st.session_state:
        st.session_state.scenario_applied = ""
    if "nav" not in st.session_state:
        st.session_state.nav = "1) Understand + Share"


def normalize_values(values: Dict[str, float]) -> Dict[str, float]:
    """Ensure we always have all COLUMNS with float values. Prevents shape/key errors."""
    safe = {}
    if isinstance(values, dict):
        for c in COLUMNS:
            try:
                safe[c] = float(values.get(c, 0.0))
            except Exception:
                safe[c] = 0.0
    else:
        # if someone accidentally put a list/tuple in session_state.values
        for i, c in enumerate(COLUMNS):
            try:
                safe[c] = float(values[i])
            except Exception:
                safe[c] = 0.0
    return safe


def load_model_and_encoder():
    """Load RF model + label encoder. Safe fallback if missing."""
    model, encoder = None, None
    try:
        if joblib is not None and os.path.exists("aqi_rf_model.joblib"):
            model = joblib.load("aqi_rf_model.joblib")
    except Exception:
        model = None
    try:
        # Use the most common filename first
        if joblib is not None and os.path.exists("label_encoder.joblib"):
            encoder = joblib.load("label_encoder.joblib")
        elif joblib is not None and os.path.exists("label_encoder_.joblib"):
            encoder = joblib.load("label_encoder_.joblib")
    except Exception:
        encoder = None
    return model, encoder


def simple_category_from_aqi(aqi: int) -> str:
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


def badge_class(label: str) -> str:
    key = label.replace(" ", "").lower()
    if key in ["good"]: return "good"
    if key in ["moderate", "satisfactory"]: return "moderate"
    if key == "poor": return "poor"
    if key in ["verypoor", "verybad"]: return "verypoor"
    return "severe"


# def predict_aqi(values: Dict[str, float], model, encoder) -> Tuple[int, str]:
#     values = normalize_values(values)
#     row = pd.DataFrame([[values[c] for c in COLUMNS]], columns=COLUMNS)

#     # Predict number or class code
#     if model is not None:
#         try:
#             pred_raw = model.predict(row)[0]
#             aqi_val = int(np.array(pred_raw).item())
#         except Exception:
#             try:
#                 aqi_val = int(pred_raw)
#             except Exception:
#                 aqi_val = int(np.clip(np.average(list(values.values())), 0, 500))
#     else:
#         w = {"PM2.5": 0.35, "PM10": 0.25, "NO2": 0.2, "SO2": 0.07, "CO": 0.05, "Ozone": 0.08}
#         aqi_val = int(sum(values[k] * w[k] for k in COLUMNS) / (sum(w.values()) or 1.0))
#         aqi_val = int(np.clip(aqi_val, 0, 500))

#     # Decode label if encoder truly encodes categories, else fallback from number
#     if encoder is not None:
#         try:
#             label = encoder.inverse_transform([aqi_val])[0]
#             if isinstance(label, (np.generic, np.integer)):
#                 label = simple_category_from_aqi(int(label))
#         except Exception:
#             label = simple_category_from_aqi(aqi_val)
#     else:
#         label = simple_category_from_aqi(aqi_val)

#     return aqi_val, label

def predict_aqi(pm25, pm10, no2, so2, co, ozone):
    """
    Predicts AQI value & category based on pollutant levels.
    Returns (aqi_value, aqi_category).
    """
    try:
        # Load model + encoder only once
        if "MODEL" not in st.session_state:
            st.session_state.MODEL, st.session_state.ENCODER = load_model()

        MODEL = st.session_state.MODEL
        ENCODER = st.session_state.ENCODER

        # Prepare input array
        X = np.array([[pm25, pm10, no2, so2, co, ozone]])

        # Predict AQI value (regression output)
        aqi_val = MODEL.predict(X)[0]

        # Predict AQI category (classification output if available)
        if hasattr(MODEL, "predict_proba"):  
            # If classification model with label encoding
            y_class = MODEL.predict(X)
            aqi_label = ENCODER.inverse_transform(y_class)[0]
        else:
            # Fallback: manually classify based on AQI range
            if aqi_val <= 50:
                aqi_label = "Good"
            elif aqi_val <= 100:
                aqi_label = "Satisfactory"
            elif aqi_val <= 200:
                aqi_label = "Moderate"
            elif aqi_val <= 300:
                aqi_label = "Poor"
            elif aqi_val <= 400:
                aqi_label = "Very Poor"
            else:
                aqi_label = "Severe"

        return round(aqi_val, 2), aqi_label

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return None, None



def make_qr_bytes(content: str, size_px: int = 160) -> bytes:
    qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_H, box_size=10, border=2)
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((size_px, size_px), resample=Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def try_log_to_sheets(values: Dict[str, float], aqi_val: int, aqi_label: str):
    if gspread is None: return
    if "gspread" not in st.secrets: return
    try:
        gc = gspread.service_account_from_dict(st.secrets["gspread"])
        sheet = gc.open("Delhi AQI Predictions").sheet1
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        values = normalize_values(values)
        row = [
            now,
            float(values["PM2.5"]), float(values["PM10"]), float(values["NO2"]),
            float(values["SO2"]), float(values["CO"]), float(values["Ozone"]),
            int(aqi_val), str(aqi_label),
        ]
        sheet.append_row(row)
        st.toast("Logged to Google Sheets.", icon="☁️")
    except Exception as e:
        st.info(f"Sheets logging skipped: {e}")


def log_to_csv(values: Dict[str, float], aqi_val: int, aqi_label: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    values = normalize_values(values)
    row = {
        "Timestamp": now,
        "PM2.5": float(values["PM2.5"]),
        "PM10": float(values["PM10"]),
        "NO2": float(values["NO2"]),
        "SO2": float(values["SO2"]),
        "CO": float(values["CO"]),
        "Ozone": float(values["Ozone"]),
        "PredictedAQI": int(aqi_val),
        "AQICategory": str(aqi_label),
    }
    df_row = pd.DataFrame([row])
    if os.path.exists("aqi_logs.csv"):
        try:
            df_old = pd.read_csv("aqi_logs.csv")
            df_new = pd.concat([df_old, df_row], ignore_index=True)
        except Exception:
            df_new = df_row
    else:
        df_new = df_row
    df_new.to_csv("aqi_logs.csv", index=False)


def values_table(values: Dict[str, float]) -> pd.DataFrame:
    values = normalize_values(values)
    return pd.DataFrame([[values[c] for c in COLUMNS]], columns=COLUMNS)


def comparison_frame(values: Dict[str, float]) -> pd.DataFrame:
    values = normalize_values(values)
    rows = []
    for p in COLUMNS:
        rows.append({
            "Pollutant": p,
            "Your Level": float(values.get(p, 0.0)),
            "Delhi Avg": float(DELHI_AVG[p]),
            "WHO Limit": float(WHO_LIMITS[p]),
        })
    return pd.DataFrame(rows)

# ──────────────────────────────
# INIT & LOAD MODEL ONCE
# ──────────────────────────────
ensure_session_defaults()
MODEL, ENCODER = load_model_and_encoder()

# ──────────────────────────────
# SIDEBAR NAVIGATION — SINGLE ROUTER
# (Fixes: no duplicate imports, no stray pages outside conditions)
# ──────────────────────────────
# with st.sidebar:
#     st.image("https://img.icons8.com/?size=100&id=12448&format=png&color=000000", width=32)
#     st.markdown("### Delhi AQI App")
    
#     options_list = [
#         "1) Understand + Share",
#         "2) Learn About AQI & Health Tips",
#         "3) Try a Sample AQI Scenario",
#         "4) Preset or Custom Inputs",
#         "5) Predict Delhi AQI Category",
#         "6) Compare with Delhi Avg & WHO",
#     ]

#     page = st.radio(
#         "Navigation",
#         options=options_list,
#         index=options_list.index(st.session_state.nav),
#         key="nav",
#     )
    
#     st.caption("Made with ❤️ for Delhi air quality. Follow the pages in order.")

# ✅ Ensure default session state
options_list = [
    "1) Understand + Share",
    "2) Learn About AQI & Health Tips",
    "3) Try a Sample AQI Scenario",
    "4) Preset or Custom Inputs",
    "5) Predict Delhi AQI Category",
    "6) Compare with Delhi Avg & WHO",
]

if "nav" not in st.session_state or st.session_state.nav not in options_list:
    # default to the prediction page if you want to be at the end of the project,
    # otherwise set to options_list[0]
    st.session_state.nav = "5) Predict Delhi AQI Category"

# default pollutant values (safe defaults so UI doesn't break)
if "values" not in st.session_state:
    st.session_state.values = {
        "pm25": 80.0,
        "pm10": 120.0,
        "no2": 40.0,
        "so2": 10.0,
        "co": 1.0,
        "ozone": 50.0,
    }

with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=12448&format=png&color=000000", width=32)
    st.markdown("### Delhi AQI App")
    # safe index lookup (we already ensured st.session_state.nav is valid)
    page = st.radio(
        "Navigation",
        options=options_list,
        index=options_list.index(st.session_state.nav),
        key="nav",
    )
    st.caption("Made with ❤️ for Delhi air quality. Follow the pages in order.")



import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
# other imports...

# --- Utility: ensure session state defaults ---
def ensure_session_defaults():
    if "values" not in st.session_state:
        st.session_state.values = {
            "PM2.5": 40.0,
            "PM10": 80.0,
            "NO2": 25.0,
            "SO2": 15.0,
            "CO": 0.8,
            "Ozone": 30.0,
        }
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "scenario_applied" not in st.session_state:
        st.session_state.scenario_applied = False
    if "last_present" not in st.session_state:
        st.session_state.last_present = None

# --- CALL HERE ---
ensure_session_defaults()

# --- Load model, encoders, and continue with pages ---
MODEL, ENCODER = load_model_and_encoder()


# ──────────────────────────────
# INIT & LOAD MODEL ONCE
# ──────────────────────────────
ensure_session_defaults()  # ✅ now works, since defined above
MODEL, ENCODER = load_model_and_encoder()

# ──────────────────────────────
# SIDEBAR NAVIGATION — SINGLE ROUTER
# ──────────────────────────────
# --- Ensure nav default exists ---
if "nav" not in st.session_state:
    st.session_state.nav = "1) Understand + Share"

# Sidebar navigation
with st.sidebar:
    st.image(
        "https://img.icons8.com/?size=100&id=12448&format=png&color=000000",
        width=32
    )
    st.markdown("### Delhi AQI App")

    page_options = [
        "1) Understand + Share",
        "2) Learn About AQI & Health Tips",
        "3) Try a Sample AQI Scenario",
        "4) Preset or Custom Inputs",
        "5) Predict Delhi AQI Category",
        "6) Compare with Delhi Avg & WHO",
    ]

    # Safe index: use get with fallback
    default_nav = st.session_state.get("nav", "1) Understand + Share")
    default_index = page_options.index(default_nav) if default_nav in page_options else 0

    # page = st.radio(
    #     "Navigation",
    #     options=page_options,
    #     index=default_index,
    #     key="nav",
    # )

    st.caption("Made with ❤️ for Delhi air quality. Follow the pages in order.")




# # ---------------- Page-1 : Home ----------------
# if page.startswith("1)"):
#     st.title("🌍 Delhi AQI Prediction App")
#     st.markdown("📌 Scan the QR code to open this app on your mobile!")

#     c1, c2 = st.columns([2, 1])
#     with c1:
#         st.subheader("Welcome!")
#         st.write("This app predicts **Delhi's Air Quality Index (AQI)** and provides health recommendations.")
#         st.write("👉 Click below to jump to prediction.")

#         # Navigation button
#         if st.button("➡️ Take Analysis"):
#             # Update page safely without direct overwrite
#             st.session_state.page = "analysis"
#             st.rerun()

#     with c2:
#         st.image(make_qr_bytes(APP_URL), caption="📱 Scan to open the app", use_container_width=True)

#     st.markdown("---")
#     st.markdown("#### Pollutants you can track")
#     for k, v in POLLUTANT_INFO.items():
#         st.markdown(f"**{k}** — {v}")


# ---------------- Page-1 : Home ----------------
if page.startswith("1)"):
    # Nice title + thin brand line
    st.markdown(
        "<h1 style='text-align:center; color:#2E86C1;'>🌍 Delhi AQI Prediction Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border:2px solid #2E86C1; margin-top:4px;'>", unsafe_allow_html=True)

    # Two-column layout: Left = intro + features + pollutant cards | Right = QR
    c1, c2 = st.columns([2, 1], vertical_alignment="top")

    with c1:
        st.subheader("✨ Welcome!")
        # Intro + features line (your request: pollutants section comes right below this)
        st.markdown(
            """
            <div style="font-size:18px; line-height:1.6; color:#444;">
              This interactive dashboard helps you understand and predict
              <b>Delhi's Air Quality Index (AQI)</b> 📊.
              <br><br>
              ✅ Real-time like predictions &nbsp;&nbsp; ✅ Pollutant-wise insights &nbsp;&nbsp; ✅ Health recommendations 🩺
            </div>
            """,
            unsafe_allow_html=True
        )

        # 🔻 Right below features: Key Pollutants Tracked
        st.markdown("### 🌟 Key Pollutants Tracked")

        # Small helper for pretty colored cards
        def _card(bg_hex: str, title_html: str, subtitle: str) -> str:
            return (
                f"<div style='background:#{bg_hex}; padding:16px; border-radius:16px;"
                f"box-shadow:0 2px 10px rgba(0,0,0,.05); border:1px solid rgba(0,0,0,.04);'>"
                f"{title_html}<br><small style='color:#333;'>{subtitle}</small></div>"
            )

        # Row 1
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.markdown(_card("FADBD8", "🌫️ <b>PM2.5</b>", "Fine particulate matter"), unsafe_allow_html=True)
        with r1c2:
            st.markdown(_card("D6EAF8", "🌪️ <b>PM10</b>", "Coarse particles"), unsafe_allow_html=True)
        with r1c3:
            st.markdown(_card("E8DAEF", "🌬️ <b>NO₂</b>", "Nitrogen dioxide"), unsafe_allow_html=True)

        # Row 2
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.markdown(_card("FCF3CF", "🔥 <b>SO₂</b>", "Sulfur dioxide"), unsafe_allow_html=True)
        with r2c2:
            st.markdown(_card("D5F5E3", "🟢 <b>CO</b>", "Carbon monoxide"), unsafe_allow_html=True)
        with r2c3:
            st.markdown(_card("FDEDEC", "☀️ <b>O₃</b>", "Ozone"), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    with c2:
        # Your QR remains exactly as before
        st.image(
            make_qr_bytes(APP_URL),
            caption="📱 Scan to open on mobile",
            use_container_width=True
        )

    # Subtle footer hint
    st.markdown("<hr style='margin:18px 0 6px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#666;'>Use the sidebar to navigate the app</div>",
        unsafe_allow_html=True
    )

# ──────────────────────────────
# 2) LEARN ABOUT AQI & HEALTH TIPS (Download)
# ──────────────────────────────
elif page.startswith("2)"):
    st.title("📚 Learn About AQI & Health Tips")
    st.markdown(
        """
        **AQI Categories (India - simplified):**
        - **Good (0–50):** Enjoy outdoor activities.
        - **Satisfactory/Moderate (51–100):** Sensitive groups take care.
        - **Moderate (101–200):** Reduce prolonged outdoor exertion.
        - **Poor (201–300):** Consider masks; limit outdoor time.
        - **Very Poor (301–400):** Avoid outdoor exertion; use purifiers.
        - **Severe (401–500):** Stay indoors; seek medical advice for symptoms.

        **General Health Tips:**
        - Track AQI daily and plan outdoor tasks on lower-AQI hours.
        - Use N95/FFP2 masks during poor days.
        - Keep windows closed during peak pollution; ventilate when cleaner.
        - Use HEPA purifiers indoors.
        - Stay hydrated; saline/nasal rinse after heavy exposure.
        """
    )

    latest_txt = ""
    if st.session_state.last_prediction is not None:
        aqi_val, aqi_label = st.session_state.last_prediction
        latest_txt = f"\nLatest Prediction: {aqi_val} ({aqi_label})"

    tips_md = (
        f"# Delhi AQI – Quick Guide\n{latest_txt}\n\n"
        "• AQI buckets and what they mean\n"
        "• Tips for masks, purifiers, timing outdoor activities\n"
        "• Monitor pollutants: PM2.5, PM10, NO2, SO2, CO, Ozone\n"
        f"\nApp: {APP_URL}\n"
    )

    st.download_button(
        "⬇️ Download This Guide (Markdown)",
        data=tips_md.encode(),
        file_name="Delhi_AQI_Guide.md",
        mime="text/markdown",
        use_container_width=True,
    )

# ──────────────────────────────
# 3) TRY A SAMPLE AQI SCENARIO
# ──────────────────────────────
elif page.startswith("3)"):
    st.title("🧪 Try a Sample AQI Scenario")

    scenarios = {
        "Winter Smog Morning": [280, 360, 95, 18, 2.2, 45],
        "Post-Diwali Night":   [420, 520, 130, 30, 3.2, 70],
        "Summer Breeze Day":   [55,  75,  22,  6,  0.6, 25],
    }

    chosen = st.selectbox("Select a scenario", list(scenarios.keys()))
    vals = scenarios[chosen]
    df_preview = pd.DataFrame([vals], columns=COLUMNS)
    st.dataframe(df_preview, use_container_width=True)

    if st.button("✅ Apply This Scenario", use_container_width=True):
        st.session_state.values = {k: float(v) for k, v in zip(COLUMNS, vals)}
        st.session_state.scenario_applied = chosen
        st.success(f"Scenario applied: {chosen}. Go to page 4 to review/edit values.")

# ──────────────────────────────
# 4) PRESET OR CUSTOM INPUTS
# ──────────────────────────────
    # elif page.startswith("4)"):
    #     st.title("🎛️ Preset or Custom Inputs")
    
    #     # Preset selector
    #     preset = st.selectbox("Choose Preset AQI Level", list(PRESETS.keys()))
    #     defaults = list(map(float, PRESETS[preset]))
    
    #     # If scenario was applied earlier, use those values as defaults
    #     if st.session_state.scenario_applied:
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
    
    #     # Update session (always normalized)
    #     st.session_state.values = normalize_values({
    #         "PM2.5": pm25, "PM10": pm10, "NO2": no2,
    #         "SO2": so2,   "CO": co,    "Ozone": o3,
    #     })
    
    #     st.markdown("### 📋 Your Entered Pollution Levels")
    #     st.dataframe(values_table(st.session_state.values), use_container_width=True)


# elif page.startswith("4)"):
#     st.title("🎛️ Preset or Custom Inputs")

#     # Preset selector
#     preset = st.selectbox("Choose Preset AQI Level", list(PRESETS.keys()))

#     # If a new preset is selected, reset scenario_applied
#     if "last_preset" not in st.session_state or st.session_state.last_preset != preset:
#         st.session_state.scenario_applied = False
#         st.session_state.last_preset = preset

#     # Defaults from preset
#     defaults = list(map(float, PRESETS[preset]))

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

#     # Update session (always normalized)
#     st.session_state.values = normalize_values({
#         "PM2.5": pm25, "PM10": pm10, "NO2": no2,
#         "SO2": so2,   "CO": co,    "Ozone": o3,
#     })

elif page.startswith("4)"):
    st.title("🎛️ Present or Custom Inputs")

    # Present selector
    present = st.selectbox("Choose Present AQI Level", list(PRESENTS.keys()))

    # If a new present is selected, reset scenario_applied
    if "last_present" not in st.session_state or st.session_state.last_present != present:
        st.session_state.scenario_applied = False
        st.session_state.last_present = present

    # Defaults from present
    defaults = list(map(float, PRESENTS[present]))

    # If scenario was applied earlier, override with those values
    if st.session_state.get("scenario_applied", False):
        defaults = [normalize_values(st.session_state.values)[c] for c in COLUMNS]

    c1, c2 = st.columns(2)
    with c1:
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=float(defaults[0]))
        no2  = st.number_input("NO2 (µg/m³)",   min_value=0.0, value=float(defaults[2]))
        co   = st.number_input("CO (mg/m³)",    min_value=0.0, value=float(defaults[4]))
    with c2:
        pm10 = st.number_input("PM10 (µg/m³)",  min_value=0.0, value=float(defaults[1]))
        so2  = st.number_input("SO2 (µg/m³)",   min_value=0.0, value=float(defaults[3]))
        o3   = st.number_input("Ozone (µg/m³)", min_value=0.0, value=float(defaults[5]))

    st.session_state.values = {
    "PM2.5": pm25,
    "PM10": pm10,
    "NO2": no2,
    "SO2": so2,
    "CO": co,
    "Ozone": o3,
}







# -------------------------
# Page 5: Predict Delhi AQI Category (isolated)
# -------------------------
# elif page.startswith("5)"):
#     import datetime
#     import numpy as np

#     st.title("🔮 Predict Delhi AQI Category")
#     st.markdown("Enter pollutant levels below and click **Predict**. This prediction is independent of other 'present/custom' controls.")

#     # Try load model/encoder using your helper if it exists (safe fallback)
#     try:
#         model, encoder = load_model_and_encoder()
#     except Exception:
#         model, encoder = None, None

#     # Use a safe canonical order for pollutants (will not read global 'values')
#     cols = COLUMNS if "COLUMNS" in globals() else ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]

#     # Page-5-specific widgets (unique keys so they don't clash)
#     col1, col2, col3 = st.columns(3)
#     col4, col5, col6 = st.columns(3)

#     p5_pm25  = col1.number_input("PM2.5 (µg/m³)", min_value=0.0, value=80.0, step=1.0, key="p5_pm25")
#     p5_pm10  = col2.number_input("PM10 (µg/m³)",  min_value=0.0, value=120.0, step=1.0, key="p5_pm10")
#     p5_no2   = col3.number_input("NO2 (µg/m³)",   min_value=0.0, value=40.0,  step=1.0, key="p5_no2")
#     p5_so2   = col4.number_input("SO2 (µg/m³)",   min_value=0.0, value=10.0,  step=1.0, key="p5_so2")
#     p5_co    = col5.number_input("CO (mg/m³)",    min_value=0.0, value=1.0,   step=0.1, key="p5_co")
#     p5_ozone = col6.number_input("Ozone (µg/m³)", min_value=0.0, value=50.0,  step=1.0, key="p5_ozone")

#     # Small helper local mapper (used if model/encoder not present)
#     def _simple_category_from_aqi(aqi: float) -> str:
#         try: aqi = float(aqi)
#         except: return "Unknown"
#         if aqi <= 50: return "Good"
#         if aqi <= 100: return "Satisfactory"
#         if aqi <= 200: return "Moderate"
#         if aqi <= 300: return "Poor"
#         if aqi <= 400: return "Very Poor"
#         return "Severe"

#     # Predict button: uses only the Page-5 widgets (guaranteed independence)
#     if st.button("🚀 Predict (use Page-5 inputs only)", use_container_width=True):
#         try:
#             input_vec = [[float(p5_pm25), float(p5_pm10), float(p5_no2), float(p5_so2), float(p5_co), float(p5_ozone)]]

#             predicted_value = None   # numeric AQI (kept internally but not shown)
#             predicted_label = None

#             if model is not None:
#                 try:
#                     raw = model.predict(input_vec)
#                     y = raw[0] if hasattr(raw, "__len__") else raw

#                     # Try to get numeric AQI
#                     try:
#                         predicted_value = float(y)
#                     except Exception:
#                         predicted_value = None

#                     # If encoder exists, try to decode class -> label (best-effort)
#                     if encoder is not None:
#                         try:
#                             # encoder likely expects integer class index or encoded label
#                             # try both approaches
#                             try:
#                                 predicted_label = encoder.inverse_transform([int(round(predicted_value))])[0]
#                             except Exception:
#                                 predicted_label = encoder.inverse_transform([int(round(float(y)))])[0]
#                         except Exception:
#                             predicted_label = None

#                     # If no encoder or decoding failed, map numeric -> category
#                     if not predicted_label:
#                         if predicted_value is not None:
#                             predicted_label = _simple_category_from_aqi(predicted_value)
#                         else:
#                             # final fallback: stringified model output (if meaningful)
#                             predicted_label = str(y)

#                 except Exception as e_model:
#                     # model failed -> fallback deterministic
#                     st.info(f"Model predict failed, using deterministic fallback: {e_model}")
#                     model = None

#             if model is None:
#                 # Deterministic fallback: weighted pseudo-AQI
#                 weights = {"PM2.5": 0.35, "PM10": 0.25, "NO2": 0.20, "SO2": 0.07, "CO": 0.05, "Ozone": 0.08}
#                 vals = {"PM2.5": p5_pm25, "PM10": p5_pm10, "NO2": p5_no2, "SO2": p5_so2, "CO": p5_co, "Ozone": p5_ozone}
#                 weighted = sum(float(vals[k]) * weights[k] for k in weights)
#                 predicted_value = float(max(0.0, min(weighted, 500.0)))
#                 predicted_label = _simple_category_from_aqi(predicted_value)

#             # Save authoritative prediction and inputs (these are the canonical Page-5 outputs)
#             st.session_state["last_inputs"] = {
#                 "PM2.5": float(p5_pm25), "PM10": float(p5_pm10), "NO2": float(p5_no2),
#                 "SO2": float(p5_so2),   "CO": float(p5_co),      "Ozone": float(p5_ozone)
#             }
#             st.session_state["last_prediction"] = {
#                 "category": str(predicted_label),
#                 "value": float(predicted_value) if predicted_value is not None else None,
#                 "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }

#             # IMPORTANT: do NOT overwrite global present/custom session values here.
#             # st.session_state.values is intentionally left unchanged to maintain independence.

#             # Show only category (no numeric AQI)
#             st.success(f"**Predicted AQI Category:** {predicted_label}")
#             st.caption("Prediction saved. Go to Page 6 to compare Predicted Levels with Delhi Avg & WHO limits.")

#         except Exception as e:
#             st.error(f"Prediction failed: {e}")



# ---------------- Page 5: Predict AQI ----------------
# import streamlit as st
# import joblib

# # Load Model + Encoder
# elif page.startswith("5)"):

#     rf_model = joblib.load("aqi_rf_model.joblib")
#     label_encoder = joblib.load("label_encoder.joblib")
    
#     st.title("🔮 Delhi AQI Prediction")
    
#     st.markdown("This page predicts **Delhi AQI Category** using your trained ML model. "
#                 "All major pollutants (PM2.5, PM10, NO2, SO2, etc.) are considered.")
    
#     # Example input (average values from dataset)
#     pm25 = 120  
#     pm10 = 180  
#     no2 = 60  
#     so2 = 20  
    
#     # Prepare features
#     features = [[pm25, pm10, no2, so2]]
    
#     # Predict button
#     if st.button("🚀 Predict AQI"):
#         prediction = rf_model.predict(features)
#         category = label_encoder.inverse_transform(prediction)[0]
    
#         # Save results to session_state
#         st.session_state.prediction_result = {
#             "AQI Value": prediction[0],
#             "Category": category,
#             "Inputs": {
#                 "PM2.5": pm25,
#                 "PM10": pm10,
#                 "NO2": no2,
#                 "SO2": so2
#             }
#         }
    
#         # Switch to next page
#         switch_page("Prediction Results")


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







# # ──────────────────────────────
# # # 6) COMPARE WITH DELHI AVERAGES & WHO LIMITS
# # ──────────────────────────────
# elif page.startswith("6)"):

#     st.title("📊 Compare Your Levels with Delhi Averages & WHO Limits")

#     values = normalize_values(st.session_state.values)
#     df_cmp = comparison_frame(values)

#     st.dataframe(df_cmp, use_container_width=True)
 
#     st.markdown("#### Visual Comparison")
#     df_long = df_cmp.melt(id_vars="Pollutant", var_name="Metric", value_name="Level")

#     for p in COLUMNS:
#         sub = df_long[df_long["Pollutant"] == p].set_index("Metric")["Level"]
#         st.markdown(f"**{p}**")
#         st.bar_chart(sub, use_container_width=True)

#     st.info("Tip: Aim to keep each pollutant at or below the WHO guideline when possible.")

# # ──────────────────────────────
# # FOOTER
# # ──────────────────────────────
# st.markdown("---")
# st.caption("© 2025 Delhi AQI App • Built with Streamlit • Clean single-router build") 



# # 6) COMPARE WITH DELHI AVERAGES & WHO LIMITS
# # ──────────────────────────────
# elif page.startswith("6)"):

#     st.title("📊 Compare Predicted Levels with Delhi Averages & WHO Limits")

#     # 🔹 Ensure values exist
#     if "values" not in st.session_state or "predicted_aqi" not in st.session_state:
#         st.error("⚠️ No predicted values found. Please complete Step 5 first.")
#         st.stop()

#     # Fetch values from session
#     values = st.session_state.values  
#     aqi_val = st.session_state.predicted_aqi
#     aqi_label = st.session_state.predicted_label

#     # Show predicted AQI (carried from Page 5)
#     st.metric(label="Predicted AQI", value=f"{aqi_val} ({aqi_label})")

#     # Build comparison frame and rename "Your Level" → "Predicted Level"
#     df_cmp = comparison_frame(values).rename(columns={"Your Level": "Predicted Level"})

#     # Display table
#     st.dataframe(df_cmp.set_index("Pollutant"), use_container_width=True)

#     st.markdown("#### Visual Comparison")
#     # Melt for plotting
#     df_long = df_cmp.melt(
#         id_vars="Pollutant",
#         value_vars=["Predicted Level", "Delhi Avg", "WHO Limit"],
#         var_name="Metric",
#         value_name="Level"
#     )

#     for p in COLUMNS:
#         sub = (
#             df_long[df_long["Pollutant"] == p]
#             .set_index("Metric")["Level"]
#             .reindex(["Predicted Level", "Delhi Avg", "WHO Limit"])
#         )
#         st.markdown(f"**{p}**")
#         st.bar_chart(sub, use_container_width=True)

#     st.info("Tip: Aim to keep each pollutant at or below the WHO guideline when possible.")

# # ──────────────────────────────
# # FOOTER
# # ──────────────────────────────
# st.markdown("---")
# st.caption("© 2025 Delhi AQI App • Built with Streamlit • Clean single-router build")


# # 6) COMPARE WITH DELHI AVERAGES & WHO LIMITS
# ──────────────────────────────
# elif page.startswith("6)"):

#     st.title("📊 Compare Predicted Levels with Delhi Averages & WHO Limits")

#     # --- 1) Get the same pollutant inputs used on Page 5 (persisted in session) ---
#     # Try to reuse existing values; if unavailable, fall back to the Page-5 defaults.
#     values = st.session_state.get("values")
#     if not isinstance(values, dict) or not values:
#         try:
#             # if your app defines ensure_session_defaults(), use it
#             ensure_session_defaults()
#             values = st.session_state.values
#         except Exception:
#             # Safe fallback matching your Page 5 defaults
#             values = {"PM2.5": 80.0, "PM10": 120.0, "NO2": 40.0, "SO2": 10.0, "CO": 1.0, "Ozone": 50.0}

#     # --- 2) Show predicted AQI category based on same inputs (no numeric value shown) ---
#     predicted_category = None
#     try:
#         model, encoder = load_model_and_encoder()
#     except Exception:
#         model, encoder = None, None

#     # Helper to map numeric AQI → category (local, so Page 6 works standalone)
#     def _simple_category_from_aqi(aqi: float) -> str:
#         try:
#             aqi = float(aqi)
#         except Exception:
#             return "Unknown"
#         if aqi <= 50:   return "Good"
#         if aqi <= 100:  return "Satisfactory"
#         if aqi <= 200:  return "Moderate"
#         if aqi <= 300:  return "Poor"
#         if aqi <= 400:  return "Very Poor"
#         return "Severe"

#     # Try predicting category using your model/encoder; fall back gracefully if anything fails.
#     try:
#         cols = COLUMNS if "COLUMNS" in globals() else ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
#         X = [[float(values.get(c, 0.0)) for c in cols]]
#         if model is not None:
#             pred_val = model.predict(X)[0]
#             if encoder is not None:
#                 # If encoder encodes categories by AQI bins
#                 predicted_category = encoder.inverse_transform([int(round(float(pred_val)))])[0]
#             else:
#                 predicted_category = _simple_category_from_aqi(pred_val)
#         else:
#             # No model available → infer category from a simple heuristic on inputs
#             # (keeps UI working without crashing)
#             weighted = 0.35*values["PM2.5"] + 0.25*values["PM10"] + 0.20*values["NO2"] + \
#                        0.07*values["SO2"] + 0.05*values["CO"] + 0.08*values["Ozone"]
#             predicted_category = _simple_category_from_aqi(weighted)
#     except Exception:
#         # Final fallback if anything above fails
#         weighted = 0.35*values.get("PM2.5",0) + 0.25*values.get("PM10",0) + 0.20*values.get("NO2",0) + \
#                    0.07*values.get("SO2",0) + 0.05*values.get("CO",0) + 0.08*values.get("Ozone",0)
#         predicted_category = _simple_category_from_aqi(weighted)

#     # Show only the category (as you requested)
#     if predicted_category:
#         st.success(f"**Predicted AQI Category:** {predicted_category}")

#     # --- 3) Build comparison table: replace "Your Level" → "Predicted Level" ---
#     try:
#         df_cmp = comparison_frame(values).rename(columns={"Your Level": "Predicted Level"})
#     except Exception:
#         # Robust fallback if comparison_frame isn't available
#         try:
#             delhi = DELHI_AVG if "DELHI_AVG" in globals() else {}
#             who   = WHO_LIMITS if "WHO_LIMITS" in globals() else {}
#             rows = []
#             cols = COLUMNS if "COLUMNS" in globals() else ["PM2.5","PM10","NO2","SO2","CO","Ozone"]
#             for p in cols:
#                 rows.append({
#                     "Pollutant": p,
#                     "Predicted Level": float(values.get(p, 0.0)),
#                     "Delhi Avg": float(delhi.get(p, 0.0)),
#                     "WHO Limit": float(who.get(p, float("inf"))),
#                 })
#             df_cmp = pd.DataFrame(rows)
#         except Exception as e:
#             st.error(f"Could not build comparison table: {e}")
#             st.stop()

#     st.dataframe(df_cmp.set_index("Pollutant"), use_container_width=True)

#     # --- 4) Visual Comparison (bar charts per pollutant) ---
#     st.markdown("#### Visual Comparison")
#     df_long = df_cmp.melt(
#         id_vars="Pollutant",
#         value_vars=["Predicted Level", "Delhi Avg", "WHO Limit"],
#         var_name="Metric",
#         value_name="Level"
#     )

#     # Use the COLUMNS order if available; else derive from the dataframe
#     _pollutants = (COLUMNS if "COLUMNS" in globals() else list(df_cmp["Pollutant"]))
#     for p in _pollutants:
#         sub = (
#             df_long[df_long["Pollutant"] == p]
#             .set_index("Metric")["Level"]
#             .reindex(["Predicted Level", "Delhi Avg", "WHO Limit"])
#         )
#         st.markdown(f"**{p}**")
#         st.bar_chart(sub, use_container_width=True)

#     st.info("Tip: Aim to keep each pollutant at or below the WHO guideline whenever possible.")


# # 6) COMPARE WITH DELHI AVERAGES & WHO LIMITS
# ──────────────────────────────
# elif page.startswith("6)"):

#     st.title("📊 Compare Predicted Levels with Delhi Averages & WHO Limits")

#     # 1) Get the same pollutant inputs used on Page 5 (persist in session)
#     #    If not present, try to create safe defaults so this page never breaks.
#     values = st.session_state.get("values")
#     if not isinstance(values, dict) or not values:
#         try:
#             ensure_session_defaults()
#             values = st.session_state.values
#         except Exception:
#             values = {"PM2.5": 80.0, "PM10": 120.0, "NO2": 40.0, "SO2": 10.0, "CO": 1.0, "Ozone": 50.0}

#     # 2) Show the predicted AQI CATEGORY derived from the same inputs as Page 5
#     #    (No numeric AQI shown.)
#     #    We prefer to reuse a label saved by Page 5 if you added it; if not, we compute it here.
#     predicted_label = st.session_state.get("predicted_label")

#     # local fallback categorizer (in case label not stored and no encoder)
#     def _simple_category_from_aqi(aqi: float) -> str:
#         try:
#             aqi = float(aqi)
#         except Exception:
#             return "Unknown"
#         if aqi <= 50:   return "Good"
#         if aqi <= 100:  return "Satisfactory"
#         if aqi <= 200:  return "Moderate"
#         if aqi <= 300:  return "Poor"
#         if aqi <= 400:  return "Very Poor"
#         return "Severe"

#     if not predicted_label:
#         # Try to reconstruct the category using your model + encoder on the same inputs
#         try:
#             model, encoder = load_model_and_encoder()
#         except Exception:
#             model, encoder = None, None

#         try:
#             cols = COLUMNS if "COLUMNS" in globals() else ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
#             X = [[float(values.get(c, 0.0)) for c in cols]]
#             if model is not None:
#                 pred_val = model.predict(X)[0]
#                 if encoder is not None:
#                     predicted_label = encoder.inverse_transform([int(round(float(pred_val)))])[0]
#                 else:
#                     predicted_label = _simple_category_from_aqi(pred_val)
#             else:
#                 # Heuristic fallback if no model is available
#                 weighted = (
#                     0.35*values.get("PM2.5",0) + 0.25*values.get("PM10",0) + 0.20*values.get("NO2",0) +
#                     0.07*values.get("SO2",0)  + 0.05*values.get("CO",0)   + 0.08*values.get("Ozone",0)
#                 )
#                 predicted_label = _simple_category_from_aqi(weighted)
#         except Exception:
#             predicted_label = None

#     if predicted_label:
#         st.success(f"**Predicted AQI Category:** {predicted_label}")
#     else:
#         st.info("Predicted category not found. Please run prediction in Step 5 once.")

#     # 3) Build the comparison table based on the **same inputs used to predict**
#     #    Replace "Your Level" → "Predicted Level"
#     try:
#         df_cmp = comparison_frame(values).rename(columns={"Your Level": "Predicted Level"})
#     except Exception:
#         # Robust fallback if comparison_frame is unavailable
#         try:
#             delhi = DELHI_AVG if "DELHI_AVG" in globals() else {}
#             who   = WHO_LIMITS if "WHO_LIMITS" in globals() else {}
#             cols = COLUMNS if "COLUMNS" in globals() else ["PM2.5","PM10","NO2","SO2","CO","Ozone"]
#             rows = []
#             for p in cols:
#                 rows.append({
#                     "Pollutant": p,
#                     "Predicted Level": float(values.get(p, 0.0)),  # the levels that produced the prediction
#                     "Delhi Avg": float(delhi.get(p, 0.0)),
#                     "WHO Limit": float(who.get(p, float("inf"))),
#                 })
#             df_cmp = pd.DataFrame(rows)
#         except Exception as e:
#             st.error(f"Could not build comparison table: {e}")
#             st.stop()

#     st.dataframe(df_cmp.set_index("Pollutant"), use_container_width=True)

#     # 4) Visual comparison — keep order consistent and show "Predicted Level" vs baselines
#     st.markdown("#### Visual Comparison")
#     df_long = df_cmp.melt(
#         id_vars="Pollutant",
#         value_vars=["Predicted Level", "Delhi Avg", "WHO Limit"],
#         var_name="Metric",
#         value_name="Level"
#     )

#     pollutants_order = (COLUMNS if "COLUMNS" in globals() else list(df_cmp["Pollutant"]))
#     for p in pollutants_order:
#         sub = (
#             df_long[df_long["Pollutant"] == p]
#             .set_index("Metric")["Level"]
#             .reindex(["Predicted Level", "Delhi Avg", "WHO Limit"])
#         )
#         st.markdown(f"**{p}**")
#         st.bar_chart(sub, use_container_width=True)

#     st.info("Tip: Aim to keep each pollutant at or below the WHO guideline whenever possible.")

# -------------------------
# Page 6: Compare with Delhi Avg & WHO Limits
# -------------------------
# elif page.startswith("6)"):
#     import pandas as pd
#     import numpy as np

#     st.title("📊 Compare Predicted Levels with Delhi Averages & WHO Limits")

#     # Ensure we have the saved Page-5 prediction and inputs
#     last_inputs = st.session_state.get("last_inputs")
#     last_prediction = st.session_state.get("last_prediction")

#     if not last_inputs or not last_prediction:
#         st.warning("⚠️ No previous prediction found. Please complete Step 5 (Predict) first and then return to this page.")
#         st.stop()

#     # Show the predicted AQI category (carried from Page 5). No numeric AQI shown unless you want it.
#     pred_cat = last_prediction.get("category", "Unknown")
#     st.success(f"**Predicted AQI Category (from Step 5):** {pred_cat}")
#     # Optional small timestamp
#     if last_prediction.get("time"):
#         st.caption(f"Predicted at: {last_prediction['time']}")

#     # Build comparison dataframe with three columns: Predicted Level / Delhi Avg / WHO Limit
#     cols = COLUMNS if "COLUMNS" in globals() else ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
#     delhi = DELHI_AVG if "DELHI_AVG" in globals() else {}
#     who = WHO_LIMITS if "WHO_LIMITS" in globals() else {}

#     rows = []
#     for p in cols:
#         pred_lvl = float(last_inputs.get(p, 0.0))
#         delhi_lvl = float(delhi.get(p, np.nan))
#         who_lvl = float(who.get(p, np.nan)) if p in who else np.nan
#         rows.append({"Pollutant": p, "Predicted Level": pred_lvl, "Delhi Avg": delhi_lvl, "WHO Limit": who_lvl})

#     df_cmp = pd.DataFrame(rows)

#     # Display table (set Pollutant as index)
#     st.dataframe(df_cmp.set_index("Pollutant"), use_container_width=True)

#     # Visual comparison: melt and plot per pollutant
#     st.markdown("#### Visual Comparison (Predicted Level vs Delhi Avg vs WHO Limit)")
#     df_long = df_cmp.melt(id_vars=["Pollutant"], value_vars=["Predicted Level", "Delhi Avg", "WHO Limit"],
#                           var_name="Metric", value_name="Level")

#     # Plot each pollutant separately to make differences clear
#     for p in cols:
#         sub = df_long[df_long["Pollutant"] == p].set_index("Metric")["Level"]
#         # reindex ensures consistent order (some values may be NaN)
#         sub = sub.reindex(["Predicted Level", "Delhi Avg", "WHO Limit"])
#         st.markdown(f"**{p}**")
#         st.bar_chart(sub, use_container_width=True)

#     # Quick interpretation: warn if Predicted Level > WHO Limit (where WHO available)
#     st.markdown("### ⚠️ Quick interpretation")
#     any_exceed = False
#     for _, r in df_cmp.iterrows():
#         p = r["Pollutant"]
#         pred_lvl = r["Predicted Level"]
#         who_lvl = r["WHO Limit"]
#         delhi_lvl = r["Delhi Avg"]
#         if not np.isnan(who_lvl) and pred_lvl > who_lvl:
#             st.error(f"**{p}** — Predicted level {pred_lvl} exceeds WHO limit {who_lvl}. Take precautions.")
#             any_exceed = True
#         elif not np.isnan(delhi_lvl) and pred_lvl > delhi_lvl:
#             st.warning(f"**{p}** — Predicted level {pred_lvl} is above Delhi average ({delhi_lvl}).")
#         else:
#             st.success(f"**{p}** — Predicted level {pred_lvl} is at/under Delhi avg and WHO where available.")

#     if not any_exceed:
#         st.info("None of the predicted pollutant levels exceed WHO limits (based on provided WHO values).")

#     # Download CSV of comparison
#     csv_bytes = df_cmp.to_csv(index=False).encode("utf-8")
#     st.download_button("⬇️ Download Predicted vs Delhi Avg & WHO (CSV)", data=csv_bytes, file_name="predicted_vs_delhi_who.csv", mime="text/csv")


# -------------------------
# Page 6: Compare with Delhi Avg & WHO Limits (uses Page-5 saved prediction only)
# -------------------------
# elif page.startswith("6)"):
#     import pandas as pd
#     import numpy as np

#     st.title("📊 Compare Predicted Levels with Delhi Averages & WHO Limits")

#     # 1) Read saved Page-5 outputs only (do NOT read present/custom inputs)
#     last_inputs = st.session_state.get("last_inputs")
#     last_prediction = st.session_state.get("last_prediction")

#     if not last_inputs or not last_prediction:
#         st.warning("⚠️ No previous prediction found. Please complete Step 5 (Predict) first; Page 6 reads the saved prediction only.")
#         st.stop()

#     # 2) Show the predicted AQI CATEGORY (no numeric AQI shown unless you opt-in)
#     pred_cat = last_prediction.get("category", "Unknown")
#     st.success(f"**Predicted AQI Category (from Step 5):** {pred_cat}")
#     if last_prediction.get("time"):
#         st.caption(f"Predicted at: {last_prediction['time']}")

#     # 3) Build comparison table using the exact Page-5 inputs (Predicted Level)
#     cols = COLUMNS if "COLUMNS" in globals() else ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
#     delhi = DELHI_AVG if "DELHI_AVG" in globals() else {}
#     who = WHO_LIMITS if "WHO_LIMITS" in globals() else {}

#     rows = []
#     for p in cols:
#         pred_lvl = float(last_inputs.get(p, 0.0))
#         delhi_lvl = float(delhi.get(p, np.nan)) if p in delhi else np.nan
#         who_lvl = float(who.get(p, np.nan)) if p in who else np.nan
#         rows.append({"Pollutant": p, "Predicted Level": pred_lvl, "Delhi Avg": delhi_lvl, "WHO Limit": who_lvl})

#     df_cmp = pd.DataFrame(rows)

#     # 4) Display table & visual comparison
#     st.dataframe(df_cmp.set_index("Pollutant"), use_container_width=True)

#     st.markdown("#### Visual Comparison (Predicted Level vs Delhi Avg vs WHO Limit)")
#     df_long = df_cmp.melt(id_vars=["Pollutant"], value_vars=["Predicted Level", "Delhi Avg", "WHO Limit"],
#                           var_name="Metric", value_name="Level")

#     for p in cols:
#         sub = df_long[df_long["Pollutant"] == p].set_index("Metric")["Level"]
#         sub = sub.reindex(["Predicted Level", "Delhi Avg", "WHO Limit"])
#         st.markdown(f"**{p}**")
#         st.bar_chart(sub, use_container_width=True)

#     # 5) Quick interpretation: warn if Predicted Level > WHO Limit (where WHO available)
#     st.markdown("### ⚠️ Quick interpretation")
#     any_exceed = False
#     for _, r in df_cmp.iterrows():
#         p = r["Pollutant"]
#         pred_lvl = r["Predicted Level"]
#         who_lvl = r["WHO Limit"]
#         delhi_lvl = r["Delhi Avg"]

#         if not np.isnan(who_lvl) and pred_lvl > who_lvl:
#             st.error(f"**{p}** — Predicted level {pred_lvl} exceeds WHO limit {who_lvl}. Take precautions.")
#             any_exceed = True
#         elif (not np.isnan(delhi_lvl)) and pred_lvl > delhi_lvl:
#             st.warning(f"**{p}** — Predicted level {pred_lvl} is above Delhi average ({delhi_lvl}).")
#         else:
#             st.success(f"**{p}** — Predicted level {pred_lvl} is at/under Delhi avg and WHO (where available).")

#     if not any_exceed:
#         st.info("None of the predicted pollutant levels exceed WHO limits (based on provided WHO values).")

#     # 6) Download CSV of the exact Page-5 prediction vs baselines
#     csv_bytes = df_cmp.to_csv(index=False).encode("utf-8")
#     st.download_button("⬇️ Download Predicted vs Delhi Avg & WHO (CSV)", data=csv_bytes,
#                        file_name="predicted_vs_delhi_who.csv", mime="text/csv")



elif page.startswith("6)"):

    import pandas as pd
    import numpy as np
    import altair as alt
    import datetime

    st.title("📊 Compare Predicted AQI with Delhi Avg & WHO Limits")

    # 1) Load Page-5 prediction from session_state
    predicted_values = st.session_state.get("predicted_values")
    predicted_label = st.session_state.get("predicted_label")
    prediction_time = st.session_state.get("prediction_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not predicted_values or not predicted_label:
        st.warning("⚠️ No predicted values found. Please complete Page 5 (Predict AQI) first.")
        st.stop()

    st.success(f"**Predicted AQI Category (from Step 5): {predicted_label}**")
    st.caption(f"Prediction made at: {prediction_time}")

    # 2) Build comparison DataFrame
    cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    DELHI_AVG = {"PM2.5": 95, "PM10": 180, "NO2": 60, "SO2": 20, "CO": 1.2, "O3": 50}
    WHO_LIMITS = {"PM2.5": 25, "PM10": 50, "NO2": 40, "SO2": 20, "CO": 4, "O3": 100}

    df_cmp = pd.DataFrame([
        {
            "Pollutant": p,
            "Predicted Level": float(predicted_values.get(p, 0.0)),
            "Delhi Avg": float(DELHI_AVG.get(p, np.nan)),
            "WHO Limit": float(WHO_LIMITS.get(p, np.nan))
        } for p in cols
    ])

    # 3) Display table
    st.dataframe(df_cmp.set_index("Pollutant"), use_container_width=True)

    # 4) Visualization using Altair
    st.markdown("#### Visual Comparison: Predicted vs Delhi Avg vs WHO")
    df_long = df_cmp.melt(id_vars="Pollutant", var_name="Metric", value_name="Level")

    chart = alt.Chart(df_long).mark_bar().encode(
        x=alt.X('Pollutant:N', title='Pollutant'),
        y=alt.Y('Level:Q', title='Concentration'),
        color=alt.Color('Metric:N', scale=alt.Scale(scheme='set1')),
        tooltip=['Pollutant', 'Metric', 'Level']
    ).properties(width=700, height=400)

    st.altair_chart(chart, use_container_width=True)

    # 5) Quick interpretation: check predicted vs WHO and Delhi Avg
    st.markdown("### ⚠️ Quick Interpretation")
    any_exceed = False
    for _, row in df_cmp.iterrows():
        p = row["Pollutant"]
        pred = row["Predicted Level"]
        delhi = row["Delhi Avg"]
        who = row["WHO Limit"]

        if not np.isnan(who) and pred > who:
            st.error(f"**{p}** — Predicted level {pred} exceeds WHO limit {who}. Take precautions!")
            any_exceed = True
        elif not np.isnan(delhi) and pred > delhi:
            st.warning(f"**{p}** — Predicted level {pred} is above Delhi average ({delhi}).")
        else:
            st.success(f"**{p}** — Predicted level {pred} is within Delhi Avg and WHO limit.")

    if not any_exceed:
        st.info("✅ None of the predicted pollutant levels exceed WHO limits.")

    # 6) Download CSV
    csv_bytes = df_cmp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Predicted vs Delhi Avg & WHO (CSV)",
        data=csv_bytes,
        file_name="predicted_vs_delhi_who.csv",
        mime="text/csv"
    )



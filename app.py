import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# ─── Page Config ───
st.set_page_config(
    page_title="Breathe Easy — AQI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #171725, #000000 70%);
        color: #e0e0e0;
    }
    
    .stSidebar [data-testid="stSidebar"] {
        background-color: rgba(14, 17, 23, 0.5);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8em;
        margin-bottom: 0px;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 210, 255, 0.15);
        border: 1px solid rgba(0, 210, 255, 0.3);
    }
    
    .aqi-card {
        background: rgba(20, 20, 30, 0.6);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        animation: fadein 0.8s ease;
    }
    
    @keyframes fadein {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .aqi-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
        pointer-events: none;
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .aqi-value {
        font-size: 6em;
        font-weight: 800;
        text-shadow: 0 0 25px currentColor;
        line-height: 1;
        margin: 10px 0;
    }
    
    .health-card {
        background: rgba(30,30,40,0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 25px;
        margin-top: 25px;
        border-left: 5px solid;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s;
    }
    
    .health-card:hover {
        transform: translateX(5px);
        background: rgba(40,40,50,0.6);
    }
    
    /* Button magic */
    .stButton > button {
        background: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1.1em;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 25px rgba(0, 210, 255, 0.6);
    }
    
    .stSlider > div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
    }
</style>
""", unsafe_allow_html=True)

# ─── AQI Helpers ───
AQI_BUCKETS = {
    (0, 50): ("Good", "#00e676", "🟢", "Air quality is satisfactory, and air pollution poses little or no risk."),
    (51, 100): ("Satisfactory", "#aeea00", "🟡", "Minor breathing discomfort to sensitive people."),
    (101, 200): ("Moderate", "#ffb300", "🟠", "Breathing discomfort to the people with lungs, asthma and heart diseases."),
    (201, 300): ("Poor", "#ff3d00", "🔴", "Breathing discomfort to most people on prolonged exposure."),
    (301, 400): ("Very Poor", "#f50057", "🚨", "Respiratory illness on prolonged exposure. Limit outdoor activity strictly."),
    (401, 500): ("Severe", "#d50000", "💀", "Health warning of emergency conditions: everyone is more likely to be affected.")
}

def get_aqi_info(aqi):
    for (lo, hi), info in AQI_BUCKETS.items():
        if lo <= aqi <= hi:
            return info
    if aqi > 500:
        return ("Severe", "#d50000", "💀", "Health emergency! AQI exceeds measurement scale.")
    return ("Unknown", "#808080", "❓", "Unable to classify.")


@st.cache_resource
def load_model():
    model = joblib.load('models/best_xgboost_tuned.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, scaler, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv('processed/processed_city_day.csv', parse_dates=['Date'])

@st.cache_data
def load_results():
    return pd.read_csv('models/model_results.csv')


# ─── Load everything ───
try:
    model, scaler, feature_cols = load_model()
    df = load_data()
    results_df = load_results()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Models not found. Error: {e}")

# ─── Sidebar ───
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 0;'>🌍 Breathe Easy</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #888;'>AI Air Quality Predictor</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["🏠 Home", "🔮 Predict AQI", "📊 Historical Trends", "🏆 Model Insights"], index=0)

# ════════════════════════════════════════════
#  HOME PAGE
# ════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("<h1 class='gradient-text'>🌍 Breathe Easy</h1>", unsafe_allow_html=True)
    st.markdown("#### Predict and understand India's Air Quality using dynamic Machine Learning")
    st.markdown("---")

    if model_loaded:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Data Points", f"{len(df):,}")
        col2.metric("🏙️ Cities Covered", df['City'].nunique())
        col3.metric("📅 Timeline", f"{df['Date'].dt.year.min()}–{df['Date'].dt.year.max()}")
        col4.metric("🤖 Optuna Models", len(results_df))

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("✨ Features")
        c1, c2, c3 = st.columns(3)
        c1.markdown("""
        <div style="background: rgba(255,255,255,0.02); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); height: 100%; transition: transform 0.3s; cursor: default;" onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
            <h2 style="margin-top: 0; color: #00d2ff; font-weight: 800;">1️⃣ Analyze</h2>
            <p style="color: #aaa; font-size: 1.1em; line-height: 1.6;">Input detailed pollutant concentrations including particulates (PM2.5), nitrogen oxides, and carbon chains to form a unique atmospheric profile.</p>
        </div>
        """, unsafe_allow_html=True)
        
        c2.markdown("""
        <div style="background: rgba(255,255,255,0.02); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); height: 100%; transition: transform 0.3s; cursor: default;" onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
            <h2 style="margin-top: 0; color: #00d2ff; font-weight: 800;">2️⃣ Predict</h2>
            <p style="color: #aaa; font-size: 1.1em; line-height: 1.6;">Our hyper-tuned XGBoost AI model predicts the exact AQI index using 20+ engineered temporal and interaction features in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        c3.markdown("""
        <div style="background: rgba(255,255,255,0.02); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); height: 100%; transition: transform 0.3s; cursor: default;" onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
            <h2 style="margin-top: 0; color: #00d2ff; font-weight: 800;">3️⃣ Advise</h2>
            <p style="color: #aaa; font-size: 1.1em; line-height: 1.6;">Receive instant visual health advisories and safety warnings based on verified government scales customized for sensitive groups.</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
#  PREDICT AQI PAGE
# ════════════════════════════════════════════
elif page == "🔮 Predict AQI" and model_loaded:
    st.markdown("<h1 class='gradient-text'>🔮 Predict Air Quality</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; font-size: 1.15em;'>Fine-tune individual ambient pollutant levels to instantly simulate AQI shifts.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([1.8, 1.2], gap="large")

    with col_left:
        cities = sorted(df['City'].unique())
        selected_city = st.selectbox("🏙️ Auto-fill from City Averages", cities, index=cities.index('Delhi') if 'Delhi' in cities else 0)

        city_latest = df[df['City'] == selected_city].sort_values('Date').iloc[-1]

        st.markdown("#### Primary Particulates")
        c1, c2 = st.columns(2)
        pm25 = c1.slider("PM2.5 (µg/m³)", 0.0, 500.0, float(city_latest.get('PM2.5', 50.0)), 1.0)
        pm10 = c2.slider("PM10 (µg/m³)", 0.0, 600.0, float(city_latest.get('PM10', 100.0)), 1.0)
        
        st.markdown("#### Key Gaseous Pollutants")
        c3, c4, c5 = st.columns(3)
        no2 = c3.slider("NO₂ (µg/m³)", 0.0, 200.0, float(city_latest.get('NO2', 30.0)), 0.5)
        so2 = c4.slider("SO₂ (µg/m³)", 0.0, 100.0, float(city_latest.get('SO2', 10.0)), 0.5)
        co = c5.slider("CO (mg/m³)", 0.0, 20.0, float(city_latest.get('CO', 1.0)), 0.1)
        
        with st.expander("Advanced Pollutants (O₃, NH₃, Volatiles)"):
            c6, c7, c8 = st.columns(3)
            o3 = c6.slider("O₃ (µg/m³)", 0.0, 200.0, float(city_latest.get('O3', 30.0)), 0.5)
            no = c7.slider("NO (µg/m³)", 0.0, 150.0, float(city_latest.get('NO', 10.0)), 0.5)
            nh3 = c8.slider("NH₃ (µg/m³)", 0.0, 100.0, float(city_latest.get('NH3', 15.0)), 0.5)

            c9, c10, c11 = st.columns(3)
            nox = c9.slider("NOx (ppb)", 0.0, 200.0, float(city_latest.get('NOx', 30.0)), 0.5)
            benzene = c10.slider("Benzene", 0.0, 50.0, float(city_latest.get('Benzene', 2.0)), 0.1)
            toluene = c11.slider("Toluene", 0.0, 100.0, float(city_latest.get('Toluene', 5.0)), 0.5)
            xylene = st.slider("Xylene", 0.0, 50.0, float(city_latest.get('Xylene', 1.0)), 0.1)

    with col_right:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🚀 Analyze & Predict AQI", use_container_width=True):
            template = city_latest.copy()
            template['PM2.5'] = pm25; template['PM10'] = pm10; template['NO'] = no
            template['NO2'] = no2; template['NOx'] = nox; template['NH3'] = nh3
            template['CO'] = co; template['SO2'] = so2; template['O3'] = o3
            template['Benzene'] = benzene; template['Toluene'] = toluene; template['Xylene'] = xylene

            template['PM25_PM10_product'] = pm25 * pm10
            template['NO2_NO_ratio'] = no2 / (no + 1e-6)
            template['PM25_PM10_ratio'] = pm25 / (pm10 + 1e-6)

            try:
                feature_values = [template[col] for col in feature_cols]
                feature_array = np.array(feature_values).reshape(1, -1)
                feature_scaled = scaler.transform(feature_array)
                prediction = model.predict(feature_scaled)[0]
                prediction = max(0, prediction)

                bucket, color, icon, advisory = get_aqi_info(prediction)

                st.markdown(f"""
                <div class="aqi-card" style="border-top: 4px solid {color};">
                    <div style="font-size: 1.1em; color: #aaa; text-transform: uppercase; letter-spacing: 2px;">Predicted Index</div>
                    <div class="aqi-value" style="color: {color};">{prediction:.0f}</div>
                    <div style="font-size: 2em; font-weight: 700; color: #fff; margin-bottom: 5px; text-shadow: 0 0 10px {color}aa;">{bucket}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="health-card" style="border-left-color: {color};">
                    <b style="font-size: 1.25em;">{icon} Health Advisory</b><br>
                    <span style="color: #d0d0d0; font-size: 1.1em; line-height: 1.5;">{advisory}</span>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.markdown("""
            <div style="text-align: center; color: #666; padding: 60px 40px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px; background: rgba(0,0,0,0.2);">
                <br><h3>Ready to Compute</h3><br>
                Adjust the precise pollutant metrics on the left and click predict to run the AI analysis.
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════
#  HISTORICAL TRENDS PAGE
# ════════════════════════════════════════════
elif page == "📊 Historical Trends" and model_loaded:
    st.markdown("<h1 class='gradient-text'>📊 Interactive Trends</h1>", unsafe_allow_html=True)
    st.markdown("---")

    cities = sorted(df['City'].unique())
    selected_cities = st.multiselect("Select cities to compare", cities,
                                      default=['Delhi', 'Mumbai', 'Bengaluru'] if 'Delhi' in cities else cities[:2])

    if selected_cities:
        filtered = df[df['City'].isin(selected_cities)]
        monthly = filtered.groupby([filtered['Date'].dt.to_period('M'), 'City'])['AQI'].mean().reset_index()
        monthly['Date'] = monthly['Date'].dt.to_timestamp()

        fig = px.line(monthly, x='Date', y='AQI', color='City',
                      template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", size=13),
                          margin=dict(l=20, r=20, t=30, b=20),
                          hovermode="x unified")
        st.markdown("### Monthly Average AQI Timeline")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Pollutant Distributions")
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2']
        avgs = filtered.groupby('City')[pollutant_cols].mean()

        fig2 = px.bar(avgs.reset_index().melt(id_vars='City'),
                       x='variable', y='value', color='City', barmode='group',
                       template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter", size=13))
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════
#  MODEL PERFORMANCE PAGE
# ════════════════════════════════════════════
elif page == "🏆 Model Insights" and model_loaded:
    st.markdown("<h1 class='gradient-text'>🏆 Algorithm Benchmarks</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Performance Metrics")
    st.dataframe(results_df.style.highlight_max(subset=['R2'], color='rgba(0,210,255,0.25)')
                                  .highlight_min(subset=['RMSE', 'MAE'], color='rgba(0,210,255,0.25)')
                                  .format({'R2': '{:.4f}', 'RMSE': '{:.2f}',
                                           'MAE': '{:.2f}', 'MAPE': '{:.2f}%'}),
                  use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = px.bar(results_df.sort_values('R2', ascending=True),
                  x='R2', y='Model', orientation='h',
                  template='plotly_dark',
                  color='R2', color_continuous_scale='Blues')
    fig.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", size=14),
                      title="R² Score Ranking")
    st.plotly_chart(fig, use_container_width=True)

# ─── Footer ───
st.markdown("---")
st.markdown("<center><p style='color: #555; font-size: 0.9em;'>Built with completely custom rendering for the <b>Breathe Easy</b> Project</p></center>", unsafe_allow_html=True)

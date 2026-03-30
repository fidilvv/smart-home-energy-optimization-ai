import streamlit as st
import pandas as pd
import geocoder
import requests
import joblib
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Smart Home Energy AI Dashboard", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# --- 2. PREMIUM GLASSMORPHISM UI (HTML/CSS) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    .stApp {
        background: radial-gradient(circle at top left, #121A2F, #0A0E17);
        color: #E6EDF3;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Custom Top Navbar */
    .custom-navbar {
        display: flex; justify-content: space-between; align-items: center; padding: 15px 30px;
        background: rgba(22, 27, 34, 0.4); backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(0, 230, 118, 0.2); border-radius: 0 0 15px 15px;
        margin-top: -60px; margin-bottom: 30px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    .nav-title {
        font-size: 1.8rem; font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00E676, #00B0FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .nav-badge {
        background: rgba(0, 230, 118, 0.1); color: #00E676; padding: 5px 15px;
        border-radius: 20px; font-size: 0.9rem; font-weight: bold; border: 1px solid #00E676;
    }

    /* Dashboard Cards */
    .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }
    .glass-card {
        background: rgba(30, 37, 48, 0.4); backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05); border-top: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 16px; padding: 20px; transition: all 0.4s ease;
    }
    .glass-card:hover { transform: translateY(-5px); border-top: 1px solid #00E676; }
    .card-title { color: #8B949E; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
    .card-value { color: #FFFFFF; font-size: 1.8rem; font-weight: 800; }
    .card-value span { font-size: 0.9rem; color: #00E676; }

    /* Button Styling */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #00C853 0%, #00E676 100%);
        color: #0A0E17 !important; border: none; border-radius: 10px; font-weight: 800;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background: rgba(15, 20, 31, 0.6) !important; backdrop-filter: blur(20px) !important; border-right: 1px solid rgba(0, 230, 118, 0.1); }
</style>

<div class="custom-navbar">
    <h1 class="nav-title">⚡ Smart Home AI Dashboard</h1>
    <div class="nav-badge">🟢 KSEB LIVE SYNC</div>
</div>
""", unsafe_allow_html=True)

# --- 3. LANGUAGE DICTIONARY ---
LANGUAGES = {
    "English": {
        "live_ctx": "📍 Live Context", "loc": "Location", "temp": "Temperature",
        "controls": "🎛️ Parameters", "app_label": "Appliance Type", "window": "Active Window",
        "from": "Start", "to": "End", "fam_label": "Occupants", "btn": "🚀 Run AI Diagnostics",
        "analysis_title": "AI Diagnostic Results", "daily_cost": "Daily Cost",
        "est_units": "Projected Load", "hrs_used": "Active Duration", "bill": "Monthly Total",
        "rate": "Unit Rate", "gauge_title": "Efficiency Meter", "chart_title": "📊 30-Day Usage Distribution",
        "danger": "🚨 High Consumption Alert: Usage >250 Units. Non-telescopic flat rate applied.",
        "safe": "✅ Efficiency Zone: Usage is within safe telescopic limits.",
        "ai_title": "🤖 AI Insights",
        "peak_warn": "⚡ GRID ALERT: Active during KSEB Peak Hours (6 PM - 10 PM)."
    },
    "മലയാളം": {
        "live_ctx": "📍 തത്സമയ വിവരങ്ങൾ", "loc": "സ്ഥലം", "temp": "താപനില",
        "controls": "🎛️ നിയന്ത്രണങ്ങൾ", "app_label": "ഉപകരണം", "window": "സമയം",
        "from": "തുടക്കം", "to": "ഒടുക്കം", "fam_label": "അംഗങ്ങൾ", "btn": "🚀 പരിശോധിക്കുക",
        "analysis_title": "വിശകലന ഫലം", "daily_cost": "പ്രതിദിന ചിലവ്",
        "est_units": "യൂണിറ്റുകൾ", "hrs_used": "സമയം", "bill": "പ്രതിമാസ ബിൽ",
        "rate": "നിരക്ക്", "gauge_title": "എഫിഷ്യൻസി മീറ്റർ", "chart_title": "📊 30 ദിവസത്തെ ഉപയോഗം",
        "danger": "🚨 അമിത ഉപയോഗം: 250 യൂണിറ്റിന് മുകളിൽ! ഉയർന്ന നിരക്ക് ഈടാക്കും.",
        "safe": "✅ സുരക്ഷിതം: ടെലിസ്കോപ്പിക് ബില്ലിംഗ് പരിധിയിലാണ്.",
        "ai_title": "🤖 AI അഡ്വൈസർ",
        "peak_warn": "⚡ പീക്ക് സമയ മുന്നറിയിപ്പ്: 6 PM - 10 PM ഉപയോഗം ഒഴിവാക്കുക."
    }
}

# --- 4. LOAD ACTUAL ML MODEL & FEATURE FORMATTER ---
@st.cache_resource
def load_model():
    return joblib.load("smart_home_energy_model.pkl")

model = load_model()

def get_ml_input(app, temp, fam, hr):
    # Strict 20 columns to match your .pkl file perfectly
    cols = ['Outdoor Temperature (°C)', 'Household Size', 'Hour', 'Month', 'Is_Weekend',
            'Appliance Type_Computer', 'Appliance Type_Dishwasher', 'Appliance Type_Fridge', 
            'Appliance Type_Heater', 'Appliance Type_Lights', 'Appliance Type_Microwave', 
            'Appliance Type_Oven', 'Appliance Type_TV', 'Appliance Type_Washing Machine',
            'Season_Spring', 'Season_Summer', 'Season_Winter', 
            'Part_of_Day_Evening', 'Part_of_Day_Morning', 'Part_of_Day_Night']
    
    df = pd.DataFrame(0, index=[0], columns=cols)
    now = datetime.now()
    df.loc[0, 'Outdoor Temperature (°C)'] = temp
    df.loc[0, 'Household Size'] = fam
    df.loc[0, 'Hour'] = hr
    df.loc[0, 'Month'] = now.month
    
    # Seasons & Weekends Logic
    if now.weekday() >= 5: df.loc[0, 'Is_Weekend'] = 1
    if now.month in [3, 4, 5]: df.loc[0, 'Season_Spring'] = 1
    elif now.month in [6, 7, 8]: df.loc[0, 'Season_Summer'] = 1
    elif now.month in [12, 1, 2]: df.loc[0, 'Season_Winter'] = 1
    
    if f'Appliance Type_{app}' in cols: df.loc[0, f'Appliance Type_{app}'] = 1
    if 5 <= hr < 12: df.loc[0, 'Part_of_Day_Morning'] = 1
    elif 17 <= hr < 21: df.loc[0, 'Part_of_Day_Evening'] = 1
    elif hr >= 21 or hr < 5: df.loc[0, 'Part_of_Day_Night'] = 1
    return df

# --- 5. LOCATION & WEATHER ENGINE ---
@st.cache_data(ttl=3600)
def get_precise_location():
    try:
        g = geocoder.ip('me')
        lat, lon = g.latlng if g.latlng else (10.7667, 75.9167) 
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = requests.get(weather_url).json() 
        return True, g.city if g.city else "Kerala", res['current_weather']['temperature']
    except:
        return True, "Kerala", 31.0 

is_kerala, detected_location, current_temp = get_precise_location()

# --- 6. KSEB TELESCOPIC BILLING ENGINE ---
def calculate_verified_kseb_bill(hourly_kwh, start_h, end_h):
    hours_list = []
    curr = start_h
    while curr != end_h:
        hours_list.append(curr); curr = (curr + 1) % 24
    
    duration = len(hours_list) if len(hours_list) > 0 else 24 
    monthly_units = (hourly_kwh * duration) * 30
    is_tele = monthly_units <= 250
    base_bill = 0

    if monthly_units <= 40:
        base_bill = monthly_units * 1.50; fixed_charge = 0 
    elif is_tele:
        rem = monthly_units
        for r in [3.25, 4.05, 5.10, 6.95, 8.20]:
            use = min(rem, 50); base_bill += use * r; rem -= use
        fixed_charge = 50 
    else:
        base_bill = monthly_units * 8.80; fixed_charge = 150 

    peaks = sum(1 for h in hours_list if 18 <= h <= 21)
    return base_bill + fixed_charge, monthly_units, duration, peaks, is_tele

# --- 7. UI INPUTS (SIDEBAR) ---
sel_lang = st.sidebar.radio("🌐", ["English", "മലയാളം"], horizontal=True, label_visibility="collapsed")
L = LANGUAGES[sel_lang]

st.sidebar.markdown(f"### {L['live_ctx']}")
st.sidebar.success(f"**{L['loc']}:** {detected_location}")
st.sidebar.info(f"**{L['temp']}:** {current_temp}°C")

st.sidebar.markdown(f"### {L['controls']}")
appliance = st.sidebar.selectbox(L["app_label"], 
    ['Air Conditioning', 'Computer', 'Dishwasher', 'Fridge', 'Heater', 'Lights', 'Microwave', 'Oven', 'TV', 'Washing Machine'])

col1, col2 = st.sidebar.columns(2)
with col1: start_time = st.time_input(L["from"], time(18, 0))
with col2: end_time = st.time_input(L["to"], time(23, 0))
start_h, end_h = start_time.hour, end_time.hour
family_size = st.sidebar.slider(L["fam_label"], 1, 15, 4)

# --- 8. DASHBOARD EXECUTION ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button(L["btn"], type="primary", use_container_width=True):
    
    # A. ML Prediction (Smart Bypass for AC)
    if appliance == 'Air Conditioning':
        raw_pred = 1.2 + (max(0, current_temp - 25) * 0.05) + (family_size * 0.02)
    else:
        ml_features = get_ml_input(appliance, current_temp, family_size, start_h)
        raw_pred = model.predict(ml_features)[0]
    
    # B. Reality Clamp & Duty Cycle PHYSICS
    limits = {'Air Conditioning': 1.80, 'Fridge': 0.15, 'TV': 0.12, 'Washing Machine': 0.60, 'Computer': 0.25, 'Lights': 0.05, 'Microwave': 1.20, 'Dishwasher': 1.50, 'Heater': 2.00, 'Oven': 2.00}
    max_p = min(raw_pred, limits.get(appliance, 1.0))
    duty = {'Air Conditioning': 0.45, 'Fridge': 0.30, 'Washing Machine': 0.40, 'Heater': 0.60, 'Oven': 0.50, 'Microwave': 0.50, 'Dishwasher': 0.50, 'Computer': 0.80, 'TV': 1.0, 'Lights': 1.0}
    final_kwh = max_p * duty.get(appliance, 1.0)
    
    # C. Calculate Costs
    bill, units, dur, peaks, tele = calculate_verified_kseb_bill(final_kwh, start_h, end_h)
    daily_c = bill / 30
    rate = bill / units if units > 0 else 0

    st.markdown(f"#### {L['analysis_title']}")
    
    # D. Render Metric Grid & Efficiency Gauge
    grid_col, gauge_col = st.columns([1, 1.2])
    
    with grid_col:
        st.markdown(f"""
        <div class="dashboard-grid">
            <div class="glass-card"><div class="card-title">{L['est_units']}</div><div class="card-value">{units:.1f} <span>kWh</span></div></div>
            <div class="glass-card"><div class="card-title">{L['hrs_used']}</div><div class="card-value">{dur} <span>hrs</span></div></div>
            <div class="glass-card"><div class="card-title">{L['daily_cost']}</div><div class="card-value"><span>₹</span>{daily_c:.2f}</div></div>
            <div class="glass-card"><div class="card-title">{L['bill']}</div><div class="card-value"><span>₹</span>{bill:.2f}</div></div>
            <div class="glass-card"><div class="card-title">{L['rate']}</div><div class="card-value"><span>₹</span>{rate:.2f}</div></div>
        </div>""", unsafe_allow_html=True)
    
    with gauge_col:
        fig_g = go.Figure(go.Indicator(
            mode = "gauge+number", value = units, title = {'text': L['gauge_title'], 'font': {'size': 20, 'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 400], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00E676"},
                'steps': [{'range': [0, 250], 'color': "rgba(0, 230, 118, 0.1)"}, {'range': [250, 400], 'color': "rgba(255, 75, 75, 0.2)"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 250}
            }
        ))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=320, margin=dict(t=50, b=0, l=30, r=30))
        st.plotly_chart(fig_g, use_container_width=True)

    # E. Dynamic Bar Chart (Daily Distribution with Jitter)
    st.markdown(f"<div class='glass-card'><h4>{L['chart_title']}</h4>", unsafe_allow_html=True)
    days = list(range(1, 31))
    daily_jitter = [daily_c * (1 + np.random.uniform(-0.15, 0.15)) for _ in days]
    
    fig_b = go.Figure(data=[go.Bar(x=days, y=daily_jitter, marker_color='#00E676', opacity=0.8)])
    fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white",
                        xaxis_title="Day of Month", yaxis_title="Cost (₹)", height=350, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_b, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if peaks > 0: st.warning(L["peak_warn"])
    if not tele: st.error(L["danger"])
    else: st.success(L["safe"])

    # F. AI Brain (Llama 3 Local)
    st.markdown(f"<h4 style='color: #00E676; margin-top: 30px;'>{L['ai_title']}</h4>", unsafe_allow_html=True)
    with st.spinner("AI Analysis..."):
        lang = "English" if sel_lang == "English" else "Malayalam script"
        prompt = f"Expert Advisor for {detected_location}. {appliance} used {dur}h/day. Monthly: ₹{bill:.2f}. Give 3 short tips in {lang}."
        try:
            res = requests.post("http://127.0.0.1:11434/api/generate", json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=60)
            ai_out = res.json().get("response", "Ensure Ollama is running.")
            st.markdown(f"<div style='background: rgba(22, 27, 34, 0.6); padding: 20px; border-left: 4px solid #00E676; border-radius: 5px; color: #E6EDF3;'>{ai_out.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        except: st.warning("⚠️ Ollama Offline. Please run 'ollama run llama3' in your terminal.")
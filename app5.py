"""
KisanMitra - Streamlit UI
Place this file in your project root (next to final_model.pkl, final_data.csv,
crop_reference.csv, state_weather.csv, models/ and data/) and run:

    streamlit run app.py

Pages (all connected with buttons, no sidebar needed):
    1. Home                 - landing page
    2. Crop Recommendation  - ranked crops for State / District / Season
    3. Crop Details         - weather fit + profit for one recommended crop
    4. Yield Prediction     - yield, production and earnings for a chosen crop
    5. Weather Explorer     - weather by state/season and which crops fit it
    6. Project Statistics   - dataset, model and weather statistics
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="KisanMitra", page_icon="🌾", layout="wide")
BASE = Path(__file__).parent

# =====================================================================
# STYLE
# =====================================================================
st.markdown("""
<style>
.block-container, [data-testid="stMainBlockContainer"] {padding-top: 4.5rem !important; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#1b5e20 0%,#43a047 55%,#9ccc65 100%);
       border-radius: 22px; padding: 48px 36px; color: #fff; text-align: center;
       box-shadow: 0 10px 30px rgba(27,94,32,.25); margin-bottom: 22px;}
.hero h1 {font-size: 3.2rem; margin: 0; color:#fff;}
.hero h3 {font-weight: 400; margin: 6px 0 14px; color:#f1f8e9;}
.hero p {max-width: 760px; margin: 0 auto; color:#f1f8e9; font-size: 1.05rem;}
.fcard {background:#fff; border:1px solid #e3ebe0; border-radius:18px; padding:22px;
        box-shadow:0 2px 8px rgba(0,0,0,.05); min-height: 170px;}
.fcard .ic {font-size:2.2rem;} .fcard h4 {margin:6px 0; color:#2e7d32;}
.fcard p {color:#5f6f60; margin:0; font-size:.93rem;}
.step {background:#f1f8e9; border-radius:14px; padding:16px; text-align:center; height:100%;}
.step b {color:#2e7d32; font-size:1.5rem;}
.kpi {background:#f1f8e9; border-radius:14px; padding:14px 10px; text-align:center;}
.kpi h2 {margin:0; color:#2e7d32;} .kpi span {color:#5f6f60; font-size:.85rem;}
.podium {border-radius:16px; padding:16px; text-align:center; border:2px solid #e3ebe0; background:#fff;}
.podium.first {border-color:#f9a825; background:#fffde7;}
.podium h3 {margin:2px 0; color:#1b5e20;}
.pill {display:inline-block; padding:2px 12px; border-radius:99px; font-weight:700; font-size:.82rem;}
.pill.High,.pill.Good,.pill.Profit {background:#c8e6c9; color:#1b5e20;}
.pill.Medium,.pill.Moderate {background:#ffe0b2; color:#a35a00;}
.pill.Low,.pill.Poor,.pill.Loss {background:#ffcdd2; color:#b71c1c;}
.note {color:#6b7a6c; font-size:.82rem;}
.wx {background:linear-gradient(135deg,#e3f2fd,#f1f8e9); border-radius:16px; padding:16px;}

@keyframes grad {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes floaty {0%,100%{transform:translateY(0) rotate(0deg)}50%{transform:translateY(-16px) rotate(6deg)}}
@keyframes scroll {from{transform:translateX(0)}to{transform:translateX(-50%)}}
@keyframes rise {from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
.hero2 {position:relative; overflow:hidden; border-radius:26px; padding:64px 40px 54px; text-align:center; color:#fff;
        background:linear-gradient(120deg,#1b5e20,#2e7d32,#66bb6a,#aed581,#2e7d32); background-size:300% 300%;
        animation:grad 14s ease infinite; box-shadow:0 14px 40px rgba(27,94,32,.3); margin-bottom:16px;}
.hero2 .fl {position:absolute; font-size:2.4rem; opacity:.28; animation:floaty 6s ease-in-out infinite;}
.hero2 .badge {display:inline-block; background:rgba(255,255,255,.2); padding:5px 16px; border-radius:99px;
               font-size:.85rem; letter-spacing:.5px; margin-bottom:14px; backdrop-filter:blur(4px);}
.hero2 h1 {font-size:4rem; margin:0; color:#fff; text-shadow:0 3px 14px rgba(0,0,0,.2); animation:rise .8s ease both;}
.hero2 h2 {font-weight:500; margin:6px 0 14px; color:#f9fbe7; font-size:1.6rem; animation:rise 1s ease both;}
.hero2 p {max-width:720px; margin:0 auto; color:#f1f8e9; font-size:1.05rem; animation:rise 1.2s ease both;}
.ticker {overflow:hidden; background:#1b5e20; color:#fff; border-radius:14px; padding:10px 0; margin-bottom:22px;}
.ticker .tr {display:inline-block; white-space:nowrap; animation:scroll 40s linear infinite;}
.ticker span {margin:0 28px; font-size:.95rem;}
.lift {background:#fff; border:1px solid #e3ebe0; border-radius:20px; padding:24px; min-height:190px;
       transition:all .25s ease; box-shadow:0 2px 8px rgba(0,0,0,.05);}
.lift:hover {transform:translateY(-6px); box-shadow:0 14px 30px rgba(46,125,50,.2); border-color:#66bb6a;}
.lift .ic {font-size:2.6rem;} .lift h4 {margin:8px 0 6px; color:#2e7d32; font-size:1.2rem;}
.lift p {color:#5f6f60; margin:0; font-size:.93rem;}
.season {border-radius:18px; padding:20px; color:#fff; min-height:150px; transition:transform .25s;}
.season:hover {transform:scale(1.03);}
.season h4 {margin:0; color:#fff; font-size:1.25rem;} .season small {opacity:.9;}
.season p {margin:8px 0 0; font-size:.88rem; opacity:.95;}
.kharif {background:linear-gradient(135deg,#0277bd,#26a69a);}
.rabi {background:linear-gradient(135deg,#ef6c00,#fbc02d);}
.zaid {background:linear-gradient(135deg,#d84315,#ff8a65);}
.fact {background:#fffde7; border-left:5px solid #f9a825; border-radius:12px; padding:14px 18px; margin:6px 0;}
.sec {font-size:1.7rem; font-weight:700; color:#1b5e20; margin:28px 0 4px;}
.sub {color:#6b7a6c; margin-bottom:14px;}
.cta {background:linear-gradient(135deg,#1b5e20,#43a047); color:#fff; border-radius:22px; padding:34px; text-align:center; margin-top:26px;}
.cta h2 {color:#fff; margin:0 0 6px;}
.vs {display:flex; gap:14px; flex-wrap:wrap;}
.vs div {flex:1; min-width:240px; border-radius:16px; padding:18px;}
.vs .bad {background:#ffebee; border:1px solid #ffcdd2;} .vs .good {background:#e8f5e9; border:1px solid #c8e6c9;}
.vs h4 {margin:0 0 8px;} .vs ul {margin:0; padding-left:18px; font-size:.92rem;}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# DATA + MODELS
# =====================================================================
ALIAS = {"andaman and nicobar island": "andaman and nicobar islands", "laddak": "ladakh",
         "laddakh": "ladakh", "the dadra and nagar haveli": "dadra and nagar haveli"}


def norm(s):
    s = str(s).strip().lower()
    return ALIAS.get(s, s)


@st.cache_resource(show_spinner="Loading models and data...")
def load():
    d = pd.read_csv(BASE / "final_data.csv")
    ref = pd.read_csv(BASE / "crop_reference.csv").set_index("Crop")
    sw = pd.read_csv(BASE / "state_weather.csv")
    yd = pd.read_csv(BASE / "data" / "clean_yield_data.csv")
    rain_n = pd.read_csv(BASE / "data" / "rain_normal.csv").set_index("State")["Rain_normal"].to_dict()
    ymodel = joblib.load(BASE / "models" / "yield_model.pkl")
    try:
        rmodel = joblib.load(BASE / "final_model.pkl")
        rerr = None
    except Exception as e:  # e.g. lightgbm missing
        rmodel, rerr = None, str(e)
    return d, ref, sw, yd, rain_n, ymodel, rmodel, rerr


d, ref, sw, yd, rain_normal, yield_model, rec_model, rec_err = load()
sw["_s"] = sw["State"].map(norm)

cats = {c: sorted(d[c].unique()) for c in ["State", "District", "Crop", "Season"]}
q = d.groupby("Crop")["Yield"].quantile([0.30, 0.60]).unstack()
q.columns = ["lo", "hi"]
next_year = int(d["Crop_Year"].max()) + 1

extra_ref = {
    "Arecanut": (45000, 150000, "High"), "Cashewnut": (12000, 60000, "Medium"),
    "Black Pepper": (50000, 150000, "High"), "Cardamom": (120000, 250000, "High"),
    "Cowpea(Lobia)": (5500, 22000, "Low"), "Guar Seed": (5500, 20000, "Low"),
    "Horse-Gram": (5000, 18000, "Low"), "Khesari": (4000, 18000, "Low"),
    "Moth": (6000, 18000, "Low"), "Tobacco": (15000, 80000, "High"),
}
low_conf = {"Cardamom", "Black Pepper", "Cashewnut", "Niger Seed", "Castor Seed",
            "Sesamum", "Tobacco", "Safflower", "Dry Chillies"}

# ---- WEATHER LOGIC (same as your original app) -----------------------
# crop: (min temp C, max temp C, min rain mm/season) - approximate, verify (ICAR / FAO EcoCrop)
ideal = {
    "Rice": (20, 35, 600), "Wheat": (10, 25, 100), "Maize": (18, 32, 300), "Gram": (15, 30, 60),
    "Rapeseed &Mustard": (10, 25, 60), "Groundnut": (22, 32, 300), "Soyabean": (20, 32, 400),
    "Sugarcane": (20, 35, 800), "Potato": (10, 25, 100), "Onion": (13, 30, 100),
    "Bajra": (25, 35, 250), "Jowar": (25, 35, 300), "Ragi": (20, 32, 300), "Barley": (10, 25, 80),
    "Urad": (22, 35, 300), "Moong(Green Gram)": (22, 35, 300), "Masoor": (10, 25, 60),
    "Arhar/Tur": (20, 35, 400), "Sunflower": (20, 32, 250), "Sesamum": (22, 35, 250),
    "Safflower": (15, 30, 100), "Niger Seed": (18, 30, 300), "Castor Seed": (20, 35, 300),
    "Linseed": (10, 25, 80), "Garlic": (10, 25, 100), "Ginger": (20, 32, 600),
    "Turmeric": (20, 32, 600), "Dry Chillies": (20, 32, 400), "Coriander": (15, 28, 100),
    "Sweet Potato": (20, 32, 400), "Tapioca": (22, 35, 600), "Banana": (20, 35, 800),
    "Arecanut": (20, 32, 1000), "Cashewnut": (22, 35, 600), "Black Pepper": (20, 32, 1000),
    "Cardamom": (15, 28, 1000), "Cowpea(Lobia)": (22, 35, 300), "Guar Seed": (25, 35, 250),
    "Horse-Gram": (22, 35, 250), "Khesari": (12, 28, 100), "Moth": (25, 38, 150),
    "Tobacco": (18, 32, 300),
}
ideal_lc = {k.lower(): (k, v) for k, v in ideal.items()}


def get_weather(state, season):
    w = sw[(sw["_s"] == norm(state)) & (sw["Season"].str.lower() == str(season).lower())]
    if w.empty:
        return None
    return {"temp": float(w["Temp"].iloc[0]), "rain": float(w["Rain"].iloc[0]),
            "hum": float(w["Humidity"].iloc[0])}


def weather_fit(crop, w):
    """Returns (fit label, (tmin,tmax,rmin), temp_ok, rain_ok) or None."""
    if w is None or crop.lower() not in ideal_lc:
        return None
    tmin, tmax, rmin = ideal_lc[crop.lower()][1]
    t_ok, r_ok = tmin <= w["temp"] <= tmax, w["rain"] >= rmin
    return ["Poor", "Moderate", "Good"][int(t_ok) + int(r_ok)], (tmin, tmax, rmin), t_ok, r_ok


def price_info(crop):
    for k in ref.index:
        if k.lower() == crop.lower() and pd.notna(ref.loc[k, "Price"]):
            return float(ref.loc[k, "Price"]), float(ref.loc[k, "Cost"]), ref.loc[k, "Effort"]
    for k, v in extra_ref.items():
        if k.lower() == crop.lower():
            return v
    return None, None, "N/A"


def inr(x):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"₹{x:,.0f}"


def cap(s):
    return str(s).title()


def pill(txt):
    return f"<span class='pill {txt}'>{txt}</span>"


# =====================================================================
# NAVIGATION
# =====================================================================
PAGES = [("home", "🏠 Home"), ("recommend", "🌱 Recommendation"), ("detail", "🔍 Crop Details"),
         ("yield", "📈 Yield Prediction"), ("weather", "🌦️ Weather Explorer"), ("stats", "📊 Statistics")]
for k, v in {"page": "home", "rows": None, "ctx": None, "sel": None, "yres": None}.items():
    st.session_state.setdefault(k, v)


def go(p):
    st.session_state.page = p


def navbar():
    cols = st.columns(len(PAGES))
    for c, (key, label) in zip(cols, PAGES):
        c.button(label, key=f"nav_{key}", use_container_width=True, on_click=go, args=(key,),
                 type="primary" if st.session_state.page == key else "secondary")
    st.divider()


def kpi(col, value, label):
    col.markdown(f"<div class='kpi'><h2>{value}</h2><span>{label}</span></div>", unsafe_allow_html=True)


# =====================================================================
# RECOMMENDATION ENGINE (same logic as your original app)
# =====================================================================
def recommend(state, district, season):
    if rec_model is None:
        return None
    sub = d[(d["State"] == state) & (d["District"] == district) & (d["Season"] == season)]
    w = get_weather(state, season)
    out = []
    for crop, g in sub.groupby("Crop"):
        g = g.sort_values("Crop_Year")
        X = pd.DataFrame([{"State": state, "District": district, "Crop": crop, "Season": season,
                           "Crop_Year": next_year, "Prev_Yield": g["Yield"].iloc[-1],
                           "Prev3_Avg": g["Yield"].tail(3).mean()}])
        for c in ["State", "District", "Crop", "Season"]:
            X[c] = pd.Categorical(X[c], categories=cats[c])
        y = float(rec_model.predict(X)[0])
        perf = "Low" if y <= q.loc[crop, "lo"] else ("Medium" if y <= q.loc[crop, "hi"] else "High")
        price, cost, effort = price_info(crop)
        profit = None if price is None else round(y * 10 * price - cost)
        wf = weather_fit(crop, w)
        out.append({"Crop": crop, "Performance": perf, "Yield (t/ha)": round(y, 2), "Effort": effort,
                    "Weather fit": wf[0] if wf else "N/A",
                    "Profit / Loss": "N/A" if profit is None else ("Profit" if profit > 0 else "Loss"),
                    "Estimated (Rs/ha)": profit, "Confidence": "Low" if crop in low_conf else "OK",
                    "_price": price, "_cost": cost})
    out.sort(key=lambda r: (r["Estimated (Rs/ha)"] is None, -(r["Estimated (Rs/ha)"] or 0)))
    return out


def color(v):
    m = {"High": "#1b5e20", "Good": "#1b5e20", "Profit": "#1b5e20", "Medium": "#e65100",
         "Moderate": "#e65100", "Low": "#c62828", "Poor": "#c62828", "Loss": "#c62828"}
    return f"color:{m[v]}; font-weight:bold" if v in m else ""


# =====================================================================
# PAGE 1: HOME
# =====================================================================
def quick_start(state, district, season):
    """Jump to the recommendation page with these values and run it automatically."""
    st.session_state.pre = (state, district, season)
    st.session_state.autorun = True
    st.session_state.page = "recommend"


def page_home():
    best = d.groupby("Crop")["Yield"].mean().sort_values(ascending=False)
    top_state = d.groupby("State")["District"].nunique().sort_values(ascending=False)
    years = f"{int(d['Crop_Year'].min())}-{int(d['Crop_Year'].max())}"

    # ---------------- HERO ----------------
    floaters = "".join(
        f"<div class='fl' style='left:{x}%;top:{y}%;animation-delay:{dl}s'>{e}</div>"
        for e, x, y, dl in [("🌾", 5, 14, 0), ("🌽", 88, 12, 1), ("🍅", 12, 70, 2), ("🥔", 82, 68, 0.5),
                            ("🌱", 46, 6, 1.5), ("☀️", 70, 80, 2.5), ("💧", 28, 82, 3)])
    st.markdown(f"""
    <div class='hero2'>{floaters}
      <div class='badge'>🇮🇳 AI-powered farming assistant</div>
      <h1>🌾 KisanMitra</h1>
      <h2>Kya ugayein? Ab data batayega.</h2>
      <p>Stop guessing your next crop. KisanMitra reads {years} of district yield data, mandi prices and
      weather to tell you what will grow well, how much work it needs and what you could earn.</p>
    </div>""", unsafe_allow_html=True)

    # ---------------- TICKER ----------------
    items = [f"🌾 {d['Crop'].nunique()} crops covered", f"📍 {d['District'].nunique()} districts",
             f"🗺️ {d['State'].nunique()} states and UTs", f"📊 {len(d):,} yield records",
             f"🏆 {best.index[0]} has the highest average yield ({best.iloc[0]:.1f} t/ha)",
             "🎯 70.9% accuracy on Low / Medium / High", "🌦️ Weather-aware recommendations",
             "💰 Profit and loss estimate for every crop"]
    t = "".join(f"<span>{i}</span>" for i in items)
    st.markdown(f"<div class='ticker'><div class='tr'>{t}{t}</div></div>", unsafe_allow_html=True)

    # ---------------- QUICK START ----------------
    st.markdown("<div class='sec'>🚀 Quick start: get your crops in 10 seconds</div>"
                "<div class='sub'>Pick your location and jump straight to results.</div>", unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        state = c1.selectbox("State", sorted(d["State"].unique()), key="h_state")
        district = c2.selectbox("District", sorted(d[d["State"] == state]["District"].unique()), key="h_dist")
        seasons = sorted(d[(d["State"] == state) & (d["District"] == district)]["Season"].unique())
        season = c3.selectbox("Season", seasons, key="h_season")
        w = get_weather(state, season)
        if w:
            m = st.columns(4)
            m[0].metric("🌡️ Temperature", f"{w['temp']:.1f} °C")
            m[1].metric("🌧️ Rain", f"{w['rain']:.0f} mm")
            m[2].metric("💧 Humidity", f"{w['hum']:.0f}%")
            n_crops = d[(d["State"] == state) & (d["District"] == district) & (d["Season"] == season)]["Crop"].nunique()
            m[3].metric("🌾 Crops analysed", n_crops)
            good = [c for c in d[(d["State"] == state) & (d["District"] == district) & (d["Season"] == season)]["Crop"].unique()
                    if (weather_fit(c, w) or ("",))[0] == "Good"]
            if good:
                st.markdown(f"<div class='wx'>✅ <b>Weather suits:</b> {', '.join(sorted(good)[:8])}"
                            f"{' and more' if len(good) > 8 else ''}</div>", unsafe_allow_html=True)
        b1, b2 = st.columns([2, 1])
        b1.button("🔎 Show my best crops now", type="primary", use_container_width=True,
                  on_click=quick_start, args=(state, district, season))
        b2.button("📈 I already have a crop", use_container_width=True, on_click=go, args=("yield",))

    # ---------------- FEATURES ----------------
    st.markdown("<div class='sec'>✨ Everything in one place</div>"
                "<div class='sub'>Four tools, one goal: a better season for you.</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, ic, title, txt, page, label in [
        (cols[0], "🌱", "Crop Recommendation", "Crops ranked by estimated profit with performance, effort and weather fit.", "recommend", "Recommend"),
        (cols[1], "📈", "Yield Prediction", "Know expected yield, production and earnings for your own land.", "yield", "Predict"),
        (cols[2], "🌦️", "Weather Explorer", "See temperature, rain and humidity and which crops suit them.", "weather", "Explore"),
        (cols[3], "📊", "Project Statistics", "Model accuracy, data coverage and trends behind the numbers.", "stats", "View stats")]:
        col.markdown(f"<div class='lift'><div class='ic'>{ic}</div><h4>{title}</h4><p>{txt}</p></div>",
                     unsafe_allow_html=True)
        col.button(f"{label} →", key=f"home_{page}", use_container_width=True, on_click=go, args=(page,))

    # ---------------- GUESSWORK vs DATA ----------------
    st.markdown("<div class='sec'>🎯 Why KisanMitra?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='vs'>
      <div class='bad'><h4>😕 Without data</h4><ul>
        <li>Choosing crops by habit or what the neighbour grows</li>
        <li>No idea if the weather suits the crop</li>
        <li>Profit or loss is known only after harvest</li>
        <li>Low yield and sudden losses</li></ul></div>
      <div class='good'><h4>😊 With KisanMitra</h4><ul>
        <li>Crops ranked by predicted performance for your district</li>
        <li>Temperature and rain checked against each crop's needs</li>
        <li>Estimated profit or loss before you sow</li>
        <li>Effort level shown, so you plan labour and water</li></ul></div>
    </div>""", unsafe_allow_html=True)

    # ---------------- HOW IT WORKS ----------------
    st.markdown("<div class='sec'>🧭 How it works</div>", unsafe_allow_html=True)
    s = st.columns(4)
    for col, ic, tt, x in [(s[0], "📍", "1. Choose location", "State, district and season"),
                           (s[1], "🤖", "2. Model predicts", "LightGBM trained on district yields"),
                           (s[2], "🌦️", "3. Weather check", "Temperature and rain vs crop needs"),
                           (s[3], "💰", "4. Profit estimate", "Yield × price − cost")]:
        col.markdown(f"<div class='step'><div style='font-size:2rem'>{ic}</div><b style='font-size:1.02rem;color:#1b3a1d'>{tt}</b>"
                     f"<br><span class='note'>{x}</span></div>", unsafe_allow_html=True)

    # ---------------- SEASONS ----------------
    st.markdown("<div class='sec'>📅 Know your seasons</div>"
                "<div class='sub'>India grows crops in three main seasons.</div>", unsafe_allow_html=True)
    sc = st.columns(3)
    for col, cls, ic, nm, mo, tx in [
        (sc[0], "kharif", "🌧️", "Kharif", "June to October", "Monsoon crops such as rice, maize, cotton and soyabean."),
        (sc[1], "rabi", "❄️", "Rabi", "October to March", "Winter crops such as wheat, gram, mustard and potato."),
        (sc[2], "zaid", "☀️", "Zaid", "March to June", "Short summer crops such as moong, watermelon and vegetables.")]:
        col.markdown(f"<div class='season {cls}'><h4>{ic} {nm}</h4><small>{mo}</small><p>{tx}</p></div>",
                     unsafe_allow_html=True)

    # ---------------- FACTS ----------------
    st.markdown("<div class='sec'>💡 Did you know?</div>", unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    f1.markdown(f"<div class='fact'>🏆 In this dataset, <b>{best.index[0]}</b> has the highest average yield at "
                f"<b>{best.iloc[0]:.1f} t/ha</b>, followed by <b>{best.index[1]}</b> and <b>{best.index[2]}</b>.</div>",
                unsafe_allow_html=True)
    f2.markdown(f"<div class='fact'>🗺️ <b>{top_state.index[0]}</b> has the most districts covered "
                f"(<b>{top_state.iloc[0]}</b>), and the data spans <b>{years}</b>.</div>", unsafe_allow_html=True)

    # ---------------- FAQ ----------------
    st.markdown("<div class='sec'>❓ Common questions</div>", unsafe_allow_html=True)
    with st.expander("How accurate is it?"):
        st.write("The model reaches R² 0.95 and about 0.97 t/ha error on held-out future years, and labels "
                 "Low / Medium / High correctly 70.9% of the time (random guessing gets 33%). "
                 "Some crops, such as Cardamom and Black Pepper, are flagged as low confidence.")
    with st.expander("Where does the data come from?"):
        st.write("Yield data from APY (data.gov.in), prices from Agmarknet, and weather from the Open-Meteo "
                 "historical API. Costs are approximate figures, so please verify them locally.")
    with st.expander("Is the weather live?"):
        st.write("No. The weather shown is a historical state-season average, so every district in a state "
                 "shares it. It tells you what is typical, not tomorrow's forecast.")
    with st.expander("Is the profit guaranteed?"):
        st.write("No. Profit is an estimate: predicted yield × price − cost per hectare. Real prices, rain "
                 "and pests change the outcome. Use it as guidance.")

    # ---------------- CTA ----------------
    st.markdown("<div class='cta'><h2>Ready for a smarter season? 🌾</h2>"
                "<div>Pick your location and see which crops can earn you the most.</div></div>",
                unsafe_allow_html=True)
    st.write("")
    c = st.columns([1, 2, 1])
    c[1].button("🌱 Start crop recommendation", key="cta_go", type="primary", use_container_width=True,
                on_click=go, args=("recommend",))


# =====================================================================
# PAGE 2: RECOMMENDATION
# =====================================================================
def page_recommend():
    st.title("🌱 Crop Recommendation")
    if rec_model is None:
        st.error(f"Recommendation model could not be loaded: {rec_err}\n\nTry: `pip install lightgbm`")
        return

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        pre = st.session_state.get("pre")
        s_opts = sorted(d["State"].unique())
        state = c1.selectbox("State", s_opts, index=s_opts.index(pre[0]) if pre and pre[0] in s_opts else 0)
        d_opts = sorted(d[d["State"] == state]["District"].unique())
        district = c2.selectbox("District", d_opts, index=d_opts.index(pre[1]) if pre and pre[1] in d_opts else 0)
        se_opts = sorted(d[(d["State"] == state) & (d["District"] == district)]["Season"].unique())
        season = c3.selectbox("Season", se_opts, index=se_opts.index(pre[2]) if pre and pre[2] in se_opts else 0)
        w = get_weather(state, season)
        if w:
            st.markdown(f"<div class='wx'>🌦️ <b>Typical {season} weather in {state}:</b> "
                        f"🌡️ {w['temp']:.1f} °C &nbsp;|&nbsp; 🌧️ {w['rain']:.0f} mm &nbsp;|&nbsp; "
                        f"💧 {w['hum']:.0f}% humidity</div>", unsafe_allow_html=True)
        auto = st.session_state.pop("autorun", False)
        if st.button("🔎 Recommend crops", type="primary", use_container_width=True) or auto:
            with st.spinner("Analysing your district..."):
                st.session_state.rows = recommend(state, district, season)
            st.session_state.ctx = (state, district, season)
            st.session_state.sel = None

    rows = st.session_state.rows
    if rows is None:
        st.info("Select your location above and press **Recommend crops**.")
        return
    if not rows:
        st.warning("No data for this district and season.")
        return

    ctx = st.session_state.ctx
    st.subheader(f"Results for {ctx[1]}, {ctx[0]} ({ctx[2]})")
    df_all = pd.DataFrame(rows)

    # ---- interactive filters
    with st.expander("🎛️ Filter results", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        perf = f1.multiselect("Performance", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        eff = f2.multiselect("Effort", sorted(df_all["Effort"].unique()), default=sorted(df_all["Effort"].unique()))
        wfit = f3.multiselect("Weather fit", ["Good", "Moderate", "Poor", "N/A"], default=["Good", "Moderate", "Poor", "N/A"])
        only_profit = f4.checkbox("Only profitable crops", value=False)
    df = df_all[df_all["Performance"].isin(perf) & df_all["Effort"].isin(eff) & df_all["Weather fit"].isin(wfit)]
    if only_profit:
        df = df[df["Profit / Loss"] == "Profit"]
    if df.empty:
        st.warning("No crops match these filters.")
        return

    # ---- top 3 podium
    top = df[df["Estimated (Rs/ha)"].notna()].head(3)
    if not top.empty:
        st.markdown("#### 🏆 Top picks by estimated profit")
        pc = st.columns(len(top))
        for i, (col, (_, r)) in enumerate(zip(pc, top.iterrows())):
            medal = ["🥇", "🥈", "🥉"][i]
            col.markdown(
                f"<div class='podium {'first' if i == 0 else ''}'>{medal}<h3>{r['Crop']}</h3>"
                f"{pill(r['Performance'])} {pill(r['Weather fit']) if r['Weather fit'] != 'N/A' else ''}<br>"
                f"<span class='note'>Est. {r['Profit / Loss'].lower()}</span><br>"
                f"<b style='font-size:1.3rem'>{inr(r['Estimated (Rs/ha)'])}</b>"
                f"<br><span class='note'>per hectare · {r['Effort']} effort</span></div>",
                unsafe_allow_html=True)

    # ---- full table
    st.markdown("#### All crops")
    show = df.drop(columns=["_price", "_cost"])
    sty = (show.style.map(color, subset=["Performance", "Profit / Loss", "Weather fit"])
           .format({"Estimated (Rs/ha)": "{:,.0f}"}, na_rep="N/A"))
    st.dataframe(sty, use_container_width=True, hide_index=True)
    st.caption("Estimated values. Price and cost are approximate. Crops marked 'Low' confidence have weaker model accuracy.")

    # ---- chart
    cd = df[df["Estimated (Rs/ha)"].notna()].head(10)
    if not cd.empty:
        ch = alt.Chart(cd).mark_bar().encode(
            x=alt.X("Estimated (Rs/ha):Q", title="Estimated profit / loss (Rs per hectare)"),
            y=alt.Y("Crop:N", sort="-x", title=None),
            color=alt.condition(alt.datum["Estimated (Rs/ha)"] > 0, alt.value("#43a047"), alt.value("#e53935")),
            tooltip=["Crop", "Performance", "Yield (t/ha)", "Weather fit", "Estimated (Rs/ha)"]).properties(height=320)
        st.altair_chart(ch, use_container_width=True)

    # ---- go to details
    st.markdown("#### Want the full picture for a crop?")
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("Choose a crop", df["Crop"].tolist(), label_visibility="collapsed")
    if c2.button("View details →", type="primary", use_container_width=True):
        st.session_state.sel = pick
        go("detail")
        st.rerun()


# =====================================================================
# PAGE 3: CROP DETAILS
# =====================================================================
def page_detail():
    st.title("🔍 Crop Details")
    rows, sel, ctx = st.session_state.rows, st.session_state.sel, st.session_state.ctx
    if not rows or not sel:
        st.info("Run a recommendation first and choose a crop to see its weather and profit breakdown.")
        st.button("Go to Crop Recommendation", on_click=go, args=("recommend",), type="primary")
        return
    state, district, season = ctx
    names = [r["Crop"] for r in rows]
    sel = st.selectbox("Crop", names, index=names.index(sel))
    st.session_state.sel = sel
    r = next(x for x in rows if x["Crop"] == sel)
    st.subheader(f"{sel} · {district}, {season}")

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("##### 🌦️ Weather suitability")
        w = get_weather(state, season)
        wf = weather_fit(sel, w)
        if w is None:
            st.write("No weather data for this state/season.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Temperature", f"{w['temp']:.1f} °C")
            m2.metric("Rain", f"{w['rain']:.0f} mm")
            m3.metric("Humidity", f"{w['hum']:.0f}%")
            if wf:
                fit, (tmin, tmax, rmin), t_ok, r_ok = wf
                st.markdown(f"Weather fit: {pill(fit)}", unsafe_allow_html=True)
                st.write(f"{'✅' if t_ok else '❌'} Ideal temperature **{tmin}–{tmax} °C** "
                         f"(now {w['temp']:.1f} °C)")
                st.write(f"{'✅' if r_ok else '❌'} Minimum rain **{rmin} mm** (now {w['rain']:.0f} mm)")
                st.progress(min(1.0, w["rain"] / max(rmin, 1)), text="Rain available vs rain needed")
                if fit == "Poor":
                    st.warning("Weather is not ideal. Consider irrigation or a better-suited crop.")
                elif fit == "Moderate":
                    st.info("One weather condition is outside the crop's ideal range.")
                else:
                    st.success("Temperature and rain both suit this crop.")
            else:
                st.caption("Ideal weather range not available for this crop.")

    with right, st.container(border=True):
        st.markdown("##### 💰 Earnings (per hectare)")
        if r["_price"] is None:
            st.write("Price / cost data not available for this crop.")
        else:
            rev = round(r["Yield (t/ha)"] * 10 * r["_price"])
            st.write(f"Predicted yield: **{r['Yield (t/ha)']} t/ha** ({r['Yield (t/ha)'] * 10:.1f} quintal)")
            a, b = st.columns(2)
            a.metric("Revenue", inr(rev))
            b.metric("Cultivation cost", inr(r["_cost"]))
            st.metric(f"Estimated {r['Profit / Loss']}", inr(abs(r["Estimated (Rs/ha)"])),
                      delta=f"{'+' if r['Estimated (Rs/ha)'] > 0 else '-'}{inr(abs(r['Estimated (Rs/ha)']))}")
            st.markdown(f"Effort: **{r['Effort']}** &nbsp; Performance: {pill(r['Performance'])}",
                        unsafe_allow_html=True)

    # What-if: price slider
    if r["_price"] is not None:
        with st.container(border=True):
            st.markdown("##### 🎚️ What-if: change the market price")
            pr = st.slider("Price (Rs / quintal)", int(r["_price"] * 0.5), int(r["_price"] * 1.5), int(r["_price"]), 50)
            area = st.number_input("Your land (hectares)", 0.1, 500.0, 1.0, 0.5)
            profit = r["Yield (t/ha)"] * 10 * pr - r["_cost"]
            st.metric(f"Estimated {'profit' if profit >= 0 else 'loss'} for {area} ha", inr(abs(profit) * area))

    # history of this crop in district
    h = d[(d["State"] == state) & (d["District"] == district) & (d["Crop"] == sel) & (d["Season"] == season)]
    if len(h) > 1:
        st.markdown("##### 📜 Past yield in this district")
        st.altair_chart(alt.Chart(h).mark_line(point=True, color="#2e7d32").encode(
            x=alt.X("Crop_Year:O", title="Year"), y=alt.Y("Yield:Q", title="Yield (t/ha)"),
            tooltip=["Crop_Year", "Yield"]).properties(height=240), use_container_width=True)

    st.markdown("##### 🏁 Top 5 crops: estimated profit/loss")
    top = [x for x in rows if x["Estimated (Rs/ha)"] is not None][:5]
    st.bar_chart(pd.DataFrame(top).set_index("Crop")["Estimated (Rs/ha)"])
    st.caption("Higher-profit crops (Banana, Ginger, Turmeric) need more investment and carry more risk.")
    c1, c2 = st.columns(2)
    c1.button("← Back to recommendations", on_click=go, args=("recommend",), use_container_width=True)
    c2.button("📈 Predict yield for my land", on_click=go, args=("yield",), use_container_width=True, type="primary")


# =====================================================================
# PAGE 4: YIELD PREDICTION
# =====================================================================
RAIN_FACTOR = {"Kam (below normal)": 0.85, "Theek (normal)": 1.0, "Zyada (above normal)": 1.0}
FERT_FACTOR = {"Kam (less than recommended)": 0.85, "Recommended jitna": 1.0, "Zyada (more than recommended)": 1.0}


def page_yield():
    st.title("📈 Yield Prediction")
    st.caption("Uses your RandomForest yield model (State, District, Season, Crop, Rainfall).")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        state = c1.selectbox("State", sorted(yd["State"].unique()), format_func=cap, key="y_state")
        dists = sorted(yd[yd["State"] == state]["District"].unique())
        district = c2.selectbox("District", dists, format_func=cap, key="y_dist")
        seasons = sorted(yd[(yd["State"] == state) & (yd["District"] == district)]["Season"].unique())
        season = c3.selectbox("Season", seasons, format_func=cap, key="y_season")
        crops = sorted(yd[(yd["State"] == state) & (yd["District"] == district) & (yd["Season"] == season)]["Crop"].unique())
        crop = c4.selectbox("Crop", crops, format_func=cap, key="y_crop")

        e1, e2, e3, e4 = st.columns(4)
        area = e1.number_input("Land area (hectares)", 0.1, 1000.0, 1.0, 0.5)
        rain_c = e2.selectbox("Expected rainfall", list(RAIN_FACTOR))
        fert_c = e3.selectbox("Fertilizer use", list(FERT_FACTOR))
        normal = float(rain_normal.get(state, 800))
        rain_val = e4.number_input("Annual rainfall (mm)", 50.0, 6000.0, float(round(normal)), 10.0,
                                   help="Defaults to this state's normal rainfall from your data.")

        w = get_weather(state, season)
        if w:
            st.markdown(f"<div class='wx'>🌦️ <b>{cap(season)} weather in {cap(state)}:</b> 🌡️ {w['temp']:.1f} °C "
                        f"&nbsp;|&nbsp; 🌧️ {w['rain']:.0f} mm &nbsp;|&nbsp; 💧 {w['hum']:.0f}%</div>",
                        unsafe_allow_html=True)

        if st.button("📈 Predict yield", type="primary", use_container_width=True):
            X = pd.DataFrame([{"State": state, "District": district, "Season": season,
                               "Crop": crop, "Rainfall": float(rain_val)}])
            base = float(np.expm1(yield_model.predict(X)[0]))
            final = base * RAIN_FACTOR[rain_c] * FERT_FACTOR[fert_c]
            hist = yd[(yd["State"] == state) & (yd["District"] == district) & (yd["Crop"] == crop)]
            # sensitivity to rainfall (uses the model itself)
            sens = []
            for rv in np.linspace(max(50, normal * 0.5), normal * 1.6, 12):
                Xs = X.assign(Rainfall=float(rv))
                sens.append({"Rainfall (mm)": round(rv), "Yield (q/ha)": float(np.expm1(yield_model.predict(Xs)[0]))})
            st.session_state.yres = dict(state=state, district=district, season=season, crop=crop, area=area,
                                         base=base, final=final, hist=hist, sens=pd.DataFrame(sens),
                                         rain=rain_val, w=w, rain_c=rain_c, fert_c=fert_c)

    r = st.session_state.yres
    if r is None:
        st.info("Fill the form above and press **Predict yield**.")
        return

    st.subheader(f"{cap(r['crop'])} · {cap(r['district'])}, {cap(r['state'])} ({cap(r['season'])})")
    hist_avg = float(r["hist"]["Yield_q_ha"].mean()) if len(r["hist"]) else None
    k = st.columns(4)
    kpi(k[0], f"{r['final']:.1f}", "Predicted yield (q/ha)")
    kpi(k[1], f"{r['final'] * r['area']:,.1f}", f"Total production (q) for {r['area']} ha")
    kpi(k[2], f"{hist_avg:.1f}" if hist_avg else "N/A", "District historical avg (q/ha)")
    if hist_avg:
        diff = (r["final"] - hist_avg) / hist_avg * 100
        kpi(k[3], f"{'▲' if diff >= 0 else '▼'} {abs(diff):.1f}%", "vs historical average")
    else:
        kpi(k[3], "—", "vs historical average")

    st.write("")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("##### 💰 Estimated earnings")
        price, cost, effort = price_info(r["crop"])
        if price is None:
            st.write("Price / cost data not available for this crop.")
        else:
            revenue, total_cost = r["final"] * price * r["area"], cost * r["area"]
            a, b, c = st.columns(3)
            a.metric("Revenue", inr(revenue))
            b.metric("Cost", inr(total_cost))
            c.metric("Profit" if revenue >= total_cost else "Loss", inr(abs(revenue - total_cost)))
            st.caption(f"At about Rs {price:,.0f}/quintal. Effort level: {effort}. Prices are approximate.")
    with right, st.container(border=True):
        st.markdown("##### 🌦️ Weather check")
        wf = weather_fit(r["crop"], r["w"])
        if wf:
            fit, (tmin, tmax, rmin), t_ok, r_ok = wf
            st.markdown(f"Weather fit: {pill(fit)}", unsafe_allow_html=True)
            st.write(f"{'✅' if t_ok else '❌'} Temperature {r['w']['temp']:.1f} °C (ideal {tmin}–{tmax} °C)")
            st.write(f"{'✅' if r_ok else '❌'} Rain {r['w']['rain']:.0f} mm (needs {rmin}+ mm)")
            st.progress(min(1.0, r["w"]["rain"] / max(rmin, 1)), text="Rain vs requirement")
        else:
            st.caption("Ideal weather range not available for this crop.")
        st.caption(f"Scenario used: rain = {r['rain_c']}, fertilizer = {r['fert_c']}. "
                   f"Base model output {r['base']:.1f} q/ha.")

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("##### 🌧️ How rainfall changes the yield")
        st.altair_chart(alt.Chart(r["sens"]).mark_line(point=True, color="#1e88e5").encode(
            x="Rainfall (mm):Q", y=alt.Y("Yield (q/ha):Q", scale=alt.Scale(zero=False)),
            tooltip=["Rainfall (mm)", "Yield (q/ha)"]).properties(height=250), use_container_width=True)
    with c2, st.container(border=True):
        st.markdown("##### 📜 Past yields in this district")
        h = r["hist"].groupby("Crop_Year", as_index=False)["Yield_q_ha"].mean()
        if len(h) > 1:
            st.altair_chart(alt.Chart(h).mark_line(point=True, color="#2e7d32").encode(
                x=alt.X("Crop_Year:O", title="Year"), y=alt.Y("Yield_q_ha:Q", title="Yield (q/ha)", scale=alt.Scale(zero=False)),
                tooltip=["Crop_Year", "Yield_q_ha"]).properties(height=250), use_container_width=True)
        else:
            st.caption("Not enough history for a chart.")

    b1, b2 = st.columns(2)
    b1.button("🌱 Get crop recommendations", on_click=go, args=("recommend",), use_container_width=True)
    b2.button("🌦️ Explore weather", on_click=go, args=("weather",), use_container_width=True)


# =====================================================================
# PAGE 5: WEATHER EXPLORER
# =====================================================================
def page_weather():
    st.title("🌦️ Weather Explorer")
    st.caption("State-season averages from Open-Meteo (your `state_weather.csv`). "
               "A crop's fit is Good when both temperature and rain are in range, Moderate when one is, Poor when neither is.")
    states = sorted(sw["State"].unique())
    c1, c2 = st.columns(2)
    state = c1.selectbox("State", states, key="w_state")
    seasons = sorted(sw[sw["State"] == state]["Season"].unique())
    season = c2.selectbox("Season", seasons, key="w_season")
    w = get_weather(state, season)
    if w is None:
        st.warning("No weather data.")
        return

    m = st.columns(3)
    m[0].metric("🌡️ Temperature", f"{w['temp']:.1f} °C")
    m[1].metric("🌧️ Rain (season)", f"{w['rain']:.0f} mm")
    m[2].metric("💧 Humidity", f"{w['hum']:.0f}%")

    t1, t2, t3 = st.tabs(["🌾 Crops that suit this weather", "📊 Compare seasons", "🗺️ Compare states"])

    with t1:
        fits = []
        for crop, (tmin, tmax, rmin) in ideal.items():
            wf = weather_fit(crop, w)
            fits.append({"Crop": crop, "Fit": wf[0], "Ideal temp (°C)": f"{tmin}-{tmax}", "Min rain (mm)": rmin})
        fdf = pd.DataFrame(fits)
        fdf["_o"] = fdf["Fit"].map({"Good": 0, "Moderate": 1, "Poor": 2})
        fdf = fdf.sort_values(["_o", "Crop"]).drop(columns="_o")
        cnt = fdf["Fit"].value_counts()
        k = st.columns(3)
        kpi(k[0], int(cnt.get("Good", 0)), "Good fit")
        kpi(k[1], int(cnt.get("Moderate", 0)), "Moderate fit")
        kpi(k[2], int(cnt.get("Poor", 0)), "Poor fit")
        choice = st.radio("Show", ["Good", "Moderate", "Poor", "All"], horizontal=True)
        shown = fdf if choice == "All" else fdf[fdf["Fit"] == choice]
        st.dataframe(shown.style.map(color, subset=["Fit"]), use_container_width=True, hide_index=True)
        st.button("🌱 Get full recommendations for my district", on_click=go, args=("recommend",), type="primary")

    with t2:
        both = sw[sw["State"] == state][["Season", "Temp", "Rain", "Humidity"]]
        mm = both.melt("Season", var_name="Measure", value_name="Value")
        st.altair_chart(alt.Chart(mm).mark_bar().encode(
            x=alt.X("Season:N", title=None), y=alt.Y("Value:Q", title=None),
            color=alt.Color("Season:N", legend=None), column=alt.Column("Measure:N", title=None),
            tooltip=["Season", "Measure", alt.Tooltip("Value:Q", format=".1f")]).properties(width=170, height=240))

    with t3:
        measure = st.selectbox("Measure", ["Rain", "Temp", "Humidity"])
        sdf = sw[sw["Season"] == season].sort_values(measure, ascending=False)
        st.altair_chart(alt.Chart(sdf).mark_bar().encode(
            x=alt.X(f"{measure}:Q"), y=alt.Y("State:N", sort="-x", title=None),
            color=alt.condition(alt.datum.State == state, alt.value("#e65100"), alt.value("#66bb6a")),
            tooltip=["State", alt.Tooltip(f"{measure}:Q", format=".1f")]).properties(height=max(300, 20 * len(sdf))),
            use_container_width=True)
        st.caption(f"Selected state ({state}) is highlighted in orange.")


# =====================================================================
# PAGE 6: PROJECT STATISTICS
# =====================================================================
def page_stats():
    st.title("📊 Project Statistics")

    st.markdown("#### Model performance (from your README)")
    k = st.columns(4)
    kpi(k[0], "0.95", "R² (LightGBM)")
    kpi(k[1], "0.97 t/ha", "Mean absolute error")
    kpi(k[2], "70.9%", "Low/Med/High accuracy")
    kpi(k[3], "33%", "Random baseline")
    st.caption("Time-based train/test split (no leakage). Weaker accuracy on Cardamom, Black Pepper, Cashewnut and a few other crops.")

    st.markdown("#### Dataset overview")
    k = st.columns(5)
    kpi(k[0], f"{len(d):,}", "Records")
    kpi(k[1], d["State"].nunique(), "States")
    kpi(k[2], d["District"].nunique(), "Districts")
    kpi(k[3], d["Crop"].nunique(), "Crops")
    kpi(k[4], f"{int(d['Crop_Year'].min())}-{int(d['Crop_Year'].max())}", "Years")
    st.write("")

    t1, t2, t3, t4 = st.tabs(["🌾 Crops", "📅 Trends", "🗺️ States & seasons", "🌦️ Weather & economics"])

    with t1:
        a, b = st.columns(2)
        top_n = a.slider("How many crops", 5, 25, 10)
        avg = d.groupby("Crop", as_index=False)["Yield"].mean().sort_values("Yield", ascending=False).head(top_n)
        a.markdown("**Highest average yield (t/ha)**")
        a.altair_chart(alt.Chart(avg).mark_bar(color="#43a047").encode(
            x="Yield:Q", y=alt.Y("Crop:N", sort="-x", title=None), tooltip=["Crop", alt.Tooltip("Yield:Q", format=".2f")]),
            use_container_width=True)
        cnt = d["Crop"].value_counts().head(top_n).rename_axis("Crop").reset_index(name="Records")
        b.markdown("**Most-recorded crops**")
        b.altair_chart(alt.Chart(cnt).mark_bar(color="#26a69a").encode(
            x="Records:Q", y=alt.Y("Crop:N", sort="-x", title=None), tooltip=["Crop", "Records"]),
            use_container_width=True)

    with t2:
        crop = st.selectbox("Crop", sorted(d["Crop"].unique()), index=sorted(d["Crop"].unique()).index("Rice") if "Rice" in d["Crop"].values else 0)
        tr = d[d["Crop"] == crop].groupby("Crop_Year", as_index=False)["Yield"].mean()
        st.altair_chart(alt.Chart(tr).mark_line(point=True, color="#2e7d32").encode(
            x=alt.X("Crop_Year:O", title="Year"), y=alt.Y("Yield:Q", title="Average yield (t/ha)", scale=alt.Scale(zero=False)),
            tooltip=["Crop_Year", alt.Tooltip("Yield:Q", format=".2f")]).properties(height=300), use_container_width=True)
        rec = d.groupby("Crop_Year").size().rename("Records").reset_index()
        st.markdown("**Records per year**")
        st.bar_chart(rec.set_index("Crop_Year"))

    with t3:
        a, b = st.columns(2)
        ss = d["Season"].value_counts().rename_axis("Season").reset_index(name="Records")
        a.markdown("**Records by season**")
        a.altair_chart(alt.Chart(ss).mark_arc(innerRadius=50).encode(
            theta="Records:Q", color="Season:N", tooltip=["Season", "Records"]), use_container_width=True)
        st_c = d.groupby("State")["District"].nunique().sort_values(ascending=False).head(12).rename_axis("State").reset_index(name="Districts")
        b.markdown("**States with most districts**")
        b.altair_chart(alt.Chart(st_c).mark_bar(color="#7cb342").encode(
            x="Districts:Q", y=alt.Y("State:N", sort="-x", title=None)), use_container_width=True)
        stt = st.selectbox("State yield profile", sorted(d["State"].unique()))
        sy = d[d["State"] == stt].groupby("Crop", as_index=False)["Yield"].mean().sort_values("Yield", ascending=False).head(10)
        st.altair_chart(alt.Chart(sy).mark_bar(color="#ffa726").encode(
            x="Yield:Q", y=alt.Y("Crop:N", sort="-x", title=None)), use_container_width=True)

    with t4:
        a, b = st.columns(2)
        sea = a.selectbox("Season", sorted(sw["Season"].unique()), key="st_sea")
        meas = b.selectbox("Measure", ["Rain", "Temp", "Humidity"], key="st_meas")
        sdf = sw[sw["Season"] == sea].sort_values(meas, ascending=False)
        st.altair_chart(alt.Chart(sdf).mark_bar(color="#42a5f5").encode(
            x=f"{meas}:Q", y=alt.Y("State:N", sort="-x", title=None),
            tooltip=["State", alt.Tooltip(f"{meas}:Q", format=".1f")]).properties(height=max(300, 18 * len(sdf))),
            use_container_width=True)
        st.markdown("**Cultivation cost vs market price (Rs)**")
        rf = ref.reset_index().dropna(subset=["Price", "Cost"])
        if not rf.empty:
            st.altair_chart(alt.Chart(rf).mark_circle(size=110).encode(
                x=alt.X("Cost:Q", title="Cost per hectare"), y=alt.Y("Price:Q", title="Price per quintal"),
                color="Effort:N", tooltip=["Crop", "Cost", "Price", "Effort"]).properties(height=320),
                use_container_width=True)

    with st.expander("Data sources and limitations"):
        st.markdown(
            "- **Yield:** APY (data.gov.in / Kaggle), 2008-2019\n"
            "- **Prices:** Agmarknet market-wise price and arrival report (a few days, not a full-season average)\n"
            "- **Weather:** Open-Meteo historical API, state-level averages shared by all districts of a state\n"
            "- **Cost, labour, irrigation:** approximate hand-made table\n\n"
            "Treat all output as guidance, not a guarantee.")
    st.button("🏠 Back to Home", on_click=go, args=("home",))


# =====================================================================
# ROUTER
# =====================================================================
navbar()
{"home": page_home, "recommend": page_recommend, "detail": page_detail,
 "yield": page_yield, "weather": page_weather, "stats": page_stats}[st.session_state.page]()
st.markdown("<p class='note' style='text-align:center;margin-top:30px'>🌾 KisanMitra · estimates are "
            "guidance only, not a guarantee.</p>", unsafe_allow_html=True)

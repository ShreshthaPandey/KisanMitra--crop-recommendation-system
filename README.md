# 🌾 KisanMitra: Crop Recommendation, Yield Prediction and Profit Estimation

> **Kya ugayein? Ab data batayega.** (What to grow? Now data will tell.)

KisanMitra is a machine-learning assistant for Indian farmers. A farmer selects a **State, District and Season**,
and the system ranks crops by predicted performance, effort needed, weather suitability and estimated profit per
hectare. A second tool predicts the **yield, production and earnings** of a chosen crop for the farmer's own land.

## Short description

Many farmers choose crops by habit or guesswork, which can lead to low yield or loss. KisanMitra uses 12 years of
district-level yield data (APY, 2008-2019), mandi prices and state-season weather to recommend crops with a
LightGBM model and to predict yield with a RandomForest model. Each result shows a Low / Medium / High performance
label, an effort level, a weather-fit rating and an estimated profit or loss, all in a simple multi-page Streamlit app.

## Features

- **Crop recommendation:** crops ranked by estimated profit for a district and season, with filters and a top-3 view.
- **Crop details:** weather suitability, revenue, cost, profit, a "what-if" price slider and past district yields.
- **Yield prediction:** yield (q/ha), total production, comparison with the district's historical average, earnings
  and a chart of how rainfall changes the yield.
- **Weather explorer:** temperature, rain and humidity by state and season, and which crops suit them.
- **Project statistics:** model accuracy, dataset coverage, yield trends, season split and cost vs price charts.
- **Creative landing page:** animated hero, quick-start planner, season guide, facts and FAQ.
- **Low-confidence flag** for crops where the model is weaker (for example Cardamom, Black Pepper, Cashewnut).

## Pages

| Page | Purpose |
|---|---|
| Home | Landing page and quick start |
| Recommendation | Ranked crops for State, District and Season |
| Crop Details | Weather fit and profit breakdown for one crop |
| Yield Prediction | Predicted yield, production and earnings for your land |
| Weather Explorer | Weather by state and season, plus crop fit |
| Statistics | Model, data and weather statistics |

All pages are connected with buttons in a top navigation bar.

## How it works

1. **Recommendation model:** LightGBM predicts next-year yield for every crop grown in the chosen district and
   season, using the previous yield and the last-3-year average.
2. **Performance label:** the predicted yield is compared with that crop's own historical percentiles
   (30th and 60th) to give Low / Medium / High.
3. **Weather fit:** each crop has an ideal temperature range and a minimum seasonal rain.
   Both met = **Good**, one met = **Moderate**, none met = **Poor**.
4. **Profit estimate:** `Yield (t/ha) x 10 x Price (Rs/quintal) - Cost (Rs/ha)`.
5. **Yield prediction model:** a RandomForest uses State, District, Season, Crop and Rainfall. The "Kam" rainfall and
   fertilizer choices apply a x0.85 adjustment to the model output.

## Results

- R² about **0.95**, MAE about **0.97 t/ha** (time-based train/test split, no leakage)
- Low / Medium / High label accuracy **70.9%** (random baseline 33%)
- Tried and rejected, with no gain: state-level weather features and neighbouring-district yield

## Data sources

| Data | Source |
|---|---|
| Crop area, production, yield | APY (data.gov.in / Kaggle), 2008-2019 |
| Prices and MSP | Agmarknet market-wise price and arrival report |
| Weather | Open-Meteo Historical API |
| Cost, labour days, irrigations | Hand-made approximate table (verify with CACP) |

## Project structure

```
.
├── app.py                    Streamlit app (multi-page UI)
├── final_model.pkl           LightGBM recommendation model
├── final_data.csv            District yield data with Prev_Yield, Prev3_Avg
├── crop_reference.csv        Price, cost and effort per crop
├── state_weather.csv         State-season temperature, rain, humidity
├── models/
│   └── yield_model.pkl       RandomForest yield model
├── data/
│   ├── clean_yield_data.csv  Data used by the yield model
│   └── rain_normal.csv       Normal annual rainfall per state
├── crop recommendation notebook.ipynb
├── crop yield notebook.ipynb
└── requirements.txt
```

## Setup and run

```bash
pip install streamlit altair pandas numpy scikit-learn lightgbm joblib
streamlit run app.py
```

Run the command from the project folder so the model and CSV files are found. Use the same scikit-learn and
lightgbm versions you trained with to avoid pickle warnings.

### Optional: MERN version

A MERN version (React frontend, Express + MongoDB login, FastAPI ML service) is available in `kisanmitra-mern/`.
It adds farmer authentication and saves each farmer's history. See the README inside that folder.

## Limitations

- Weather is a state-season historical average, so all districts in a state share it. It is not a live forecast.
- Prices come from a few days of mandi data, not a full-season average.
- Cost, labour and ideal weather ranges are approximate and should be verified locally.
- Rainfall and fertilizer scenarios in the yield tool are simple multipliers, not model outputs.
- Treat all results as guidance, not a guarantee.

## Future work

- Live weather forecast integration
- District-level weather
- Soil data (NPK, pH) as model inputs
- Hindi and regional language support
- Live mandi price feed

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, Streamlit, Altair. Optional: React, Node.js, Express, MongoDB, FastAPI.

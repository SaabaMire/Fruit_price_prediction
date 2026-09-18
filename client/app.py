import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# Load Models & Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_models():
    random_forest = joblib.load(ROOT_DIR / "model" / "rf_fruit_model.pkl")
    linear_regression = joblib.load(ROOT_DIR / "model" / "lr_fruit_model.pkl")
    return random_forest, linear_regression


@st.cache_data
def load_dataset():
    return pd.read_csv(ROOT_DIR / "dataset" / "Clean_fruit_dataset_encoded.csv")


rf_model, lr_model = load_models()
df_encoded = load_dataset()


# Streamlit Page Configuration

st.set_page_config(
    page_title="🍎 Fruit Price Predictor",
    page_icon="🍇",
    layout="centered"
)
# Custom CSS Styling

st.markdown("""
    <style>
        /* Soft gradient background */
        body {
            background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
            font-family: "Inter", sans-serif;
        }

        .main {
            background-color: white;
            border-radius: 18px;
            padding: 40px;
            box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.08);
            margin-top: 40px;
        }

        h1 {
            text-align: center;
            color: #34495e;
            font-weight: 700;
        }

        .stButton>button {
            background: linear-gradient(90deg, #9be15d, #00e3ae);
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
            font-size: 1.05rem;
            border: none;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #00e3ae, #9be15d);
        }

        .stSelectbox label, .stNumberInput label, .stSlider label, .stRadio label {
            color: #2c3e50;
            font-weight: 600;
        }

        .result-card {
            background: black;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.05);
            margin-top: 20px;
        }

        .footer {
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)


# Header

st.markdown("<h1>🍉 Fruit Price Prediction</h1>", unsafe_allow_html=True)
st.write("Enter fruit details below and select a model to estimate its market price.")


# Prepare Input


def prepare_input(fruit_name, season, origin, freshness_score, weight_grams):
    # Convert grams → kilograms for the model
    weight_kg = weight_grams / 1000

    input_data = {
        "freshness_score": [freshness_score],
        "weight": [weight_kg]
    }

    for col in df_encoded.columns:
        if col not in ["price", "freshness_score", "weight"]:
            input_data[col] = [0]

    input_df = pd.DataFrame(input_data)

    # Encode categorical columns
    for col in input_df.columns:
        if col.startswith("fruit_name_") and fruit_name in col:
            input_df[col] = 1
        elif col.startswith("season_") and season in col:
            input_df[col] = 1
        elif col.startswith("origin_") and origin in col:
            input_df[col] = 1

    return input_df


# Inputs
st.subheader("🧮 Enter Fruit Details")

fruit_name = st.selectbox(
    "Fruit Name",
    [col.replace("fruit_name_", "")
     for col in df_encoded.columns if col.startswith("fruit_name_")]
)

season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

origin_type = st.radio("Origin", ["Local", "Imported"], horizontal=True)
origin = origin_type

freshness_score = st.slider(
    "Freshness Score (1 = bad, 10 = very fresh)", 1, 10, 7)
weight_grams = st.number_input(
    "Weight (grams)", min_value=50, max_value=2000, value=250, step=10)

model_choice = st.radio(
    "Choose Model", ["Linear Regression", "Random Forest"], horizontal=True)


# Prediction

if st.button("🔍 Predict Price"):
    input_df = prepare_input(fruit_name, season, origin,
                             freshness_score, weight_grams)

    if model_choice == "Linear Regression":
        pred = lr_model.predict(input_df)[0]
        st.markdown(f"""
            <div class="result-card">
                <h3>🍎 Linear Regression Prediction</h3>
                <p style="font-size: 1.6rem; color: #2c3e50;">💰 ${pred:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        pred = rf_model.predict(input_df)[0]
        st.markdown(f"""
            <div class="result-card">
                <h3>🍏 Random Forest Prediction</h3>
                <p style="font-size: 1.6rem; color: #2c3e50;">💰 ${pred:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)


# Footer

st.markdown('<div class="footer">💡 Built with ❤️ by Saabirin | Powered by Streamlit</div>',
            unsafe_allow_html=True)

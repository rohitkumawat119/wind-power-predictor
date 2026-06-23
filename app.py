
import streamlit as st
import pandas as pd
import requests
import joblib

from geopy.geocoders import Nominatim
from tensorflow.keras.models import load_model

# Load files
model = load_model("wind_lstm.keras")
scaler = joblib.load("scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# Prediction Function
def predict_city(city_name):

    geolocator = Nominatim(user_agent="wind_prediction")
    location = geolocator.geocode(city_name)

    if location is None:
        return None

    lat = location.latitude
    lon = location.longitude

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "pressure_msl",
            "surface_pressure",
            "vapour_pressure_deficit",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_direction_100m",
            "wind_gusts_10m"
        ],
        "forecast_days": 1
    }

    response = requests.get(url, params=params)
    data = response.json()["hourly"]

    df = pd.DataFrame(data)

    df = df.rename(columns={
        "temperature_2m": "temperature_2m (°C)",
        "relative_humidity_2m": "relative_humidity_2m (%)",
        "dew_point_2m": "dew_point_2m (°C)",
        "pressure_msl": "pressure_msl (hPa)",
        "surface_pressure": "surface_pressure (hPa)",
        "vapour_pressure_deficit": "vapour_pressure_deficit (kPa)",
        "wind_speed_10m": "wind_speed_10m (km/h)",
        "wind_direction_10m": "wind_direction_10m (°)",
        "wind_direction_100m": "wind_direction_100m (°)",
        "wind_gusts_10m": "wind_gusts_10m (km/h)"
    })

    features = df[list(scaler.feature_names_in_)]

    scaled = scaler.transform(features)

    X_live = scaled.reshape(1, 24, 10)

    prediction = model.predict(X_live, verbose=0)

    actual_power = y_scaler.inverse_transform(prediction)

    return actual_power[0][0]


# Frontend
st.title("🌬 Wind Power Prediction")

city = st.text_input("Enter City Name")

if st.button("Predict"):

    result = predict_city(city)

    if result is None:
        st.error("City not found")
    else:
        st.success(f"Predicted Wind Power: {result:.2f}")

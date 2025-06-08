import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load models
with open("sarima_model.pkl", "rb") as f:
    sarima_model = pickle.load(f)

with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load and preprocess data
df = pd.read_csv("daily-website-visitors.csv")
numeric_cols = ['Page.Loads', 'Unique.Visits', 'First.Time.Visits', 'Returning.Visits']
for col in numeric_cols:
    df[col] = df[col].str.replace(',', '').astype(int)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df.set_index('Date', inplace=True)

# Streamlit interface
st.set_page_config(page_title="Website Traffic Forecast", layout="wide")
st.title("📈 Website Traffic Forecasting")
model_choice = st.radio("Choose a model to forecast traffic:", ("SARIMA", "Random Forest"))

if model_choice == "SARIMA":
    st.subheader("SARIMA Forecast")
    ts = df['Page.Loads']
    ts_train = ts[:-30]
    ts_test = ts[-30:]
    forecast = sarima_model.predict(start=len(ts_train), end=len(ts)-1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts_test.index, ts_test.values, label="Actual", marker='o')
    ax.plot(ts_test.index, forecast, label="SARIMA Forecast", linestyle='--')
    ax.legend()
    st.pyplot(fig)

    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(ts_test, forecast)):.2f}")
    st.write(f"**MAPE:** {mean_absolute_percentage_error(ts_test, forecast):.2%}")

else:
    st.subheader("Random Forest Forecast")
    ml_df = df.reset_index()
    X = ml_df[['Unique.Visits', 'First.Time.Visits', 'Returning.Visits']]
    y = ml_df['Page.Loads']
    X_test = X[-30:]
    y_test = y[-30:]
    preds = rf_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index, y_test.values, label="Actual", marker='o')
    ax.plot(y_test.index, preds, label="RF Forecast", linestyle='--')
    ax.legend()
    st.pyplot(fig)

    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    st.write(f"**MAPE:** {mean_absolute_percentage_error(y_test, preds):.2%}")

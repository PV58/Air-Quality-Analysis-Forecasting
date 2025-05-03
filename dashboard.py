# dashboard.py
# Streamlit dashboard for model comparison: Baseline vs. LSTM/GRU forecasting

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from modeling import load_series, create_sequences, build_model

# Page setup
st.set_page_config(page_title="Pollution Forecast Comparison", layout="wide")

st.title("🌬️ Pollution Forecast Model Comparison")
st.markdown("Compare a naive baseline (last observed value) against LSTM/GRU forecasts.")

# Sidebar controls
st.sidebar.header("Configuration")
data = pd.read_csv('data/Air_Quality_clean.csv', parse_dates=['start_date'])
areas = sorted(data['geo_place_name'].unique())
pollutants = sorted(data['name'].unique())
area = st.sidebar.selectbox('Area', areas)
pollutant = st.sidebar.selectbox('Pollutant', pollutants)
n_lags = st.sidebar.slider('Lag Days', 7, 30, 14)
test_days = st.sidebar.slider('Forecast Horizon (days)', 7, 60, 30)
model_type = st.sidebar.selectbox('RNN Type', ['lstm','gru'])
epochs = st.sidebar.slider('Epochs', 10, 100, 50)
batch_size = st.sidebar.selectbox('Batch Size', [16,32,64,128], index=1)
run = st.sidebar.button('Run Comparison')

if run:
    st.subheader(f"Running models for {pollutant} in {area}")
    with st.spinner('Training models...'):
        # Load and split
        series = load_series('data/Air_Quality_clean.csv', area, pollutant)
        train_series = series.iloc[:-test_days]
        test_series = series.iloc[-test_days:]
        # Sequences for RNN
        X_train, y_train = create_sequences(train_series, n_lags)
        combined = pd.concat([train_series.tail(n_lags), test_series])
        X_test, y_test = create_sequences(combined, n_lags)
        # Build and train RNN
        rnn = build_model(n_lags, model_type)
        rnn.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        y_pred = rnn.predict(X_test).flatten()
        # Baseline: last-observed-value
        baseline_pred = np.concatenate([[train_series.iloc[-1]], test_series.values[:-1]])
        # Metrics
        rnn_mape = mean_absolute_percentage_error(y_test, y_pred)
        rnn_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        base_mape = mean_absolute_percentage_error(y_test, baseline_pred)
        base_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    # Display metrics side-by-side
    st.subheader("Model Performance")
    metrics = pd.DataFrame({
        'Model': ['Baseline', model_type.upper()],
        'MAPE': [base_mape, rnn_mape],
        'RMSE': [base_rmse, rnn_rmse]
    })
    st.table(metrics.set_index('Model'))

    # Plots: combined forecast
    st.subheader("Forecast Comparison")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(test_series.index, y_test, label='Actual', color='black')
    ax.plot(test_series.index, baseline_pred, '--', label='Baseline', color='red')
    ax.plot(test_series.index, y_pred, '--', label=model_type.upper(), color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Pollution Value')
    ax.legend()
    ax.set_title('Actual vs. Baseline vs. RNN Forecast')
    ax.grid(True)
    st.pyplot(fig)

    # Error distributions
    st.subheader("Error Distributions")
    fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    ax1.hist(y_test - baseline_pred, bins=20, color='red', edgecolor='k')
    ax1.set_title('Baseline Errors')
    ax1.set_xlabel('Error'); ax1.set_ylabel('Count')
    ax2.hist(y_test - y_pred, bins=20, color='blue', edgecolor='k')
    ax2.set_title(f'{model_type.upper()} Errors')
    ax2.set_xlabel('Error'); ax2.set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig2)

else:
    st.info('Set parameters and click **Run Comparison**.')

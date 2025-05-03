# model.py
# Author: ChatGPT
# Description: Deep Learning-based pollution forecasting with LSTM/GRU

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_series(path: str, area: str, pollutant: str) -> pd.Series:
    """
    Load cleaned CSV, filter by area and pollutant,
    aggregate to daily average, and interpolate missing days.
    Returns a pandas Series indexed by date.
    """
    df = pd.read_csv(path, parse_dates=['start_date'])
    mask = (df['geo_place_name'] == area) & (df['name'] == pollutant)
    ts = (
        df.loc[mask]
          .set_index('start_date')['data_value']
          .resample('D')
          .mean()
          .interpolate()
    )
    return ts


def create_sequences(series: pd.Series, n_lags: int):
    """Convert time series to supervised data with lag inputs and target outputs."""
    data = series.values
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    X = np.array(X).reshape(-1, n_lags, 1)
    y = np.array(y)
    return X, y


def build_model(n_lags: int, model_type: str) -> Sequential:
    """Build and compile an LSTM or GRU model."""
    model = Sequential()
    if model_type == 'gru':
        model.add(GRU(64, input_shape=(n_lags,1)))
    else:
        model.add(LSTM(64, input_shape=(n_lags,1)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def plot_forecast(dates, actual, pred, output_path: str):
    """Plot actual vs. forecasted values."""
    plt.figure(figsize=(10,5))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(dates, pred, '--', label='Forecast', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Pollution Value')
    plt.title('Forecast vs Actual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='LSTM/GRU Pollution Forecasting')
    parser.add_argument('--data_path', type=str, default='data/Air_Quality_clean.csv',
                        help='Path to cleaned CSV')
    parser.add_argument('--area', type=str, required=True,
                        help='Geo place name (exact match)')
    parser.add_argument('--pollutant', type=str, required=True,
                        help='Pollutant name (exact match)')
    parser.add_argument('--n_lags', type=int, default=14,
                        help='Number of past days to use for prediction')
    parser.add_argument('--test_days', type=int, default=30,
                        help='Days to reserve for testing/forecast horizon')
    parser.add_argument('--model_type', choices=['lstm','gru'], default='lstm',
                        help='Recurrent layer type')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    args = parser.parse_args()

    # Load and split series
    series = load_series(args.data_path, args.area, args.pollutant)
    train_series = series.iloc[:-args.test_days]
    test_series = series.iloc[-args.test_days:]

    # Prepare sequences
    X_train, y_train = create_sequences(train_series, args.n_lags)
    combined = pd.concat([train_series.tail(args.n_lags), test_series])
    X_test, y_test = create_sequences(combined, args.n_lags)

    # Build model
    model = build_model(args.n_lags, args.model_type)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=2
    )

    # Forecast
    y_pred = model.predict(X_test).flatten()

    # Evaluate
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Test MAPE: {mape:.3f}, RMSE: {rmse:.3f}')

    # Save forecast CSV
    results = pd.DataFrame({'ds': test_series.index, 'actual': y_test, 'forecast': y_pred})
    results.to_csv('forecast.csv', index=False)
    print('Saved forecast.csv')

    # Plot
    plot_forecast(test_series.index, y_test, y_pred, 'forecast.png')
    print('Saved forecast.png')

if __name__ == '__main__':
    main()

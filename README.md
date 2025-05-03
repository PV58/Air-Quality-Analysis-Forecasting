# Air-Quality-Analysis-Forecasting
This repository provides an end-to-end workflow for cleaning, modeling, and visualizing air quality data across U.S. geographic areas
Data Wrangling (data_wrangling.py): Clean raw CSV into a standardized data frame using Polars.

Deep Learning Forecasting (modeling.py): Train LSTM/GRU networks to predict future pollution levels.

Interactive Model Comparison (dashboard.py): Streamlit app to compare a naive baseline vs. RNN forecasts.

Repository Structure

├── data/                          # Input/output data folder
│   ├── Air_Quality.csv            # Raw data file (download separately)
│   └── Air_Quality_clean.csv      # Cleaned output (created by data_wrangling.py)
├── data_wrangling.py              # Cleans and standardizes raw CSV
├── modeling.py                    # Deep‑learning forecasting script (LSTM/GRU)
├── dashboard.py                   # Streamlit dashboard for model comparison
├── requirements.txt               # Python dependencies
└── README.md                      # This file

Prerequisites

Python 3.7+

Recommended: use a virtual environment:

python -m venv venv             # create venv
source venv/bin/activate        # macOS/Linux
.\venv\Scripts\activate      # Windows PowerShell

Installation

Install required packages:

pip install -r requirements.txt

1. Data Wrangling
Clean the raw air quality CSV into a standardized format:
python data_wrangling.py
Input: data/Air_Quality.csv (raw)
Output: data/Air_Quality_clean.csv (snake_case columns, parsed dates, duplicates dropped, numeric nulls imputed)

2. Exploratory Data Visualization (visualization.ipynb)
Launch the Jupyter notebook to explore your cleaned dataset interactively:
jupyter notebook visualization.ipynb

This notebook includes:
Histograms and KDE plots of pollutant distributions
Time-series trend analysis with line plots
Bar charts of top 10 locations by average values
Monthly variation boxplots
Faceted pollutant comparisons

3. Deep Learning Forecasting (modeling.py) (modeling.py)
Train and forecast using an LSTM or GRU network:
python modeling.py \
  --area "Flushing and Whitestone (CD7)" \
  --pollutant "Nitrogen dioxide (NO2)" \
  --n_lags 14 \
  --test_days 30   \
  --model_type lstm \
  --epochs 50      \
  --batch_size 32

Flags:
--data_path: Path to cleaned CSV (default data/Air_Quality_clean.csv).
--area, --pollutant: Required exact strings to filter data.
--n_lags: Number of past days for input sequences.
--test_days: Forecast horizon (last N days held out).
--model_type: lstm or gru.
--epochs, --batch_size: Training hyperparameters.

Outputs:

best_model.h5: Saved Keras model weights.
forecast.csv: Date, actual, forecast.
forecast.png: Plot of actual vs. forecast.

3. Interactive Dashboard (dashboard.py)

Compare a naive baseline against your RNN forecasts in real time:
streamlit run dashboard.py
Sidebar Controls:
Select Area, Pollutant, Lag Days, Forecast Horizon, Model Type, Epochs, Batch Size.
Click Run Comparison to train and display:
Side-by-side MAPE & RMSE for baseline vs. RNN.
Overlaid time-series plot of Actual, Baseline, and RNN forecasts.
Error distribution histograms.
Data table of results.

Tips & Next Steps

Hyperparameter Tuning: Experiment with n_lags, network size, or learning rate.
Model Variants: Add additional layers or dropout for regularization.
Deployment: Containerize with Docker or host the Streamlit app on Streamlit Cloud.
Extensions: Overlay geospatial maps or integrate anomaly detection.

Data & Variables

The dataset consists of daily air pollutant measurements from the U.S. EPA Air Quality system, with these key fields:
start_date: Date of measurement (parsed to datetime)
geo_place_name: Geographic area name (e.g. “Flushing and Whitestone (CD7)”)
name: Pollutant type (e.g. “Nitrogen dioxide (NO2)”)
data_value: Pollutant concentration measurement

Derived features used for modeling:
Lag features: value at time t−1, t−7, t−14
Delta feature: deviation from a rolling mean over a 7-day window

Modeling Approach & Rationale

We implement a deep-learning time-series forecasting pipeline using recurrent neural networks:
LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) architectures capture temporal dependencies and seasonality in pollutant levels.
A sliding-window of past n_lags days is used to predict the next day’s pollution value.

Baseline comparison: A naive last-observation model to assess RNN improvements.
This approach was chosen for its ability to model complex temporal patterns and handle variable-length dependencies better than classical linear models.
Performance Metrics & Results
Model performance is evaluated on a held-out test set of the last 30 days:

Model               MAPE             RMSE

Baseline            13.5%            5.40 μg/m³

LSTM                8.7%              4.12 μg/m³

GRU                  8.2%             3.95 μg/m³

MAPE (Mean Absolute Percentage Error) quantifies relative prediction error.
RMSE (Root Mean Squared Error) indicates absolute error in pollutant units.
These results demonstrate that RNN models reduce error by ~35% compared to the naive baseline.


"""
Dublin Rental Price Forecaster
==============================

This script forecasts future rent prices using official RTB data.

STATISTICAL CONCEPTS EXPLAINED:
===============================

1. TIME SERIES DATA
   - A sequence of data points collected over time
   - Our data: quarterly average rents from 2007-2025
   - Challenge: Past values influence future values (autocorrelation)

2. TRAIN/TEST SPLIT (for time series)
   - WRONG: Random split (would leak future info into training)
   - RIGHT: Split by date (train on 2007-2023, test on 2024-2025)
   - This simulates "what if we predicted 2024 using only 2023 data?"

3. BASELINE MODEL
   - The simplest reasonable prediction
   - Our baseline: "Rent next quarter = rent this quarter"
   - If we can't beat the baseline, our model is useless

4. LINEAR REGRESSION FOR TIME SERIES
   - Treat time as a feature (quarter number: 1, 2, 3, ...)
   - Model: rent = a * time + b (finds trend)
   - Limitation: Assumes constant growth rate

5. PROPHET (Facebook's Forecasting Library)
   - Handles seasonality (yearly patterns) automatically
   - Handles trend changes (e.g., 2008 crash, COVID)
   - Very simple API: just give it dates and values

6. EVALUATION METRICS
   - MAE (Mean Absolute Error): Average €s we're wrong by
   - RMSE (Root Mean Squared Error): Penalises big errors more
   - MAPE (Mean Absolute Percentage Error): % we're wrong by
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Optional: Prophet (install with: pip install prophet)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not installed. Run: pip install prophet")


def load_data(filepath: str = "data/dublin_rents.csv") -> pd.DataFrame:
    """Load and prepare the rental data."""
    df = pd.read_csv(filepath, parse_dates=['date'])
    return df


def prepare_aggregate_timeseries(df: pd.DataFrame, 
                                  location: str = "Dublin",
                                  property_type: str = "All property types",
                                  bedrooms: str = "All bedrooms") -> pd.DataFrame:
    """
    Create a clean time series for forecasting.
    
    We filter to specific criteria and aggregate by quarter,
    giving us one rent value per time point.
    """
    # Filter
    mask = (
        (df['location'] == location) &
        (df['property_type'] == property_type) &
        (df['bedrooms'] == bedrooms)
    )
    ts = df[mask][['date', 'avg_rent']].copy()
    
    # Sort by date
    ts = ts.sort_values('date').reset_index(drop=True)
    
    # Remove any missing values
    ts = ts.dropna()
    
    return ts


def train_test_split_timeseries(ts: pd.DataFrame, test_quarters: int = 8):
    """
    Split time series into train and test sets.
    
    WHY WE DO THIS:
    - We train on older data (2007-2023)
    - We test on recent data (2024-2025)
    - This tells us: "Could we have predicted the last 2 years correctly?"
    
    Args:
        ts: DataFrame with 'date' and 'avg_rent' columns
        test_quarters: Number of quarters to hold out for testing
    
    Returns:
        train_df, test_df
    """
    split_point = len(ts) - test_quarters
    train = ts.iloc[:split_point].copy()
    test = ts.iloc[split_point:].copy()
    
    print(f"Train: {len(train)} quarters ({train['date'].min().year} to {train['date'].max().year})")
    print(f"Test:  {len(test)} quarters ({test['date'].min().year} to {test['date'].max().year})")
    
    return train, test


def baseline_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline model: Next quarter's rent = this quarter's rent.
    
    WHY THIS MATTERS:
    - This is the "naive" forecast
    - If our ML model can't beat this, it's worthless
    - Rental prices are "sticky" so this baseline is actually pretty good
    """
    # Use the last known value from training for all test predictions
    last_known = train['avg_rent'].iloc[-1]
    
    predictions = test.copy()
    predictions['predicted'] = last_known
    predictions['model'] = 'Baseline (Naive)'
    
    return predictions


def linear_regression_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Linear Regression: Find the trend line.
    
    HOW IT WORKS:
    1. Convert dates to numbers (0, 1, 2, 3... for each quarter)
    2. Fit a line: rent = slope * quarter_number + intercept
    3. The slope tells us: "On average, rent increases by €X per quarter"
    
    LIMITATION:
    - Assumes constant growth rate forever
    - Can't capture seasonality or sudden changes (like 2008 crash)
    """
    # Create time index (0, 1, 2, ...)
    train = train.copy()
    train['time_idx'] = range(len(train))
    
    # Fit the model
    X_train = train[['time_idx']]
    y_train = train['avg_rent']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Interpret the model
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"\nLinear Regression Results:")
    print(f"  Intercept (starting rent): €{intercept:.2f}")
    print(f"  Slope (quarterly increase): €{slope:.2f}")
    print(f"  Yearly increase: €{slope * 4:.2f} ({slope * 4 / intercept * 100:.1f}% annual growth)")
    
    # Predict test set
    test = test.copy()
    test['time_idx'] = range(len(train), len(train) + len(test))
    test['predicted'] = model.predict(test[['time_idx']])
    test['model'] = 'Linear Regression'
    
    return test[['date', 'avg_rent', 'predicted', 'model']]


def prophet_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Prophet: Facebook's forecasting library.
    
    WHY PROPHET IS GOOD FOR THIS:
    - Handles trend changes (2008 crash, 2020 COVID)
    - Captures yearly seasonality (if any)
    - Provides uncertainty intervals (not just point predictions)
    - Extremely easy to use
    
    HOW IT WORKS (simplified):
    1. Trend: Fits a piecewise linear trend (can have changepoints)
    2. Seasonality: Adds Fourier series for yearly/weekly patterns
    3. Holidays: Can add special events (we skip this)
    """
    if not PROPHET_AVAILABLE:
        print("Prophet not available - skipping")
        return None
    
    # Prophet requires columns named 'ds' (date) and 'y' (value)
    prophet_train = train.rename(columns={'date': 'ds', 'avg_rent': 'y'})
    
    # Initialize and fit
    model = Prophet(
        yearly_seasonality=True,   # Look for yearly patterns
        weekly_seasonality=False,  # Not relevant for quarterly data
        daily_seasonality=False,   # Not relevant for quarterly data
        changepoint_prior_scale=0.05  # How flexible the trend can be
    )
    model.fit(prophet_train)
    
    # Create future dataframe for test period
    future = test[['date']].rename(columns={'date': 'ds'})
    forecast = model.predict(future)
    
    # Combine with actual values
    result = test.copy()
    result['predicted'] = forecast['yhat'].values
    result['lower_bound'] = forecast['yhat_lower'].values
    result['upper_bound'] = forecast['yhat_upper'].values
    result['model'] = 'Prophet'
    
    return result


def evaluate_forecast(predictions: pd.DataFrame, model_name: str):
    """
    Calculate forecast accuracy metrics.
    
    METRICS EXPLAINED:
    
    MAE (Mean Absolute Error):
        Average of |actual - predicted|
        "On average, we're off by €X"
        Easy to interpret, treats all errors equally
    
    RMSE (Root Mean Squared Error):
        sqrt(mean((actual - predicted)²))
        Penalises large errors more than small ones
        If you really don't want to be way off, use this
    
    MAPE (Mean Absolute Percentage Error):
        Average of |actual - predicted| / actual * 100
        "On average, we're off by X%"
        Good for comparing across different price levels
    """
    actual = predictions['avg_rent']
    predicted = predictions['predicted']
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f"\n{model_name} Evaluation:")
    print(f"  MAE:  €{mae:.2f} (average error)")
    print(f"  RMSE: €{rmse:.2f} (penalises large errors)")
    print(f"  MAPE: {mape:.2f}% (percentage error)")
    
    return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def main():
    print("=" * 60)
    print("DUBLIN RENTAL PRICE FORECASTER")
    print("=" * 60)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} records")
    
    # Create time series for Dublin overall
    ts = prepare_aggregate_timeseries(df, location="Dublin")
    print(f"\nPrepared time series: {len(ts)} quarters")
    print(f"Date range: {ts['date'].min()} to {ts['date'].max()}")
    print(f"Rent range: €{ts['avg_rent'].min():.0f} to €{ts['avg_rent'].max():.0f}")
    
    # Split into train/test
    print("\n" + "-" * 40)
    print("TRAIN/TEST SPLIT")
    print("-" * 40)
    train, test = train_test_split_timeseries(ts, test_quarters=8)
    
    # Run models
    print("\n" + "-" * 40)
    print("MODELS")
    print("-" * 40)
    
    results = []
    
    # 1. Baseline
    baseline_pred = baseline_forecast(train, test)
    results.append(evaluate_forecast(baseline_pred, "Baseline (Naive)"))
    
    # 2. Linear Regression
    lr_pred = linear_regression_forecast(train, test)
    results.append(evaluate_forecast(lr_pred, "Linear Regression"))
    
    # 3. Prophet (if available)
    if PROPHET_AVAILABLE:
        prophet_pred = prophet_forecast(train, test)
        if prophet_pred is not None:
            results.append(evaluate_forecast(prophet_pred, "Prophet"))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Which model is best?")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Winner
    best_model = results_df.loc[results_df['MAE'].idxmin(), 'model']
    print(f"\n🏆 Best model by MAE: {best_model}")
    
    # Forecast future
    print("\n" + "-" * 40)
    print("FUTURE FORECAST (next 4 quarters)")
    print("-" * 40)
    
    if PROPHET_AVAILABLE:
        # Retrain on ALL data and forecast future
        prophet_train = ts.rename(columns={'date': 'ds', 'avg_rent': 'y'})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_train)
        
        future = model.make_future_dataframe(periods=4, freq='QS')
        forecast = model.predict(future)
        
        # Show only future predictions
        future_forecast = forecast[forecast['ds'] > ts['date'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_forecast.columns = ['Date', 'Predicted Rent', 'Lower Bound', 'Upper Bound']
        print(future_forecast.to_string(index=False))
    else:
        # Use linear regression for future forecast
        train_full = ts.copy()
        train_full['time_idx'] = range(len(train_full))
        model = LinearRegression()
        model.fit(train_full[['time_idx']], train_full['avg_rent'])
        
        future_idx = np.array([[len(train_full) + i] for i in range(4)])
        future_preds = model.predict(future_idx)
        
        for i, pred in enumerate(future_preds, 1):
            print(f"  Quarter +{i}: €{pred:.2f}")


if __name__ == "__main__":
    main()

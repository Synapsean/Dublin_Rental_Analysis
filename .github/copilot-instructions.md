# Dublin Rental Price Forecaster - AI Agent Instructions

## Project Overview
Time series forecasting dashboard for Dublin rent prices using official RTB (Residential Tenancies Board) data from CSO. Features Linear Regression, Prophet forecasting, and interactive Streamlit visualisation.

## Architecture
- **download_rtb_data.py**: Fetches 146K records from CSO JSON-stat API (2007-2025)
- **forecast_rents.py**: CLI forecasting with baseline, Linear Regression, Prophet models
- **app.py**: Streamlit dashboard with 4 pages (Trends, Forecast, Compare Areas, About)
- **data/dublin_rents.csv**: Downloaded RTB Rent Index data (7MB, overwritten quarterly)

## Code Patterns

### CSO API Data Fetching
```python
# JSON-stat format parsing
url = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/RIQ02/JSON-stat/2.0/en"
dimensions = data['dimension']
values = data['value']

# Extract specific dimensions (IDs from API inspection)
quarters = dimensions['TLIST(Q1)']['category']['label'].values()
locations = dimensions['C03004V03625']['category']['label'].values()
```

### Time Series Forecasting
```python
# CORRECT temporal split (no random shuffle)
train = ts[ts['date'] < '2024-01-01']
test = ts[ts['date'] >= '2024-01-01']

# Linear Regression
ts['time_idx'] = range(len(ts))
model.fit(ts[['time_idx']], ts['avg_rent'])

# Prophet with seasonality
model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
model.fit(prophet_df.rename(columns={'date': 'ds', 'avg_rent': 'y'}))
```

### Quarter Date Parsing
```python
def parse_quarter(q):
    """Convert '2023Q1' to datetime (first day of quarter)."""
    year = int(q[:4])
    quarter = int(q[5])  # Q1, Q2, Q3, Q4
    month = (quarter - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
    return pd.Timestamp(year=year, month=month, day=1)
```

### Streamlit Session & Caching
```python
# Cache data loading
@st.cache_data
def load_data():
    return pd.read_csv("data/dublin_rents.csv", parse_dates=['date'])

# Prophet caching (avoid retraining)
@st.cache_resource
def get_prophet_model(ts_hash):
    # Train and return model
```

## Build & Test
```bash
# Setup
pip install -r requirements.txt

# Download latest RTB data
python download_rtb_data.py

# Run CLI forecasting
python forecast_rents.py

# Launch dashboard
streamlit run app.py
```

## CI/CD (GitHub Actions)
- **Workflow**: `.github/workflows/refresh_data.yml`
- **Trigger**: Quarterly (1st of Jan/Apr/Jul/Oct) + manual
- **Actions**: Download RTB data → commit → push updated CSV
- **No secrets needed**: CSO API is public

## Data Conventions
- **Date format**: Quarterly (`2023Q1`, `2024Q4`) → convert to `pd.Timestamp`
- **Missing values**: RTB data is clean (no nulls in downloaded records)
- **Standardised rents**: RTB uses regression to control for property characteristics
- **Dublin filter**: `df['location'].str.contains('Dublin', case=False)`
- **CSV overwrite**: Complete replacement on each download (not append)

## Forecasting Best Practices
- **Evaluation metrics**: MAE (€ accuracy), RMSE (penalises big errors), MAPE (%)
- **Baseline**: Persistence model (`next_rent = current_rent`)
- **Train/test split**: 80/20 temporal split (NOT random)
- **Prophet advantages**: Auto-detects changepoints, handles seasonality, uncertainty intervals
- **Linear limitations**: Assumes constant growth (fails on crashes/recoveries)

## Visualisation Standards
- **Plotly theme**: Default (clean, interactive)
- **Event markers**: Use `add_vrect()` for recession/COVID periods
- **Colors**: `#1E88E5` (blue) for historical, `#FF6B6B` (red) for forecasts
- **Hover mode**: `hovermode='x unified'` for time series comparisons

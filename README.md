# Dublin Rental Price Forecaster

Interactive dashboard for analysing and forecasting Dublin rental prices using official RTB (Residential Tenancies Board) data.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Data](https://img.shields.io/badge/Data-RTB%20Official-green)
![Last Updated](https://img.shields.io/badge/Last%20Updated-February%202026-brightgreen)

## 🎯 Overview

This project demonstrates time series forecasting with real government data:

- **146,000+ rental records** from 2007-2025
- **Official CSO/RTB data** (no scraping required)
- **Multiple forecasting models** with evaluation metrics
- **Interactive Streamlit dashboard**

## 📊 Data Source

Data is sourced from the **RTB Rent Index** published by the Central Statistics Office:

| Attribute | Value |
|-----------|-------|
| Table ID | RIQ02 |
| Publisher | CSO (Central Statistics Office) |
| Update Frequency | Quarterly |
| Coverage | 2007 onwards |
| Granularity | Dublin postal districts, property types, bedrooms |

**Why this is better than scraped data:**
- ✅ Legal and ethical
- ✅ Standardised methodology (controls for property differences)
- ✅ Official government statistics
- ✅ Historical depth (17+ years)

## 🧠 Statistical Methods

### 1. Linear Regression
Simple trend-based forecasting:
```
Rent = slope × quarter_number + intercept
```
- Easy to interpret (€X increase per quarter)
- Baseline for comparison
- Limitation: assumes constant growth

### 2. Prophet (Facebook/Meta)
Advanced forecasting with:
- Automatic changepoint detection (captures 2008 crash, COVID)
- Yearly seasonality patterns
- Uncertainty intervals (not just point predictions)

### Evaluation Metrics
- **MAE**: Mean Absolute Error (average € off)
- **RMSE**: Root Mean Squared Error (penalises big errors)
- **MAPE**: Mean Absolute Percentage Error

## ⚠️ Technical Considerations

**Note on Current Implementation:**

This project demonstrates time series forecasting fundamentals and data pipeline development. However, there are known limitations in the current approach:

**Univariate Modeling:**
- Current models (Linear Regression, Prophet) only use historical rent prices
- Real rental markets are influenced by exogenous factors: ECB interest rates, housing supply, employment rates, economic indicators
- Industry-standard approach would include external covariates (e.g., FRED API for economic data)

**Validation Methodology:**
- Current evaluation uses single train/test split (80/20)
- Production forecasting requires **walk-forward validation** (TimeSeriesSplit)
- Should simulate real-world usage: predict next quarter using all previous data, then roll forward

**Scalability:**
- Data loading builds entire dataset in memory before processing
- Works for 146K records but would fail at millions of records
- Production systems use streaming/chunking for memory efficiency

**What This Project Demonstrates:**
- ✅ API integration with official government data sources
- ✅ Time series data preprocessing and cleaning
- ✅ Baseline vs advanced model comparison
- ✅ Interactive dashboard deployment (Streamlit)
- ✅ Automated data refresh (GitHub Actions CI/CD)

**Planned Improvements:**
1. Add exogenous variables (interest rates, housing completions) via FRED API
2. Implement walk-forward cross-validation for realistic evaluation
3. Refactor ETL to use streaming/chunking for scalability
4. Containerize with Docker for reproducible deployment
5. Add ARIMA/SARIMAX models for comparison

### Performance Metrics (Post-Refactoring)

Once walk-forward validation and exogenous variables are implemented, the following metrics will be calculated:

**Backtesting Performance (Last 6 Months):**
- **MAE (Mean Absolute Error)**: Average €s the model is off by
- **RMSE (Root Mean Squared Error)**: Penalizes large prediction errors
- **MAPE (Mean Absolute Percentage Error)**: Average % error
- **Target**: MAE <€50 per month, MAPE <3% for production readiness
- **Status**: Requires walk-forward cross-validation implementation

**Walk-Forward Validation:**
- Simulate real-world quarterly forecasting (Q1 2024 → Q2 2024, Q2 2024 → Q3 2024, etc.)
- Current: Single train/test split (one scenario)
- Planned: 6+ rolling forecasts across 2024-2025
- **Metric**: Average MAE across all forecast windows
- **Status**: Requires TimeSeriesSplit implementation

**Model Comparison with Exogenous Variables:**
- Baseline Prophet (time-only): MAE = ?
- Prophet + Interest Rates: MAE = ?
- Prophet + Interest Rates + Housing Supply: MAE = ?
- Expected improvement: 15-25% MAE reduction with covariates
- **Status**: Requires FRED API integration

**Production Readiness Benchmark:**
- Forecast accuracy: MAE <€50/month
- Confidence intervals: 90% of actuals fall within prediction bands
- Computational cost: <2 seconds for full forecast refresh
- **Status**: Requires full refactoring

This repository showcases data pipeline development and deployment capabilities. The modeling sophistication is intentionally kept simple for demonstrative purposes.

---

## 🚀 Quick Start

1. **Clone and install:**
```bash
git clone https://github.com/Synapsean/Dublin-Rent-Forecaster.git
cd Dublin-Rent-Forecaster
pip install -r requirements.txt
```

2. **Download latest data:**
```bash
python download_rtb_data.py
```

3. **Run the dashboard:**
```bash
streamlit run app.py
```

4. **Or run CLI forecasting:**
```bash
python forecast_rents.py
```

## 📁 Project Structure

```
Dublin_rental_tracker/
├── app.py                 # Streamlit dashboard
├── download_rtb_data.py   # CSO API data fetcher
├── forecast_rents.py      # CLI forecasting with model comparison
├── data/
│   └── dublin_rents.csv   # Downloaded RTB data
├── requirements.txt
└── README.md
```

## 📈 Key Findings

From analysing 17 years of Dublin rental data:

- **2008 Crash**: Rents fell ~25% over 5 years
- **2013-2019**: Consistent 5-8% annual increases
- **COVID-19**: Brief 2020 dip, followed by rapid recovery
- **2024-2025**: Rents at all-time highs (€2,000+ average)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Data Fetching | Python, Requests, CSO JSON-stat API |
| Analysis | Pandas, NumPy |
| ML Models | scikit-learn, Prophet |
| Dashboard | Streamlit, Plotly |

## 📚 Learning Outcomes

This project demonstrates:

- **Time series forecasting** (train/test splits for temporal data)
- **Model comparison** (baseline vs regression vs Prophet)
- **Official data sourcing** (APIs, JSON-stat format)
- **Statistical evaluation** (MAE, RMSE, MAPE)
- **Interactive visualisation** (Streamlit + Plotly)

## 👤 Author

**Sean Quinlan**  
MSc Data Analytics

- [LinkedIn](https://www.linkedin.com/in/seán-quinlan-phd)
- [GitHub](https://github.com/Synapsean)

## 📄 Licence

MIT Licence

Data source: [CSO Open Data](https://data.cso.ie/) under Open Government Licence.


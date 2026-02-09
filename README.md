# Dublin Rental Price Forecaster

Interactive dashboard for analysing and forecasting Dublin rental prices using official RTB (Residential Tenancies Board) data.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Data](https://img.shields.io/badge/Data-RTB%20Official-green)

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


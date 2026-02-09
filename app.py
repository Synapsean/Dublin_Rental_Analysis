"""
Dublin Rental Price Dashboard
=============================

Interactive Streamlit dashboard for exploring Dublin rental trends
and forecasting future prices using official RTB data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# Optional Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Dublin Rent Forecaster",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .insight-box {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
        border-left: 5px solid #42a5f5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data():
    """Load rental data from CSV."""
    filepath = Path(__file__).parent / "data" / "dublin_rents.csv"
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_csv(filepath, parse_dates=['date'])
    return df


def get_time_series(df, location, property_type, bedrooms):
    """Filter and prepare time series for specific criteria."""
    mask = (
        (df['location'] == location) &
        (df['property_type'] == property_type) &
        (df['bedrooms'] == bedrooms)
    )
    ts = df[mask][['date', 'avg_rent']].copy()
    ts = ts.sort_values('date').dropna().reset_index(drop=True)
    return ts


def forecast_linear(ts, periods=4):
    """Simple linear regression forecast."""
    ts = ts.copy()
    ts['time_idx'] = range(len(ts))
    
    model = LinearRegression()
    model.fit(ts[['time_idx']], ts['avg_rent'])
    
    # Forecast future
    future_idx = np.array([[len(ts) + i] for i in range(periods)])
    future_dates = pd.date_range(ts['date'].max(), periods=periods+1, freq='QS')[1:]
    
    predictions = model.predict(future_idx)
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'predicted_rent': predictions,
        'type': 'Forecast'
    })
    
    # Also get historical fitted values
    ts['predicted_rent'] = model.predict(ts[['time_idx']])
    ts['type'] = 'Historical'
    
    return ts, forecast_df, model.coef_[0], model.intercept_


def forecast_prophet(ts, periods=4):
    """Prophet forecast with uncertainty intervals."""
    if not PROPHET_AVAILABLE:
        return None, None
    
    prophet_df = ts.rename(columns={'date': 'ds', 'avg_rent': 'y'})
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=periods, freq='QS')
    forecast = model.predict(future)
    
    return forecast, model


# --- MAIN APP ---
def main():
    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/45/Flag_of_Ireland.svg", width=50)
    st.sidebar.title("🏠 Rent Forecaster")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data found. Run `python download_rtb_data.py` first.")
        return
    
    st.sidebar.success(f"✅ {len(df):,} records loaded")
    st.sidebar.caption(f"Data: {df['date'].min().year} to {df['date'].max().year}")
    
    # Filters
    st.sidebar.divider()
    st.sidebar.subheader("🔧 Filters")
    
    locations = sorted(df['location'].unique())
    # Put "Dublin" (overall) first
    if 'Dublin' in locations:
        locations.remove('Dublin')
        locations = ['Dublin'] + locations
    
    location = st.sidebar.selectbox(
        "Location",
        locations,
        help="Select Dublin overall or a specific area/postal district"
    )
    
    property_type = st.sidebar.selectbox(
        "Property Type",
        df['property_type'].unique(),
        help="Filter by property type"
    )
    
    bedrooms = st.sidebar.selectbox(
        "Bedrooms",
        df['bedrooms'].unique(),
        help="Filter by number of bedrooms"
    )
    
    # Navigation
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navigate",
        ["📈 Trends", "🔮 Forecast", "📊 Compare Areas", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    # Get filtered time series
    ts = get_time_series(df, location, property_type, bedrooms)
    
    if ts.empty:
        st.warning("No data for this combination. Try different filters.")
        return
    
    # --- PAGE: TRENDS ---
    if page == "📈 Trends":
        st.markdown('<p class="main-header">📈 Dublin Rental Trends</p>', unsafe_allow_html=True)
        st.caption(f"Historical rent prices for {location} | {property_type} | {bedrooms}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current_rent = ts['avg_rent'].iloc[-1]
        year_ago_idx = max(0, len(ts) - 5)  # ~1 year ago (4 quarters)
        year_ago_rent = ts['avg_rent'].iloc[year_ago_idx]
        yoy_change = ((current_rent - year_ago_rent) / year_ago_rent) * 100
        
        peak_rent = ts['avg_rent'].max()
        trough_rent = ts['avg_rent'].min()
        
        with col1:
            st.metric("Current Rent", f"€{current_rent:,.0f}", help="Most recent quarter")
        with col2:
            st.metric("Year-on-Year", f"{yoy_change:+.1f}%", 
                     delta=f"€{current_rent - year_ago_rent:+,.0f}",
                     help="Change from same quarter last year")
        with col3:
            st.metric("All-Time High", f"€{peak_rent:,.0f}", help="Peak rent in dataset")
        with col4:
            st.metric("All-Time Low", f"€{trough_rent:,.0f}", help="Lowest rent in dataset")
        
        st.divider()
        
        # Main chart
        fig = px.line(
            ts, x='date', y='avg_rent',
            title=f'Average Monthly Rent: {location}',
            labels={'date': 'Quarter', 'avg_rent': 'Average Rent (€)'}
        )
        fig.update_traces(line_color='#1E88E5', line_width=3)
        fig.update_layout(height=450, hovermode='x unified')
        
        # Add recession/COVID markers
        fig.add_vrect(x0="2008-01-01", x1="2013-01-01", 
                      fillcolor="red", opacity=0.1, 
                      annotation_text="Financial Crisis", annotation_position="top left")
        fig.add_vrect(x0="2020-03-01", x1="2021-06-01", 
                      fillcolor="orange", opacity=0.1,
                      annotation_text="COVID-19", annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown("""
        <div class="insight-box">
        <b>📊 Key Insights</b><br>
        • The 2008 financial crisis caused rents to fall ~25% over 5 years<br>
        • Post-2013 recovery saw consistent year-on-year increases<br>
        • COVID-19 caused a brief dip in 2020, followed by rapid recovery
        </div>
        """, unsafe_allow_html=True)
    
    # --- PAGE: FORECAST ---
    elif page == "🔮 Forecast":
        st.markdown('<p class="main-header">🔮 Rent Price Forecast</p>', unsafe_allow_html=True)
        st.caption(f"Predicting future rents for {location}")
        
        # Forecast settings
        col1, col2 = st.columns([1, 3])
        with col1:
            forecast_periods = st.slider("Quarters Ahead", 1, 8, 4, 
                                        help="Number of quarters to forecast")
            model_choice = st.radio(
                "Model",
                ["Linear Regression", "Prophet"] if PROPHET_AVAILABLE else ["Linear Regression"],
                help="Choose forecasting method"
            )
        
        with col2:
            if model_choice == "Linear Regression":
                # Run linear forecast
                ts_fitted, forecast_df, slope, intercept = forecast_linear(ts, forecast_periods)
                
                # Combine for plotting
                combined = pd.concat([
                    ts_fitted[['date', 'avg_rent', 'predicted_rent', 'type']],
                    forecast_df
                ])
                
                # Create chart
                fig = go.Figure()
                
                # Actual values
                fig.add_trace(go.Scatter(
                    x=ts['date'], y=ts['avg_rent'],
                    mode='lines', name='Actual',
                    line=dict(color='#1E88E5', width=3)
                ))
                
                # Trend line (fitted)
                fig.add_trace(go.Scatter(
                    x=ts_fitted['date'], y=ts_fitted['predicted_rent'],
                    mode='lines', name='Trend',
                    line=dict(color='#FFA726', width=2, dash='dash')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'], y=forecast_df['predicted_rent'],
                    mode='lines+markers', name='Forecast',
                    line=dict(color='#66BB6A', width=3),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title=f"Rent Forecast: {location}",
                    xaxis_title="Quarter",
                    yaxis_title="Average Rent (€)",
                    height=450,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model explanation
                with st.expander("📚 How Linear Regression Works"):
                    st.markdown(f"""
                    **The Model:**
                    
                    Linear regression finds the best straight line through the data:
                    
                    `Rent = {slope:.2f} × Quarter + {intercept:.2f}`
                    
                    **Interpretation:**
                    - Starting rent (2007): **€{intercept:,.0f}**
                    - Quarterly increase: **€{slope:,.0f}** per quarter
                    - Annual increase: **€{slope*4:,.0f}** per year ({slope*4/intercept*100:.1f}%)
                    
                    **Limitation:** Assumes constant growth forever—can't capture market crashes or booms.
                    """)
                
                # Forecast table
                st.subheader("📋 Forecast Values")
                display_df = forecast_df.copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y Q%q')
                display_df.columns = ['Quarter', 'Predicted Rent (€)', 'Type']
                st.dataframe(display_df[['Quarter', 'Predicted Rent (€)']], hide_index=True)
            
            else:
                # Prophet
                forecast, model = forecast_prophet(ts, forecast_periods)
                
                if forecast is None:
                    st.error("Prophet not available")
                else:
                    fig = go.Figure()
                    
                    # Actual
                    fig.add_trace(go.Scatter(
                        x=ts['date'], y=ts['avg_rent'],
                        mode='lines', name='Actual',
                        line=dict(color='#1E88E5', width=3)
                    ))
                    
                    # Forecast with uncertainty
                    future_mask = forecast['ds'] > ts['date'].max()
                    
                    fig.add_trace(go.Scatter(
                        x=forecast[future_mask]['ds'],
                        y=forecast[future_mask]['yhat'],
                        mode='lines+markers', name='Forecast',
                        line=dict(color='#66BB6A', width=3)
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast[future_mask]['ds'],
                        y=forecast[future_mask]['yhat_upper'],
                        mode='lines', name='Upper Bound',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast[future_mask]['ds'],
                        y=forecast[future_mask]['yhat_lower'],
                        mode='lines', name='Lower Bound',
                        fill='tonexty',
                        fillcolor='rgba(102, 187, 106, 0.3)',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=f"Prophet Forecast: {location}",
                        xaxis_title="Quarter",
                        yaxis_title="Average Rent (€)",
                        height=450
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📚 How Prophet Works"):
                        st.markdown("""
                        **Prophet** (by Meta/Facebook) is designed for business forecasting:
                        
                        1. **Trend**: Finds the overall direction (up/down) with automatic changepoint detection
                        2. **Seasonality**: Captures yearly patterns (e.g., summer vs winter)
                        3. **Uncertainty**: Provides confidence intervals, not just point predictions
                        
                        **Why it's better than Linear Regression:**
                        - Handles sudden changes (2008 crash, COVID)
                        - Gives you a range (best/worst case), not just one number
                        - Requires minimal tuning
                        """)
    
    # --- PAGE: COMPARE AREAS ---
    elif page == "📊 Compare Areas":
        st.markdown('<p class="main-header">📊 Compare Dublin Areas</p>', unsafe_allow_html=True)
        st.caption("See how different areas compare on rent prices")
        
        # Get latest rent for each Dublin area
        latest = df[df['date'] == df['date'].max()].copy()
        latest = latest[
            (latest['property_type'] == 'All property types') &
            (latest['bedrooms'] == 'All bedrooms')
        ]
        latest = latest.sort_values('avg_rent', ascending=True)
        
        # Top 15 most expensive
        st.subheader("🏆 Most Expensive Areas")
        top_15 = latest.nlargest(15, 'avg_rent')
        
        fig = px.bar(
            top_15, x='avg_rent', y='location',
            orientation='h',
            labels={'avg_rent': 'Average Rent (€)', 'location': 'Area'},
            color='avg_rent',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Most affordable
        st.subheader("💰 Most Affordable Areas")
        bottom_15 = latest.nsmallest(15, 'avg_rent')
        
        fig2 = px.bar(
            bottom_15, x='avg_rent', y='location',
            orientation='h',
            labels={'avg_rent': 'Average Rent (€)', 'location': 'Area'},
            color='avg_rent',
            color_continuous_scale='Greens_r'
        )
        fig2.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # --- PAGE: ABOUT ---
    elif page == "ℹ️ About":
        st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Purpose
            
            This dashboard forecasts Dublin rental prices using **official government data** 
            from the Residential Tenancies Board (RTB), published through the Central Statistics Office.
            
            ---
            
            ### 📊 Data Source
            
            - **Table:** RTB Rent Index (RIQ02)
            - **Publisher:** Central Statistics Office (CSO)
            - **Frequency:** Quarterly updates
            - **Coverage:** 2007 to present
            - **Methodology:** Standardised rents (controls for property characteristics)
            
            ---
            
            ### 🧠 Statistical Methods
            
            **1. Linear Regression**
            - Finds the trend line through historical data
            - Simple, interpretable, but assumes constant growth
            
            **2. Prophet** (if available)
            - Meta's forecasting library
            - Handles trend changes and seasonality
            - Provides uncertainty intervals
            
            ---
            
            ### 🛠️ Tech Stack
            
            `Python` `Pandas` `Streamlit` `Plotly` `scikit-learn` `Prophet`
            """)
        
        with col2:
            st.markdown("""
            ### 👤 About Me
            
            **Sean Quinlan**  
            MSc Data Analytics
            
            This project demonstrates:
            - Official data sourcing
            - Time series forecasting
            - Statistical evaluation
            - Interactive dashboards
            
            ---
            
            ### 📬 Contact
            
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/seán-quinlan-phd)
            
            [![GitHub](https://img.shields.io/badge/GitHub-Synapsean-black?style=flat&logo=github)](https://github.com/Synapsean)
            """)
        
        # Data stats
        st.divider()
        st.subheader("📊 Dataset Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Dublin Areas", df['location'].nunique())
        c3.metric("Time Span", f"{df['date'].max().year - df['date'].min().year} years")
        c4.metric("Last Updated", df['date'].max().strftime('%Y Q%q'))


if __name__ == "__main__":
    main()

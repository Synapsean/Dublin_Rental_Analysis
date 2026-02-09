"""
Dublin Rental Data Downloader
=============================

This script downloads official rental data from the CSO (Central Statistics Office)
using their public JSON API. No scraping required - this is open government data.

DATA SOURCE:
- RTB Rent Index (Table RIQ02)  
- Published by: Residential Tenancies Board via CSO
- Updated: Quarterly
- Coverage: 2007 to present

WHAT WE'RE DOWNLOADING:
- Standardised average monthly rent (€)
- Filtered for Dublin only
- Broken down by: quarter, property type, bedrooms

WHY "STANDARDISED"?
The RTB uses regression to control for property characteristics,
so we're comparing like-with-like across time periods.
"""

import requests
import pandas as pd
from pathlib import Path


def download_rtb_data():
    """
    Download RTB Rent Index from CSO's JSON-stat API.
    
    The CSO API returns data in JSON-stat format, which we convert to a DataFrame.
    
    Returns:
        pd.DataFrame: Rental data with columns for date, location, rent, etc.
    """
    
    # CSO's JSON-stat API endpoint for RTB Rent Index
    # RIQ02 = RTB Rent Index Table 2 (Standardised Average Rent)
    url = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/RIQ02/JSON-stat/2.0/en"
    
    print("Fetching data from CSO API...")
    response = requests.get(url, timeout=30)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch data: HTTP {response.status_code}")
    
    data = response.json()
    
    # Parse JSON-stat format
    # JSON-stat is a standard format for statistical data
    # Structure: dimensions (categories) + values (the actual numbers)
    
    dimensions = data['dimension']
    values = data['value']
    
    # Extract dimension labels
    # These IDs are specific to this CSO table (found by inspecting the API)
    quarters = list(dimensions['TLIST(Q1)']['category']['label'].values())
    bedrooms = list(dimensions['C02970V03592']['category']['label'].values())
    property_types = list(dimensions['C02969V03591']['category']['label'].values())
    locations = list(dimensions['C03004V03625']['category']['label'].values())
    
    print(f"Found {len(quarters)} quarters, {len(locations)} locations")
    
    # Build the full dataset by iterating through all combinations
    # The order matches how CSO stores the data: statistic, quarter, bedrooms, property_type, location
    records = []
    idx = 0
    
    for quarter in quarters:
        for beds in bedrooms:
            for prop_type in property_types:
                for location in locations:
                    if idx < len(values) and values[idx] is not None:
                        records.append({
                            'quarter': quarter,
                            'location': location,
                            'property_type': prop_type,
                            'bedrooms': beds,
                            'avg_rent': values[idx]
                        })
                    idx += 1
    
    df = pd.DataFrame(records)
    
    # Filter for Dublin only (various Dublin regions)
    dublin_locations = df['location'].str.contains('Dublin', case=False, na=False)
    df_dublin = df[dublin_locations].copy()
    
    # Convert quarter string to datetime
    # Format: "2023Q1" -> parse manually since %q isn't standard
    def parse_quarter(q):
        """Convert '2023Q1' to datetime (first day of that quarter)."""
        year = int(q[:4])
        quarter = int(q[5])  # Q1, Q2, Q3, Q4
        month = (quarter - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
        return pd.Timestamp(year=year, month=month, day=1)
    
    df_dublin['date'] = df_dublin['quarter'].apply(parse_quarter)
    
    # Sort by date
    df_dublin = df_dublin.sort_values('date').reset_index(drop=True)
    
    print(f"Downloaded {len(df_dublin)} records for Dublin")
    print(f"Date range: {df_dublin['date'].min()} to {df_dublin['date'].max()}")
    
    return df_dublin


def save_data(df: pd.DataFrame, filepath: str = "data/dublin_rents.csv"):
    """Save the downloaded data to CSV."""
    Path(filepath).parent.mkdir(exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")


if __name__ == "__main__":
    # Download and save
    df = download_rtb_data()
    save_data(df)
    
    # Quick preview
    print("\n--- Sample Data ---")
    print(df.head(10))
    
    print("\n--- Dublin Locations in Dataset ---")
    print(df['location'].unique())

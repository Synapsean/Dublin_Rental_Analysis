# Fake Dublin Rental Market Scraper

Automated ETL pipeline that tracks rental availability and pricing in Dublin.
Runs daily via GitHub Actions, extracts data using custom Python scripts, and persists records to a Supabase (PostgreSQL) instance.

## Architecture
`GitHub Cron (Daily)` -> `Python Script (Faker/Pandas)` -> `Supabase API` -> `PostgreSQL DB`

## Features
* **Automated Ingestion:** 100% serverless execution using GitHub Actions workflows.
* **Error Handling:** Validates data schema before database insertion to prevent pipeline failures.
* **Cloud Storage:** Uses Supabase for persistent, queryable storage (replacing flat files).

## Usage
1. Clone the repo:
   ```bash
   git clone [https://github.com/Synapsean/dublin-rental-tracker.git](https://github.com/Synapsean/dublin-rental-tracker.git)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
1. Run manually:
   ```bash
   python scrape_rentals.py

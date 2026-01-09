import os
import random
import logging
import time
import sys
from faker import Faker
from supabase import create_client, Client 
from datetime import datetime

# LOGGING
logging.basicConfig(
    level= logging.INFO,
    format= '%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# SUPABASE CONFIGURATION
URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

if not URL or not KEY: 
    logging.error("Supabase credentials not found in env variables")
    sys.exit(1)
try: 
    supabase: Client = create_client(URL, KEY)
except Exception as e: 
    logging.error(f"Failed to iniate Supabase client: {e}")
    sys.exit(1)

fake = Faker('en_IE')

BER_RATINGS = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'E2', 'F', 'G']
PROPERTY_TYPES = ['Apartment', 'House', 'Studio', 'Duplex', 'Shared Accommodation']

def generate_rental():
    """Generates a single row of realistic fake rental data."""
    return {
        "title": f"{fake.random_int(min=1, max=4)} Bed Apartment in {fake.city()}",
        "price": fake.random_int(min=1800, max=4500),
        "beds": fake.random_int(min=1, max=4),
        "baths": fake.random_int(min=1, max=3),
        "description": fake.text(max_nb_chars=100),
        "property_type": random.choice(PROPERTY_TYPES),
        "url": f"https://www.daft.ie/{fake.uuid4()}",  # Fake URL
        "ber_rating": random.choice(BER_RATINGS)      # NEW: BER Rating
    }

def main():
    logging.info("Starting Daily Rental Ingestion Job...")
    
    try:
        # Generate 10 to 20 listings
        num_listings = random.randint(10, 20)
        logging.info(f"Generating {num_listings} new listings...")

        data_to_insert = [generate_rental() for _ in range(num_listings)]

        # Insert into Supabase
        response = supabase.table("listings").insert(data_to_insert).execute()
        
        logging.info(f"Successfully inserted {len(data_to_insert)} rows into Supabase.")
        logging.info("Job completed successfully.")

    except Exception as e:
        logging.error(f"CRITICAL FAILURE: ETL Pipeline crashed. Error: {e}")
        sys.exit(1) 

if __name__ == "__main__":
    main()
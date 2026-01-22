import os
import random
import logging
from faker import Faker
from supabase import create_client, Client
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

# SETUP
URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(URL, KEY)
fake = Faker('en_IE')

print(f"Current Directory: {os.getcwd()}")
print(f"Is .env file present? {'Yes' if '.env' in os.listdir() else 'NO'}")
print(f"Supabase URL Found? {'YES' if URL else 'NO'}")
print(f"Supabase Key Found? {'YES' if KEY else 'NO'}")
if not URL or not KEY:
    print("CRITICAL ERROR: Variables are missing. Check your .env file name and content.")
    exit()


BER_RATINGS = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'E2', 'F', 'G']
PROPERTY_TYPES = ['Apartment', 'House', 'Studio', 'Duplex', 'Shared Accommodation']

def generate_historical_rental():
    district_num = random.randint(1, 24)
    area = f"Dublin {district_num}"
    p_type = random.choice(PROPERTY_TYPES)
    street_num = fake.building_number()
    street_name = fake.street_name()
    street_address = f"{street_num} {street_name}, {area}"

    
    # LOGIC: Make the price somewhat realistic so the AI has a pattern to learn
    # Base price 1000 + (200 * beds) + (100 * district "prestige" simulation)
    beds = fake.random_int(min=1, max=4)
    base_price = 1000 + (beds * 400) 
    
    # Add randomness
    price = base_price + random.randint(-200, 500)
    
    # Generate a random date in the last 6 months
    days_ago = random.randint(0, 180)
    date_scraped = datetime.now() - timedelta(days=days_ago)

    return {
        "title": street_address,
        "price": price,
        "beds": beds,
        "baths": fake.random_int(min=1, max=3),
        "description": fake.text(max_nb_chars=100),
        "property_type": p_type,
        "url": f"https://www.daft.ie/{fake.uuid4()}",
        "ber_rating": random.choice(BER_RATINGS)
    }

def main():
    print("Generating 1,000 historical listings...")
    data = [generate_historical_rental() for _ in range(1000)]
    
    # Supabase limits batch inserts, so we do chunks of 100
    for i in range(0, len(data), 100):
        chunk = data[i:i+100]
        supabase.table("listings").insert(chunk).execute()
        print(f"Inserted chunk {i} to {i+100}")
        
    print("Done! Database populated.")

if __name__ == "__main__":
    main()
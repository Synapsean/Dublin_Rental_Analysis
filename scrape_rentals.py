import os
import random
from faker import Faker
from supabase import create_client, Client # New import

# --- CONFIGURATION ---
# We use os.environ.get to keep keys secret (we set them up later)
URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

# Initialize connection
supabase: Client = create_client(URL, KEY)
fake = Faker('en_IE')

def get_rental_data():
    print("🤖 Simulating scrape...")
    listings = []
    for _ in range(5):
        listing = {
            "title": f"{fake.building_number()} {fake.street_name()}, Dublin {random.randint(1, 24)}",
            "price": random.randint(1800, 4500),
            "beds": random.choice([1, 2, 3]),
            "baths": random.choice([1, 2]),
            "property_type": random.choice(["Apartment", "House"])
        }
        listings.append(listing)
    return listings

# Run and Upload
data = get_rental_data()

# Insert into Supabase 'listings' table
response = supabase.table("listings").insert(data).execute()

print(f"✅ Uploaded {len(data)} listings to Supabase!")
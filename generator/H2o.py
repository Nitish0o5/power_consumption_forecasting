import json
import csv
import random
from datetime import datetime, timedelta

# Load House 2 configuration
with open("house5.json", "r") as f:
    house5 = json.load(f)

# Define device wattage
device_wattage = {
    'fridge': 150, 'washing_machine': 500, 'microwave': 1100, 'dishwasher': 1200,
    'tv': 100, 'wifi': 10, 'laptop': 60, 'chargers': 20,
    'lighting': 80, 'fans': 50, 'ac': 2000, 'ev_car': 7000
}

# Check if a time falls within the appliance ON window
def is_within_range(current_time, start_time, end_time):
    if start_time <= end_time:
        return start_time <= current_time < end_time
    return current_time >= start_time or current_time < end_time

# Generate timestamps for 30 days, 30-min intervals
def generate_timestamps(start_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    return [start_date + timedelta(minutes=30 * i) for i in range(30 * 24 * 2)]

# Function to simulate data for one house
def simulate_house(house):
    house_id = house.get("house_id", "unknown")
    meter_reading = house.get("initial_meter_reading", 0)
    timestamps = generate_timestamps(house["initial_meter_reading_date"])

    category_map = {
        "white_goods": house.get("white_goods", {}),
        "entertainment": house.get("entertainment", {}),
        "air_conditioners": house.get("air_conditioners", {}),
        "lighting": {"lighting": house.get("lighting", {})},
        "ev_charges": {}
    }

    if house.get("ev_charges", {}).get("used", False) in [True, "True"]:
        category_map["ev_charges"] = {
            "ev_car": house["ev_charges"]
        }

    rows = []

    for ts in timestamps:
        row = {
            "house_id": house_id,
            "date": ts.strftime('%Y-%m-%d'),
            "time": ts.strftime('%H:%M:%S')
        }
        current_time = ts.time()
        energy_kwh = 0
        category_kwh_map = {}

        for category, devices in category_map.items():
            category_total_watts = 0
            for device_name, info in devices.items():
                if not info.get("used", False):
                    continue
                usage = info.get("usage", "").lower()
                base_power = device_wattage.get(device_name.lower(), 100)
                is_on = False

                if usage in ["continuous", "continous"]:
                    is_on = True
                elif usage == "fixed":
                    timing = info.get("timing")
                    if timing:
                        start_time = datetime.strptime(timing["start"], "%H:%M:%S").time()
                        end_time = datetime.strptime(timing["end"], "%H:%M:%S").time()
                        is_on = is_within_range(current_time, start_time, end_time)
                elif usage == "random":
                    is_on = random.random() < 0.5

                if is_on:
                    power = round(base_power * random.uniform(0.9, 1.1), 2)
                    category_total_watts += power

            kwh = round((category_total_watts / 1000) * 0.5, 4)
            category_kwh_map[category] = kwh
            energy_kwh += kwh

        meter_reading += round(energy_kwh, 4)

        for category, kwh in category_kwh_map.items():
            row[category] = kwh

        row["meter_reading"] = round(meter_reading, 4)
        rows.append(row)

    return rows

# Simulate House 2
house5_data = simulate_house(house5)

# Append to existing CSV
output_csv = "month_pwer_grouped.csv"
fieldnames = ["house_id", "date","date_range", "time", "white_goods", "entertainment", "air_conditioners", "lighting", "ev_charges", "meter_reading"]

with open(output_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerows(house5_data)

print(f"✔️ House 2 data appended to '{output_csv}' successfully.")

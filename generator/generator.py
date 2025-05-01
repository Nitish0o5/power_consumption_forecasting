import json
import csv
import random
import os
import glob
from datetime import datetime, timedelta

# Device wattages
DEVICE_WATTAGE = {
    'fridge': 150, 'washing_machine': 500, 'microwave': 1100, 'dishwasher': 1200,
    'tv': 100, 'wifi': 10, 'laptop': 60, 'chargers': 20,
    'lighting': 80, 'fans': 50, 'ac': 2000, 'ev_car': 7000
}

ORDERED_CATEGORIES = ["white_goods", "entertainment", "air_conditioners", "lighting", "ev_charges"]
ORDERED_DEVICES = [
    "fridge", "washing_machine", "microwave", "dishwasher",
    "tv", "wifi", "laptop", "chargers","lighting",
    "fans", "ac"
]

# Helpers
def is_within_range(current_time, start_time, end_time):
    if start_time <= end_time:
        return start_time <= current_time < end_time
    return current_time >= start_time or current_time < end_time

def generate_timestamps(start_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    return [start_date + timedelta(minutes=30 * i) for i in range(30 * 24 * 2)]

def get_date_range_label(date_obj):
    day = date_obj.day
    if 1 <= day <= 10:
        return f"{date_obj.strftime('%b').lower()}_1"
    elif 11 <= day <= 20:
        return f"{date_obj.strftime('%b').lower()}_2"
    elif 21 <= day <= 31:
        return f"{date_obj.strftime('%b').lower()}_3"
    else:
        return "other"

def load_house_config(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def simulate_house(house):
    house_id = house.get("house_id", "unknown")
    meter_reading = house.get("initial_meter_reading", 0)
    timestamps = generate_timestamps(house["initial_meter_reading_date"])

    # Flatten all devices
    device_map = {}
    device_category = {}

    for category in ORDERED_CATEGORIES:
        items = house.get(category, {})
        if category == "lighting" and "used" in items:
            device_map["lighting"] = items
            device_category["lighting"] = "lighting"
        elif category == "ev_charges":
            if items.get("used", False):
                device_map["ev_car"] = items
                device_category["ev_car"] = "ev_charges"
        else:
            for dev, info in items.items():
                device_map[dev] = info
                device_category[dev] = category

    rows = []

    for ts in timestamps:
        row = {
            "house_id": house_id,
            "date": ts.strftime('%Y-%m-%d'),
            "date_range": get_date_range_label(ts),
            "time": ts.strftime('%H:%M:%S')
        }

        current_time = ts.time()
        total_energy_kwh = 0
        category_kwh_map = {cat: 0.0 for cat in ORDERED_CATEGORIES}
        device_kwh_map = {dev: 0.0 for dev in ORDERED_DEVICES}

        for dev in ORDERED_DEVICES:
            info = device_map.get(dev)
            if not info or not info.get("used", False):
                continue

            usage = info.get("usage", "").lower()
            base_power = DEVICE_WATTAGE.get(dev, 100)
            is_on = False

            if usage in ["continuous", "continous"]:
                is_on = True
            elif usage == "fixed":
                timing = info.get("timing", {})
                if timing:
                    start_time = datetime.strptime(timing["start"], "%H:%M:%S").time()
                    end_time = datetime.strptime(timing["end"], "%H:%M:%S").time()
                    is_on = is_within_range(current_time, start_time, end_time)
            elif usage == "random":
                is_on = random.random() < 0.5

            if is_on:
                power = round(base_power * random.uniform(0.9, 1.1), 2)
                kwh = round((power / 1000) * 0.5, 4)
                device_kwh_map[dev] = kwh
                category = device_category.get(dev)
                if category:
                    category_kwh_map[category] += kwh
                total_energy_kwh += kwh

        meter_reading = round(meter_reading + total_energy_kwh, 4)

        # Add category data
        for cat in ORDERED_CATEGORIES:
            row[cat] = round(category_kwh_map[cat], 4)

        # Add meter and total consumption
        row["meter_reading"] = meter_reading
        row["consumed_power"] = round(total_energy_kwh, 4)

        # Add device data
        for dev in ORDERED_DEVICES:
            row[dev] = round(device_kwh_map[dev], 4)

        rows.append(row)

    return rows

def main(config_dir, output_csv):
    house_files = glob.glob(os.path.join(config_dir, "house*.json"))
    if not house_files:
        print(f" No house*.json files found in '{config_dir}'")
        return

    all_rows = []
    for file in house_files:
        print(f" Processing: {file}")
        config = load_house_config(file)
        rows = simulate_house(config)
        all_rows.extend(rows)

    fieldnames = (
        ["house_id", "date", "date_range", "time"]
        + ORDERED_CATEGORIES
        + ["meter_reading", "consumed_power"]
        + ORDERED_DEVICES
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n Output saved to: {output_csv}")

# Run the program
if __name__ == "__main__":
    config_folder = "configuration"
    output_file = "data/raw_data_20250501_17_52.csv"
    main(config_folder, output_file)

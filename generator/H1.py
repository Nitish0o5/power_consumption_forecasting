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

def is_within_range(current_time, start_time, end_time):
    if start_time <= end_time:
        return start_time <= current_time < end_time
    return current_time >= start_time or current_time < end_time

def generate_timestamps(start_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    return [start_date + timedelta(minutes=30 * i) for i in range(30 * 24 * 2)]

def load_house_config(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_date_range_label(date_obj):
    if datetime(2025, 1, 1) <= date_obj <= datetime(2025, 1, 10):
        return "jan_1"
    elif datetime(2025, 1, 11) <= date_obj <= datetime(2025, 1, 20):
        return "jan_2"
    elif datetime(2025, 1, 21) <= date_obj <= datetime(2025, 1, 30):
        return "jan_3"
    else:
        return "other"

def simulate_house(house):
    house_id = house.get("house_id", "unknown")
    meter_reading = house.get("initial_meter_reading", 0)
    timestamps = generate_timestamps(house["initial_meter_reading_date"])

    # Build device + category maps
    category_device_map = {
        "white_goods": house.get("white_goods", {}),
        "entertainment": house.get("entertainment", {}),
        "air_conditioners": house.get("air_conditioners", {}),
    }

    device_map = {}
    device_category_lookup = {}

    for category, devices in category_device_map.items():
        for name, info in devices.items():
            device_map[name] = info
            device_category_lookup[name] = category

    # Add lighting
    if house.get("lighting", {}).get("used", False):
        device_map["lighting"] = house["lighting"]
        device_category_lookup["lighting"] = "lighting"

    # Add EV if used
    if house.get("ev_charges", {}).get("used", False):
        device_map["ev_car"] = house["ev_charges"]
        device_category_lookup["ev_car"] = "ev_charges"

    all_device_names = sorted(device_map.keys())
    all_category_names = sorted(set(device_category_lookup.values()))

    rows = []

    for ts in timestamps:
        row = {
            "house_id": house_id,
            "date": ts.strftime('%Y-%m-%d'),
            "time": ts.strftime('%H:%M:%S'),
            "date_range": get_date_range_label(ts)
        }

        current_time = ts.time()
        total_energy_kwh = 0

        # Track per-device and per-category kWh
        device_kwh_map = {}
        category_kwh_map = {cat: 0.0 for cat in all_category_names}

        for device_name in all_device_names:
            info = device_map[device_name]
            used = info.get("used", False)
            if not used:
                device_kwh_map[device_name] = 0.0
                continue

            usage = info.get("usage", "").lower()
            base_power = DEVICE_WATTAGE.get(device_name.lower(), 100)
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
            else:
                kwh = 0.0

            device_kwh_map[device_name] = kwh
            category = device_category_lookup[device_name]
            category_kwh_map[category] += kwh
            total_energy_kwh += kwh

        # Add device-level columns
        for device in all_device_names:
            row[device] = device_kwh_map.get(device, 0.0)

        # Add category-level columns
        for category in all_category_names:
            row[category] = round(category_kwh_map.get(category, 0.0), 4)

        row["meter_reading"] = round(meter_reading + total_energy_kwh, 4)
        row["consumed_power"] = round(total_energy_kwh, 4)
        meter_reading = row["meter_reading"]

        rows.append(row)

    return rows, all_device_names, all_category_names

def main(config_dir, output_csv):
    house_files = glob.glob(os.path.join(config_dir, "house*.json"))
    if not house_files:
        print(f"❌ No matching files found in '{config_dir}'.")
        return

    all_data = []
    all_devices = set()
    all_categories = set()

    for file in house_files:
        print(f"📂 Processing: {file}")
        house_config = load_house_config(file)
        house_data, devices, categories = simulate_house(house_config)
        all_data.extend(house_data)
        all_devices.update(devices)
        all_categories.update(categories)

    sorted_devices = sorted(all_devices)
    sorted_categories = sorted(all_categories)

    fieldnames = (
        ["house_id", "date", "time", "date_range"]
        + sorted_devices
        + sorted_categories
        + ["meter_reading", "consumed_power"]
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"\n✅ All house data saved to '{output_csv}'.")

# Run the program
if __name__ == "__main__":
    config_folder = "configuration"
    output_file = "data/raw_data_all_devices_and_categories.csv"
    main(config_folder, output_file)

import json
import csv
import random
import os
import glob
import calendar
from datetime import datetime, timedelta

# Device wattages
DEVICE_WATTAGE = {
    'fridge': 150, 'washing_machine': 500, 'microwave': 1100, 'dishwasher': 1200,
    'tv': 100, 'wifi': 10, 'laptop': 60, 'chargers': 20,
    'lighting': 80, 'fans': 50, 'ac': 2000, 'ev_car': 7000,
    'water_heater': 3000, 'mortar': 750
}

ORDERED_CATEGORIES = [
    "white_goods", "entertainment", "air_conditioners", "lighting", "ev_charges", "utility_appliances"
]

ORDERED_DEVICES = [
    "fridge", "washing_machine", "microwave", "dishwasher",
    "tv", "wifi", "laptop", "chargers", "lighting",
    "fans", "ac", "water_heater", "mortar"
]

DEFAULT_CONFIG = {
    "initial_meter_reading": 0,
    "initial_meter_reading_date": "2024-01-01",
    "month_variations": {
        "january": 1.1, "february": 1.0, "march": 1.1,
        "april": 1.3, "may": 1.4, "june": 1.3,
        "july": 1.2, "august": 1.1, "september": 1.0,
        "october": 0.9, "november": 1.0, "december": 1.2
    }
}

def load_house_config(file_path):
    try:
        with open(file_path, "r") as f:
            config = json.load(f)
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
    except Exception as e:
        print(f"Error loading house config from {file_path}: {str(e)}")
        return DEFAULT_CONFIG.copy()

def load_seasonal_config(config_dir):
    seasonal_config_path = os.path.join(config_dir, "house_seasonal_config.json")
    if not os.path.exists(seasonal_config_path):
        print(f"Warning: Seasonal config not found at {seasonal_config_path}")
        return None
    try:
        with open(seasonal_config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading seasonal config: {str(e)}")
        return None

def generate_timestamps(start_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    timestamps = []
    current_date = start_date
    while current_date.year < 2026:
        days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]
        for day in range(1, days_in_month + 1):
            current_date = current_date.replace(day=day)
            for hour in range(24):
                for minute in [0, 30]:
                    timestamp = current_date.replace(hour=hour, minute=minute, second=0)
                    timestamps.append(timestamp)
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1, day=1)
    return timestamps

def get_date_range_label(date_obj):
    day = date_obj.day
    month = date_obj.strftime('%b').lower()
    days_in_month = calendar.monthrange(date_obj.year, date_obj.month)[1]
    range_size = days_in_month // 3
    if day <= range_size:
        return f"{month}_1"
    elif day <= range_size * 2:
        return f"{month}_2"
    else:
        return f"{month}_3"

def get_season_for_date(date_obj, seasonal_config):
    if not seasonal_config:
        return None
    month = date_obj.month
    for season, data in seasonal_config["seasons"].items():
        if month in data["months"]:
            return season
    return None

def get_device_multiplier(device, season, seasonal_config):
    if not seasonal_config or not season:
        return 1.0
    season_data = seasonal_config["seasons"].get(season, {})
    return season_data.get("device_multipliers", {}).get(device, 1.0)

def get_peak_multiplier(device, current_time, season, seasonal_config):
    if not seasonal_config or not season:
        return 1.0
    season_data = seasonal_config["seasons"].get(season, {})
    time_patterns = season_data.get("time_patterns", {}).get(device, {})
    if not time_patterns:
        return 1.0
    start = datetime.strptime(time_patterns["peak_hours"][0], "%H:%M:%S").time()
    end = datetime.strptime(time_patterns["peak_hours"][1], "%H:%M:%S").time()
    if start <= end:
        return time_patterns["peak_multiplier"] if start <= current_time < end else 1.0
    else:
        return time_patterns["peak_multiplier"] if (current_time >= start or current_time < end) else 1.0

def simulate_house(house):
    try:
        house_id = house.get("house_id", "unknown")
        meter_reading = house.get("initial_meter_reading", DEFAULT_CONFIG["initial_meter_reading"])
        start_date = house.get("initial_meter_reading_date", DEFAULT_CONFIG["initial_meter_reading_date"])
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            start_date = DEFAULT_CONFIG["initial_meter_reading_date"]
        timestamps = generate_timestamps(start_date)
        seasonal_config = load_seasonal_config(os.path.dirname(house.get("config_path", "")))

        device_map, device_category = {}, {}
        for category in ORDERED_CATEGORIES:
            items = house.get(category, {})
            if category == "lighting":
                total = {"used": False, "usage": "fixed", "timing": {}}
                for room, info in items.items():
                    if info.get("used"):
                        total["used"] = True
                        if info.get("usage", "").lower() == "random":
                            total["usage"] = "random"
                        if "timing" in info:
                            total["timing"] = info["timing"]
                device_map["lighting"] = total
                device_category["lighting"] = category
            elif category == "ev_charges" and items.get("used"):
                device_map["ev_car"] = items
                device_category["ev_car"] = category
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
            season = get_season_for_date(ts, seasonal_config)
            total_kwh = 0
            category_kwh = {cat: 0.0 for cat in ORDERED_CATEGORIES}
            device_kwh = {dev: 0.0 for dev in ORDERED_DEVICES}

            for dev in ORDERED_DEVICES:
                info = device_map.get(dev)
                if not info or not info.get("used"):
                    continue

                usage = info.get("usage", "").lower()
                base_power = DEVICE_WATTAGE.get(dev, 100)
                is_on = False

                if usage in ["continuous", "continous"]:
                    is_on = True
                elif usage == "fixed" and "timing" in info:
                    timing = info["timing"]
                    start = datetime.strptime(timing.get("start", "00:00:00"), "%H:%M:%S").time()
                    end = datetime.strptime(timing.get("end", "23:59:59"), "%H:%M:%S").time()
                    is_on = start <= current_time < end if start <= end else current_time >= start or current_time < end
                elif usage == "random":
                    is_on = random.random() < 0.5

                if is_on:
                    seasonal_mult = get_device_multiplier(dev, season, seasonal_config)
                    peak_mult = get_peak_multiplier(dev, current_time, season, seasonal_config)
                    power = round(base_power * random.uniform(0.9, 1.1) * seasonal_mult * peak_mult, 2)
                    kwh = round((power / 1000) * 0.5, 4)
                    device_kwh[dev] = kwh
                    category = device_category.get(dev)
                    if category:
                        category_kwh[category] += kwh
                    total_kwh += kwh

            meter_reading = round(meter_reading + total_kwh, 4)
            for cat in ORDERED_CATEGORIES:
                row[cat] = round(category_kwh[cat], 4)
            row["meter_reading"] = meter_reading
            row["consumed_power"] = round(total_kwh, 4)
            for dev in ORDERED_DEVICES:
                row[dev] = round(device_kwh[dev], 4)
            rows.append(row)
        return rows
    except Exception as e:
        print(f"Error in simulate_house: {str(e)}")
        return []

def main(config_dir, output_csv):
    try:
        house_files = glob.glob(os.path.join(config_dir, "house*.json"))
        if not house_files:
            print(f"No house*.json files found in '{config_dir}'")
            return
        all_rows = []
        for file in house_files:
            print(f"Processing: {file}")
            config = load_house_config(file)
            config["config_path"] = file
            rows = simulate_house(config)
            all_rows.extend(rows)

        if not all_rows:
            print("No data generated. Check configurations.")
            return

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
        print(f"\nOutput saved to: {output_csv}")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    config_folder = "configuration"
    output_file = "data/raw_data_20250508_20_25.csv"
    main(config_folder, output_file)

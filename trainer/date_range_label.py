import pickle
import calendar

# Define the function
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

# Save the function to a pickle file
with open("fastapi/model/get_date_range_label.pkl", "wb") as f:
    pickle.dump(get_date_range_label, f)

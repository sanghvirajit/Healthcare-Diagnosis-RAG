import requests
import json

# Local API
url = "http://127.0.0.1:8000/diagnose_parkinson/"

# Load the gait analysis JSON data
file_path = "data/parkinson_patient_data.json"
with open(file_path, "r") as f:
    gait_data = json.load(f)

# Send POST request
response = requests.post(url, json=gait_data)

# Print response
if response.status_code == 200:
    print(response.json())
else:
    print(response.status_code)
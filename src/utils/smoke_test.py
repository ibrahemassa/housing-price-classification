import os
import sys
import requests

EC2_IP = os.getenv("EC2_IP")

if not EC2_IP:
    print("EC2_IP environment variable not set")
    sys.exit(1)

URL = f"http://{EC2_IP}:8081/predict"

payload = {
    "district": "Kadikoy",
    "address": "Test Address",
    "HeatingType": "NaturalGas",
    "FloorLocation": "3rdFloor",
    "GrossSquareMeters": 120,
    "NetSquareMeters": 100,
    "BuildingAge": 5,
    "NumberOfRooms": 3,
    "AdCreationDate": "2023-01",
    "Subscription": "unknown"
}

try:
    response = requests.post(URL, json=payload, timeout=10)
except requests.RequestException as e:
    print(f"Smoke test failed: {e}")
    sys.exit(1)

if response.status_code != 200:
    print(f"Smoke test failed with status {response.status_code}")
    sys.exit(1)

print("Smoke test passed")

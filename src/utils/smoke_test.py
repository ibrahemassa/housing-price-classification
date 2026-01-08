import os
import sys

import requests

EC2_IP = os.getenv("EC2_IP")

if not EC2_IP:
    print("EC2_IP environment variable not set")
    sys.exit(1)

URL = f"http://{EC2_IP}:8081/predict"

payload = {
  "GrossSquareMeters": 200,
  "NetSquareMeters": 120,
  "BuildingAge": 3,
  "NumberOfRooms": 4.0,
  "district": "besiktas",
  "HeatingType": "Kombi Doğalgaz",
  "StructureType": "Betonarme",
  "FloorLocation": "Çatı Katı",
  "address": "İstanbul>besiktas>yildiz",
  "AdCreationDate": "2025-02",
  "Subscription": "None"
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

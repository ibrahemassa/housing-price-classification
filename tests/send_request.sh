#!/bin/bash

API_URL="http://localhost:8081/predict"
DATA_FILE="test_request.json"

echo "Sending requests to $API_URL"
echo "--------------------------------"

jq -c '.[]' "$DATA_FILE" | while read -r row; do
  echo "Sending:"
  echo "$row"

  curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "$row"

  echo ""
  echo "--------------------------------"
  sleep 0.2
done


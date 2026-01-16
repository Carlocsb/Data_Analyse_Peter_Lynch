#!/bin/bash
echo "🕕 Warte auf 22:00 Uhr (Europe/Berlin) für den täglichen Datenabruf..."
while true; do
  now=$(date +%H:%M)
  echo "⌚ Aktuelle Zeit: $now"
  if [ "$now" = "22:00" ]; then
    echo "🚀 Starte ingest_yf.py um $(date)"
    python /app/ingest_yf.py
    echo "✅ Fertig. Warte bis zum nächsten Tag..."
    sleep 3600
  fi
  sleep 60
done

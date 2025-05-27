#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

# Default values
TICKER="NVDA"
DAYS=30
YEARS=2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--ticker)
      TICKER="$2"
      shift 2
      ;;
    -d|--days)
      DAYS="$2"
      shift 2
      ;;
    -y|--years)
      YEARS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running stock predictor for $TICKER with $DAYS days forecast using $YEARS years of historical data..."

# Run the prediction script
python src/predict_stock.py "$TICKER" --days "$DAYS" --years "$YEARS"

echo "Prediction complete!" 
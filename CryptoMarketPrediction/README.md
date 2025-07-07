# DRW - Crypto Market Prediction

## Overview
This project is for the DRW Crypto Market Prediction competition on Kaggle. The goal is to predict cryptocurrency market movements using historical market data.

## Project Structure
```
CryptoMarketPrediction/
├── data/                           # Competition data files
├── crypto_market_analysis.ipynb    # Main analysis notebook
└── README.md                      # This file
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install pandas numpy polars matplotlib seaborn kaggle
```

### 2. Download Competition Data
Make sure you have Kaggle CLI configured with your API key, then run:
```bash
kaggle competitions download -c drw-crypto-market-prediction -p data
```

### 3. Run the Analysis
Open and run the `crypto_market_analysis.ipynb` notebook to:
- Load and explore the dataset
- Examine data structure and quality
- Get recommendations for next steps

## Competition Details
- **Competition**: DRW - Crypto Market Prediction
- **Type**: Time Series Prediction
- **Domain**: Cryptocurrency Markets
- **Objective**: Predict future market movements

## Next Steps
After running the initial analysis notebook, consider:
1. Time series analysis and trend identification
2. Feature engineering with technical indicators
3. Model development and evaluation
4. Backtesting and risk assessment

## Files
- `crypto_market_analysis.ipynb`: Initial data exploration and analysis
- `data/`: Directory containing competition datasets 
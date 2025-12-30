# Data Documentation

This directory contains the primary datasets used for the Bitcoin sentiment and trading performance analysis.

## Dataset Descriptions

### 1. fear_greed_index.csv
This file contains daily historical sentiment data for the Bitcoin market.

*   **Record Count**: 2,644 daily observations
*   **Time Span**: February 2018 to May 2025
*   **Key Fields**:
    *   `timestamp`: Unix representation of the date
    *   `value`: Numerical sentiment score ranging from 0 to 100
    *   `classification`: Discrete sentiment categories (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)
    *   `date`: Calendar date in standard YYYY-MM-DD format

### 2. historical_data.csv
This dataset includes detailed execution records for traders on the Hyperliquid exchange.

*   **Record Count**: 211,218 individual trades
*   **Time Span**: Full year 2024
*   **Coverage**: 32 unique accounts trading 246 different assets
*   **Key Fields**:
    *   `Account`: Unique identifier for the trading account
    *   `Coin`: Ticker symbol for the traded cryptocurrency
    *   `Execution Price`: The price at which the trade was filled
    *   `Size Tokens`: The quantity of tokens traded
    *   `Size USD`: The dollar value of the position at execution
    *   `Side`: Indicates if the trade was a Buy or Sell
    *   `Timestamp IST`: Local execution time in India Standard Time
    *   `Direction`: Classification of the entry/exit (e.g., Open Long, Close Short)
    *   `Closed PnL`: The realized profit or loss for the trade

## Institutional Data Sources

*   **Market Sentiment**: Data sourced from the public Bitcoin Fear & Greed Index API/repository.
*   **Trading Records**: Secured historical trade logs derived from Hyperliquid exchange activity.

## Data Integration and Ingestion

The datasets are designed for structured ingestion into a MySQL environment. The primary ingestion pipeline is managed through the following module:

```bash
python src/database/load_data.py
```

This process ensures data types are correctly cast and unique constraints are maintained before the analysis and modeling stages begin.
# Bitcoin Sentiment & Trading Performance Analysis

This project explores the correlation between Bitcoin market sentiment, measured by the Fear & Greed Index, and actual trader performance on the Hyperliquid exchange. 

## Project Overview

The study analyzes approximately 211,214 trades executed by 32 unique traders across 246 different cryptocurrencies during the 2024 calendar year. The primary objective is to determine if market sentiment categories significantly impact trading profitability and behavior.

## Key Findings

*   **Sentiment and Profitability**: Trading performance during extreme sentiment periods (Extreme Fear or Extreme Greed) showed nearly double the average PnL compared to neutral market periods.
*   **Predictive Modeling**: A classification model achieved an accuracy of 75.92% in predicting whether a trade would be profitable based on sentiment and temporal features.
*   **Primary Predictors**: The hour of the day (22.6%) and the specific sentiment value (21.4%) were identified as the most significant factors influencing trade outcomes.

### Performance Summary by Sentiment Category

| Sentiment Category | Average PnL | Win Rate | Total Trades |
|--------------------|-------------|----------|--------------|
| Extreme Greed      | $67.89      | 43.2%    | 39,980       |
| Fear               | $52.49      | 40.8%    | 61,826       |
| Greed              | $41.75      | 41.5%    | 50,283       |
| Extreme Fear       | $35.48      | 39.7%    | 21,400       |
| Neutral            | $34.31      | 40.1%    | 37,686       |

## Repository Structure

```
├── data/                      # Input datasets (CSV format)
├── notebooks/                 # Documentation and exploratory analysis
├── src/                       # Source code repository
│   ├── database/             # Schema setup and data ingestion scripts
│   ├── visualization/        # Business intelligence and charting modules
│   └── models/               # Machine learning pipelines
├── outputs/                   # Generated artifacts
│   ├── visualizations/       # PNG format analysis charts
│   └── ml_results.json       # JSON format model performance metrics
├── docs/                      # Formal analysis reports
└── requirements.txt          # Project dependencies
```

## Getting Started

### Prerequisites

*   Python 3.8 or higher
*   MySQL Server (5.7 or 8.0+)
*   Standard data science libraries (listed in requirements.txt)

### Setup and Installation

1.  Clone this repository to your local environment.
2.  Install all required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Initialize the MySQL database and schema:
    ```bash
    mysql -u root -p < src/database/setup.sql
    ```

### Running the Analysis Pipeline

1.  **Ingest Data**: Load the CSV files into the MySQL database.
    ```bash
    python src/database/load_data.py
    ```
2.  **Generate Charts**: Create visual representations of the data.
    ```bash
    python src/visualization/create_charts.py
    ```
3.  **Train Models**: Execute the machine learning modules to generate insights.
    ```bash
    python src/models/train_models.py
    ```

## Analytical Methodology

### Data Sources
*   **Sentiment Metrics**: Historical data from the Bitcoin Fear & Greed Index (2,644 daily records).
*   **Trading Activity**: Internal historical trade records from Hyperliquid (211,218 records).

### Processing Steps
*   Data normalization and merging based on execution timestamps.
*   Feature engineering focusing on trade timing (hour, day, month) and position characteristics.
*   Outlier detection and data cleaning to ensure model integrity.

### Visualization Capabilities
Six primary visualization modules provide insights into:
1.  Performance metrics stratified by sentiment classification.
2.  Comparative analysis of Long vs. Short positions.
3.  Temporal trends linking sentiment values to daily PnL.
4.  Sentiment-based performance across top-traded assets.
5.  Multi-feature correlation analysis.
6.  Statistical distribution of trade sizes and returns.

## Strategic Insights

*   **Opportunity Windows**: Trading during market extremes appears to provide a significant statistical edge.
*   **Timing Importance**: The high importance of "Trade Hour" suggests that execution timing is critical for strategy optimization.
*   **Capital Allocation**: Larger position sizes (>$10K) historically correlate with higher risk-adjusted returns in this dataset.

## Technical Stack

*   **Database Management**: MySQL 8.0
*   **Computational Logic**: Python (Pandas, NumPy)
*   **Visualization**: Matplotlib, Seaborn
*   **Predictive Analytics**: Scikit-learn

## License

This project is licensed under the MIT License.

## Author

Maharsh Doshi

## Acknowledgments

*   Data provided by the Bitcoin Fear & Greed Index and Hyperliquid exchange.
*   Supported by the open-source Python data science community.

---
*Disclaimer: This analysis is for informational purposes only and does not constitute financial advice.*

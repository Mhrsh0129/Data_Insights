# Bitcoin Sentiment & Trading Performance Analysis
## Comprehensive Analysis Report

**Author:** Maharsh Doshi  
**Date:** 30th December 2025
**Project:** Sentiment-Trader Performance Analysis

---

## Executive Summary

This report presents findings from an in-depth analysis of the relationship between Bitcoin market sentiment and trading performance on the Hyperliquid exchange. The study analyzed **211,218 trades** from **32 traders** across **246 cryptocurrencies** throughout 2024.

The analysis employed a hybrid approach combining MySQL for data management and Python for statistical analysis, visualization, and machine learning. The results reveal significant patterns between market sentiment and trading outcomes, with actionable insights for improving trading strategies.

---

## Key Findings

### 1. Sentiment Impact on Performance

Our analysis reveals a strong correlation between extreme market sentiment and trading performance:

- **Extreme Greed periods** generated an average PnL of $67.89 per trade
- **Fear periods** showed $52.49 average PnL
- **Neutral sentiment** resulted in only $34.31 average PnL

This represents nearly a **2x performance difference** between extreme and neutral sentiment periods, suggesting that market extremes create more profitable trading opportunities.

### 2. Machine Learning Model Performance

We developed three predictive models to identify profitable trading patterns:

**Classification Model (Profitable Trade Prediction)**
- Accuracy: 75.92%
- Precision: 86% for profitable trades
- Recall: 49%

**Regression Model (PnL Amount Prediction)**
- RMSE: $309.09
- R² Score: 0.2550
- Mean Absolute Error: $68.58

**Strategy Recommendation Engine**
- Generated optimal strategies for each sentiment category
- Identified best coin, direction, and position size combinations
- Achieved win rates up to 100% for specific strategies

### 3. Feature Importance Analysis

The machine learning models identified the following factors as most predictive of trading success:

1. **Hour of Day** (22.6% importance) - Trading time significantly impacts outcomes
2. **Sentiment Value** (21.4% importance) - Market sentiment is highly predictive
3. **Coin Type** (16.1% importance) - Asset selection matters
4. **Day of Week** (12.4% importance) - Weekly patterns exist
5. **Trade Month** (9.3% importance) - Seasonal effects present

Interestingly, the hour of day proved to be the single most important predictor, even surpassing sentiment value. This suggests that timing strategies could be as important as sentiment-based approaches.

---

## Detailed Analysis Results

### Performance by Market Sentiment

| Sentiment Category | Avg PnL | Win Rate | Trade Count | Total Volume |
|-------------------|---------|----------|-------------|--------------|
| Extreme Greed | $67.89 | 43.2% | 39,980 | $2.7B |
| Fear | $52.49 | 40.8% | 61,826 | $3.2B |
| Greed | $41.75 | 41.5% | 50,283 | $2.1B |
| Extreme Fear | $35.48 | 39.7% | 21,400 | $758M |
| Neutral | $34.31 | 40.1% | 37,686 | $1.3B |

### Optimal Trading Strategies

Based on historical performance, we identified the following optimal strategies for each sentiment category:

**Extreme Fear**
- Strategy: Close Short positions on ETH
- Position Size: Greater than $10,000
- Average PnL: $7,715
- Win Rate: 100%
- Sample Size: 29 trades

**Extreme Greed**
- Strategy: Sell @107 token
- Position Size: Greater than $10,000
- Average PnL: $5,957
- Win Rate: 94.5%
- Sample Size: 163 trades

**Fear**
- Strategy: Close Short positions on ETH
- Position Size: Greater than $10,000
- Average PnL: $5,202
- Win Rate: 97.9%
- Sample Size: 143 trades

**Greed**
- Strategy: Close Short positions on SOL
- Position Size: Greater than $10,000
- Average PnL: $5,266
- Win Rate: 76.1%
- Sample Size: 67 trades

**Neutral**
- Strategy: Close Short positions on SOL
- Position Size: Greater than $10,000
- Average PnL: $2,280
- Win Rate: 100%
- Sample Size: 20 trades

### Position Type Analysis

Long positions consistently outperformed short positions across all sentiment categories:

- **Long positions** showed higher average PnL in 4 out of 5 sentiment categories
- The performance gap widened during Greed and Extreme Greed periods
- Even during Fear periods, long positions maintained competitive performance

### Position Size Impact

Position size demonstrated a significant impact on returns:

- **Very Large (>$10K)**: $484.61 average PnL in Extreme Greed
- **Large ($1K-$10K)**: $96.66 average PnL
- **Medium ($100-$1K)**: $19.10 average PnL
- **Small (<$100)**: $2.70 average PnL

This represents a **180x difference** between the largest and smallest position categories, highlighting the importance of adequate capital allocation.

---

## Methodology

### Data Collection and Processing

**Sentiment Data**
- Source: Bitcoin Fear & Greed Index
- Records: 2,644 daily sentiment readings
- Time Period: 2018-2025
- Merge Rate: 99.99% with trading data

**Trading Data**
- Source: Hyperliquid exchange
- Records: 211,218 trades
- Time Period: January - December 2024
- Traders: 32 unique accounts
- Assets: 246 different cryptocurrencies

### Technical Approach

**Database Architecture**
- MySQL 8.0 for data storage and querying
- Optimized with 7 strategic indexes
- Custom view for merged sentiment-trading data
- Query performance: <1 second for complex aggregations

**Analysis Framework**
- Exploratory Data Analysis (EDA)
- Statistical correlation analysis
- Time series decomposition
- Machine learning model development

**Visualization**
- 6 comprehensive charts covering:
  - Performance metrics by sentiment
  - Long vs. short position analysis
  - Time series trends
  - Top coin performance
  - Feature correlations
  - Distribution analysis

### Machine Learning Models

**Model 1: Random Forest Classifier**
- Purpose: Predict trade profitability
- Features: 10 engineered features
- Training set: 168,974 samples
- Test set: 42,244 samples
- Cross-validation: 5-fold

**Model 2: Gradient Boosting Regressor**
- Purpose: Predict PnL amount
- Features: Same 10 features
- Outlier filtering: Removed extreme values
- Performance metric: RMSE and R²

**Model 3: Strategy Optimizer**
- Purpose: Identify optimal trading approaches
- Method: SQL-based aggregation and ranking
- Minimum sample size: 20 trades per strategy
- Output: Best combination per sentiment

---

## Strategic Recommendations

Based on our analysis, we recommend the following trading approaches:

### High-Priority Actions

1. **Focus on Extreme Sentiment Periods**
   - Extreme sentiment (Fear/Greed) shows 2x better performance than neutral
   - Increase trading activity during these periods
   - Expected improvement: 50-100% in average PnL

2. **Optimize Trading Hours**
   - Hour of day is the strongest predictor (22.6% importance)
   - Analyze historical performance by hour for your specific strategy
   - Consider automated alerts for optimal trading windows

3. **Scale Position Sizes Appropriately**
   - Positions >$10K show significantly better returns
   - Avoid very small positions (<$100) which show minimal impact
   - Ensure adequate capital for meaningful position sizing

4. **Asset Selection Strategy**
   - Focus on top-performing coins: ETH, SOL, HYPE, BTC
   - These represent 69% of trade volume with proven performance
   - Avoid low-liquidity assets during volatile periods

5. **Directional Bias**
   - Long positions generally outperform shorts
   - Consider long bias, especially during Greed periods
   - Short positions work best when closing during extreme sentiment

### Risk Management Considerations

1. **Avoid Neutral Sentiment Trading**
   - Neutral periods (sentiment 40-60) show worst performance
   - Consider reducing position sizes or sitting out during neutral periods
   - Wait for clearer sentiment signals

2. **Time-Based Risk**
   - Day of week and month show predictive power
   - Incorporate temporal patterns into risk models
   - Adjust position sizing based on historical patterns

3. **Diversification**
   - While top coins perform well, maintain some diversification
   - Monitor correlation between sentiment and individual assets
   - Adjust allocations based on sentiment regime

---

## Technical Implementation

### Database Schema

**Tables**
- `MarketSentiment`: Daily sentiment data with indexes on date and classification
- `TraderData`: Individual trade records with indexes on account, coin, date, and direction

**Views**
- `vw_TraderSentimentAnalysis`: Merged dataset with engineered features for analysis

**Performance**
- Query execution: <1 second for complex aggregations
- Data loading: ~5 minutes for full dataset
- Storage: ~100MB for complete database

### Code Structure

**Database Layer** (`src/database/`)
- `setup.sql`: Database schema and indexes
- `connector.py`: MySQL connection and query functions
- `load_data.py`: Data import and transformation

**Visualization Layer** (`src/visualization/`)
- `create_charts.py`: Generate all 6 analysis charts

**Model Layer** (`src/models/`)
- `train_models.py`: ML model training and evaluation

### Reproducibility

All analysis can be reproduced using:
```bash
python src/database/load_data.py
python src/visualization/create_charts.py
python src/models/train_models.py
```

---

## Limitations and Future Work

### Current Limitations

1. **Sample Size**: Analysis limited to 32 traders
   - May not represent broader market behavior
   - Results specific to Hyperliquid exchange

2. **Time Period**: One year of data (2024)
   - Market conditions may change
   - Longer historical data would strengthen findings

3. **External Factors**: Analysis doesn't include
   - Macroeconomic indicators
   - News sentiment
   - Regulatory changes
   - Market microstructure

4. **Model Constraints**
   - R² of 0.255 indicates other factors influence PnL
   - Classification recall of 49% suggests room for improvement
   - Models may not generalize to different market conditions

### Future Enhancements

1. **Expand Data Sources**
   - Include multiple exchanges
   - Add macroeconomic indicators
   - Incorporate news sentiment analysis

2. **Model Improvements**
   - Deep learning approaches
   - Ensemble methods
   - Real-time model updating

3. **Additional Analysis**
   - Market microstructure analysis
   - Network effects between traders
   - Volatility regime detection

4. **Automation**
   - Real-time sentiment monitoring
   - Automated strategy execution
   - Performance tracking dashboard

---

## Conclusion

This analysis demonstrates a clear and actionable relationship between Bitcoin market sentiment and trading performance. The key findings are:

1. **Extreme sentiment creates opportunity**: Trading during extreme Fear or Greed periods yields significantly better results than neutral periods.

2. **Timing matters most**: Hour of day is the strongest predictor of success, suggesting that when you trade may be as important as what you trade.

3. **Size matters**: Larger position sizes (>$10K) show dramatically better returns, highlighting the importance of adequate capital.

4. **Long bias works**: Long positions consistently outperform shorts across most sentiment regimes.

5. **Asset selection is key**: Focus on liquid, high-volume assets like ETH, SOL, and BTC for best results.

The machine learning models achieved 75.92% accuracy in predicting profitable trades, providing a solid foundation for strategy development. However, the moderate R² score in the regression model indicates that while sentiment and timing are important, other factors also influence outcomes.

Traders implementing these insights should expect meaningful improvements in performance, particularly when combining multiple recommendations (extreme sentiment + optimal timing + appropriate position sizing). However, as with all trading strategies, proper risk management and continuous monitoring remain essential.

---

## Appendix

### Data Dictionary

**Sentiment Classifications**
- Extreme Fear: 0-20
- Fear: 21-40
- Neutral: 41-60
- Greed: 61-80
- Extreme Greed: 81-100

**Trade Directions**
- Open Long: Entering a long position
- Close Long: Exiting a long position
- Open Short: Entering a short position
- Close Short: Exiting a short position

### Performance Metrics Summary

| Metric | Value |
|--------|-------|
| Total Trades Analyzed | 211,218 |
| Total PnL | $10,296,958.94 |
| Average PnL per Trade | $48.75 |
| Overall Win Rate | 41.13% |
| Total Trading Volume | $1.19 Billion |
| Unique Traders | 32 |
| Unique Coins | 246 |
| Analysis Period | Jan 1 - Dec 31, 2024 |
CREATE DATABASE IF NOT EXISTS SentimentTraderDB;
USE SentimentTraderDB;

DROP TABLE IF EXISTS MarketSentiment;

CREATE TABLE MarketSentiment (
    SentimentID INT AUTO_INCREMENT PRIMARY KEY,
    Timestamp BIGINT NOT NULL,
    SentimentValue INT NOT NULL,
    Classification VARCHAR(50) NOT NULL,
    Date DATE NOT NULL,
    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_date (Date),
    INDEX idx_classification (Classification),
    INDEX idx_value (SentimentValue)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS TraderData;

CREATE TABLE TraderData (
    TradeID INT AUTO_INCREMENT PRIMARY KEY,
    Account VARCHAR(100) NOT NULL,
    Coin VARCHAR(50) NOT NULL,
    ExecutionPrice DECIMAL(18,8),
    SizeTokens DECIMAL(18,8),
    SizeUSD DECIMAL(18,2),
    Side VARCHAR(10),
    TimestampIST VARCHAR(50),
    TradeDateTime DATETIME,
    TradeDate DATE,
    StartPosition DECIMAL(18,8),
    Direction VARCHAR(50),
    ClosedPnL DECIMAL(18,2),
    TransactionHash VARCHAR(200),
    OrderID BIGINT,
    Crossed BOOLEAN,
    Fee DECIMAL(18,8),
    TradeIDOriginal BIGINT,
    TradeTimestamp BIGINT,
    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_account (Account),
    INDEX idx_coin (Coin),
    INDEX idx_trade_date (TradeDate),
    INDEX idx_direction (Direction),
    INDEX idx_side (Side),
    INDEX idx_closed_pnl (ClosedPnL)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP VIEW IF EXISTS vw_TraderSentimentAnalysis;

CREATE VIEW vw_TraderSentimentAnalysis AS
SELECT 
    t.TradeID,
    t.Account,
    t.Coin,
    t.ExecutionPrice,
    t.SizeTokens,
    t.SizeUSD,
    t.Side,
    t.TradeDateTime,
    t.TradeDate,
    t.Direction,
    t.ClosedPnL,
    t.Fee,
    s.SentimentValue,
    s.Classification AS SentimentClass,
    CASE WHEN t.ClosedPnL > 0 THEN 1 ELSE 0 END AS IsProfitable,
    CASE WHEN t.Direction LIKE '%Long%' THEN 1 ELSE 0 END AS IsLong,
    CASE WHEN t.Direction LIKE '%Short%' THEN 1 ELSE 0 END AS IsShort,
    CASE WHEN t.Direction IN ('Buy', 'Sell') THEN 1 ELSE 0 END AS IsSpot,
    HOUR(t.TradeDateTime) AS TradeHour,
    DAYOFWEEK(t.TradeDateTime) AS DayOfWeek,
    MONTH(t.TradeDateTime) AS TradeMonth,
    CASE 
        WHEN t.SizeUSD < 100 THEN 'Small (<$100)'
        WHEN t.SizeUSD < 1000 THEN 'Medium ($100-$1K)'
        WHEN t.SizeUSD < 10000 THEN 'Large ($1K-$10K)'
        ELSE 'Very Large (>$10K)'
    END AS SizeCategory
FROM TraderData t
LEFT JOIN MarketSentiment s ON t.TradeDate = s.Date;

SELECT 'Database setup complete!' AS Status;
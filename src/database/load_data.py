import mysql.connector
import pandas as pd
from datetime import datetime

MYSQL_USER = 'root'
MYSQL_PASSWORD = '@Maha2004'

def load_data_manually():
    print("="*80)
    print("LOADING DATA INTO MySQL")
    print("="*80)
    
    conn = mysql.connector.connect(
        host='localhost',
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database='SentimentTraderDB'
    )
    cursor = conn.cursor()
    
    print("\n1. Loading sentiment data...")
    df_sentiment = pd.read_csv('fear_greed_index.csv')
    print(f"   Read {len(df_sentiment)} rows from CSV")
    
    insert_query = """
    INSERT INTO MarketSentiment (Timestamp, SentimentValue, Classification, Date)
    VALUES (%s, %s, %s, %s)
    """
    
    for idx, row in df_sentiment.iterrows():
        try:
            cursor.execute(insert_query, (
                int(row['timestamp']),
                int(row['value']),
                str(row['classification']),
                pd.to_datetime(row['date']).date()
            ))
        except Exception as e:
            if 'Duplicate' not in str(e):
                print(f"   Error on row {idx}: {e}")
    
    conn.commit()
    print(f"   ✓ Loaded sentiment data")
    
    print("\n2. Loading trader data (this will take a few minutes)...")
    df_trader = pd.read_csv('historical_data.csv')
    print(f"   Read {len(df_trader)} rows from CSV")
    
    insert_query = """
    INSERT INTO TraderData (
        Account, Coin, ExecutionPrice, SizeTokens, SizeUSD, Side,
        TimestampIST, TradeDateTime, TradeDate, StartPosition, Direction,
        ClosedPnL, TransactionHash, OrderID, Crossed, Fee, TradeIDOriginal, TradeTimestamp
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    batch_size = 1000
    for i in range(0, len(df_trader), batch_size):
        batch = df_trader.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            try:
                trade_dt = pd.to_datetime(row['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
                trade_date = trade_dt.date() if pd.notna(trade_dt) else None
                
                cursor.execute(insert_query, (
                    str(row['Account']),
                    str(row['Coin']),
                    float(row['Execution Price']) if pd.notna(row['Execution Price']) else None,
                    float(row['Size Tokens']) if pd.notna(row['Size Tokens']) else None,
                    float(row['Size USD']) if pd.notna(row['Size USD']) else None,
                    str(row['Side']) if pd.notna(row['Side']) else None,
                    str(row['Timestamp IST']),
                    trade_dt if pd.notna(trade_dt) else None,
                    trade_date,
                    float(row['Start Position']) if pd.notna(row['Start Position']) else None,
                    str(row['Direction']) if pd.notna(row['Direction']) else None,
                    float(row['Closed PnL']) if pd.notna(row['Closed PnL']) else None,
                    str(row['Transaction Hash']) if pd.notna(row['Transaction Hash']) else None,
                    int(row['Order ID']) if pd.notna(row['Order ID']) else None,
                    bool(row['Crossed']) if pd.notna(row['Crossed']) else None,
                    float(row['Fee']) if pd.notna(row['Fee']) else None,
                    int(row['Trade ID']) if pd.notna(row['Trade ID']) else None,
                    int(row['Timestamp']) if pd.notna(row['Timestamp']) else None
                ))
            except Exception as e:
                if 'Duplicate' not in str(e):
                    print(f"   Error on row {idx}: {str(e)[:100]}")
        
        conn.commit()
        print(f"   Progress: {min(i+batch_size, len(df_trader))}/{len(df_trader)} rows")
    
    print(f"   ✓ Loaded trader data")
    
    print("\n3. Verifying data...")
    cursor.execute("SELECT COUNT(*) FROM MarketSentiment")
    sentiment_count = cursor.fetchone()[0]
    print(f"   MarketSentiment: {sentiment_count} rows")
    
    cursor.execute("SELECT COUNT(*) FROM TraderData")
    trader_count = cursor.fetchone()[0]
    print(f"   TraderData: {trader_count} rows")
    
    cursor.execute("SELECT COUNT(*) FROM vw_TraderSentimentAnalysis")
    view_count = cursor.fetchone()[0]
    print(f"   vw_TraderSentimentAnalysis: {view_count} rows")
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*80)
    print("DATA LOADING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    try:
        load_data_manually()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

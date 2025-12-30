import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime
import os

class SQLConnector:
    
    def __init__(self, host='localhost', database='SentimentTraderDB', 
                 user='root', password=''):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cursor = None
        
    def connect(self):
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.conn.cursor(dictionary=True, buffered=True)
            print(f"✓ Connected to MySQL: {self.host}/{self.database}")
            return True
        except mysql.connector.Error as e:
            if e.errno == 1049:
                print(f"Database '{self.database}' doesn't exist. Creating it...")
                try:
                    conn_temp = mysql.connector.connect(
                        host=self.host,
                        user=self.user,
                        password=self.password
                    )
                    cursor_temp = conn_temp.cursor()
                    cursor_temp.execute(f"CREATE DATABASE {self.database}")
                    cursor_temp.close()
                    conn_temp.close()
                    print(f"✓ Database '{self.database}' created")
                    return self.connect()
                except Exception as e2:
                    print(f"✗ Failed to create database: {e2}")
                    return False
            else:
                print(f"✗ Connection failed: {e}")
                return False
    
    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✓ Disconnected from database")
    
    def execute_query(self, query):
        try:
            df = pd.read_sql(query, self.conn)
            return df
        except Exception as e:
            print(f"✗ Query failed: {e}")
            return None
    
    def execute_script(self, script_path):
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            
            statements = sql_script.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if not statement or statement.startswith('--'):
                    continue
                try:
                    self.cursor.execute(statement)
                    self.conn.commit()
                except Exception as e:
                    if 'already exists' not in str(e).lower():
                        print(f"  Warning: {str(e)[:100]}")
            
            print(f"✓ Executed script: {os.path.basename(script_path)}")
            return True
        except Exception as e:
            print(f"✗ Script execution failed: {e}")
            return False
    
    def load_csv_to_table(self, csv_path, table_name):
        try:
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} rows from {os.path.basename(csv_path)}")
            
            if table_name == 'MarketSentiment':
                df.columns = ['Timestamp', 'SentimentValue', 'Classification', 'Date']
                df['Date'] = pd.to_datetime(df['Date'])
            
            elif table_name == 'TraderData':
                df['TradeDateTime'] = pd.to_datetime(df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
                df['TradeDate'] = df['TradeDateTime'].dt.date
                df.rename(columns={'Timestamp': 'TradeTimestamp'}, inplace=True)
            
            from sqlalchemy import create_engine
            engine = create_engine(
                f'mysql+mysqlconnector://{self.user}:{self.password}@{self.host}/{self.database}'
            )
            
            df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=1000)
            print(f"✓ Loaded {len(df)} rows into {table_name}")
            return True
            
        except Exception as e:
            print(f"✗ CSV load failed: {e}")
            return False
    
    def get_all_data(self):
        query = "SELECT * FROM vw_TraderSentimentAnalysis"
        return self.execute_query(query)
    
    def get_performance_by_sentiment(self):
        query = """
        SELECT 
            SentimentClass,
            COUNT(*) AS TotalTrades,
            AVG(ClosedPnL) AS AvgPnL,
            SUM(ClosedPnL) AS TotalPnL,
            CAST(SUM(CASE WHEN IsProfitable = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS DECIMAL(5,2)) AS WinRate,
            AVG(SizeUSD) AS AvgTradeSize
        FROM vw_TraderSentimentAnalysis
        WHERE SentimentClass IS NOT NULL
        GROUP BY SentimentClass
        ORDER BY AvgPnL DESC
        """
        return self.execute_query(query)
    
    def get_long_vs_short(self):
        query = """
        SELECT 
            SentimentClass,
            CASE WHEN IsLong = 1 THEN 'Long' WHEN IsShort = 1 THEN 'Short' END AS PositionType,
            COUNT(*) AS TotalTrades,
            AVG(ClosedPnL) AS AvgPnL,
            SUM(ClosedPnL) AS TotalPnL
        FROM vw_TraderSentimentAnalysis
        WHERE SentimentClass IS NOT NULL AND (IsLong = 1 OR IsShort = 1)
        GROUP BY SentimentClass, CASE WHEN IsLong = 1 THEN 'Long' WHEN IsShort = 1 THEN 'Short' END
        ORDER BY SentimentClass, PositionType
        """
        return self.execute_query(query)
    
    def get_top_coins_performance(self, top_n=5):
        query = f"""
        SELECT v.Coin, v.SentimentClass, COUNT(*) AS TotalTrades,
               AVG(v.ClosedPnL) AS AvgPnL, SUM(v.ClosedPnL) AS TotalPnL
        FROM vw_TraderSentimentAnalysis v
        INNER JOIN (
            SELECT Coin FROM vw_TraderSentimentAnalysis
            GROUP BY Coin ORDER BY COUNT(*) DESC LIMIT {top_n}
        ) tc ON v.Coin = tc.Coin
        WHERE v.SentimentClass IS NOT NULL
        GROUP BY v.Coin, v.SentimentClass
        ORDER BY v.Coin, v.SentimentClass
        """
        return self.execute_query(query)
    
    def get_time_series_data(self):
        query = """
        SELECT 
            TradeDate,
            COUNT(*) AS DailyTrades,
            SUM(ClosedPnL) AS DailyPnL,
            AVG(ClosedPnL) AS AvgPnL,
            SUM(SizeUSD) AS DailyVolume,
            AVG(SentimentValue) AS AvgSentiment,
            MAX(SentimentClass) AS DominantSentiment
        FROM vw_TraderSentimentAnalysis
        WHERE TradeDate IS NOT NULL
        GROUP BY TradeDate
        ORDER BY TradeDate
        """
        return self.execute_query(query)
    
    def get_ml_features(self):
        query = """
        SELECT 
            SentimentValue,
            SizeUSD,
            IsLong,
            IsShort,
            IsSpot,
            TradeHour,
            DayOfWeek,
            TradeMonth,
            ClosedPnL,
            IsProfitable,
            Coin,
            Direction
        FROM vw_TraderSentimentAnalysis
        WHERE SentimentValue IS NOT NULL
        """
        return self.execute_query(query)


def setup_database(user='root', password=''):
    print("="*80)
    print("HYBRID SOLUTION: MySQL DATABASE SETUP AND DATA LOADING")
    print("="*80)
    print()
    
    connector = SQLConnector(user=user, password=password)
    if not connector.connect():
        print("\nFailed to connect to MySQL. Please check:")
        print("  1. MySQL is running")
        print("  2. Username and password are correct")
        print("  3. User has CREATE DATABASE permissions")
        return None
    
    print("\nStep 1: Creating database and tables...")
    script_path = os.path.join(os.path.dirname(__file__), '01_setup_database.sql')
    connector.execute_script(script_path)
    
    connector.disconnect()
    connector = SQLConnector(user=user, password=password)
    connector.connect()
    
    print("\nStep 2: Loading CSV data...")
    base_dir = os.path.dirname(__file__)
    
    print("\n  Loading sentiment data...")
    sentiment_csv = os.path.join(base_dir, 'fear_greed_index.csv')
    connector.load_csv_to_table(sentiment_csv, 'MarketSentiment')
    
    print("\n  Loading trader data (this may take a minute)...")
    trader_csv = os.path.join(base_dir, 'historical_data.csv')
    connector.load_csv_to_table(trader_csv, 'TraderData')
    
    print("\n" + "="*80)
    print("DATABASE SETUP COMPLETE!")
    print("="*80)
    print("\nYou can now use the connector for queries and analysis.")
    
    return connector


if __name__ == "__main__":
    import getpass
    
    print("MySQL Connection Test")
    print("="*80)
    user = input("MySQL username (default: root): ").strip() or 'root'
    password = getpass.getpass("MySQL password (press Enter if none): ")
    
    connector = setup_database(user=user, password=password)
    
    if connector:
        print("\nTesting queries...")
        
        df = connector.get_performance_by_sentiment()
        if df is not None:
            print("\nPerformance by Sentiment:")
            print(df)
        
        connector.disconnect()

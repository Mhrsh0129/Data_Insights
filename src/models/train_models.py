import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sql_connector import SQLConnector
import json
import warnings
warnings.filterwarnings('ignore')

class MLEngine:
    
    def __init__(self, connector):
        self.connector = connector
        self.rf_clf = None
        self.gb_reg = None
        self.le_coin = None
        self.le_sentiment = None
        self.feature_cols = None
        
    def train_all_models(self):
        print("="*80)
        print("MACHINE LEARNING: TRAINING MODELS FROM SQL DATA")
        print("="*80)
        print()
        
        print("Loading data from SQL Server...")
        df = self.connector.get_ml_features()
        print(f"Loaded {len(df)} records")
        
        print("\nEngineering features...")
        df['is_profitable'] = df['IsProfitable']
        df['log_size'] = np.log1p(df['SizeUSD'])
        
        self.le_coin = LabelEncoder()
        self.le_sentiment = LabelEncoder()
        
        df['coin_encoded'] = self.le_coin.fit_transform(df['Coin'])
        
        df_clean = df.dropna(subset=['SentimentValue'])
        print(f"Clean dataset: {len(df_clean)} records")
        
        self.train_classification_model(df_clean)
        self.train_regression_model(df_clean)
        self.generate_strategy_recommendations(df_clean)
        
        self.save_model_results()
        
        print("\n" + "="*80)
        print("MACHINE LEARNING TRAINING COMPLETE!")
        print("="*80)
    
    def train_classification_model(self, df):
        print("\n" + "="*80)
        print("MODEL 1: BINARY CLASSIFICATION - PREDICT PROFITABLE TRADE")
        print("="*80)
        
        self.feature_cols = [
            'SentimentValue', 'SizeUSD', 'log_size',
            'IsLong', 'IsShort', 'IsSpot', 
            'TradeHour', 'DayOfWeek', 'TradeMonth',
            'coin_encoded'
        ]
        
        X = df[self.feature_cols]
        y = df['is_profitable']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        print("\nTraining Random Forest Classifier...")
        self.rf_clf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        )
        self.rf_clf.fit(X_train, y_train)
        
        y_pred = self.rf_clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        self.classification_accuracy = accuracy
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Not Profitable', 'Profitable']))
        
        self.feature_importance_clf = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.rf_clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(self.feature_importance_clf.to_string(index=False))
        
        return accuracy
    
    def train_regression_model(self, df):
        print("\n" + "="*80)
        print("MODEL 2: REGRESSION - PREDICT PnL AMOUNT")
        print("="*80)
        
        df_reg = df[(df['ClosedPnL'] > -10000) & (df['ClosedPnL'] < 10000)]
        print(f"\nFiltered dataset: {len(df_reg):,} records")
        
        X_reg = df_reg[self.feature_cols]
        y_reg = df_reg['ClosedPnL']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        print("\nTraining Gradient Boosting Regressor...")
        self.gb_reg = GradientBoostingRegressor(
            n_estimators=100, 
            max_depth=5, 
            random_state=42
        )
        self.gb_reg.fit(X_train, y_train)
        
        y_pred = self.gb_reg.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        
        print(f"\nModel Performance:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: ${mae:.2f}")
        
        self.feature_importance_reg = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.gb_reg.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (Regression):")
        print(self.feature_importance_reg.to_string(index=False))
        
        return rmse, r2
    
    def generate_strategy_recommendations(self, df):
        print("\n" + "="*80)
        print("MODEL 3: STRATEGY RECOMMENDATIONS BY SENTIMENT")
        print("="*80)
        
        query = """
        WITH RankedStrategies AS (
            SELECT 
                SentimentClass,
                Direction,
                Coin,
                SizeCategory,
                COUNT(*) AS TradeCount,
                AVG(ClosedPnL) AS AvgPnL,
                SUM(ClosedPnL) AS TotalPnL,
                CAST(SUM(CASE WHEN IsProfitable = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS DECIMAL(5,2)) AS WinRate,
                ROW_NUMBER() OVER (PARTITION BY SentimentClass ORDER BY AVG(ClosedPnL) DESC) AS PnLRank
            FROM vw_TraderSentimentAnalysis
            WHERE SentimentClass IS NOT NULL
            GROUP BY SentimentClass, Direction, Coin, SizeCategory
            HAVING COUNT(*) >= 20
        )
        SELECT 
            SentimentClass,
            Direction AS BestDirection,
            Coin AS BestCoin,
            SizeCategory AS BestSizeCategory,
            TradeCount,
            AvgPnL,
            TotalPnL,
            WinRate
        FROM RankedStrategies
        WHERE PnLRank = 1
        ORDER BY SentimentClass
        """
        
        strategies = self.connector.execute_query(query)
        
        print("\nOptimal Strategy for Each Sentiment:")
        print(strategies.to_string(index=False))
        
        self.strategies = strategies
        return strategies
    
    def predict_scenarios(self):
        print("\n" + "="*80)
        print("PREDICTION EXAMPLES")
        print("="*80)
        
        scenarios = [
            {
                'name': 'Extreme Greed + Long BTC + Large Size',
                'SentimentValue': 85,
                'SizeUSD': 5000,
                'IsLong': 1,
                'IsShort': 0,
                'IsSpot': 0,
                'Coin': 'BTC'
            },
            {
                'name': 'Extreme Fear + Short BTC + Medium Size',
                'SentimentValue': 15,
                'SizeUSD': 500,
                'IsLong': 0,
                'IsShort': 1,
                'IsSpot': 0,
                'Coin': 'BTC'
            },
            {
                'name': 'Neutral + Spot ETH + Small Size',
                'SentimentValue': 50,
                'SizeUSD': 100,
                'IsLong': 0,
                'IsShort': 0,
                'IsSpot': 1,
                'Coin': 'ETH'
            }
        ]
        
        print("\nScenario Predictions:")
        print("-" * 80)
        
        for scenario in scenarios:
            coin_encoded = 0
            if scenario['Coin'] in self.le_coin.classes_:
                coin_encoded = self.le_coin.transform([scenario['Coin']])[0]
            
            features = np.array([[
                scenario['SentimentValue'],
                scenario['SizeUSD'],
                np.log1p(scenario['SizeUSD']),
                scenario['IsLong'],
                scenario['IsShort'],
                scenario['IsSpot'],
                12,
                3,
                6,
                coin_encoded
            ]])
            
            profit_prob = self.rf_clf.predict_proba(features)[0][1]
            predicted_pnl = self.gb_reg.predict(features)[0]
            
            print(f"\n{scenario['name']}:")
            print(f"  Probability of Profit: {profit_prob*100:.1f}%")
            print(f"  Predicted PnL: ${predicted_pnl:.2f}")
            print(f"  Recommendation: {'TRADE' if profit_prob > 0.5 else 'AVOID'}")
    
    def save_model_results(self):
        results = {
            'classification_model': {
                'accuracy': float(self.classification_accuracy) if hasattr(self, 'classification_accuracy') else 0,
                'feature_importance': self.feature_importance_clf.to_dict('records') if hasattr(self, 'feature_importance_clf') else []
            },
            'regression_model': {
                'feature_importance': self.feature_importance_reg.to_dict('records') if hasattr(self, 'feature_importance_reg') else []
            },
            'strategies': self.strategies.to_dict('records') if hasattr(self, 'strategies') else []
        }
        
        with open('ml_model_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✓ Model results saved to: ml_model_results.json")


if __name__ == "__main__":
    print("Connecting to database...")
    connector = SQLConnector(user='root', password='@Maha2004')
    
    if connector.connect():
        ml_engine = MLEngine(connector)
        ml_engine.train_all_models()
        ml_engine.predict_scenarios()
        connector.disconnect()
    else:
        print("Failed to connect to database. Please run sql_connector.py first.")

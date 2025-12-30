import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sql_connector import SQLConnector
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

class VisualizationEngine:
    
    def __init__(self, connector):
        self.connector = connector
        self.output_dir = 'visualizations'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_all_visualizations(self):
        print("="*80)
        print("CREATING VISUALIZATIONS FROM SQL DATA")
        print("="*80)
        print()
        
        self.viz_1_performance_by_sentiment()
        self.viz_2_long_vs_short()
        self.viz_3_time_series()
        self.viz_4_top_coins()
        self.viz_5_correlation_heatmap()
        self.viz_6_distribution_analysis()
        
        print("\n" + "="*80)
        print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
        print("="*80)
        print(f"\nSaved in: {self.output_dir}/")
    
    def viz_1_performance_by_sentiment(self):
        print("Creating Visualization 1: Performance by Sentiment...")
        
        df = self.connector.get_performance_by_sentiment()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Trader Performance by Market Sentiment Classification', 
                     fontsize=16, fontweight='bold')
        
        colors = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c']
        
        ax1 = axes[0, 0]
        df_sorted = df.sort_values('AvgPnL')
        ax1.barh(df_sorted['SentimentClass'], df_sorted['AvgPnL'], color=colors)
        ax1.set_title('Average PnL by Sentiment', fontweight='bold')
        ax1.set_xlabel('Average Closed PnL ($)')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        for i, v in enumerate(df_sorted['AvgPnL'].values):
            ax1.text(v, i, f' ${v:.2f}', va='center')
        
        ax2 = axes[0, 1]
        df_sorted = df.sort_values('TotalTrades', ascending=False)
        ax2.bar(range(len(df_sorted)), df_sorted['TotalTrades'], color=colors)
        ax2.set_xticks(range(len(df_sorted)))
        ax2.set_xticklabels(df_sorted['SentimentClass'], rotation=45)
        ax2.set_title('Number of Trades by Sentiment', fontweight='bold')
        ax2.set_ylabel('Number of Trades')
        for i, v in enumerate(df_sorted['TotalTrades'].values):
            ax2.text(i, v, f'{int(v):,}', ha='center', va='bottom')
        
        ax3 = axes[1, 0]
        df_sorted = df.sort_values('WinRate')
        ax3.barh(df_sorted['SentimentClass'], df_sorted['WinRate'], color=colors)
        ax3.set_title('Win Rate by Sentiment', fontweight='bold')
        ax3.set_xlabel('Win Rate (%)')
        ax3.axvline(x=50, color='black', linestyle='--', alpha=0.5)
        for i, v in enumerate(df_sorted['WinRate'].values):
            ax3.text(v, i, f' {v:.1f}%', va='center')
        
        ax4 = axes[1, 1]
        df_sorted = df.sort_values('TotalPnL')
        ax4.barh(df_sorted['SentimentClass'], df_sorted['TotalPnL'], color=colors)
        ax4.set_title('Total PnL by Sentiment', fontweight='bold')
        ax4.set_xlabel('Total Closed PnL ($)')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        for i, v in enumerate(df_sorted['TotalPnL'].values):
            ax4.text(v, i, f' ${v:,.0f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/1_performance_by_sentiment.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {self.output_dir}/1_performance_by_sentiment.png")
        plt.close()
    
    def viz_2_long_vs_short(self):
        print("Creating Visualization 2: Long vs Short Performance...")
        
        df = self.connector.get_long_vs_short()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Long vs Short Performance by Market Sentiment', 
                     fontsize=16, fontweight='bold')
        
        pivot_pnl = df.pivot(index='SentimentClass', columns='PositionType', values='AvgPnL')
        pivot_count = df.pivot(index='SentimentClass', columns='PositionType', values='TotalTrades')
        pivot_total = df.pivot(index='SentimentClass', columns='PositionType', values='TotalPnL')
        
        ax1 = axes[0, 0]
        pivot_pnl.plot(kind='bar', ax=ax1, color=['#2196F3', '#FF5722'])
        ax1.set_title('Average PnL: Long vs Short', fontweight='bold')
        ax1.set_ylabel('Average Closed PnL ($)')
        ax1.set_xlabel('Sentiment')
        ax1.legend(title='Position Type')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax2 = axes[0, 1]
        pivot_count.plot(kind='bar', ax=ax2, color=['#2196F3', '#FF5722'])
        ax2.set_title('Trade Count: Long vs Short', fontweight='bold')
        ax2.set_ylabel('Number of Trades')
        ax2.set_xlabel('Sentiment')
        ax2.legend(title='Position Type')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3 = axes[1, 0]
        pivot_total.plot(kind='bar', ax=ax3, color=['#2196F3', '#FF5722'])
        ax3.set_title('Total PnL: Long vs Short', fontweight='bold')
        ax3.set_ylabel('Total Closed PnL ($)')
        ax3.set_xlabel('Sentiment')
        ax3.legend(title='Position Type')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax4 = axes[1, 1]
        sentiments = pivot_pnl.index
        x = np.arange(len(sentiments))
        width = 0.35
        ax4.bar(x - width/2, pivot_pnl['Long'], width, label='Long', color='#2196F3')
        ax4.bar(x + width/2, pivot_pnl['Short'], width, label='Short', color='#FF5722')
        ax4.set_title('Avg PnL Comparison', fontweight='bold')
        ax4.set_ylabel('Average PnL ($)')
        ax4.set_xlabel('Sentiment')
        ax4.set_xticks(x)
        ax4.set_xticklabels(sentiments, rotation=45)
        ax4.legend()
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/2_long_vs_short_sentiment.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {self.output_dir}/2_long_vs_short_sentiment.png")
        plt.close()
    
    def viz_3_time_series(self):
        print("Creating Visualization 3: Time Series Analysis...")
        
        df = self.connector.get_time_series_data()
        df['TradeDate'] = pd.to_datetime(df['TradeDate'])
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Time Series: Sentiment and Trading Performance', 
                     fontsize=16, fontweight='bold')
        
        ax1 = axes[0]
        ax1.plot(df['TradeDate'], df['DailyPnL'], color='#1976D2', linewidth=1.5)
        ax1.fill_between(df['TradeDate'], 0, df['DailyPnL'], 
                          where=df['DailyPnL']>=0, color='#4CAF50', alpha=0.3)
        ax1.fill_between(df['TradeDate'], 0, df['DailyPnL'], 
                          where=df['DailyPnL']<0, color='#F44336', alpha=0.3)
        ax1.set_title('Daily Closed PnL Over Time', fontweight='bold')
        ax1.set_ylabel('Daily PnL ($)')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(df['TradeDate'], df['AvgSentiment'], color='#FF6F00', linewidth=1.5)
        ax2.fill_between(df['TradeDate'], df['AvgSentiment'], alpha=0.3, color='#FF6F00')
        ax2.set_title('Market Sentiment Value Over Time', fontweight='bold')
        ax2.set_ylabel('Sentiment Value')
        ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Neutral (50)')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[2]
        ax3.bar(df['TradeDate'], df['DailyVolume'], color='#7B1FA2', alpha=0.6)
        ax3.set_title('Daily Trading Volume (USD)', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volume ($)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/3_time_series_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {self.output_dir}/3_time_series_analysis.png")
        plt.close()
    
    def viz_4_top_coins(self):
        print("Creating Visualization 4: Top Coins Performance...")
        
        df = self.connector.get_top_coins_performance(top_n=5)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Top 5 Coins Performance by Market Sentiment', 
                     fontsize=16, fontweight='bold')
        
        coins = df['Coin'].unique()
        colors_map = {'Extreme Fear': '#d32f2f', 'Fear': '#f57c00', 'Neutral': '#fbc02d', 
                      'Greed': '#7cb342', 'Extreme Greed': '#388e3c'}
        
        for idx, coin in enumerate(coins[:5]):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            coin_data = df[df['Coin'] == coin]
            colors = [colors_map.get(s, '#999999') for s in coin_data['SentimentClass']]
            
            ax.bar(range(len(coin_data)), coin_data['AvgPnL'], color=colors)
            ax.set_title(f'{coin}', fontweight='bold', fontsize=12)
            ax.set_xticks(range(len(coin_data)))
            ax.set_xticklabels(coin_data['SentimentClass'], rotation=45, ha='right')
            ax.set_ylabel('Avg PnL ($)')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
        
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/4_top_coins_sentiment.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {self.output_dir}/4_top_coins_sentiment.png")
        plt.close()
    
    def viz_5_correlation_heatmap(self):
        print("Creating Visualization 5: Correlation Analysis...")
        
        df = self.connector.get_ml_features()
        
        corr_data = df[['SentimentValue', 'SizeUSD', 'IsLong', 'IsShort', 
                        'IsSpot', 'TradeHour', 'DayOfWeek', 'TradeMonth', 
                        'ClosedPnL', 'IsProfitable']].copy()
        
        correlation = corr_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix: Sentiment vs Trading Metrics', 
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/5_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {self.output_dir}/5_correlation_heatmap.png")
        plt.close()
    
    def viz_6_distribution_analysis(self):
        print("Creating Visualization 6: Distribution Analysis...")
        
        df = self.connector.get_all_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distribution Analysis by Sentiment', fontsize=16, fontweight='bold')
        
        sentiments = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
        colors_map = {'Extreme Fear': '#d32f2f', 'Fear': '#f57c00', 'Neutral': '#fbc02d', 
                      'Greed': '#7cb342', 'Extreme Greed': '#388e3c'}
        
        ax1 = axes[0, 0]
        for sentiment in sentiments:
            data = df[df['SentimentClass'] == sentiment]['ClosedPnL']
            data = data[(data > -1000) & (data < 1000)]
            ax1.hist(data, bins=50, alpha=0.5, label=sentiment, color=colors_map[sentiment])
        ax1.set_title('PnL Distribution by Sentiment (Filtered)', fontweight='bold')
        ax1.set_xlabel('Closed PnL ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        ax2 = axes[0, 1]
        pnl_by_sentiment = [df[df['SentimentClass'] == s]['ClosedPnL'] for s in sentiments]
        bp = ax2.boxplot(pnl_by_sentiment, labels=sentiments, patch_artist=True)
        for patch, sentiment in zip(bp['boxes'], sentiments):
            patch.set_facecolor(colors_map[sentiment])
        ax2.set_title('PnL Distribution Box Plot', fontweight='bold')
        ax2.set_xlabel('Sentiment')
        ax2.set_ylabel('Closed PnL ($)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(-5000, 5000)
        
        ax3 = axes[1, 0]
        for sentiment in sentiments:
            data = df[df['SentimentClass'] == sentiment]['SizeUSD']
            data = data[data < 10000]
            ax3.hist(data, bins=50, alpha=0.5, label=sentiment, color=colors_map[sentiment])
        ax3.set_title('Trade Size Distribution (Filtered)', fontweight='bold')
        ax3.set_xlabel('Trade Size (USD)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        ax4 = axes[1, 1]
        ax4.hist(df['SentimentValue'].dropna(), bins=50, color='#FF6F00', edgecolor='black')
        ax4.set_title('Overall Sentiment Value Distribution', fontweight='bold')
        ax4.set_xlabel('Sentiment Value')
        ax4.set_ylabel('Frequency')
        ax4.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Neutral (50)')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/6_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {self.output_dir}/6_distribution_analysis.png")
        plt.close()


if __name__ == "__main__":
    print("Connecting to database...")
    connector = SQLConnector(user='root', password='@Maha2004')
    
    if connector.connect():
        viz_engine = VisualizationEngine(connector)
        viz_engine.create_all_visualizations()
        connector.disconnect()
    else:
        print("Failed to connect to database. Please run sql_connector.py first to setup the database.")
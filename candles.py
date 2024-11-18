import yfinance as yf
import talib as ta
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.dates import date2num
import matplotlib.dates as mdates

# Define candlestick patterns
patterns = {
    "Bullish Reversal": {
        "Hammer": ta.CDLHAMMER,
        "Inverted Hammer": ta.CDLINVERTEDHAMMER,
        # Add other bullish patterns...
    },
    "Bearish Reversal": {
        "Hanging Man": ta.CDLHANGINGMAN,
        "Shooting Star": ta.CDLSHOOTINGSTAR,
        # Add other bearish patterns...
    }
}

def detect_patterns(df):
    for category, patterns_dict in patterns.items():
        for pattern_name, pattern_func in patterns_dict.items():
            df[pattern_name] = pattern_func(df['Open'], df['High'], df['Low'], df['Close'])
    return df

def candles_driver(df, ticker, temp_dir_path):
    df = detect_patterns(df)
    df_last_50 = df.tail(50).copy()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    df_last_50['Date'] = pd.to_datetime(df_last_50.index)
    df_last_50['Date_num'] = date2num(df_last_50['Date'])
    
    for _, row in df_last_50.iterrows():
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        ax.bar(row['Date_num'], row['Close'] - row['Open'], width=0.6, bottom=row['Open'], color=color)
        ax.plot([row['Date_num'], row['Date_num']], [row['Low'], row['High']], color='black')
    
    # Add detected patterns
    for pattern_name, _ in patterns.items():
        for pattern_subtype in df_last_50.columns:
            pattern_data = df_last_50[df_last_50[pattern_subtype] != 0]
            if not pattern_data.empty:
                ax.scatter(pattern_data['Date_num'], pattern_data['Close'], s=100, label=pattern_subtype)
    
    # Finalize plot
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.title(f"{ticker} - Last 50 Days with Detected Patterns")
    plt.grid(True)
    plt.legend()
    # Save the plot
    plot_path = os.path.join(temp_dir_path, 'candlestick_patterns.png')
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return df, plot_path

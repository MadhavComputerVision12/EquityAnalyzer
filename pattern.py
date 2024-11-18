import pandas as pd
import numpy as np
import yfinance as yf
from get_data import *
import plotly.tools as tls
import plotly.io as pio
import matplotlib.pyplot as plt

def detect_head_shoulder(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

    mask_head_shoulder = (
        (df['High'].shift(1) < df['High']) &
        (df['High'].shift(-1) < df['High']) &
        (df['high_roll_max'] == df['High'])
    )

    mask_inv_head_shoulder = (
        (df['Low'].shift(1) > df['Low']) &
        (df['Low'].shift(-1) > df['Low']) &
        (df['low_roll_min'] == df['Low'])
    )

    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df

def detect_multiple_tops_bottoms(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['close_roll_max'] = df['Close'].rolling(window=roll_window).max()
    df['close_roll_min'] = df['Close'].rolling(window=roll_window).min()

    mask_top = (df['High'] == df['high_roll_max']) & (df['Close'].shift(1) > df['Close'])
    mask_bottom = (df['Low'] == df['low_roll_min']) & (df['Close'].shift(1) < df['Close'])

    df['multiple_top_bottom_pattern'] = np.nan
    df.loc[mask_top, 'multiple_top_bottom_pattern'] = 'Multiple Top'
    df.loc[mask_bottom, 'multiple_top_bottom_pattern'] = 'Multiple Bottom'
    return df

def calculate_support_resistance(df, window=3):
    roll_window = window
    std_dev = 2

    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mean_high = df['High'].rolling(window=roll_window).mean()
    std_high = df['High'].rolling(window=roll_window).std()
    mean_low = df['Low'].rolling(window=roll_window).mean()
    std_low = df['Low'].rolling(window=roll_window).std()

    df['support'] = mean_low - std_dev * std_low
    df['resistance'] = mean_high + std_dev * std_high
    return df

def detect_triangle_pattern(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

    mask_asc = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    mask_desc = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['Close'] < df['Close'].shift(1))

    df['triangle_pattern'] = np.nan
    df.loc[mask_asc, 'triangle_pattern'] = 'Ascending Triangle'
    df.loc[mask_desc, 'triangle_pattern'] = 'Descending Triangle'
    return df

def detect_wedge(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)

    mask_wedge_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_wedge_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)

    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'
    return df

def detect_channel(df, window=3):
    roll_window = window
    channel_range = 0.1

    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)

    mask_channel_up = (
        (df['high_roll_max'] >= df['High'].shift(1)) &
        (df['low_roll_min'] <= df['Low'].shift(1)) &
        (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min']) / 2) &
        (df['trend_high'] == 1) &
        (df['trend_low'] == 1)
    )

    mask_channel_down = (
        (df['high_roll_max'] <= df['High'].shift(1)) &
        (df['low_roll_min'] >= df['Low'].shift(1)) &
        (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min']) / 2) &
        (df['trend_high'] == -1) &
        (df['trend_low'] == -1)
    )

    df['channel_pattern'] = np.nan
    df.loc[mask_channel_up, 'channel_pattern'] = 'Channel Up'
    df.loc[mask_channel_down, 'channel_pattern'] = 'Channel Down'
    return df

def detect_double_top_bottom(df, window=3, threshold=0.05):
    roll_window = window
    range_threshold = threshold

    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

    mask_double_top = (
        (df['High'] == df['high_roll_max']) &
        (df['High'].shift(1) < df['High']) &
        (df['High'].shift(-1) < df['High']) &
        ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1)) / 2) &
        ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1)) / 2)
    )

    mask_double_bottom = (
        (df['Low'] == df['low_roll_min']) &
        (df['Low'].shift(1) > df['Low']) &
        (df['Low'].shift(-1) > df['Low']) &
        ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1)) / 2) &
        ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1)) / 2)
    )

    df['double_pattern'] = np.nan
    df.loc[mask_double_top, 'double_pattern'] = 'Double Top'
    df.loc[mask_double_bottom, 'double_pattern'] = 'Double Bottom'
    return df

def detect_trendline(df, window=2):
    df['slope'] = np.nan
    df['intercept'] = np.nan

    for i in range(window, len(df)):
        x = np.array(range(i-window, i))
        y = df['Close'][i-window:i]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], 'slope'] = m
        df.at[df.index[i], 'intercept'] = c

    mask_support = df['slope'] > 0
    mask_resistance = df['slope'] < 0

    df['support'] = np.nan
    df['resistance'] = np.nan
    df.loc[mask_support, 'support'] = df['Close'] * df['slope'] + df['intercept']
    df.loc[mask_resistance, 'resistance'] = df['Close'] * df['slope'] + df['intercept']

    return df

def find_pivots(df):
    high_diffs = df['High'].diff()
    low_diffs = df['Low'].diff()

    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)

    df['signal'] = ''
    df.loc[higher_high_mask, 'signal'] = 'HH'
    df.loc[lower_low_mask, 'signal'] = 'LL'
    df.loc[lower_high_mask, 'signal'] = 'LH'
    df.loc[higher_low_mask, 'signal'] = 'HL'

    return df

import pandas as pd
import numpy as np

def detect_all_patterns(df, window=3):
    # Head and Shoulder & Inverse Head and Shoulder
    df = detect_head_shoulder(df, window=window)
    
    # Multiple Tops and Bottoms
    df = detect_multiple_tops_bottoms(df, window=window)
    
    # Support and Resistance Levels
    df = calculate_support_resistance(df, window=window)
    
    # Triangle Patterns (Ascending and Descending)
    df = detect_triangle_pattern(df, window=window)
    
    # Wedge Patterns (Wedge Up and Wedge Down)
    df = detect_wedge(df, window=window)
    
    # Channel Patterns (Channel Up and Channel Down)
    df = detect_channel(df, window=window)
    
    # Double Top and Double Bottom Patterns
    df = detect_double_top_bottom(df, window=window)
    
    # Trendlines Detection
    df = detect_trendline(df, window=window)
    
    # Pivots (Higher Highs, Lower Lows, Lower Highs, Higher Lows)
    df = find_pivots(df)

    return df

import mplfinance as mpf
import numpy as np

def initialize_pattern_columns(df):
    pattern_columns = [
        'head_shoulder_pattern', 'multiple_top_bottom_pattern', 'triangle_pattern',
        'wedge_pattern', 'channel_pattern', 'double_pattern'
    ]
    for col in pattern_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib.dates import date2num

def patterns_driver(df, ticker, save_path="pattern_chart.png"):
    df = detect_all_patterns(df)
    # Restrict to the last 50 days of data
    df_last_50 = df.tail(50).copy()
    
    # Convert dates to numerical format for plotting
    df_last_50['Date'] = pd.to_datetime(df_last_50.index)
    df_last_50['Date_num'] = date2num(df_last_50['Date'])
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the candlestick chart
    width = 0.6  # Width of the candlestick bars
    for index, row in df_last_50.iterrows():
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        ax.plot([row['Date_num'], row['Date_num']], [row['Low'], row['High']], color='black')  # High-low line
        ax.bar(row['Date_num'], row['Close'] - row['Open'], width, bottom=row['Open'], color=color)  # Open-close bar

    # Plot the detected patterns as scatter plots
    patterns = {
        'Head and Shoulder': ('head_shoulder_pattern', 'r', '^'),
        'Inverse Head and Shoulder': ('head_shoulder_pattern', 'g', 'v'),
        'Multiple Top': ('multiple_top_bottom_pattern', 'orange', '^'),
        'Multiple Bottom': ('multiple_top_bottom_pattern', 'purple', 'v'),
        'Ascending Triangle': ('triangle_pattern', 'blue', '^'),
        'Descending Triangle': ('triangle_pattern', 'cyan', 'v'),
        'Wedge Up': ('wedge_pattern', 'lime', '^'),
        'Wedge Down': ('wedge_pattern', 'brown', 'v'),
        'Channel Up': ('channel_pattern', 'gold', '^'),
        'Channel Down': ('channel_pattern', 'silver', 'v'),
        'Double Top': ('double_pattern', 'pink', '^'),
        'Double Bottom': ('double_pattern', 'darkblue', 'v')
    }

    for label, (col, color, marker) in patterns.items():
        pattern_data = df_last_50[df_last_50[col] == label]
        ax.scatter(pattern_data['Date_num'], pattern_data['Close'], color=color, marker=marker, s=100, label=label)

    # Formatting the plot
    ax.xaxis_date()  # Interpret the x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.title(f"{ticker} - Last 50 Days with Detected Patterns")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(save_path)
    print(f"Chart saved to {save_path}")

    # Display the plot
    plotly_fig = tls.mpl_to_plotly(plt.gcf())

    # Convert Plotly figure to HTML
    plot_html = pio.to_html(plotly_fig, full_html=False)

    return df,plot_html

# Example usage after detecting patterns
#plot_patterns_with_matplotlib(result_df, ticker="BATAINDIA.NS", save_path="BATAINDIA_patterns.png")

# camarilla_levels.py

import os
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

# Constants for plotting
PLOT_STYLE = 'charles'
LINE_STYLE = '--'
LINE_COLOR = 'red'
TEXT_COLOR = 'black'
FONT_SIZE = 10
DPI = 100
NUM_CANDLES = 50

def calculate_camarilla_levels(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Camarilla levels based on the highest, lowest, and last closing prices.

    :param df: DataFrame containing the stock price data with 'High', 'Low', 'Close' columns.
    :return: A Series containing the calculated Camarilla levels.
    """
    high = df['High'].max()
    low = df['Low'].min()
    close = df['Close'].iloc[-1]
    
    # Calculate Camarilla levels
    r4 = close + (high - low) * 1.1 / 0.55
    r3 = close + (high - low) * 1.1 / 1.1
    r2 = close + (high - low) * 1.1 / 6
    r1 = close + (high - low) * 1.1 / 12
    s1 = close - (high - low) * 1.1 / 12
    s2 = close - (high - low) * 1.1 / 6
    s3 = close - (high - low) * 1.1 / 1.1
    s4 = close - (high - low) * 1.1 / 0.55
    
    levels = [r4, r3, r2, r1, s1, s2, s3, s4]
    return pd.Series(levels, index=['R4', 'R3', 'R2', 'R1', 'S1', 'S2', 'S3', 'S4'])

def plot_camarilla_levels(df: pd.DataFrame, temp_dir_path: str) -> tuple:
    """
    Plot stock prices and overlay Camarilla levels on the chart.

    :param df: DataFrame containing the stock price data.
    :param temp_dir_path: Directory path where the plot image will be saved.
    :return: A tuple containing the Camarilla levels as a Series and the file path of the saved plot image.
    """
    # Calculate Camarilla levels
    camarilla_levels = calculate_camarilla_levels(df)

    # Focus on the last NUM_CANDLES candles
    df = df.tail(NUM_CANDLES)
    df = df.tail(30)
    # Create a candlestick plot
    fig, ax = mpf.plot(df, type='candle', style=PLOT_STYLE, title='Share Prices with Camarilla Levels',
                       ylabel='Price', returnfig=True)
    
    # Overlay Camarilla levels on the plot
    for level in camarilla_levels:
        ax[0].axhline(level, linestyle=LINE_STYLE, alpha=0.5, color=LINE_COLOR)
    
    # Add text labels to the right of each level line
    x_position = ax[0].get_xlim()[1]
    for i, level in enumerate(camarilla_levels):
        ax[0].text(x_position, level, f'Camarilla {i+1}', va='center', ha='right', 
                   fontsize=FONT_SIZE, color=TEXT_COLOR)
    
    # Save the plot to a file
    plot_path = os.path.join(temp_dir_path, 'camarilla_levels_plot.png')
    fig.savefig(plot_path, bbox_inches='tight', dpi=DPI)
    plt.close()  # Close the figure to free memory
    print(camarilla_levels)
    return camarilla_levels, plot_path

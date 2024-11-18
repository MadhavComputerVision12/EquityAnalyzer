import numpy as np
import pandas as pd
import mplfinance as mpf
import os
import matplotlib.pyplot as plt

# Function to calculate EMA manually
def exponential_moving_average(prices, period):
    ema = np.full(len(prices), np.nan)
    sma = np.mean(prices[:period])
    ema[period - 1] = sma
    weighting_factor = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * weighting_factor + ema[i - 1]
    return ema

def ema_driver(data, temp_dir_path):
    df = pd.DataFrame(data)
    periods = [5, 20, 50, 100, 200]
    
    for period in periods:
        df[f'EMA_{period}'] = exponential_moving_average(df['Close'], period)
    df = df[-50:]
    ema_colors = ['blue', 'red', 'green', 'orange', 'purple']
    ema_plots = [mpf.make_addplot(df[f'EMA_{period}'], color=color) for period, color in zip(periods, ema_colors)]
    
    fig, _ = mpf.plot(df, type='candle', style='charles', addplot=ema_plots,
                      title="Share Prices with EMAs", ylabel='Price', returnfig=True)
    plt.legend()
    # Save the plot
    plot_path = os.path.join(temp_dir_path, 'ema_plot.png')
    fig.savefig(plot_path, bbox_inches='tight',dpi = 100)
    plt.close()
    return df, plot_path

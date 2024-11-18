import pandas as pd
from datetime import datetime, timedelta
import mplfinance as mpf
import matplotlib.pyplot as plt
import tempfile
import os

def ma_driver(data, temp_dir_path):
    # Convert the 'Date' column to datetime if it's not already and set it as the index
    df = pd.DataFrame(data)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    print(df.head())  # Check if data is fetched correctly

    # Filter data for one year back from today
    one_year_ago = datetime.now() - timedelta(days=365)
    one_year_data = df.loc[df.index >= one_year_ago].copy()

    # Calculate Simple Moving Averages (SMA)
    one_year_data['SMA_5'] = one_year_data['Close'].rolling(window=5).mean()
    one_year_data['SMA_20'] = one_year_data['Close'].rolling(window=20).mean()
    one_year_data['SMA_50'] = one_year_data['Close'].rolling(window=50).mean()
    one_year_data['SMA_100'] = one_year_data['Close'].rolling(window=100).mean()
    one_year_data['SMA_200'] = one_year_data['Close'].rolling(window=200).mean()
    one_year_data = one_year_data.tail(50)
    # Create additional plots for each moving average with specified colors
    apds = [
        mpf.make_addplot(one_year_data['SMA_5'], color='blue'),
        mpf.make_addplot(one_year_data['SMA_20'], color='red'),
        mpf.make_addplot(one_year_data['SMA_50'], color='green'),
        mpf.make_addplot(one_year_data['SMA_100'], color='violet'),
        mpf.make_addplot(one_year_data['SMA_200'], color='black')
    ]

    # Plot the candlestick chart with moving averages
    fig, axlist = mpf.plot(one_year_data, type='candle', volume=True, addplot=apds,
                           title='Candlestick Chart with 5, 20, 50, 100, and 200-day SMA',
                           style='yahoo', returnfig=True)

    # Manually add the legend with the specified colors
    axlist[0].legend(['5-Day SMA', '20-Day SMA', '50-Day SMA', '100-Day SMA', '200-Day SMA'],
                     loc='upper left', fontsize='x-small')

    # Save the plot to the temporary directory
    plot_path = os.path.join(temp_dir_path, 'candlestick_sma_plot.png')
    # Save the plot with lower DPI to reduce file size
    fig.savefig(plot_path, bbox_inches='tight', dpi=100)

    plt.close()
    # Return the DataFrame and the plot path
    return one_year_data, plot_path

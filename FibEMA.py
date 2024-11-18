import pandas as pd
import mplfinance as mpf
import os
import matplotlib.pyplot as plt

# Function to calculate EMA for given length
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# Function to add Fibonacci EMAs to DataFrame
def add_fib_ema(df):
    fib_lengths = [144, 233, 377, 610, 987, 1597, 2584]
    for length in fib_lengths:
        df[f'EMA_{length}'] = ema(df['Close'], length)
    print("FibEMA__________")
    print(df.iloc[-1])
    return df

def fibema_driver(data, temp_dir_path):
    df = pd.DataFrame(data)
    df_with_fib_ema = add_fib_ema(df)
    df_with_fib_ema = df_with_fib_ema[-50:]
    ema_colors = ['red', 'red', 'orange', 'orange', 'green', 'green', 'blue']
    ema_linewidths = [1, 2, 1, 2, 1, 2, 1]
    
    ema_plots = [
        mpf.make_addplot(df_with_fib_ema[f'EMA_{length}'], color=color, width=linewidth)
        for length, color, linewidth in zip([144, 233, 377, 610, 987, 1597, 2584], ema_colors, ema_linewidths)
    ]
    
    fig, _ = mpf.plot(df_with_fib_ema, type='candle', addplot=ema_plots, volume=True, style='yahoo',
                      title="Candlestick with Fibonacci EMAs", returnfig=True)
    
    # Save the plot
    plot_path = os.path.join(temp_dir_path, 'fibonacci_ema_plot.png')
    fig.savefig(plot_path, bbox_inches='tight',dpi = 100)
    plt.close()
    
    return df_with_fib_ema, plot_path

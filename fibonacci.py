import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to calculate Fibonacci retracement levels
def fibonacci_retracement(df):
    Low = df["Close"].min()
    High = df["Close"].max()
    Diff = High - Low
    
    # Fibonacci Levels
    Fib100 = High
    Fib764 = Low + (Diff * 0.764)
    Fib618 = Low + (Diff * 0.618)
    Fib50 = Low + (Diff * 0.5)
    Fib382 = Low + (Diff * 0.382)
    Fib236 = Low + (Diff * 0.236)
    Fib0 = Low
    
    return Fib100, Fib764, Fib618, Fib50, Fib382, Fib236, Fib0

# Function to plot Fibonacci retracement levels
def plot_fibonacci_retracement(df,temp_dir):
    # Calculate Fibonacci levels
    Fib100, Fib764, Fib618, Fib50, Fib382, Fib236, Fib0 = fibonacci_retracement(df)
    df = df.tail(50)
    # Plot adjusted close prices and Fibonacci levels
    plt.figure(figsize=(15, 8))  # Adjust figure size
    plt.plot(df["Close"], color="black", label="Price")
    
    # Plot Fibonacci lines
    plt.axhline(y=Fib100, color="limegreen", linestyle="-", label="100%")
    plt.axhline(y=Fib764, color="slateblue", linestyle="-", label="76.4%")
    plt.axhline(y=Fib618, color="mediumvioletred", linestyle="-", label="61.8%")
    plt.axhline(y=Fib50, color="gold", linestyle="-", label="50%")
    plt.axhline(y=Fib382, color="darkturquoise", linestyle="-", label="38.2%")
    plt.axhline(y=Fib236, color="darkturquoise", linestyle="-", label="23.6%")
    plt.axhline(y=Fib0, color="lightcoral", linestyle="-", label="0%")
    
    # Add labels and title
    plt.ylabel("Price")
    plt.xticks(rotation=90)
    plt.title("Fibonacci Retracement Levels")
    plt.legend()
    
    # Show the plot
    # Define the file path where the figure will be saved
    plot_path = os.path.join(temp_dir, 'fibonacci_retracement_plot.png')

    # Save the figure
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print([Fib100, Fib764, Fib618, Fib50, Fib382, Fib236, Fib0])
    return [Fib100, Fib764, Fib618, Fib50, Fib382, Fib236, Fib0],plot_path


# Example usage:
# Assuming df is your DataFrame with 'adjusted_close' column
# df = pd.DataFrame({"adjusted_close": ...})  # Populate with your data
# plot_fibonacci_retracement(df)

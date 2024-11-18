import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta



def get_adjusted_date():
    # Get the current date and time
    current_datetime = datetime.now()

    # Define the time for 3:30 AM today (early morning, not PM)
    three_thirty_am = current_datetime.replace(hour=3, minute=30, second=0, microsecond=0)

    # If the current time is earlier than 3:30 AM, subtract 1 day
    if current_datetime < three_thirty_am:
        adjusted_date = current_datetime - timedelta(days=1)
    else:
        adjusted_date = current_datetime

    # Format the date as yyyy-mm-dd
    
    return adjusted_date.strftime('%Y-%m-%d')


def get_data(symbol):
    if symbol == 'NIFTY_50':
        symbol = "^NSEI"
    elif symbol == 'BANK_NIFTY':
        symbol = '^NSEBANK'
    else:
        symbol = symbol+'.NS'
    nifty_ticker = symbol
    df = yf.download(nifty_ticker, start='2020-03-01', end=get_adjusted_date())
    return df
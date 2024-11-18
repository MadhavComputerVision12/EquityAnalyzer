from get_data import *
from MA import *
from EMA import *
from FibEMA import *
from fibonacci import *
from candles import *
from pattern import *
from camerilla import *
from astro import *


def caller_function(symbol,temp_dir_path):
    data = get_data(symbol)
    ma,ma_fig = ma_driver(data,temp_dir_path)
    ema,ema_fig = ema_driver(data,temp_dir_path)
    fibonacci,fib_fig = plot_fibonacci_retracement(data,temp_dir_path)
    candles,candle_fig = candles_driver(data, symbol, temp_dir_path)
    patterns,pattern_fig = patterns_driver(data, symbol, temp_dir_path)
    fibema,fibema_fig = fibema_driver(data,temp_dir_path)
    camerilla,camerilla_fig = plot_camarilla_levels(data,temp_dir_path)
    return data,ma,ma_fig,ema,ema_fig,fibonacci,fib_fig,candles,candle_fig,fibema,fibema_fig,camerilla,camerilla_fig

from flask import Flask, render_template, request, url_for
import os
import tempfile
from caller_function import *

app = Flask(__name__)
app.secret_key = b'Royalh'

# Create a static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')

@app.route('/input_process', methods=['POST'])
def input_process():
    if request.method == 'POST':
        symbol = request.form['symbol']
        data, ma, ma_fig, ema, ema_fig, fibonacci, fib_fig, candles, candle_fig, fibema, fibema_fig, camerilla, camerilla_fig = caller_function(symbol, 'static')
        ma = None

        plot_paths = {
            'ma': url_for('static', filename='candlestick_sma_plot.png'),
            'ema': url_for('static', filename='ema_plot.png'),
            'fibonacci': url_for('static', filename='fibonacci_retracement_plot.png'),
            'candles': url_for('static', filename='candlestick_patterns.png'),
            'fibema': url_for('static', filename='fibonacci_ema_plot.png'),
            'camerilla': url_for('static', filename='camarilla_levels_plot.png')
        }
        
        return render_template('index.html', symbol=symbol, plot_paths=plot_paths)

if __name__ == '__main__':
    app.run(debug=True)
